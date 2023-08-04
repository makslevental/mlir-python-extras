import inspect
import logging
from textwrap import dedent
from typing import Optional, Sequence

import libcst as cst
import libcst.matchers as m
from bytecode import ConcreteBytecode, ConcreteInstr
from mlir.dialects.scf import IfOp, ForOp
from mlir.ir import InsertionPoint, Value, OpResultList, OpResult

import mlir_utils.types as T
from mlir_utils.ast.canonicalize import (
    StrictTransformer,
    Canonicalizer,
    BytecodePatcher,
    OpCode,
)
from mlir_utils.ast.util import ast_call
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.scf import yield_ as yield__
from mlir_utils.util import (
    region_op,
    maybe_cast,
    _update_caller_vars,
    get_result_or_results,
    get_user_code_loc,
)

logger = logging.getLogger(__name__)


def _for(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    if isinstance(start, int):
        start = constant(start, index=True)
    if isinstance(stop, int):
        stop = constant(stop, index=True)
    if isinstance(step, int):
        step = constant(step, index=True)
    if loc is None:
        loc = get_user_code_loc()
    return ForOp(start, stop, step, iter_args, loc=loc, ip=ip)


for_ = region_op(_for, terminator=yield__)


def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    for_op = _for(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = maybe_cast(for_op.induction_variable)
    iter_args = tuple(map(maybe_cast, for_op.inner_iter_args))
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args
        elif len(iter_args) == 1:
            yield iv, iter_args[0]
        else:
            yield iv


def yield_(*args):
    if len(args) == 1 and isinstance(args[0], OpResultList):
        args = list(args[0])
    y = yield__(args)
    parent_op = y.operation.parent.opview
    if len(parent_op.results_):
        results = get_result_or_results(parent_op)
        assert (
            isinstance(results, (OpResult, OpResultList))
            or isinstance(results, list)
            and all(isinstance(r, OpResult) for r in results)
        ), f"api has changed: {results=}"
        if isinstance(results, OpResult):
            results = [results]
        unpacked_args = args
        if any(isinstance(a, OpResultList) for a in unpacked_args):
            assert len(unpacked_args) == 1
            unpacked_args = list(unpacked_args[0])

        assert len(results) == len(unpacked_args), f"{results=}, {unpacked_args=}"
        for i, r in enumerate(results):
            if r.type == T._placeholder_opaque_t():
                r.set_type(unpacked_args[i].type)

        results = maybe_cast(results)
        if len(results) > 1:
            return results
        return results[0]


def _if(cond, results_=None, *, has_else=False, loc=None, ip=None):
    if results_ is None:
        results_ = []
    if results_:
        has_else = True
    if loc is None:
        loc = get_user_code_loc()
    return IfOp(cond, results_, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)


class IfStack:
    __current_if_op: list[IfOp] = []
    __if_ip: list[InsertionPoint] = []

    @staticmethod
    def _repr_current_stacks():
        return dedent(
            f"""\
            {IfStack.__current_if_op}
            {IfStack.__if_ip}
        """
        )

    @staticmethod
    def __push_block_ip(block):
        ip = InsertionPoint(block)
        ip.__enter__()
        IfStack.__if_ip.append(ip)

    @staticmethod
    def push(cond: Value, results_=None, has_else=False):
        if results_ is None:
            results_ = []
        if results_:
            has_else = True
        assert isinstance(cond, Value), f"cond must be a mlir.Value: {cond=}"
        if_op = _if(cond, results_, has_else=has_else)
        cond.owner.move_before(if_op)

        IfStack.__current_if_op.append(if_op)
        IfStack.__push_block_ip(if_op.then_block)

        return maybe_cast(get_result_or_results(if_op))

    @staticmethod
    def pop_branch():
        ip = IfStack.__if_ip.pop()
        ip.__exit__(None, None, None)

    @staticmethod
    def push_else():
        if_op = IfStack.__current_if_op[-1]
        assert len(
            if_op.regions[1].blocks
        ), f"can't have else without bb in second region of {if_op=}"
        IfStack.__push_block_ip(if_op.else_block)
        return maybe_cast(get_result_or_results(if_op))

    @staticmethod
    def pop():
        if len(IfStack.__if_ip):
            ip = IfStack.__if_ip.pop()
            ip.__exit__(None, None, None)
        IfStack.__current_if_op.pop()

    @staticmethod
    def yield_(*args):
        if_op = IfStack.__current_if_op[-1]
        results = get_result_or_results(if_op)
        assert isinstance(
            results, (OpResult, OpResultList)
        ), f"api has changed: {results=}"
        if isinstance(results, OpResult):
            results = [results]
        unpacked_args = args
        if any(isinstance(a, OpResultList) for a in unpacked_args):
            assert len(unpacked_args) == 1
            unpacked_args = list(unpacked_args[0])

        assert len(results) == len(unpacked_args), f"{results=}, {unpacked_args=}"
        for i, r in enumerate(results):
            if r.type == T._placeholder_opaque_t():
                r.set_type(unpacked_args[i].type)

        yield_(*args)


# forward here for readability (and easier ast manipulation below)
def stack_if(*args, **kwargs):
    return IfStack.push(*args, **kwargs)


def unstack_if(cond: Value, results_=None, has_else=False):
    if results_ is None:
        results_ = []
    if results_:
        has_else = True
    assert isinstance(cond, Value), f"cond must be a mlir.Value: {cond=}"
    if_op = _if(cond, results_, has_else=has_else)
    cond.owner.move_before(if_op)

    ip = InsertionPoint(if_op.then_block)
    ip.__enter__()

    return ip, if_op


def end_branch():
    IfStack.pop_branch()


def unstack_end_branch(ip):
    ip.__exit__(None, None, None)


def else_():
    return IfStack.push_else()


def unstack_else_if(if_op):
    assert len(
        if_op.regions[1].blocks
    ), f"can't have else without bb in second region of {if_op=}"

    ip = InsertionPoint(if_op.else_block)
    ip.__enter__()
    return ip


def end_if():
    IfStack.pop()


def unstack_end_if(ip):
    ip.__exit__(None, None, None)


def insert_body_maybe_semicolon(
    node: cst.CSTNode, index: int, new_node: cst.CSTNode, before=False
):
    indented_block = node.body
    assert isinstance(
        indented_block, cst.IndentedBlock
    ), f"expected IndentedBlock, got {indented_block=}"
    body = list(indented_block.body)
    maybe_statement = body[index]
    if isinstance(maybe_statement, cst.SimpleStatementLine):
        # can append (with semicolon) to the simplestatement
        if before:
            maybe_statement_body = [cst.Expr(new_node)] + list(maybe_statement.body)
        else:
            maybe_statement_body = list(maybe_statement.body) + [cst.Expr(new_node)]
        return node.deep_replace(
            maybe_statement,
            maybe_statement.with_changes(body=maybe_statement_body),
        )

    # else have to create new statement at index (or append to body if -1)
    new_statement = cst.SimpleStatementLine([cst.Expr(new_node)])
    indented_block_body = list(indented_block.body)
    if index == -1:
        indented_block_body.append(new_statement)
    else:
        indented_block_body.insert(index, new_statement)
    return node.with_changes(
        body=indented_block.with_changes(body=indented_block_body),
    )


class ReplaceYieldWithSCFYield(StrictTransformer):
    @m.call_if_inside(m.If())
    @m.leave(m.Yield(value=m.Tuple()))
    def tuple_yield_inside_conditional(
        self, original_node: cst.Yield, _updated_node: cst.Yield
    ):
        args = [cst.Arg(e.value) for e in original_node.value.elements]
        return ast_call(yield_.__name__, args)

    @m.call_if_inside(m.If())
    @m.leave(m.Yield(value=~m.Tuple()))
    def single_yield_inside_conditional(
        self, original_node: cst.Yield, _updated_node: cst.Yield
    ):
        args = [cst.Arg(original_node.value)] if original_node.value else []
        return ast_call(yield_.__name__, args)

    @m.call_if_not_inside(m.If())
    @m.leave(m.Yield(value=m.Tuple()))
    def tuple_yield(self, original_node: cst.Yield, _updated_node: cst.Yield):
        args = [cst.Arg(e.value) for e in original_node.value.elements]
        return ast_call(yield_.__name__, args)

    @m.call_if_not_inside(m.If())
    @m.leave(m.Yield(value=~m.Tuple()))
    def single_yield(self, original_node: cst.Yield, _updated_node: cst.Yield):
        args = [cst.Arg(original_node.value)] if original_node.value else []
        return ast_call(yield_.__name__, args)


class InsertEmptyYield(StrictTransformer):
    @m.leave(m.If() | m.Else())
    def leave_(
        self, _original_node: cst.If | cst.Else, updated_node: cst.If | cst.Else
    ) -> cst.If | cst.Else:
        indented_block = updated_node.body
        last_statement = indented_block.body[-1]
        if not m.matches(last_statement, m.SimpleStatementLine()):
            return insert_body_maybe_semicolon(updated_node, -1, cst.Yield())
        elif m.matches(last_statement, m.SimpleStatementLine()) and not m.findall(
            last_statement, m.Yield()
        ):
            return insert_body_maybe_semicolon(updated_node, -1, cst.Yield())
        # VERY IMPORTANT: you have to return the updated node if you believe
        # at any point there was a mutation anywhere in the tree below
        return updated_node


class CanonicalizeElIfs(StrictTransformer):
    @m.leave(m.If(orelse=m.If()))
    def leave_if_with_elif(
        self, _original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        indented_block = updated_node.orelse.body
        last_statement = indented_block.body[-1]
        if m.matches(last_statement, m.SimpleStatementLine()) and m.matches(
            last_statement.body[-1], m.Assign(value=m.Yield())
        ):
            assign_targets = last_statement.body[-1].targets
            last_statement = cst.SimpleStatementLine(
                [
                    cst.Assign(
                        targets=assign_targets,
                        value=cst.Yield(
                            cst.Tuple([cst.Element(a.target) for a in assign_targets])
                            if len(assign_targets) > 1
                            else assign_targets[0].target
                        ),
                    )
                ]
            )
            body = [updated_node.orelse, last_statement]
        else:
            body = [updated_node.orelse]
        return updated_node.with_changes(orelse=cst.Else(cst.IndentedBlock(body)))


class ReplaceSCFCond(StrictTransformer):
    @m.leave(m.If(test=m.Call(func=m.Name(stack_if.__name__))))
    def insert_with_results(
        self, original_node: cst.If, _updated_node: cst.If
    ) -> cst.If:
        return original_node

    @m.leave(m.If(test=~m.Call(func=m.Name(stack_if.__name__))))
    def insert_with_results(
        self, original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        indented_block = updated_node.body
        last_statement = indented_block.body[-1]
        assert m.matches(
            last_statement, m.SimpleStatementLine()
        ), f"conditional must end with a statement"
        yield_expr = m.findall(last_statement, m.Call(func=m.Name(yield_.__name__)))
        assert (
            len(yield_expr) == 1
        ), f"conditional with must explicitly {yield_.__name__} on last line: {yield_expr}"
        yield_expr = yield_expr[0]
        results = [cst.Element(ast_call(T._placeholder_opaque_t.__name__))] * len(
            yield_expr.args
        )
        results = cst.Tuple(results)

        test = original_node.test
        new_test = ast_call(
            stack_if.__name__,
            args=[
                cst.Arg(test),
                cst.Arg(results),
                cst.Arg(
                    cst.Name(str(bool(original_node.orelse))),
                    keyword=cst.Name("has_else"),
                ),
            ],
        )
        new_test = test.deep_replace(test, new_test)
        return updated_node.with_changes(test=new_test)


class InsertEndIfs(StrictTransformer):
    @m.leave(m.If(orelse=None))
    def no_else(self, _original_node: cst.If, updated_node: cst.If) -> cst.If:
        # every if branch needs a scf_endif_branch
        # no else, then need to end the whole if in the body of the true branch
        return insert_body_maybe_semicolon(updated_node, -1, ast_call(end_if.__name__))

    @m.leave(m.If(orelse=m.Else()))
    def has_else(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        # every if branch needs a scf_endif_branch
        updated_node = insert_body_maybe_semicolon(
            updated_node, -1, ast_call(end_branch.__name__)
        )

        # insert the else at beginning of else
        orelse = updated_node.orelse
        orelse = insert_body_maybe_semicolon(
            orelse, 0, ast_call(else_.__name__), before=True
        )
        # and end the if after the else branch
        orelse = insert_body_maybe_semicolon(orelse, -1, ast_call(end_if.__name__))
        return updated_node.with_changes(orelse=orelse)


class RemoveJumpsAndInsertGlobals(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
        src_lines = inspect.getsource(f).splitlines()
        early_returns = []
        for i, c in enumerate(code):
            c: ConcreteInstr
            if c.opcode == int(OpCode.RETURN_VALUE):
                early_returns.append(i)

            if c.opcode in {
                # this is the first test condition jump from python <= 3.10
                # "POP_JUMP_IF_FALSE",
                # this is the test condition jump from python >= 3.11
                int(OpCode.POP_JUMP_FORWARD_IF_FALSE),
            }:
                code[i] = ConcreteInstr(
                    str(OpCode.POP_TOP), lineno=c.lineno, location=c.location
                )

            if c.opcode in {
                # this is the jump after each arm in a conditional
                int(OpCode.JUMP_FORWARD),
                # this is the jump at the end of a for loop
                # "JUMP_BACKWARD",
                # in principle this should be no-oped too but for whatever reason it leads to a stack-size
                # miscalculation (inside bytecode). we don't really need it though because
                # affine_range returns an iterator with length 1
            }:
                # only remove the jump if generated by an if stmt (not a `with` stmt)
                if "with" not in src_lines[c.lineno - code.first_lineno]:
                    code[i] = ConcreteInstr(
                        str(OpCode.NOP), lineno=c.lineno, location=c.location
                    )

        # early returns cause branches in conditionals to not be visited
        for idx in early_returns[:-1]:
            c = code[idx]
            code[idx] = ConcreteInstr(
                str(OpCode.NOP), lineno=c.lineno, location=c.location
            )

        f.__globals__[else_.__name__] = else_
        f.__globals__[end_branch.__name__] = end_branch
        f.__globals__[end_if.__name__] = end_if
        f.__globals__[stack_if.__name__] = stack_if
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[T._placeholder_opaque_t.__name__] = T._placeholder_opaque_t
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
