import inspect
import logging
from textwrap import dedent
from typing import Optional, Sequence

import libcst as cst
import libcst.matchers as m
from bytecode import ConcreteBytecode, ConcreteInstr
from mlir.dialects import scf
from mlir.ir import InsertionPoint, Value, OpResultList

from mlir_utils.ast.canonicalize import (
    StrictTransformer,
    Canonicalizer,
    BytecodePatcher,
)
from mlir_utils.ast.util import ast_call
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.scf import yield_ as yield__
from mlir_utils.util import (
    region_op,
    maybe_cast,
    _update_caller_vars,
    get_result_or_results,
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
    return scf.ForOp(start, stop, step, iter_args, loc=loc, ip=ip)


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
    if len(iter_args):
        previous_frame = inspect.currentframe().f_back
        replacements = tuple(map(maybe_cast, for_op.results_))
        _update_caller_vars(previous_frame, iter_args, replacements)


def yield_(*args):
    if len(args) == 1 and isinstance(args[0], OpResultList):
        args = list(args[0])
    yield__(args)


def _if(cond, results_=None, *, has_else=False, loc=None, ip=None):
    if results_ is None:
        results_ = []
    if results_:
        has_else = True
    return scf.IfOp(cond, results_, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)


class IfStack:
    __current_if_op: list[scf.IfOp] = []
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


# forward here for readability
def stack_if(*args, **kwargs):
    return IfStack.push(*args, **kwargs)


def end_branch():
    IfStack.pop_branch()


def else_():
    return IfStack.push_else()


def end_if():
    IfStack.pop()


class ReplaceSCFYield(StrictTransformer):
    @m.leave(m.Yield(value=m.Tuple()))
    def tuple_yield(self, original_node: cst.Yield, updated_node: cst.Yield):
        args = [cst.Arg(e.value) for e in original_node.value.elements]
        return ast_call(yield_.__name__, args)

    @m.leave(m.Yield(value=~m.Tuple()))
    def single_yield(self, original_node: cst.Yield, updated_node: cst.Yield):
        args = [cst.Arg(original_node.value)] if original_node.value else []
        return ast_call(yield_.__name__, args)


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


class InsertSCFYield(StrictTransformer):
    @m.leave(m.If() | m.Else())
    def leave_(
        self, _original_node: cst.If | cst.Else, updated_node: cst.If | cst.Else
    ) -> cst.If | cst.Else:
        indented_block = updated_node.body
        last_statement = indented_block.body[-1]
        if not m.matches(last_statement, m.SimpleStatementLine([m.Expr(m.Yield())])):
            return insert_body_maybe_semicolon(
                updated_node, -1, ast_call(yield_.__name__)
            )
        # VERY IMPORTANT: you have to return the updated node if you believe
        # at any point there was a mutation anywhere in the tree below
        return updated_node


class CanonicalizeElIfs(StrictTransformer):
    @m.leave(m.If(orelse=m.If(test=m.NamedExpr())))
    def leave_if_with_elif_named(
        self, _original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        return updated_node.with_changes(
            orelse=cst.Else(
                cst.IndentedBlock(
                    [
                        updated_node.orelse,
                        cst.SimpleStatementLine(
                            [cst.Expr(cst.Yield(updated_node.orelse.test.target))]
                        ),
                    ]
                )
            )
        )

    @m.leave(m.If(orelse=m.If(test=~m.NamedExpr())))
    def leave_if_with_elif(
        self, _original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        return updated_node.with_changes(
            orelse=cst.Else(cst.IndentedBlock([updated_node.orelse]))
        )


class ReplaceSCFCond(StrictTransformer):
    @m.leave(m.If(test=m.NamedExpr(value=m.Call(func=m.Name(stack_if.__name__)))))
    def insert_with_results(
        self, original_node: cst.If, _updated_node: cst.If
    ) -> cst.If:
        return original_node

    @m.leave(m.If(test=m.NamedExpr(value=m.Comparison())))
    def insert_with_results(
        self, original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        test = original_node.test
        results = cst.Tuple(
            [
                cst.Element(cst.Name(n))
                for n in self.func_sym_table[original_node.test.target.value]
            ]
        )
        compare = test.value
        assert isinstance(
            compare, cst.Comparison
        ), f"expected cst.Compare from {compare=}"
        new_compare = ast_call(
            stack_if.__name__, args=[cst.Arg(compare), cst.Arg(results)]
        )
        new_test = test.deep_replace(compare, new_compare)
        return updated_node.with_changes(test=new_test)

    @m.leave(m.If(test=m.Comparison()))
    def insert_no_results(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        test = original_node.test
        args = [cst.Arg(test)]
        if original_node.orelse:
            args += [cst.Arg(cst.Tuple([])), cst.Arg(cst.Name(str(True)))]
        new_test = ast_call(stack_if.__name__, args=args)
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
            if c.name == "RETURN_VALUE":
                early_returns.append(i)

            if c.name in {
                # this is the first test condition jump from python <= 3.10
                "POP_JUMP_IF_FALSE",
                # this is the test condition jump from python >= 3.11
                "POP_JUMP_FORWARD_IF_FALSE",
            }:
                code[i] = ConcreteInstr("POP_TOP", lineno=c.lineno, location=c.location)

            if c.name in {
                # this is the jump after each arm in a conditional
                "JUMP_FORWARD",
                # this is the jump at the end of a for loop
                # "JUMP_BACKWARD",
                # in principle this should be no-oped too but for whatever reason it leads to a stack-size
                # miscalculation (inside bytecode). we don't really need it though because
                # affine_range returns an iterator with length 1
            }:
                # only remove the jump if generated by an if stmt (not a with stmt)
                if "with" not in src_lines[c.lineno - code.first_lineno]:
                    code[i] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

        # early returns cause branches in conditionals to not be visited
        for idx in early_returns[:-1]:
            c = code[idx]
            code[idx] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

        # TODO(max): this is bad
        f.__globals__["stack_if"] = stack_if
        f.__globals__["end_branch"] = end_branch
        f.__globals__["end_if"] = end_if
        f.__globals__["else_"] = else_
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        CanonicalizeElIfs,
        InsertSCFYield,
        ReplaceSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
