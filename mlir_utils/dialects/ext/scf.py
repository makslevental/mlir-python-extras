import inspect
import logging
import warnings
from typing import Optional, Sequence

import libcst as cst
import libcst.matchers as m
from bytecode import ConcreteBytecode, ConcreteInstr
from libcst.metadata import QualifiedNameProvider
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


class IpStack:
    #     __current_if_op: list[IfOp]
    __if_ips: list[InsertionPoint]

    def __init__(self, block):
        self.__if_ips = []
        self.push_block_ip(block)

    def push_block_ip(self, block):
        ip = InsertionPoint(block)
        ip.__enter__()
        self.__if_ips.append(ip)

    def pop_branch(self):
        ip = self.__if_ips.pop()
        ip.__exit__(None, None, None)

    def __len__(self):
        return len(self.__if_ips)

    def __add__(self, other: "IpStack"):
        self.__if_ips.extend(other.__if_ips)
        other.__if_ips = []
        return self


def unstack_if(cond: Value, results_=None) -> tuple[IpStack, IfOp]:
    if results_ is None:
        results_ = []
    assert isinstance(cond, Value), f"cond must be a mlir.Value: {cond=}"
    if_op = _if(cond, results_)
    cond.owner.move_before(if_op)

    return IpStack(if_op.then_block), if_op


def unstack_end_branch(ips_ifop: tuple[IpStack, IfOp]) -> tuple[IpStack, IfOp]:
    ips, ifop = ips_ifop
    ips.pop_branch()
    return ips, ifop


def unstack_else(prev_ips_ifop: tuple[IpStack, IfOp]) -> tuple[IpStack, IfOp]:
    prev_ips, ifop = prev_ips_ifop
    if not len(ifop.regions[1].blocks):
        ifop.regions[1].blocks.append(*[])

    prev_ips.push_block_ip(ifop.else_block)
    return prev_ips, ifop


def unstack_else_if(prev_ips_ifop: tuple[IpStack, IfOp], cond: Value, results_=None):
    prev_ips, prev_ifop = unstack_else(prev_ips_ifop)
    next_if_ip, next_if_op = unstack_if(cond, results_)
    return prev_ips + next_if_ip, next_if_op


def get_last_statement(original_node):
    statements = m.findall(original_node, m.SimpleStatementLine())
    assert len(statements), "no statements...?"
    return statements[-1]


def insert_in_deep_last_statement(
    original_node: cst.CSTNode,
    new_node: cst.CSTNode,
) -> cst.CSTNode:
    last_statement = get_last_statement(original_node)
    new_last_statement = last_statement.with_changes(
        body=list(last_statement.body) + [cst.Expr(new_node)]
    )
    return original_node.deep_replace(last_statement, new_last_statement)


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


def maybe_insert_yield_at_end_or_deep(node):
    maybe_last_statement = node.body[-1]
    if m.matches(maybe_last_statement, m.SimpleStatementLine()):
        if len(m.findall(maybe_last_statement, m.Yield())) > 0:
            return node

        # if last thing in body is a simplestatement then you can talk the yield (with a semicolon)
        # onto the end
        new_maybe_last_statement = insert_in_deep_last_statement(
            maybe_last_statement, cst.Yield()
        )
        node = node.deep_replace(maybe_last_statement, new_maybe_last_statement)
    else:
        # this branch is different (i.e., doesn't check for a match)
        # because if the last thing is an indented block, there's no way the user could've intentionally placed
        # a yield there that handles this conditional (even if they placed a yield to handle a conditional in that
        # last block)
        node = insert_in_deep_last_statement(node, cst.Yield())

    return node


class InsertEmptyYield(StrictTransformer):
    @m.leave(m.If())
    def leave_if(self, _original_node: cst.If, updated_node: cst.If) -> cst.If:
        new_body = maybe_insert_yield_at_end_or_deep(updated_node.body)
        new_orelse = updated_node.orelse
        if new_orelse:
            new_orelse_body = maybe_insert_yield_at_end_or_deep(new_orelse.body)
            new_orelse = new_orelse.with_changes(body=new_orelse_body)
        return updated_node.with_changes(body=new_body, orelse=new_orelse)

    @m.leave(m.For())
    def leave_for(self, _original_node: cst.For, updated_node: cst.For) -> cst.For:
        new_body = maybe_insert_yield_at_end_or_deep(updated_node.body)
        return updated_node.with_changes(body=new_body)


class CheckMatchingYields(StrictTransformer):
    @m.leave(m.If())
    def leave_(self, original_node: cst.If, _updated_node: cst.If) -> cst.If:
        n_ifs = len(m.findall(original_node, m.If()))
        n_elses = len(m.findall(original_node, m.Else()))
        n_yields = len(m.findall(original_node, m.Call(func=m.Name(yield_.__name__))))
        if n_ifs + n_elses <= n_yields:
            warnings.warn(
                f"unmatched if/elses and yields: {n_ifs=} {n_elses=} {n_yields=}; line {self.get_pos(original_node).start.line}"
            )
        return original_node


def check_unstack_if(original_node, metadata_resolver):
    return m.matches(
        original_node,
        m.If(
            test=m.NamedExpr(
                target=m.MatchMetadataIfTrue(
                    QualifiedNameProvider,
                    lambda qualnames: any(
                        unstack_if.__name__ in n.name
                        or unstack_else_if.__name__ in n.name
                        for n in qualnames
                    ),
                )
            )
        ),
        metadata_resolver=metadata_resolver,
    )


class CanonicalizeElIfTests(StrictTransformer):
    @m.call_if_inside(m.If(orelse=m.If()))
    @m.leave(m.If())
    def leave_last_elif(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        assert check_unstack_if(
            original_node, self
        ), f"if must already have had test replaced with unstack_if"
        parent = self.get_parent(original_node)
        if (
            not check_unstack_if(parent, self)
            # you need this because call_if_inside matches self as well as parent
            or parent.orelse != original_node
        ):
            return updated_node

        test = updated_node.test
        new_test_call = ast_call(
            unstack_else_if.__name__,
            args=[cst.Arg(parent.test.target)] + list(updated_node.test.value.args),
        )
        new_test = test.with_changes(value=new_test_call)
        return updated_node.with_changes(test=new_test)


class ReplaceSCFCond(StrictTransformer):
    @m.leave(
        m.If(
            test=m.Call(
                func=m.Name(unstack_if.__name__) | m.Name(unstack_else_if.__name__)
            )
        )
    )
    def insert_with_results(
        self, original_node: cst.If, _updated_node: cst.If
    ) -> cst.If:
        return original_node

    @m.leave(m.If())
    def leave_if(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        indented_block = updated_node.body
        last_statement = indented_block.body[-1]
        results = []
        if m.matches(last_statement, m.SimpleStatementLine()):
            yield_expr = m.findall(last_statement, m.Call(func=m.Name(yield_.__name__)))
            assert len(
                yield_expr
            ), f"conditional must explicitly {yield_.__name__} on last line: {yield_expr}"
            yield_expr = yield_expr[0]
            results = [cst.Element(ast_call(T._placeholder_opaque_t.__name__))] * len(
                yield_expr.args
            )
        results = cst.Tuple(results)

        test = original_node.test
        new_test = ast_call(
            unstack_if.__name__,
            args=[cst.Arg(test), cst.Arg(results)],
        )
        pos = self.get_pos(original_node)
        new_test = cst.NamedExpr(
            cst.Name(f"__{unstack_if.__name__}__{pos.start.line}"), new_test
        )
        new_test = test.deep_replace(test, new_test)
        return updated_node.with_changes(test=new_test)


def in_last_statement_maybe_interleave_with_yields(node, new_node):
    last_statement = get_last_statement(node)
    last_statement_body = list(last_statement.body)
    for i, b in enumerate(last_statement_body[:-1]):
        next_b = last_statement_body[i + 1]
        # two adjacent yields (this happens when InsertEmptyYield inserts a yield in a deep statement
        if m.matches(b, m.Expr(m.Call(func=m.Name(yield_.__name__)))) and m.matches(
            next_b, m.Expr(m.Call(func=m.Name(yield_.__name__)))
        ):
            last_statement_body.insert(i + 1, new_node)
            break
    else:
        last_statement_body.append(new_node)
    return node.deep_replace(
        last_statement,
        last_statement.with_changes(body=last_statement_body),
    )


class InsertEndIfs(StrictTransformer):
    @m.leave(m.If())
    def leave_if(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        assert check_unstack_if(
            original_node, self
        ), f"if must already have had test replaced with unstack_if"

        assign = cst.Assign(
            targets=[cst.AssignTarget(updated_node.test.target)],
            value=ast_call(
                unstack_end_branch.__name__, [cst.Arg(updated_node.test.target)]
            ),
        )

        new_body = in_last_statement_maybe_interleave_with_yields(
            updated_node.body, assign
        )

        new_orelse = None
        if updated_node.orelse:
            new_orelse = in_last_statement_maybe_interleave_with_yields(
                updated_node.orelse, assign
            )
            parent = self.get_parent(original_node)
            if not check_unstack_if(parent, self) or parent.orelse != original_node:
                return updated_node.with_changes(body=new_body, orelse=new_orelse)

            # basically adds a yield for scf.elseif that yields the correct result (i.e., whatever is yielded in the inner
            # block
            maybe_assigned_yield_in_body = ast_call(yield_.__name__)
            last_statement_in_body = updated_node.body.body[-1]

            # if the inner block yields a named result, "re-yield" it
            if m.matches(last_statement_in_body, m.SimpleStatementLine()) and m.matches(
                last_statement_in_body.body[0],
                m.Assign(value=m.Call(func=m.Name(yield_.__name__))),
            ):
                maybe_assigned_yield_in_body = last_statement_in_body.body[0]
                # re-yield but you don't need to name it, i.e. it doesn't need to be visible at the python/frontend level
                # i.e., if a user sets a breakpoint
                maybe_assigned_yield_in_body = ast_call(
                    yield_.__name__,
                    [cst.Arg(t.target) for t in maybe_assigned_yield_in_body.targets],
                )

            maybe_assigned_yield_in_body = cst.Expr(maybe_assigned_yield_in_body)
            new_orelse = in_last_statement_maybe_interleave_with_yields(
                new_orelse, maybe_assigned_yield_in_body
            )
        return updated_node.with_changes(body=new_body, orelse=new_orelse)


class InsertPreElses(StrictTransformer):
    @m.leave(m.If(orelse=m.Else()))
    def leave_if_else(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        assert check_unstack_if(
            original_node, self
        ), f"if must already have had test replaced with unstack_if"

        assign = cst.Assign(
            targets=[cst.AssignTarget(updated_node.test.target)],
            value=ast_call(unstack_else.__name__, [cst.Arg(updated_node.test.target)]),
        )
        new_body = insert_in_deep_last_statement(updated_node.body, assign)
        return updated_node.with_changes(body=new_body)


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

        # TODO(max): this is bad
        f.__globals__[unstack_else.__name__] = unstack_else
        f.__globals__[unstack_end_branch.__name__] = unstack_end_branch
        f.__globals__[unstack_else_if.__name__] = unstack_else_if
        f.__globals__[unstack_if.__name__] = unstack_if
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[T._placeholder_opaque_t.__name__] = T._placeholder_opaque_t
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        CheckMatchingYields,
        ReplaceSCFCond,
        CanonicalizeElIfTests,
        InsertEndIfs,
        InsertPreElses,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
