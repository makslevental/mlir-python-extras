import ast
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Sequence

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
from mlir_utils.ast.util import ast_call, set_lineno
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.scf import yield_ as yield__
from mlir_utils.util import (
    region_op,
    maybe_cast,
    get_result_or_results,
    get_user_code_loc,
    region_adder,
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


def _if(cond, results=None, *, has_else=False, loc=None, ip=None):
    if results is None:
        results = []
    if results:
        has_else = True
    if loc is None:
        loc = get_user_code_loc()
    return IfOp(cond, results, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)


@contextmanager
def if_ctx_manager(cond, results=None, *, has_else=False, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if_op = _if(cond, results, has_else=has_else, loc=loc, ip=ip)
    with InsertionPoint(if_op.regions[0].blocks[0]):
        yield if_op


@contextmanager
def else_ctx_manager(if_op):
    if len(if_op.regions[1].blocks) == 0:
        if_op.regions[1].blocks.append(*[])
    with InsertionPoint(if_op.regions[1].blocks[0]):
        yield


@region_adder(terminator=yield__)
def else_(ifop):
    return ifop.regions[1]


def is_yield_(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    )


def is_yield(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Yield)
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Yield)
    )


def append_hidden_node(node_body, new_node):
    last_statement = node_body[-1]
    new_node = ast.fix_missing_locations(
        set_lineno(new_node, last_statement.end_lineno)
    )
    node_body.append(new_node)
    return node_body


class InsertEmptyYield(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        updated_node = self.generic_visit(updated_node)

        new_yield = ast.Expr(ast.Yield())
        if not is_yield(updated_node.body[-1]):
            updated_node.body = append_hidden_node(
                updated_node.body, deepcopy(new_yield)
            )
        if updated_node.orelse and not is_yield(updated_node.orelse[-1]):
            updated_node.orelse = append_hidden_node(
                updated_node.orelse, deepcopy(new_yield)
            )

        return updated_node

    def visit_For(self, updated_node: ast.For) -> ast.For:
        updated_node = self.generic_visit(updated_node)
        new_yield = ast.Expr(ast.Yield())
        if not is_yield(updated_node.body[-1]):
            updated_node.body = append_hidden_node(updated_node.body, new_yield)
        return updated_node


def forward_yield_from_nested_if(node_body):
    last_statement = node_body[0].body[-1]
    if isinstance(last_statement.targets[0], ast.Tuple):
        res = ast.Tuple(
            [ast.Name(t.id, ast.Load()) for t in last_statement.targets[0].elts],
            ast.Load(),
        )
        targets = [
            ast.Tuple(
                [ast.Name(t.id, ast.Store()) for t in last_statement.targets[0].elts],
                ast.Store(),
            )
        ]
    else:
        res = ast.Name(last_statement.targets[0].id, ast.Load())
        targets = [ast.Name(last_statement.targets[0].id, ast.Store())]
    forwarding_yield = ast.Assign(
        targets=targets,
        value=ast.Yield(res),
    )
    return append_hidden_node(node_body, forwarding_yield)


class CanonicalizeElIfs(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        # postorder
        updated_node = self.generic_visit(updated_node)
        needs_forward = lambda body: (
            body
            and isinstance(body[0], ast.If)
            and is_yield(body[0].body[-1])
            and not is_yield(body[-1])
        )
        if needs_forward(updated_node.body):
            updated_node.body = forward_yield_from_nested_if(updated_node.body)

        if needs_forward(updated_node.orelse):
            updated_node.orelse = forward_yield_from_nested_if(updated_node.orelse)
        return updated_node


class ReplaceYieldWithSCFYield(StrictTransformer):
    def visit_Yield(self, node: ast.Yield) -> ast.Expr:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        call = ast.copy_location(ast_call(yield_.__name__, args), node)
        call = ast.fix_missing_locations(call)
        return call


class ReplaceIfWithWith(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.With | list[ast.With, ast.With]:
        is_elif = (
            len(updated_node.orelse) >= 1
            and isinstance(updated_node.orelse[0], ast.If)
            and updated_node.body[-1].end_lineno + 1 == updated_node.orelse[0].lineno
        )

        updated_node = self.generic_visit(updated_node)
        last_statement = updated_node.body[-1]
        assert is_yield_(last_statement) or is_yield(
            last_statement
        ), f"{last_statement=}"

        test = updated_node.test
        results = [
            ast_call(T._placeholder_opaque_t.__name__)
            for _ in range(len(last_statement.value.args))
        ]
        results = ast.fix_missing_locations(
            ast.copy_location(ast.Tuple(results, ctx=ast.Load()), test)
        )

        if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Store())
        withitem = ast.withitem(
            context_expr=ast_call(if_ctx_manager.__name__, args=[test, results]),
            optional_vars=if_op_name,
        )
        then_with = ast.With(items=[withitem])
        then_with = ast.copy_location(then_with, updated_node)
        then_with = ast.fix_missing_locations(then_with)
        then_with.body = updated_node.body

        if updated_node.orelse:
            if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Load())
            withitem = ast.withitem(
                context_expr=ast_call(else_ctx_manager.__name__, args=[if_op_name])
            )
            else_with = ast.With(items=[withitem])
            if is_elif:
                else_with = ast.copy_location(else_with, updated_node.orelse[0])
            else:
                else_with = set_lineno(else_with, updated_node.orelse[0].lineno - 1)
            else_with = ast.fix_missing_locations(else_with)
            else_with.body = updated_node.orelse
            return [then_with, else_with]
        else:
            return then_with


class RemoveJumpsAndInsertGlobals(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
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

        # early returns cause branches in conditionals to not be visited
        for idx in early_returns[:-1]:
            c = code[idx]
            code[idx] = ConcreteInstr(
                str(OpCode.NOP), lineno=c.lineno, location=c.location
            )

        # TODO(max): this is bad
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[if_ctx_manager.__name__] = if_ctx_manager
        f.__globals__[else_ctx_manager.__name__] = else_ctx_manager
        f.__globals__[T._placeholder_opaque_t.__name__] = T._placeholder_opaque_t
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
