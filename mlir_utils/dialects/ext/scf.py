import ast
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Sequence, Union

from bytecode import ConcreteBytecode, ConcreteInstr
from mlir.dialects._ods_common import get_op_results_or_values, get_default_loc_context
from mlir.dialects.linalg.opdsl.lang.emitter import _is_index_type
from mlir.dialects.scf import IfOp, ForOp, ForallOp, ParallelOp, InParallelOp, ReduceOp
from mlir.ir import (
    InsertionPoint,
    Value,
    OpResultList,
    OpResult,
    Operation,
    OpView,
    IndexType,
    _denseI64ArrayAttr,
)

import mlir_utils.types as T
from mlir_utils.ast.canonicalize import (
    StrictTransformer,
    Canonicalizer,
    BytecodePatcher,
    OpCode,
)
from mlir_utils.ast.util import ast_call, set_lineno
from mlir_utils.dialects.ext.arith import constant, index_cast
from mlir_utils.dialects.scf import yield_ as yield__, return_
from mlir_utils.util import (
    region_op,
    maybe_cast,
    get_result_or_results,
    get_user_code_loc,
    region_adder,
    is_311,
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
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = constant(p, index=True)
        if not _is_index_type(p.type):
            p = index_cast(p)
        params[i] = p

    if loc is None:
        loc = get_user_code_loc()
    return ForOp(*params, iter_args, loc=loc, ip=ip)


for_ = region_op(_for, terminator=yield__)


class ForallOp(ForallOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        shared_outs: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        if shared_outs is not None:
            results = [o.type for o in shared_outs]
        else:
            results = shared_outs = []
        iv_types = [IndexType.get()] * len(lower_bounds)
        dynamic_lower_bounds = []
        dynamic_upper_bounds = []
        dynamic_steps = []
        context = get_default_loc_context(loc)
        attributes = {
            "staticLowerBound": _denseI64ArrayAttr(lower_bounds, context),
            "staticUpperBound": _denseI64ArrayAttr(upper_bounds, context),
            "staticStep": _denseI64ArrayAttr(steps, context),
        }

        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=[
                    get_op_results_or_values(o)
                    for o in [
                        dynamic_lower_bounds,
                        dynamic_upper_bounds,
                        dynamic_steps,
                        # lower_bounds,
                        # upper_bounds,
                        # steps,
                        shared_outs,
                    ]
                ],
                attributes=attributes,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*iv_types, *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def arguments(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments


class InParallelOp(InParallelOp):
    def __init__(self, *, loc=None, ip=None):
        super().__init__(
            self.build_generic(regions=1, results=[], operands=[], loc=loc, ip=ip)
        )
        self.regions[0].blocks.append(*[])

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]


def _parfor(op_ctor, iter_args_name):
    def _base(
        lower_bounds, upper_bounds=None, steps=None, *, loc=None, ip=None, **kwargs
    ):
        iter_args = kwargs.get(iter_args_name)
        if loc is None:
            loc = get_user_code_loc()

        if upper_bounds is None:
            upper_bounds = lower_bounds
            lower_bounds = [0] * len(upper_bounds)
        if steps is None:
            steps = [1] * len(lower_bounds)

        params = [lower_bounds, upper_bounds, steps]
        for i, p in enumerate(params):
            for j, pp in enumerate(p):
                if isinstance(pp, int):
                    pp = constant(pp, index=True)
                if not _is_index_type(pp.type):
                    pp = index_cast(pp)
                p[j] = pp
            params[i] = p

        if loc is None:
            loc = get_user_code_loc()

        return op_ctor(*params, iter_args, loc=loc, ip=ip)

    return _base


@region_op
def in_parallel():
    return InParallelOp()


def parallel_insert_slice(
    source,
    dest,
    static_offsets=None,
    static_sizes=None,
    static_strides=None,
    offsets=None,
    sizes=None,
    strides=None,
):
    from . import tensor

    @in_parallel
    def foo():
        tensor.parallel_insert_slice(
            source,
            dest,
            offsets,
            sizes,
            strides,
            static_offsets,
            static_sizes,
            static_strides,
        )


forall_ = region_op(_parfor(ForallOp, iter_args_name="shared_outs"))


def _parfor_cm(op_ctor, iter_args_name):
    def _base(*args, **kwargs):
        for_op = _parfor(op_ctor, iter_args_name=iter_args_name)(*args, **kwargs)
        block = for_op.regions[0].blocks[0]
        block_args = tuple(map(maybe_cast, block.arguments))
        with InsertionPoint(block):
            yield block_args

    return _base


forall = _parfor_cm(ForallOp, iter_args_name="shared_outs")


def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
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


class ParallelOp(ParallelOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        inits: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        if inits is None:
            inits = []
        results = [i.type for i in inits]
        iv_types = [IndexType.get()] * len(lower_bounds)
        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=[
                    get_op_results_or_values(o)
                    for o in [lower_bounds, upper_bounds, steps, inits]
                ],
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*iv_types)

    @property
    def body(self):
        return self.regions[0].blocks[0]

    @property
    def induction_variables(self):
        return self.body.arguments


parange_ = region_op(_parfor(ParallelOp, iter_args_name="inits"), terminator=yield__)
parange = _parfor_cm(ParallelOp, iter_args_name="inits")


class ReduceOp(ReduceOp):
    def __init__(self, operand, *, loc=None, ip=None):
        super().__init__(
            self.build_generic(
                regions=1, results=[], operands=[operand], loc=loc, ip=ip
            )
        )
        self.regions[0].blocks.append(*[operand.type, operand.type])


def reduce_(operand, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return ReduceOp(operand, loc=loc, ip=ip)


reduce = region_op(reduce_, terminator=lambda xs: return_(*xs))


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

        for i, r in enumerate(results):
            if r.type == T.placeholder_opaque():
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
            ast_call(T.placeholder_opaque.__name__)
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
                int(OpCode.POP_JUMP_FORWARD_IF_FALSE)
                if is_311()
                else int(OpCode.POP_JUMP_IF_FALSE),
            }:
                code[i] = ConcreteInstr(
                    str(OpCode.POP_TOP), lineno=c.lineno, location=c.location
                )

        # TODO(max): this is bad
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[if_ctx_manager.__name__] = if_ctx_manager
        f.__globals__[else_ctx_manager.__name__] = else_ctx_manager
        f.__globals__[T.placeholder_opaque.__name__] = T.placeholder_opaque
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
