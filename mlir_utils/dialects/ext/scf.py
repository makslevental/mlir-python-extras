import ast
import inspect
from typing import Optional, Sequence

from bytecode import ConcreteBytecode, ConcreteInstr
from mlir.dialects import scf
from mlir.ir import InsertionPoint, Value

from mlir_utils.ast.canonicalize import (
    StrictTransformer,
    Canonicalizer,
    BytecodePatcher,
)
from mlir_utils.ast.util import ast_call
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.scf import yield_ as yield__
from mlir_utils.dialects.util import region_op, maybe_cast, _update_caller_vars


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
    yield__(args)


def _if(cond, results_=None, *, has_else=False, loc=None, ip=None):
    if results_ is None:
        results_ = []
    return scf.IfOp(cond, results_, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)

_current_if_op: list[scf.IfOp] = []
_if_ip: InsertionPoint = None


def stack_if(cond: Value, results_=None, *, has_else=False):
    assert isinstance(cond, Value)
    global _if_ip, _current_if_op
    if_op = _if(cond, results_, has_else=has_else)
    cond.owner.move_before(if_op)
    _current_if_op.append(if_op)
    _if_ip = InsertionPoint(if_op.then_block)
    _if_ip.__enter__()
    return True


def stack_if_else():
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].add_else())
    _if_ip.__enter__()
    return True


def stack_else():
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].add_else())
    _if_ip.__enter__()
    return True


def stack_else_if(cond, results_=None, *, has_else=False):
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].add_else())
    _if_ip.__enter__()
    return stack_if(cond, results_, has_else=has_else)


def stack_endif_branch():
    global _if_ip
    _if_ip.__exit__(None, None, None)


def stack_endif():
    global _current_if_op
    _current_if_op.pop()


_for_ip = None


class ReplaceSCFYield(StrictTransformer):
    def visit_Yield(self, node: ast.Yield) -> ast.Call:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        return ast_call(yield_.__name__, args)


class InsertEndIfs(StrictTransformer):
    def visit_If(self, node):
        for i, b in enumerate(node.body):
            node.body[i] = self.visit(b)
        for i, b in enumerate(node.orelse):
            node.orelse[i] = self.visit(b)

        if yield_in_body := next(
            (n for n in node.body if ast.unparse(n).startswith("yield")), None
        ):
            yield_call = yield_in_body.value
            if yield_call.args:
                # TODO
                print(yield_in_body)

        node.test = ast_call(stack_if.__name__, args=[node.test])
        # every if branch needs a scf_endif_branch
        if yield_in_body is None:
            node.body.append(ast.Expr(ast_call(yield_.__name__)))
        node.body.append(ast.Expr(ast_call(stack_endif_branch.__name__)))
        # no else, then need to end the whole if in the body of the true branch
        if not node.orelse:
            node.body.append(ast.Expr(ast_call(stack_endif.__name__)))
        else:
            # otherwise end the if after the else branch
            node.orelse.insert(0, ast.Expr(ast_call(stack_else.__name__)))
            node.orelse.append(ast.Expr(ast_call(stack_endif_branch.__name__)))
            node.orelse.append(ast.Expr(ast_call(stack_endif.__name__)))

        return node


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
        f.__globals__["stack_endif_branch"] = stack_endif_branch
        f.__globals__["stack_endif"] = stack_endif
        return code


class SCFCanonicalizer(Canonicalizer):
    @property
    def ast_rewriters(self):
        return [ReplaceSCFYield, InsertEndIfs]

    @property
    def bytecode_patchers(self):
        return [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
