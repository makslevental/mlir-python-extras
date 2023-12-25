import contextlib
import inspect
import warnings
from functools import wraps

from ..ir import Type, InsertionPoint, OpResultList, OpView
from ..dialects._ods_common import get_op_result_or_op_results

try:
    from ..ir import TypeID
except ImportError:
    warnings.warn(
        f"TypeID not supported by host bindings; value casting won't work correctly"
    )
    TypeID = object

from .util import get_user_code_loc, Successor


# builds the decorator
def make_maybe_no_args_decorator(decorator):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(decorator)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_dec


@contextlib.contextmanager
def bb(*preds: tuple[Successor | OpView]):
    current_ip = InsertionPoint.current
    op = current_ip.block.owner
    op_region = op.regions[0]
    args = []
    if len(preds):
        if isinstance(preds[0], OpView):
            args = preds[0].operands
        elif isinstance(preds[0], Successor):
            args = preds[0].operands
        else:
            raise NotImplementedError(f"{preds[0]=} not supported.")
    arg_locs = list(filter(None, [get_user_code_loc()] * len(args)))
    if len(arg_locs) == 0:
        arg_locs = None
    block = op_region.blocks.append(*[a.type for a in args], arg_locs=arg_locs)
    for p in preds:
        if isinstance(p, OpView):
            p.operation.successors[0] = block
        elif isinstance(p, Successor):
            for i, b in enumerate(p.block.owner.successors):
                if i == p.pos:
                    p.op.successors[i] = block
                    p.block = block
                    break
    with InsertionPoint(block):
        yield block, list(block.arguments)


def op_region_builder(op, op_region, terminator=None):
    def builder_wrapper(body_builder):
        # add a block with block args having types ...
        if len(op_region.blocks) == 0:
            sig = inspect.signature(body_builder)
            types = [p.annotation for p in sig.parameters.values()]
            if not (
                len(types) == len(sig.parameters)
                and all(isinstance(t, Type) for t in types)
            ):
                raise ValueError(
                    f"for {body_builder=} either missing a type annotation or type annotation isn't a mlir type: {sig}"
                )

            arg_locs = list(filter(None, [get_user_code_loc()] * len(sig.parameters)))
            if len(arg_locs) == 0:
                arg_locs = None
            op_region.blocks.append(*types, arg_locs=arg_locs)

        with InsertionPoint(op_region.blocks[0]):
            results = body_builder(*list(op_region.blocks[0].arguments))

        with InsertionPoint(list(op_region.blocks)[-1]):
            if terminator is not None:
                res = []
                if isinstance(results, (tuple, list)):
                    res.extend(results)
                elif results is not None:
                    res.append(results)
                terminator(res)

        res = get_op_result_or_op_results(op)
        if isinstance(res, (OpResultList, list, tuple)):
            return tuple(res)
        else:
            return res

    return builder_wrapper


def region_adder(terminator=None):
    def wrapper(op_region_adder):
        def region_adder_decorator(op, *args, **kwargs):
            region = op_region_adder(op, *args, **kwargs)

            return op_region_builder(op, region, terminator)

        return region_adder_decorator

    return wrapper


def region_op(op_constructor, terminator=None):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)
        op_region = op.regions[0]

        return op_region_builder(op, op_region, terminator)

    # this is like make_maybe_no_args_decorator but a little different because the decorators here
    # are already wrapped (or something like that)
    @wraps(op_decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return op_decorator()(args[0])
        else:
            return op_decorator(*args, **kwargs)

    return maybe_no_args


class ModuleMeta(type):
    def __new__(cls, name, bases, classdict, **kwargs):
        ip = classdict.pop("ip")
        loc = classdict.pop("loc")
        module_terminator = classdict.pop("module_terminator", None)
        new = super().__new__(cls, name, bases, classdict)
        if module_terminator is not None:
            module_terminator(loc=loc, ip=ip)
        for k, v in classdict.items():
            if callable(v):
                v.qualname = name
        ip.__exit__(None, None, None)
        return new
