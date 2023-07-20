import ctypes
import inspect
import warnings
from collections import defaultdict
from functools import wraps
from typing import Callable

import mlir
from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values
from mlir.ir import InsertionPoint, Value, Type

try:
    from mlir.ir import TypeID
except ImportError:
    warnings.warn(
        f"TypeID not supported by {mlir=}; value casting won't work correctly"
    )
    TypeID = object


def get_result_or_results(op):
    if op is None:
        return
    if isinstance(op, Value):
        return op
    return (
        get_op_results_or_values(op)
        if len(op.operation.results) > 1
        else get_op_result_or_value(op)
        if len(op.operation.results) > 0
        else op
    )


def make_maybe_no_args_decorator(decorator):
    @wraps(decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return decorator()(args[0])
        else:
            return decorator(*args, **kwargs)

    return maybe_no_args


__VALUE_CASTERS: defaultdict[
    TypeID, list[Callable[[Value], Value | None]]
] = defaultdict(list)


def register_value_caster(
    typeid: TypeID, caster: Callable[[Value], Value], priority: int = None
):
    if not isinstance(typeid, TypeID):
        raise ValueError(f"{typeid=} is not a TypeID")
    if priority is None:
        __VALUE_CASTERS[typeid].append(caster)
    else:
        __VALUE_CASTERS[typeid].insert(priority, caster)


def has_value_caster(typeid: TypeID):
    if not isinstance(typeid, TypeID):
        raise ValueError(f"{typeid=} is not a TypeID")
    if not typeid in __VALUE_CASTERS:
        return False
    return True


def get_value_caster(typeid: TypeID):
    if not has_value_caster(typeid):
        raise ValueError(f"no registered caster for {typeid=}")
    return __VALUE_CASTERS[typeid]


def maybe_cast(val: Value):
    """Maybe cast an ir.Value to one of Tensor, Scalar.

    Args:
      val: The ir.Value to maybe cast.
    """
    from mlir_utils.dialects.ext.arith import Scalar

    if not isinstance(val, Value):
        return val

    if has_value_caster(val.type.typeid):
        for caster in get_value_caster(val.type.typeid):
            if casted := caster(val):
                return casted
        raise ValueError(f"no successful casts for {val=}")
    if Scalar.isinstance(val):
        return Scalar(val)
    return val


# builds the decorator
def region_op(op_constructor):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)

        def builder_wrapper(body_builder):
            # add a block with block args having types ...
            sig = inspect.signature(body_builder)
            types = [p.annotation for p in sig.parameters.values()]
            if not (
                len(types) == len(sig.parameters)
                and all(isinstance(t, Type) for t in types)
            ):
                raise ValueError(
                    f"for {body_builder=} either missing a type annotation or type annotation isn't a mlir type: {sig}"
                )

            op.regions[0].blocks.append(*types)
            with InsertionPoint(op.regions[0].blocks[0]):
                body_builder(
                    *[maybe_cast(a) for a in op.regions[0].blocks[0].arguments]
                )

            return maybe_cast(get_result_or_results(op))

        return builder_wrapper

    return make_maybe_no_args_decorator(op_decorator)


def _update_caller_vars(previous_frame, args, replacements):
    """Update caller vars passed as args.

    This function uses CPython API  to update the values
    of the caller's args (not the caller of this function but the caller of caller of this function).
    It does this by searching for a match in the caller's f_locals based on identity (A is A) and then
    updating all corresponding values in the f_locals dict. Finally, it uses PyFrame_LocalsToFast to signal
    to the CPython runtime that an update has been made to f_locals.

    Args:
      previous_frame: The frame in which vars will be updated.
      args: The args to the callee.
      replacements: The values that should replace the values of the vars in the caller.
    """

    if len(args) != len(replacements):
        raise ValueError(f"updates must be 1-1: {args=} {replacements=}")
    # find the name of the iter args in the previous frame
    var_names = [
        [
            var_name
            for var_name, var_val in previous_frame.f_locals.items()
            if var_val is arg
        ]
        for arg in args
    ]
    for i, var_names in enumerate(var_names):
        for var_name in var_names:
            # previous_frame.f_locals[var_name] = maybe_cast(replacements[i])
            previous_frame.f_locals[var_name] = replacements[i]
    # signal to update
    ctypes.pythonapi.PyFrame_LocalsToFast(
        ctypes.py_object(previous_frame), ctypes.c_int(1)
    )
