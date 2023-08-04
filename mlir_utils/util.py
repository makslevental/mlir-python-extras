import ctypes
import inspect
import traceback
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Sequence

import mlir
from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values
from mlir.ir import (
    InsertionPoint,
    Value,
    Type,
    OpResultList,
    Location,
    OpResult,
    OpView,
)

try:
    from mlir.ir import TypeID
except ImportError:
    warnings.warn(
        f"TypeID not supported by {mlir=}; value casting won't work correctly"
    )
    TypeID = object


def get_result_or_results(
    op: None | OpView | Value,
) -> None | Value | OpResultList | OpResult | OpView:
    if op is None:
        return
    if isinstance(op, Value):
        return op
    return (
        list(get_op_results_or_values(op))
        if len(op.operation.results) > 1
        else get_op_result_or_value(op)
        if len(op.operation.results) > 0
        else op
    )


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


def maybe_cast(val: Value | list[Value]):
    """Maybe cast an ir.Value to one of Tensor, Scalar.

    Args:
      val: The ir.Value to maybe cast.
    """
    from mlir_utils.dialects.ext.arith import Scalar

    if isinstance(val, list):
        return list(map(maybe_cast, val))

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
def region_op(op_constructor, terminator=None):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)

        def builder_wrapper(body_builder):
            # add a block with block args having types ...
            if len(op.regions[0].blocks) == 0:
                sig = inspect.signature(body_builder)
                types = [p.annotation for p in sig.parameters.values()]
                if not (
                    len(types) == len(sig.parameters)
                    and all(isinstance(t, Type) for t in types)
                ):
                    raise ValueError(
                        f"for {body_builder=} either missing a type annotation or type annotation isn't a mlir type: {sig}"
                    )

                arg_locs = [get_user_code_loc()] * len(sig.parameters)
                op.regions[0].blocks.append(*types, arg_locs=arg_locs)
            with InsertionPoint(op.regions[0].blocks[0]):
                results = body_builder(
                    *[maybe_cast(a) for a in op.regions[0].blocks[0].arguments]
                )
                if terminator is not None:
                    res = []
                    if isinstance(results, (tuple, list)):
                        res.extend(results)
                    elif results is not None:
                        res.append(results)
                    terminator(res)

            res = get_result_or_results(op)
            if isinstance(res, OpResultList):
                return tuple(map(maybe_cast, res))
            else:
                return maybe_cast(res)

        return builder_wrapper

    # this is like make_maybe_no_args_decorator but a little different because the decorators here
    # are already wrapped (or something like that)
    @wraps(op_decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return op_decorator()(args[0])
        else:
            return op_decorator(*args, **kwargs)

    return maybe_no_args


def _update_caller_vars(previous_frame, args: Sequence, replacements: Sequence):
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
            previous_frame.f_locals[var_name] = replacements[i]
            # signal to update
            # for some reason you can only update one at a time?
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(previous_frame), ctypes.c_int(1)
            )


def get_user_code_loc():
    import mlir_utils

    mlir_utis_root_path = Path(mlir_utils.__path__[0])

    prev_frame = inspect.currentframe().f_back
    while Path(prev_frame.f_code.co_filename).is_relative_to(mlir_utis_root_path):
        prev_frame = prev_frame.f_back
    frame_info = inspect.getframeinfo(prev_frame)
    return Location.file(
        frame_info.filename, frame_info.lineno, frame_info.positions.col_offset
    )
