import ctypes
from functools import wraps

from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values
from mlir.ir import InsertionPoint, Value


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


def maybe_cast(val: Value):
    """Maybe cast an ir.Value to one of Tensor, Scalar.

    Args:
      val: The ir.Value to maybe cast.
    """
    from mlir_utils.dialects.ext.tensor import Tensor
    from mlir_utils.dialects.ext.arith import Scalar

    if not isinstance(val, Value):
        return val

    if Tensor.isinstance(val):
        return Tensor(val)
    if Scalar.isinstance(val):
        return Scalar(val)
    return val


# builds the decorator
def region_op(op_constructor):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        block_arg_types = kwargs.pop("block_args", [])
        op = op_constructor(*args, **kwargs)

        def builder_wrapper(body_builder):
            # add a block with block args having types ...
            op.regions[0].blocks.append(*[t for t in block_arg_types])
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
