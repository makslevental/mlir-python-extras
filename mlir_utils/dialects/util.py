from functools import wraps

import numpy as np
from mlir.dialects import arith
from mlir.dialects._ods_common import get_op_result_or_value, get_op_results_or_values
from mlir.ir import (
    InsertionPoint,
    IntegerType,
    F64Type,
    RankedTensorType,
    IndexType,
    F16Type,
    F32Type,
)


def get_result_or_results(op):
    if op is None:
        return
    return (
        get_op_results_or_values(op)
        if len(op.operation.results) > 1
        else get_op_result_or_value(op)
        if len(op.operation.results) > 0
        else None
    )


def make_maybe_no_args_decorator(decorator):
    @wraps(decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return decorator()(args[0])
        else:
            return decorator(*args, **kwargs)

    return maybe_no_args


# builds the decorator
def region_op(op_constructor):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)

        def builder_wrapper(body_builder):
            @wraps(body_builder)
            def wrapper(*args):
                # add a block with block args having types ...
                op.regions[0].blocks.append(*[a.type for a in args])
                with InsertionPoint(op.regions[0].blocks[0]):
                    return get_result_or_results(
                        body_builder(*op.regions[0].blocks[0].arguments)
                    )

            wrapper.op = op
            return wrapper

        return builder_wrapper

    return make_maybe_no_args_decorator(op_decorator)


def infer_mlir_type(
    py_val: int | float | bool | np.ndarray,
) -> IntegerType | F64Type | RankedTensorType:
    if isinstance(py_val, bool):
        return IntegerType.get_signless(1)
    elif isinstance(py_val, int):
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    elif isinstance(py_val, np.ndarray):
        dtype_ = {
            np.int8: IntegerType.get_signless(8),
            np.int16: IntegerType.get_signless(16),
            np.int32: IntegerType.get_signless(32),
            np.int64: IntegerType.get_signless(64),
            np.uintp: IndexType.get(),
            np.longlong: IndexType.get(),
            np.float16: F16Type.get(),
            np.float32: F32Type.get(),
            np.float64: F64Type.get(),
        }[py_val.dtype.type]
        return RankedTensorType.get(py_val.shape, dtype_)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def constant(py_val: int | float | bool | np.ndarray):
    return arith.ConstantOp(infer_mlir_type(py_val), py_val)
