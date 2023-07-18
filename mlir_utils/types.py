from typing import Union

import numpy as np
from mlir.ir import (
    IntegerType,
    F64Type,
    RankedTensorType,
    IndexType,
    F16Type,
    F32Type,
)

index = IndexType.get()
bool_ = IntegerType.get_signless(1)
i8 = IntegerType.get_signless(8)
i16 = IntegerType.get_signless(16)
i32 = IntegerType.get_signless(32)
i64 = IntegerType.get_signless(64)
f16 = F16Type.get()
f32 = F32Type.get()
f64 = F64Type.get()

NP_DTYPE_TO_MLIR_TYPE = lambda: {
    np.int8: i8,
    np.int16: i16,
    np.int32: i32,
    np.int64: i64,
    # this is techincally wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map this to index type
    np.longlong: index,
    np.uintp: index,
    np.float16: f16,
    np.float32: f32,
    np.float64: f64,
}

MLIR_TYPE_TO_NP_DTYPE = lambda: {v: k for k, v in NP_DTYPE_TO_MLIR_TYPE().items()}


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F64Type, RankedTensorType]:
    """Infer MLIR type (`ir.Type`) from supported python values.

    Note ints and floats are mapped to 64-bit types.

    Args:
      py_val: Python value that's either a numerical value or numpy array.

    Returns:
      MLIR type corresponding to py_val.
    """
    if isinstance(py_val, bool):
        return bool_
    elif isinstance(py_val, int):
        return i64
    elif isinstance(py_val, float):
        return f64
    elif isinstance(py_val, np.ndarray):
        dtype = NP_DTYPE_TO_MLIR_TYPE()[py_val.dtype.type]
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )
