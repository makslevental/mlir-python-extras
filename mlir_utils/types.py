import sys
from functools import partial
from typing import Union

import numpy as np
from mlir.ir import (
    Attribute,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E4M3B11FNUZType,
    IndexType,
    IntegerType,
    MemRefType,
    NoneType,
    OpaqueType,
    RankedTensorType,
    Type,
    UnrankedMemRefType,
    UnrankedTensorType,
    VectorType,
)

_index_t = lambda: IndexType.get()
_bool_t = lambda: IntegerType.get_signless(1)

_i8_t = lambda: IntegerType.get_signless(8)
_i16_t = lambda: IntegerType.get_signless(16)
_i32_t = lambda: IntegerType.get_signless(32)
_i64_t = lambda: IntegerType.get_signless(64)

_si8_t = lambda: IntegerType.get_signed(8)
_si16_t = lambda: IntegerType.get_signed(16)
_si32_t = lambda: IntegerType.get_signed(32)
_si64_t = lambda: IntegerType.get_signed(64)

_ui8_t = lambda: IntegerType.get_unsigned(8)
_ui16_t = lambda: IntegerType.get_unsigned(16)
_ui32_t = lambda: IntegerType.get_unsigned(32)
_ui64_t = lambda: IntegerType.get_unsigned(64)

_f16_t = lambda: F16Type.get()
_f32_t = lambda: F32Type.get()
_f64_t = lambda: F64Type.get()
_bf16_t = lambda: BF16Type.get()

_f8e5m2_t = lambda: Float8E5M2Type.get()
_f8e4m3_t = lambda: Float8E4M3FNType.get()
_f8e4m3b11fnuz_t = lambda: Float8E4M3B11FNUZType.get()

_cmp16_t = lambda: ComplexType.get(_f16_t())
_cmp32_t = lambda: ComplexType.get(_f32_t())
_cmp64_t = lambda: ComplexType.get(_f64_t())

_none_t = lambda: NoneType.get()

opaque_t = lambda dialect_namespace, buffer: OpaqueType.get(dialect_namespace, buffer)


def _placeholder_opaque_t():
    return opaque_t("scf", "placeholder")


_name_to_type = {
    "index_t": _index_t,
    "bool_t": _bool_t,
    "i8_t": _i8_t,
    "i16_t": _i16_t,
    "i32_t": _i32_t,
    "i64_t": _i64_t,
    "si8_t": _si8_t,
    "si16_t": _si16_t,
    "si32_t": _si32_t,
    "si64_t": _si64_t,
    "ui8_t": _ui8_t,
    "ui16_t": _ui16_t,
    "ui32_t": _ui32_t,
    "ui64_t": _ui64_t,
    "f16_t": _f16_t,
    "f32_t": _f32_t,
    "f64_t": _f64_t,
    "bf16_t": _bf16_t,
    "f8e5m2_t": _f8e5m2_t,
    "f8e4m3_t": _f8e4m3_t,
    "f8e4m3b11fnuz_t": _f8e4m3b11fnuz_t,
    "cmp16_t": _cmp16_t,
    "cmp32_t": _cmp32_t,
    "cmp64_t": _cmp64_t,
    "none_t": _none_t,
}


def __getattr__(name):
    if name in _name_to_type:
        return _name_to_type[name]()
    # this kicks it to the default module attribute lookup (i.e., functions defined below and such)
    return None


_np_dtype_to_mlir_type_ctor = {
    np.int8: _i8_t,
    np.int16: _i16_t,
    np.int32: _i32_t,
    # windows
    np.intc: _i32_t,
    np.int64: _i64_t,
    # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: _index_t,
    np.uintp: _index_t,
    np.float16: _f16_t,
    np.float32: _f32_t,
    np.float64: _f64_t,
}

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    return _np_dtype_to_mlir_type_ctor[np_dtype]()


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype[mlir_type]


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F32Type, F64Type, RankedTensorType]:
    """Infer MLIR type (`ir.Type`) from supported python values.

    Note ints and floats are mapped to 64-bit types.

    Args:
      py_val: Python value that's either a numerical value or numpy array.

    Returns:
      MLIR type corresponding to py_val.
    """
    if isinstance(py_val, bool):
        return _bool_t()
    elif isinstance(py_val, int):
        if -(2 ** 31) <= py_val < 2 ** 31:
            return _i32_t()
        elif 2 ** 31 <= py_val < 2 ** 32:
            return _ui32_t()
        elif -(2 ** 63) <= py_val < 2 ** 63:
            return _i64_t()
        elif 2 ** 63 <= py_val < 2 ** 64:
            return _ui64_t()
        else:
            raise RuntimeError(f"Nonrepresentable integer {py_val}.")
    elif isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return _f32_t()
        else:
            return _f64_t()
    elif isinstance(py_val, np.ndarray):
        dtype = np_dtype_to_mlir_type(py_val.dtype.type)
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def shaped_t(*args, element_type: Type = None, type_constructor=None):
    if type_constructor is None:
        raise ValueError("shaped_t is an abstract base class - cannot be constructed")
    if (element_type is None and args and not isinstance(args[-1], Type)) or (
        args and isinstance(args[-1], Type) and element_type is not None
    ):
        raise ValueError(
            f"either element_type must be provided explicitly XOR last arg to tensor type constructor must be the element type"
        )
    if element_type is not None:
        type = element_type
        sizes = args
    else:
        type = args[-1]
        sizes = args[:-1]
    if sizes:
        return type_constructor(sizes, type)
    else:
        return type_constructor(type)


def vector_t(*args, element_type: Type = None):
    return shaped_t(*args, element_type=element_type, type_constructor=VectorType.get)


def tensor_t(*args, element_type: Type = None):
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped_t(
            *args, element_type=element_type, type_constructor=UnrankedTensorType.get
        )
    else:
        return shaped_t(
            *args, element_type=element_type, type_constructor=RankedTensorType.get
        )


def memref_t(*args, element_type: Type = None, memory_space: int = None):
    if memory_space is None:
        memory_space = 0
    memory_space = Attribute.parse(str(memory_space))
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped_t(
            *args,
            element_type=element_type,
            type_constructor=partial(UnrankedMemRefType.get, memory_space=memory_space),
        )
    else:
        return shaped_t(
            *args,
            element_type=element_type,
            type_constructor=partial(MemRefType.get, memory_space=memory_space),
        )
