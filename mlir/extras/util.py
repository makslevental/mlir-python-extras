import contextlib
import ctypes
import inspect
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..ir import (
    Block,
    Context,
    Location,
    OpResult,
    OpResultList,
    OpView,
    Operation,
    Value,
    _GlobalDebug,
    IntegerType,
    F32Type,
    F64Type,
    RankedTensorType,
)
from ..extras import types as T

try:
    from ..ir import TypeID
except ImportError:
    warnings.warn(
        f"TypeID not supported by host bindings; value casting won't work correctly"
    )
    TypeID = object


def get_user_code_loc(user_base: Optional[Path] = None):
    from .. import extras

    if Context.current is None:
        return

    mlir_extras_root_path = Path(extras.__path__[0])

    prev_frame = inspect.currentframe().f_back
    if user_base is None:
        user_base = Path(prev_frame.f_code.co_filename)

    while prev_frame.f_back and (
        Path(prev_frame.f_code.co_filename).is_relative_to(mlir_extras_root_path)
        or Path(prev_frame.f_code.co_filename).is_relative_to(sys.prefix)
        or Path(prev_frame.f_code.co_filename).is_relative_to(user_base)
    ):
        prev_frame = prev_frame.f_back
    frame_info = inspect.getframeinfo(prev_frame)
    if sys.version_info.minor >= 11:
        return Location.file(
            frame_info.filename, frame_info.lineno, frame_info.positions.col_offset
        )
    elif sys.version_info.minor == 10:
        return Location.file(frame_info.filename, frame_info.lineno, col=0)
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


@contextlib.contextmanager
def enable_debug():
    _GlobalDebug.flag = True
    yield
    _GlobalDebug.flag = False


def shlib_ext():
    if platform.system() == "Darwin":
        shlib_ext = "dylib"
    elif platform.system() == "Linux":
        shlib_ext = "so"
    elif platform.system() == "Windows":
        shlib_ext = "lib"
    else:
        raise NotImplementedError(f"unknown platform {platform.system()}")

    return shlib_ext


def shlib_prefix():
    if platform.system() in {"Darwin", "Linux"}:
        shlib_pref = "lib"
    elif platform.system() == "Windows":
        shlib_pref = ""
    else:
        raise NotImplementedError(f"unknown platform {platform.system()}")

    return shlib_pref


def find_ops(op, pred: Callable[[OpView], bool], single=False):
    matching = []

    def find(op):
        if single and len(matching):
            return
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if pred(o):
                        matching.append(o)
                    find(o)

    find(op)
    if single:
        matching = matching[0]
    return matching


@dataclass
class Successor:
    op: OpView | Operation
    operands: list[Value]
    block: Block
    pos: int


_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,
    # windows
    np.intc: T.i32,
    np.int64: T.i64,
    # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: T.index,
    np.uintp: T.index,
    np.float16: T.f16,
    np.float32: T.f32,
    np.float64: T.f64,
}

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)


_mlir_type_to_ctype = {
    T.bool: ctypes.c_bool,
    T.i8: ctypes.c_byte,
    T.i64: ctypes.c_int,
    T.f32: ctypes.c_float,
    T.f64: ctypes.c_double,
}


def mlir_type_to_ctype(mlir_type):
    __mlir_type_to_ctype = {k(): v for k, v in _mlir_type_to_ctype.items()}
    return _mlir_type_to_ctype.get(mlir_type)


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
        return T.bool()
    elif isinstance(py_val, int):
        if -(2**31) <= py_val < 2**31:
            return T.i32()
        elif 2**31 <= py_val < 2**32:
            return T.ui32()
        elif -(2**63) <= py_val < 2**63:
            return T.i64()
        elif 2**63 <= py_val < 2**64:
            return T.ui64()
        else:
            raise RuntimeError(f"Nonrepresentable integer {py_val}.")
    elif isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return T.f32()
        else:
            return T.f64()
    elif isinstance(py_val, np.ndarray):
        dtype = np_dtype_to_mlir_type(py_val.dtype.type)
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def memref_type_to_np_dtype(memref_type):
    _memref_type_to_np_dtype = {
        T.memref(element_type=T.f16()): np.float16,
        T.memref(element_type=T.f32()): np.float32,
        T.memref(T.f64()): np.float64,
        T.memref(element_type=T.bool()): np.bool_,
        T.memref(T.i8()): np.int8,
        T.memref(T.i32()): np.int32,
        T.memref(T.i64()): np.int64,
    }
    return _memref_type_to_np_dtype.get(memref_type)
