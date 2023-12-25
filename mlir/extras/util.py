import contextlib
import ctypes
import inspect
import platform
import sys
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union, Sequence

import numpy as np

from .meta import op_region_builder
from ..ir import (
    Block,
    Context,
    F32Type,
    F64Type,
    InsertionPoint,
    IntegerType,
    Location,
    OpResult,
    OpResultList,
    OpView,
    Operation,
    RankedTensorType,
    Value,
    _GlobalDebug,
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
        if -(2 ** 31) <= py_val < 2 ** 31:
            return T.i32()
        elif 2 ** 31 <= py_val < 2 ** 32:
            return T.ui32()
        elif -(2 ** 63) <= py_val < 2 ** 63:
            return T.i64()
        elif 2 ** 63 <= py_val < 2 ** 64:
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


def region_adder(terminator=None):
    def wrapper(op_region_adder):
        def region_adder_decorator(op, *args, **kwargs):
            region = op_region_adder(op, *args, **kwargs)

            return op_region_builder(op, region, terminator)

        return region_adder_decorator

    return wrapper


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
