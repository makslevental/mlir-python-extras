import contextlib
import inspect
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..dialects._ods_common import get_op_result_or_value, get_op_results_or_values
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
)

try:
    from ..ir import TypeID
except ImportError:
    warnings.warn(
        f"TypeID not supported by host bindings; value casting won't work correctly"
    )
    TypeID = object


def get_result_or_results(
    op: type(None) | OpView | Value,
) -> type(None) | Value | OpResultList | OpResult | OpView:
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


def get_user_code_loc(user_base: Optional[Path] = None):
    from .. import utils

    try:
        Context.current
    except ValueError as e:
        assert e.args[0] == "No current Context"
        return None

    mlir_utils_root_path = Path(utils.__path__[0])

    prev_frame = inspect.currentframe().f_back
    if user_base is None:
        user_base = Path(prev_frame.f_code.co_filename)

    while prev_frame.f_back and (
        Path(prev_frame.f_code.co_filename).is_relative_to(mlir_utils_root_path)
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
