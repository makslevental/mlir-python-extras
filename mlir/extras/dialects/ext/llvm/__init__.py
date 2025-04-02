from typing import Union, Optional

from ....util import infer_mlir_type

# noinspection PyUnresolvedReferences
from .....dialects.llvm import *
from .....ir import Type, Value, IntegerAttr, FloatAttr
from .....dialects._ods_common import get_op_result_or_op_results

ValueRef = Value


def llvm_ptr_t():
    return Type.parse("!llvm.ptr")


try:
    from . import amdgcn
except ImportError:
    pass


def mlir_constant(
    value: Union[int, float, bool], type: Optional[Type] = None, *, loc=None, ip=None
) -> Value:
    if type is None:
        type = infer_mlir_type(value, vector=False)

    if isinstance(value, int):
        value = IntegerAttr.get(type, value)
    elif isinstance(value, float):
        value = FloatAttr.get(type, value)
    else:
        raise NotImplementedError(f"{value} is not a valid type")

    return get_op_result_or_op_results(
        ConstantOp(res=value.type, value=value, loc=loc, ip=ip)
    )
