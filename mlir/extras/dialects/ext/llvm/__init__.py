# noinspection PyUnresolvedReferences
from .....dialects.llvm import *
from .....ir import Type, Value

ValueRef = Value

def llvm_ptr_t():
    return Type.parse("!llvm.ptr")

try:
    from . import amdgcn
except ImportError:
    pass
