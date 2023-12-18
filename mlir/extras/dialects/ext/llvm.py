from ....ir import Type
from ....dialects.llvm import *


def llvm_ptr_t():
    return Type.parse("!llvm.ptr")
