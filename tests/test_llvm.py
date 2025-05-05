from textwrap import dedent

import mlir.extras.types as T
import pytest

from mlir.extras.dialects.ext import llvm
from mlir.extras.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
from util import llvm_bindings_not_installed, llvm_amdgcn_bindings_not_installed

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@pytest.mark.skipif(
    llvm_bindings_not_installed() or llvm_amdgcn_bindings_not_installed(),
    reason="llvm bindings not installed or llvm_amdgcn bindings not installed",
)
def test_call_instrinsic(ctx: MLIRContext):
    @func(emit=True)
    def sum(a: T.i32(), b: T.i32(), c: T.f32()):
        e = llvm.amdgcn.cvt_pk_i16(a, b)
        f = llvm.amdgcn.frexp_mant(c)

    correct = dedent(
        """
    module {
      func.func @sum(%arg0: i32, %arg1: i32, %arg2: f32) {
        %0 = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.i16"(%arg0, %arg1) : (i32, i32) -> vector<2xi16>
        %1 = llvm.call_intrinsic "llvm.amdgcn.frexp.mant"(%arg2) : (f32) -> f32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)
