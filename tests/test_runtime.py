import ctypes
import re
from textwrap import dedent

import numpy as np
import pytest
from mlir.runtime import get_unranked_memref_descriptor

from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.func import func, declare
from mlir_utils.dialects.ext.scf import (
    canonicalizer,
)
from mlir_utils.runtime.passes import Pipeline
from mlir_utils.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import memref_t, f32_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@pytest.fixture
def backend() -> LLVMJITBackend:
    return LLVMJITBackend()


def test_tensor_additions(ctx: MLIRContext, backend: LLVMJITBackend, capfd):
    # TODO(max): ValueError: foo requires closure of length 0, not 1
    unranked_memref_f32_t = memref_t(element_type=f32_t)
    print_memref_32 = declare("printMemrefF32", [unranked_memref_f32_t])

    @func
    @canonicalize(using=canonicalizer)
    def foo(x: unranked_memref_f32_t):
        print_memref_32(x)
        return

    foo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
    )
    correct = dedent(
        """\
    module attributes {llvm.data_layout = ""} {
      llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
      llvm.func @foo(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
        llvm.call @printMemrefF32(%arg0, %arg1) : (i64, !llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr) attributes {llvm.emit_c_interface} {
        %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
        %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr)> 
        %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)> 
        llvm.call @foo(%1, %2) : (i64, !llvm.ptr) -> ()
        llvm.return
      }
    }
    """
    )
    filecheck(correct, ctx.module)
    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))
    backend.load(module).foo(AA)
    correct = dedent(
        """\
    Unranked Memref base@ =  rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data = 
    [[1,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1]]
    """
    )
    out, err = capfd.readouterr()
    filecheck(correct, re.sub(r"0x\w+", "", out))
