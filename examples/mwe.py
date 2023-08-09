import ctypes

import numpy as np
from mlir.runtime import get_ranked_memref_descriptor

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import mlir_utils.dialects.ext.memref
import mlir_utils.types as T
from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.context import MLIRContext, mlir_mod_ctx
from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.ext.scf import canonicalizer, range_
from mlir_utils.runtime.passes import Pipeline
from mlir_utils.runtime.refbackend import LLVMJITBackend


def setting_memref(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32)

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(
        A: ranked_memref_kxk_f32,
        B: ranked_memref_kxk_f32,
        C: ranked_memref_kxk_f32,
    ):
        for i in range_(0, K):
            for j in range_(0, K):
                C[i, j] = A[i, j] * B[i, j]

    memfoo.emit()
    print(ctx.module)

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )

    invoker = backend.load(module)
    A = np.random.randint(0, 10, (K, K)).astype(np.float32)
    B = np.random.randint(0, 10, (K, K)).astype(np.float32)
    C = np.zeros((K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))
    BB = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(B)))
    CC = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(C)))

    invoker.memfoo_capi_wrapper(AA, BB, CC)
    assert np.array_equal(A * B, C)


with mlir_mod_ctx() as ctx:
    setting_memref(ctx, LLVMJITBackend())
