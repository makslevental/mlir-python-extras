import math
from pathlib import Path

import mlir.extras.types as T
import numpy as np
from mlir.dialects import builtin
from mlir.ir import UnitAttr

from mlir import _mlir_libs
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import (
    mlir_mod_ctx,
    MLIRContext,
)
from mlir.extras.dialects.ext import arith, memref, scf, gpu
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.gpu import block_id, thread_id, block_dim, GPUModuleMeta
from mlir.extras.dialects.ext.scf import range_
from mlir.extras.runtime.passes import Pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir.extras.util import find_ops, enable_debug as enable_debug

CUDA_RUNTIME_LIB_PATH = Path(_mlir_libs.__file__).parent / f"libmlir_cuda_runtime.so"
assert CUDA_RUNTIME_LIB_PATH.exists()


def print_ptx(compiled_module):
    ptx = find_ops(compiled_module, lambda o: o.name == "gpu.binary", single=True)
    ptx = str(ptx.objects).replace("\\0A", "\n").replace("\\09", "\t")
    print(ptx)


@func
def naive[
    M, K, N, dtype, BLOCK_SIZE: 32
](
    lhs: "T.memref(M, K, dtype)",
    rhs: "T.memref(K, N, dtype)",
    res: "T.memref(M, N, dtype)",
):
    ulhs = memref.cast(T.memref(dtype), lhs)
    urhs = memref.cast(T.memref(dtype), rhs)
    ures = memref.cast(T.memref(dtype), res)

    gpu.host_register(ulhs)
    gpu.host_register(urhs)
    gpu.host_register(ures)

    index = T.index()

    @gpu.launch(
        grid_size=[math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE), 1],
        block_size=[BLOCK_SIZE, BLOCK_SIZE, 1],
    )
    @canonicalize(using=scf.canonicalizer)
    def kernel(
        bx: index,
        by: index,
        _bz: index,
        tx: index,
        ty: index,
        _tz: index,
        num_bx: index,
        num_by: index,
        _num_bz: index,
        _num_tx: index,
        _num_ty: index,
        _num_tz: index,
    ):
        x = bx * num_bx + tx
        y = by * num_by + ty

        tmp = arith.constant(0, type=dtype)
        for k, tmp in range_(K, iter_args=[tmp]):
            tmp += lhs[x, k] * rhs[k, y]
            tmp = yield tmp
        res[x, y] = tmp

    return res


def naive_with_gpu_func(M, K, N, dtype, BLOCK_SIZE=32):
    @builtin.module(attrs={"gpu.container_module": UnitAttr.get()})
    def mod():
        class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):

            @gpu.func(emit=True)
            @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
            def mat_product_kernel(
                A: T.memref(M, K, dtype),
                B: T.memref(K, N, dtype),
                C: T.memref(M, N, dtype),
            ):
                x = block_dim.x * block_id.x + thread_id.x
                y = block_dim.y * block_id.y + thread_id.y

                tmp = arith.constant(0, type=dtype)
                for k, tmp in range_(K, iter_args=[tmp]):
                    tmp += A[x, k] * B[k, y]
                    tmp = yield tmp
                C[x, y] = tmp

        @func(emit=True)
        @canonicalize(using=scf.canonicalizer)
        def naive(
            A: T.memref(M, K, dtype),
            B: T.memref(K, N, dtype),
            C: T.memref(M, N, dtype),
        ):
            # this is a cuda stream
            w = gpu.wait()
            dA, _ = gpu.alloc(A.shape, A.dtype, [w])
            dB, _ = gpu.alloc(B.shape, B.dtype, [w])
            dC, _ = gpu.alloc(C.shape, C.dtype, [w])

            gpu.memcpy(dA, A, [w])
            gpu.memcpy(dB, B, [w])
            gpu.memcpy(dC, C, [w])

            MyClass1.mat_product_kernel[
                grid_size := [math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE), 1],
                block_size := [BLOCK_SIZE, BLOCK_SIZE, 1],
                async_dependencies := [w],
            ](dA, dB, dC)
            gpu.wait([w])

            gpu.dealloc(dA, [w])
            gpu.dealloc(dB, [w])

            gpu.memcpy(C, dC, [w])
            gpu.wait([w])
            gpu.dealloc(dC, [w])

            return C

    return mod.opview


def main(ctx: MLIRContext):
    M, K, N = 32, 32, 32
    dtype = T.f32()
    npy_dtype = np.float32

    module = naive_with_gpu_func(M, K, N, dtype)

    # naive[M, K, N, dtype].emit()
    # module = ctx.module

    print(module)
    print(module.operation.verify())

    backend = LLVMJITBackend([CUDA_RUNTIME_LIB_PATH])
    compiled_module = backend.compile(
        module,
        Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            **{
                "cubin-chip": "sm_80",
                "cubin-features": "+ptx83",
                "cubin-format": "isa",
            },
        ),
        kernel_name="naive",
        enable_ir_printing=True,
    )
    print(compiled_module)
    print_ptx(compiled_module)

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)
    res: np.ndarray = backend.load(compiled_module, opt_level=3).naive_capi_wrapper(
        A, B, C
    )

    if not np.array_equal(res, A @ B):
        print(res)


with (
    mlir_mod_ctx() as ctx,
    # enable_debug()
):
    main(ctx)
