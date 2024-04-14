import ast
import math
import re
from pathlib import Path

import mlir.extras.types as T
import numpy as np
import cupy as cp
from cupy.cuda.function import Module
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


def get_ptx(compiled_module):
    binary = find_ops(compiled_module, lambda o: o.name == "gpu.binary", single=True)
    r = re.findall(r'"(.*?)"', str(binary.objects[1]))
    return r[-1]


def print_ptx(compiled_module):
    ptx = get_ptx(compiled_module)
    ptx = str(ptx).replace("\\0A", "\n").replace("\\09", "\t")
    print(ptx)


def build_cuda_func(compiled_module, kernel_name="mat_product_kernel"):
    ptx = get_ptx(compiled_module)
    ptx = re.sub(r"\\(\w\w)", lambda m: r"\x" + m.groups(0)[0].lower(), ptx)
    ptx = ast.literal_eval(rf"b'{ptx}'")
    mod = Module()
    mod.load(ptx)
    cuda_func = mod.get_function(kernel_name)
    return cuda_func


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
    BLOCK_SIZE = 32
    dtype = T.f32()
    npy_dtype = np.float32

    module = naive_with_gpu_func(M, K, N, dtype)

    # naive[M, K, N, dtype].emit()
    # module = ctx.module

    # print(module)
    # print(module.operation.verify())

    backend = LLVMJITBackend([CUDA_RUNTIME_LIB_PATH])
    compiled_module = backend.compile(
        module,
        Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            **{
                "cubin-chip": "sm_80",
                "cubin-features": "+ptx83",
                # "cubin-format": "isa",
                # "cubin-format": "fatbin",
                "cubin-format": "bin",
            },
        ),
        kernel_name="naive",
        # enable_ir_printing=True,
    )
    # print(compiled_module)
    # print_ptx(compiled_module)

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)

    dA = cp.asarray(A)
    dB = cp.asarray(B)
    dC = cp.asarray(C)

    args = (
        mat_product_kernel_param_0,
        mat_product_kernel_param_1,
        mat_product_kernel_param_2,
        mat_product_kernel_param_3,
        mat_product_kernel_param_4,
        mat_product_kernel_param_5,
        mat_product_kernel_param_6,
        mat_product_kernel_param_7,
        mat_product_kernel_param_8,
        mat_product_kernel_param_9,
        mat_product_kernel_param_10,
        mat_product_kernel_param_11,
        mat_product_kernel_param_12,
        mat_product_kernel_param_13,
        mat_product_kernel_param_14,
        mat_product_kernel_param_15,
        mat_product_kernel_param_16,
        mat_product_kernel_param_17,
        mat_product_kernel_param_18,
        mat_product_kernel_param_19,
        mat_product_kernel_param_20,
    ) = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23)

    # mat_product_kernel_param_15 = dA.data.ptr
    # mat_product_kernel_param_1 = dB.data.ptr
    # mat_product_kernel_param_8 = dC.data.ptr

    cuda_func = build_cuda_func(compiled_module)
    cuda_func(
        (math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE), 1),
        (BLOCK_SIZE, BLOCK_SIZE, 1),
        args,
    )

    print(dC)

    # res: np.ndarray = backend.load(compiled_module, opt_level=3).naive_capi_wrapper(
    #     A, B, C
    # )
    #
    # if not np.array_equal(res, A @ B):
    #     print(res)


with (
    mlir_mod_ctx() as ctx,
    # enable_debug()
):
    main(ctx)
