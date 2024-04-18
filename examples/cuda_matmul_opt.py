from __future__ import annotations

import contextlib
import math

import cupy as cp
import mlir.extras.types as T
import numpy as np
from cupy.cuda import Module

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import (
    mlir_mod_ctx,
    MLIRContext,
)
from mlir.extras.dialects.ext import arith, memref, gpu, scf
from mlir.extras.dialects.ext.gpu import (
    block_id,
    thread_id,
    block_dim,
    get_compile_object_bytes,
)
from mlir.extras.dialects.ext.scf import range_
from mlir.extras.runtime.passes import Pipeline, run_pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.util import find_ops, enable_debug as enable_debug

# just so it doesn't get DCE'd by black/reformat
_ = memref


def build_cuda_func(compiled_module, kernel_name="naive"):
    ptx = get_compile_object_bytes(compiled_module)
    mod = Module()
    mod.load(ptx)
    return mod.get_function(kernel_name)


@contextlib.contextmanager
def time_cuda():
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    yield start_gpu, end_gpu
    end_gpu.record()
    end_gpu.synchronize()


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def naive[
    M, K, N, dtype
](A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)):
    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    x = block_dim.x * block_id.x + thread_id.x
    y = block_dim.y * block_id.y + thread_id.y
    for k, tmp in range_(K, iter_args=[tmp]):
        tmp += A[x, k] * B[k, y]
        tmp = yield tmp
    C[x, y] = tmp + one


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def naive_global_mem_coalesce[
    M, K, N, dtype, BLOCK_SIZE
](A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)):
    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    c_row = block_id.x * BLOCK_SIZE + (thread_id.x / BLOCK_SIZE)
    c_col = block_id.y * BLOCK_SIZE + (thread_id.x % BLOCK_SIZE)
    for k, tmp in range_(K, iter_args=[tmp]):
        tmp += A[c_row, k] * B[k, c_col]
        tmp = yield tmp
    C[c_row, c_col] = tmp + one


def main(ctx: MLIRContext, M, K, N, BLOCK_SIZE=32, repeat_times=50):
    dtype = T.f32()
    npy_dtype = np.float32

    gpu.set_container_module(ctx.module)

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        naive_global_mem_coalesce[M, K, N, dtype, BLOCK_SIZE].emit()

    # print(ctx.module)
    ctx.module.operation.verify()

    compiled_module = run_pipeline(
        ctx.module,
        Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            # https://github.com/llvm/llvm-project/blob/ace69e6b942b8fa7e610d70be2a92e801ceea481/mlir/include/mlir/Dialect/GPU/Pipelines/Passes.h#L18
            **{
                "cubin-chip": "sm_80",
                "cubin-features": "+ptx83",
                "cubin-format": "isa",
                "kernel-bare-ptr-calling-convention": "1",
                # "cubin-format": "fatbin",
                # "cubin-format": "bin",
            },
        ),
    )
    kernel_name = matmul_mod.opview.body.operations[0].attributes["sym_name"].value
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print(compiled_module)
    # print_ptx(compiled_module)

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)

    dA = cp.asarray(A)
    dB = cp.asarray(B)
    dC = cp.asarray(C)

    grid_dims = (math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE), 1)
    # naive
    # block_dims = (BLOCK_SIZE, BLOCK_SIZE, 1)
    # global coalesce
    block_dims = (BLOCK_SIZE * BLOCK_SIZE, 1, 1)

    cuda_func(grid_dims, block_dims, (dA.data.ptr, dB.data.ptr, dC.data.ptr))
    if not cp.array_equal(dC, dA @ dB + 1):
        print(dA @ dB + 1)
        print(dC)
        assert False

    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cuda_func(grid_dims, block_dims, (dA.data.ptr, dB.data.ptr, dC.data.ptr))

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t_gpu={t_gpu / repeat_times:.6f} ms")


for s in [128, 256, 512, 1024]:
    with (
        mlir_mod_ctx() as ctx,
        # enable_debug()
    ):
        main(ctx, s, s, s)
