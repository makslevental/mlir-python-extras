from __future__ import annotations

import math
from pathlib import Path

import mlir.extras.types as T
import numpy as np

from mlir import _mlir_libs
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import (
    mlir_mod_ctx,
    MLIRContext,
)
from mlir.extras.dialects.ext import arith, memref, scf, gpu
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.scf import range_
from mlir.extras.runtime.passes import Pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend
from mlir.extras.util import find_ops

CUDA_RUNTIME_LIB_PATH = Path(_mlir_libs.__file__).parent / f"libmlir_cuda_runtime.so"
assert CUDA_RUNTIME_LIB_PATH.exists()


def print_ptx(compiled_module):
    ptx = find_ops(compiled_module, lambda o: o.name == "gpu.binary", single=True)
    ptx = str(ptx.objects).replace("\\0A", "\n").replace("\\09", "\t")
    print(ptx)


@func
def naive[
    M, K, N, dtype, BLOCK_SIZE: 32
](lhs: T.memref(M, K, dtype), rhs: T.memref(K, N, dtype), res: T.memref(M, N, dtype),):
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


def main(ctx: MLIRContext):
    M, K, N = 32, 32, 32
    npy_dtype = np.float32
    naive[M, K, N, T.f32()].emit()
    # print(ctx.module)

    backend = LLVMJITBackend([CUDA_RUNTIME_LIB_PATH])
    compiled_module = backend.compile(
        ctx.module,
        Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            **{
                "cubin-chip": "sm_80",
                "cubin-features": "+ptx76",
                "cubin-format": "isa",
            },
        ),
        kernel_name="naive",
    )
    print_ptx(compiled_module)

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.random.randint(0, 10, (M, N)).astype(npy_dtype)
    res: np.ndarray = backend.load(compiled_module).naive_capi_wrapper(A, B, C)

    assert np.array_equal(res, A @ B)


with mlir_mod_ctx() as ctx:
    main(ctx)
