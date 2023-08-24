from textwrap import dedent

import pytest

import mlir_utils.types as T
from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.ext.gpu import (
    thread_attr as thread,
    block_id_x,
    block_id_y,
    GPUModuleMeta,
    gpu_func,
    set_container_module,
)
from mlir_utils.dialects.ext.memref import alloc
from mlir_utils.dialects.ext.memref import load, store
from mlir_utils.dialects.ext.scf import canonicalizer
from mlir_utils.dialects.ext.scf import forall, in_parallel_
from mlir_utils.dialects.gpu import host_register
from mlir_utils.dialects.math import fma
from mlir_utils.dialects.memref import cast

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic(ctx: MLIRContext):
    unranked_memref_f32 = T.memref(element_type=T.f32)
    mem = cast(unranked_memref_f32, alloc((10, 10), element_type=T.f32))
    host_register(mem)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xf32>
      %cast = memref.cast %alloc : memref<10x10xf32> to memref<*xf32>
      gpu.host_register %cast : memref<*xf32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_forall_insert_slice_no_region_with_for_with_gpu_mapping(ctx: MLIRContext):
    x = alloc((10, 10), T.f32)
    y = alloc((10, 10), T.f32)
    alpha = constant(1, T.f32)

    for i, j in forall(
        [1, 1],
        [2, 2],
        [3, 3],
        device_mapping=[thread("x"), thread("y")],
    ):
        a = load(x, (i, j))
        b = load(y, (i, j))
        c = fma(alpha, a, b)
        store(c, y, (i, j))

        in_parallel_()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xf32>
      %alloc_0 = memref.alloc() : memref<10x10xf32>
      %cst = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2_2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c3_3 = arith.constant 3 : index
      scf.forall (%arg0, %arg1) = (1, 1) to (2, 2) step (3, 3) {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<10x10xf32>
        %1 = memref.load %alloc_0[%arg0, %arg1] : memref<10x10xf32>
        %2 = math.fma %cst, %0, %1 : f32
        memref.store %2, %alloc_0[%arg0, %arg1] : memref<10x10xf32>
      } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
    }
    """
    )
    filecheck(correct, ctx.module)


def test_class(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32),
            B: T.memref(N, K, T.f32),
            C: T.memref(M, K, T.f32),
        ):
            x = block_id_x()
            y = block_id_y()
            a = A[x, y]
            b = B[x, y]
            C[x, y] = a * b
            return

    correct = dedent(
        """\
    module {
      gpu.module @MyClass1 [#nvvm.target]  {
        gpu.func @mat_product_kernel(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) kernel {
          %0 = gpu.block_id  x
          %1 = gpu.block_id  y
          %2 = memref.load %arg0[%0, %1] : memref<4x16xf32>
          %3 = memref.load %arg1[%0, %1] : memref<16x8xf32>
          %4 = arith.mulf %2, %3 : f32
          memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
          gpu.return
        }
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_class_call(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    set_container_module(ctx.module)

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32),
            B: T.memref(N, K, T.f32),
            C: T.memref(M, K, T.f32),
        ):
            x = block_id_x()
            y = block_id_y()
            a = A[x, y]
            b = B[x, y]
            C[x, y] = a * b
            return

    a = alloc((M, N), T.f32)
    b = alloc((N, K), T.f32)
    c = alloc((M, K), T.f32)

    MyClass1.mat_product_kernel(a, b, c, grid_size=[4, 4, 1], block_size=[1, 1, 1])

    correct = dedent(
        """\
    module attributes {gpu.container_module} {
      gpu.module @MyClass1 [#nvvm.target]  {
        gpu.func @mat_product_kernel(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) kernel {
          %0 = gpu.block_id  x
          %1 = gpu.block_id  y
          %2 = memref.load %arg0[%0, %1] : memref<4x16xf32>
          %3 = memref.load %arg1[%0, %1] : memref<16x8xf32>
          %4 = arith.mulf %2, %3 : f32
          memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
          gpu.return
        }
      }
      %alloc = memref.alloc() : memref<4x16xf32>
      %alloc_0 = memref.alloc() : memref<16x8xf32>
      %alloc_1 = memref.alloc() : memref<4x8xf32>
      %c4 = arith.constant 4 : index
      %c4_2 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      gpu.launch_func  @MyClass1::@mat_product_kernel blocks in (%c4, %c4_2, %c1) threads in (%c1_3, %c1_4, %c1_5)  args(%alloc : memref<4x16xf32>, %alloc_0 : memref<16x8xf32>, %alloc_1 : memref<4x8xf32>)
    }
    """
    )

    filecheck(correct, ctx.module)


def test_class_call_from_func(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    set_container_module(ctx.module)

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32),
            B: T.memref(N, K, T.f32),
            C: T.memref(M, K, T.f32),
        ):
            x = block_id_x()
            y = block_id_y()
            a = A[x, y]
            b = B[x, y]
            C[x, y] = a * b
            return

        def test(self):
            pass

    @func(emit=True)
    @canonicalize(using=canonicalizer)
    def main():
        a = alloc((M, N), T.f32)
        b = alloc((N, K), T.f32)
        c = alloc((M, K), T.f32)

        MyClass1.mat_product_kernel["grid_size":[4, 4, 1], "block_size":[1, 1, 1]](
            a, b, c
        )

    ctx.module.operation.verify()

    correct = dedent(
        """\
    module attributes {gpu.container_module} {
      gpu.module @MyClass1 [#nvvm.target]  {
        gpu.func @mat_product_kernel(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) kernel {
          %0 = gpu.block_id  x
          %1 = gpu.block_id  y
          %2 = memref.load %arg0[%0, %1] : memref<4x16xf32>
          %3 = memref.load %arg1[%0, %1] : memref<16x8xf32>
          %4 = arith.mulf %2, %3 : f32
          memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
          gpu.return
        }
      }
      func.func @main() {
        %alloc = memref.alloc() : memref<4x16xf32>
        %alloc_0 = memref.alloc() : memref<16x8xf32>
        %alloc_1 = memref.alloc() : memref<4x8xf32>
        %c4 = arith.constant 4 : index
        %c4_2 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c1_3 = arith.constant 1 : index
        %c1_4 = arith.constant 1 : index
        %c1_5 = arith.constant 1 : index
        gpu.launch_func  @MyClass1::@mat_product_kernel blocks in (%c4, %c4_2, %c1) threads in (%c1_3, %c1_4, %c1_5)  args(%alloc : memref<4x16xf32>, %alloc_0 : memref<16x8xf32>, %alloc_1 : memref<4x8xf32>)
        return
      }
    }
    """
    )

    filecheck(correct, ctx.module)
