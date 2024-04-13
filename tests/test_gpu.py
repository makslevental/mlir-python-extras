import sys
from textwrap import dedent

import pytest
from mlir.dialects._gpu_enum_gen import AllReduceOperation

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.llvm import llvm_ptr_t

from mlir.extras.dialects.ext.gpu import (
    thread_attr as thread,
    block_id_x,
    block_id_y,
    GPUModuleMeta,
    func as gpu_func,
    set_container_module,
    launch,
    all_reduce_,
)
from mlir.extras.dialects.ext.memref import alloc
from mlir.extras.dialects.ext.memref import load, store
from mlir.extras.dialects.ext.scf import canonicalizer
from mlir.extras.dialects.ext.scf import forall, in_parallel_
from mlir.dialects.gpu import host_register
from mlir.extras.dialects.ext.gpu import all_reduce, wait
from mlir.dialects.llvm import mlir_zero
from mlir.dialects.math import fma
from mlir.dialects.memref import cast
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic(ctx: MLIRContext):
    unranked_memref_f32 = T.memref(element_type=T.f32())
    mem = cast(unranked_memref_f32, alloc(10, 10, element_type=T.f32()))
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
    x = alloc(10, 10, T.f32())
    y = alloc(10, 10, T.f32())
    alpha = constant(1, T.f32())

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
        @gpu_func(emit=True)
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32()),
            B: T.memref(N, K, T.f32()),
            C: T.memref(M, K, T.f32()),
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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_class_call(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    set_container_module(ctx.module)

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func(emit=True)
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32()),
            B: T.memref(N, K, T.f32()),
            C: T.memref(M, K, T.f32()),
        ):
            x = block_id_x()
            y = block_id_y()
            a = A[x, y]
            b = B[x, y]
            C[x, y] = a * b
            return

    a = alloc(M, N, T.f32())
    b = alloc(N, K, T.f32())
    c = alloc(M, K, T.f32())

    # this is to avoid python 3.8 parser
    eval(
        "MyClass1.mat_product_kernel[grid_size:= [4, 4, 1], block_size:= [1, 1, 1]](a, b, c)"
    )

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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_class_call_from_func(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    set_container_module(ctx.module)

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func(emit=True)
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32()),
            B: T.memref(N, K, T.f32()),
            C: T.memref(M, K, T.f32()),
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
        a = alloc(M, N, T.f32())
        b = alloc(N, K, T.f32())
        c = alloc(M, K, T.f32())

        MyClass1
        eval(
            "MyClass1.mat_product_kernel[grid_size:= [4, 4, 1], block_size:= [1, 1, 1]](a, b, c)"
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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_async_object(ctx: MLIRContext):
    scale = 1
    M, N, K = 4 * scale, 16 * scale, 8 * scale

    set_container_module(ctx.module)

    class MyClass1(metaclass=GPUModuleMeta, targets=["#nvvm.target"]):
        @gpu_func(emit=True)
        @canonicalize(using=canonicalizer)
        def mat_product_kernel(
            A: T.memref(M, N, T.f32()),
            B: T.memref(N, K, T.f32()),
            C: T.memref(M, K, T.f32()),
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
        a = alloc(M, N, T.f32())
        b = alloc(N, K, T.f32())
        c = alloc(M, K, T.f32())

        w = wait()
        stream = mlir_zero(llvm_ptr_t())
        MyClass1
        eval(
            "MyClass1.mat_product_kernel[grid_size:= [4, 4, 1], block_size:= [1, 1, 1]](a, b, c, async_dependencies=[w], stream=stream)"
        )

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
        %0 = gpu.wait async
        %1 = llvm.mlir.zero : !llvm.ptr
        %c4 = arith.constant 4 : index
        %c4_2 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c1_3 = arith.constant 1 : index
        %c1_4 = arith.constant 1 : index
        %c1_5 = arith.constant 1 : index
        %2 = gpu.launch_func async [%0]<%1 : !llvm.ptr> @MyClass1::@mat_product_kernel blocks in (%c4, %c4_2, %c1) threads in (%c1_3, %c1_4, %c1_5)  args(%alloc : memref<4x16xf32>, %alloc_0 : memref<16x8xf32>, %alloc_1 : memref<4x8xf32>)
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_launch_op(ctx: MLIRContext):
    @func(emit=True)
    def main():
        data = alloc(2, 6, T.i32())
        sum = alloc(2, T.i32())

        power_csts = [constant(0)] + [constant(2**i) for i in range(5)]
        odd_csts = [constant(3), constant(6), constant(7), constant(10), constant(11)]
        cast_data = cast(T.memref(T.i32()), data)
        host_register(cast_data)
        cast_sum = cast(T.memref(T.i32()), sum)
        host_register(cast_sum)

        for i in range(6):
            data[0, i] = power_csts[i]

        data[1, 0] = power_csts[2]
        for i in range(0, 5):
            data[1, i + 1] = odd_csts[i]

        @launch(grid_size=[2, 1, 1], block_size=[6, 1, 1])
        def kernel(bx, by, bz, tx, ty, tz, *grid_block_sizes):
            val = data[bx, tx]

            @all_reduce(val, uniform=True)
            def reduced(lhs: T.i32(), rhs: T.i32()):
                return lhs

            sum[bx] = reduced
            return

    module = run_pipeline(ctx.module, Pipeline().cse())
    correct = dedent(
        """\
    module {
      func.func @main() {
        %alloc = memref.alloc() : memref<2x6xi32>
        %alloc_0 = memref.alloc() : memref<2xi32>
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c3_i32 = arith.constant 3 : i32
        %c6_i32 = arith.constant 6 : i32
        %c7_i32 = arith.constant 7 : i32
        %c10_i32 = arith.constant 10 : i32
        %c11_i32 = arith.constant 11 : i32
        %cast = memref.cast %alloc : memref<2x6xi32> to memref<*xi32>
        gpu.host_register %cast : memref<*xi32>
        %cast_1 = memref.cast %alloc_0 : memref<2xi32> to memref<*xi32>
        gpu.host_register %cast_1 : memref<*xi32>
        %c0 = arith.constant 0 : index
        memref.store %c0_i32, %alloc[%c0, %c0] : memref<2x6xi32>
        %c1 = arith.constant 1 : index
        memref.store %c1_i32, %alloc[%c0, %c1] : memref<2x6xi32>
        %c2 = arith.constant 2 : index
        memref.store %c2_i32, %alloc[%c0, %c2] : memref<2x6xi32>
        %c3 = arith.constant 3 : index
        memref.store %c4_i32, %alloc[%c0, %c3] : memref<2x6xi32>
        %c4 = arith.constant 4 : index
        memref.store %c8_i32, %alloc[%c0, %c4] : memref<2x6xi32>
        %c5 = arith.constant 5 : index
        memref.store %c16_i32, %alloc[%c0, %c5] : memref<2x6xi32>
        memref.store %c2_i32, %alloc[%c1, %c0] : memref<2x6xi32>
        memref.store %c3_i32, %alloc[%c1, %c1] : memref<2x6xi32>
        memref.store %c6_i32, %alloc[%c1, %c2] : memref<2x6xi32>
        memref.store %c7_i32, %alloc[%c1, %c3] : memref<2x6xi32>
        memref.store %c10_i32, %alloc[%c1, %c4] : memref<2x6xi32>
        memref.store %c11_i32, %alloc[%c1, %c5] : memref<2x6xi32>
        %c6 = arith.constant 6 : index
        gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c2, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c6, %arg10 = %c1, %arg11 = %c1) {
          %0 = memref.load %alloc[%arg0, %arg3] : memref<2x6xi32>
          %1 = gpu.all_reduce  %0 uniform {
          ^bb0(%arg12: i32, %arg13: i32):
            gpu.yield %arg12 : i32
          } : (i32) -> i32
          memref.store %1, %alloc_0[%arg0] : memref<2xi32>
          gpu.terminator
        }
        return
      }
    }
    """
    )

    filecheck(correct, module)


def test_launch_op_reduce_op(ctx: MLIRContext):
    @func(emit=True)
    def main():
        data = alloc(2, 6, T.i32())
        sum = alloc(2, T.i32())

        power_csts = [constant(0)] + [constant(2**i) for i in range(5)]
        odd_csts = [constant(3), constant(6), constant(7), constant(10), constant(11)]
        cast_data = cast(T.memref(T.i32()), data)
        host_register(cast_data)
        cast_sum = cast(T.memref(T.i32()), sum)
        host_register(cast_sum)

        for i in range(6):
            data[0, i] = power_csts[i]

        data[1, 0] = power_csts[2]
        for i in range(0, 5):
            data[1, i + 1] = odd_csts[i]

        @launch(grid_size=[2, 1, 1], block_size=[6, 1, 1])
        def kernel(bx, by, bz, tx, ty, tz, *grid_block_sizes):
            val = data[bx, tx]

            reduced = all_reduce_(val, op=AllReduceOperation.AND, uniform=True)

            sum[bx] = reduced
            return

    module = run_pipeline(ctx.module, Pipeline().cse())

    correct = dedent(
        """\
    module {
      func.func @main() {
        %alloc = memref.alloc() : memref<2x6xi32>
        %alloc_0 = memref.alloc() : memref<2xi32>
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c3_i32 = arith.constant 3 : i32
        %c6_i32 = arith.constant 6 : i32
        %c7_i32 = arith.constant 7 : i32
        %c10_i32 = arith.constant 10 : i32
        %c11_i32 = arith.constant 11 : i32
        %cast = memref.cast %alloc : memref<2x6xi32> to memref<*xi32>
        gpu.host_register %cast : memref<*xi32>
        %cast_1 = memref.cast %alloc_0 : memref<2xi32> to memref<*xi32>
        gpu.host_register %cast_1 : memref<*xi32>
        %c0 = arith.constant 0 : index
        memref.store %c0_i32, %alloc[%c0, %c0] : memref<2x6xi32>
        %c1 = arith.constant 1 : index
        memref.store %c1_i32, %alloc[%c0, %c1] : memref<2x6xi32>
        %c2 = arith.constant 2 : index
        memref.store %c2_i32, %alloc[%c0, %c2] : memref<2x6xi32>
        %c3 = arith.constant 3 : index
        memref.store %c4_i32, %alloc[%c0, %c3] : memref<2x6xi32>
        %c4 = arith.constant 4 : index
        memref.store %c8_i32, %alloc[%c0, %c4] : memref<2x6xi32>
        %c5 = arith.constant 5 : index
        memref.store %c16_i32, %alloc[%c0, %c5] : memref<2x6xi32>
        memref.store %c2_i32, %alloc[%c1, %c0] : memref<2x6xi32>
        memref.store %c3_i32, %alloc[%c1, %c1] : memref<2x6xi32>
        memref.store %c6_i32, %alloc[%c1, %c2] : memref<2x6xi32>
        memref.store %c7_i32, %alloc[%c1, %c3] : memref<2x6xi32>
        memref.store %c10_i32, %alloc[%c1, %c4] : memref<2x6xi32>
        memref.store %c11_i32, %alloc[%c1, %c5] : memref<2x6xi32>
        %c6 = arith.constant 6 : index
        gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c2, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c6, %arg10 = %c1, %arg11 = %c1) {
          %0 = memref.load %alloc[%arg0, %arg3] : memref<2x6xi32>
          %1 = gpu.all_reduce  and %0 uniform {
          } : (i32) -> i32
          memref.store %1, %alloc_0[%arg0] : memref<2xi32>
          gpu.terminator
        }
        return
      }
    }
    """
    )

    filecheck(correct, module)
