import re
from pathlib import Path
from textwrap import dedent

import pytest
from mlir import _mlir_libs
from mlir.dialects.transform.extras import named_sequence

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import arith, memref, scf, gpu, linalg, transform
from mlir.dialects.transform import any_op_t
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.nvgpu import tensormap_descriptor
from mlir.dialects.transform.structured import MatchInterfaceEnum
from mlir.dialects.memref import cast
from mlir.dialects.nvgpu import tma_create_descriptor
from mlir.dialects import builtin
from mlir.extras.runtime.passes import run_pipeline, Pipeline
from mlir.ir import StringAttr, UnitAttr

from mlir.extras.runtime.refbackend import LLVMJITBackend
from mlir.extras.util import find_ops

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic(ctx: MLIRContext):
    @func
    def create_tensor_map(
        device_ptr_2d: T.memref(64, 128, element_type=T.f32()),
    ):
        crd0 = arith.constant(64, index=True)
        crd1 = arith.constant(128, index=True)
        device_ptr_2d_unranked = cast(T.memref(element_type=T.f32()), device_ptr_2d)
        tensor_map_2d = tensormap_descriptor(T.memref(32, 32, T.f32(), memory_space=3))
        tensor_map_2d = tma_create_descriptor(
            tensor_map_2d, device_ptr_2d_unranked, [crd0, crd1]
        )

    create_tensor_map.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func @create_tensor_map(%arg0: memref<64x128xf32>) {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %cast = memref.cast %arg0 : memref<64x128xf32> to memref<*xf32>
        %0 = nvgpu.tma.create.descriptor %cast box[%c64, %c128] : memref<*xf32> -> <tensor = memref<32x32xf32, 3>, swizzle = none, l2promo = none, oob = nan, interleave = none>
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_transform_mma_sync_matmul_f16_f16_accum(ctx: MLIRContext, capfd):
    range_ = scf.range_

    M, K, N = 16, 16, 8

    # forward reference...
    # TODO(max): figure out closures...
    printMemrefF32_ = []

    @func
    def compute_linspace_val(ridx: T.index(), cidx: T.index(), stride_cidx: T.index()):
        r = arith.index_cast(ridx, to=T.i32())
        c = arith.index_cast(cidx, to=T.i32())
        stride_c = arith.index_cast(stride_cidx, to=T.i32())
        v2 = r * stride_c
        v3 = c + v2
        v4 = arith.sitofp(T.f16(), v3)
        factor = arith.constant(64.0, T.f16())
        v5 = arith.divf(v4, factor)
        return v5

    # TODO(max): this looks like generics...
    @func
    @canonicalize(using=scf.canonicalizer)
    def print_lhs_as_memref_32(lhs: T.memref(M, K, T.f16())):
        M = memref.dim(lhs, 0)
        K = memref.dim(lhs, 1)
        tmp_alloc = memref.alloc(M, K, T.f32())
        for m in range_(0, M):
            for k in range_(0, K):
                f16 = lhs[m, k]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[m, k] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def print_rhs_as_memref_32(rhs: T.memref(K, N, T.f16())):
        K = memref.dim(rhs, 0)
        N = memref.dim(rhs, 1)
        tmp_alloc = memref.alloc(K, N, T.f32())
        for k in range_(0, K):
            for n in range_(0, N):
                f16 = rhs[k, n]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[k, n] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def print_res_as_memref_32(res: T.memref(M, N, T.f16())):
        c0 = arith.constant(0, index=True)
        c1 = arith.constant(1, index=True)
        M = memref.dim(res, c0)
        N = memref.dim(res, c1)
        tmp_alloc = memref.alloc(M, N, T.f32())
        for m in range_(0, M):
            for n in range_(0, N):
                f16 = res[m, n]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[m, n] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def main():
        lhs = memref.alloc(M, K, T.f16())
        rhs = memref.alloc(K, N, T.f16())
        res = memref.alloc(M, N, T.f16())

        M_ = memref.dim(res, 0)
        N_ = memref.dim(res, 1)
        K_ = memref.dim(lhs, 1)

        _f1 = arith.constant(1.0e00, T.f16())
        _f0 = arith.constant(0.0e00, T.f16())
        _c32 = arith.constant(32, T.index())

        # Initialize the lhs matrix with a linspace function.
        for r in range_(0, M_):
            for c in range_(0, K_):
                idx = compute_linspace_val(r, c, K_)
                lhs[r, c] = idx

        # Initialize the rhs matrix with a linspace function.
        for r in range_(0, K_):
            for c in range_(0, N_):
                idx = compute_linspace_val(r, c, N_)
                rhs[r, c] = idx

        # Initialize the res matrix with a linspace function.
        for r in range_(0, M_):
            for c in range_(0, N_):
                idx = compute_linspace_val(r, c, N_)
                res[r, c] = idx

        ulhs = memref.cast(T.memref(T.f16()), lhs)
        urhs = memref.cast(T.memref(T.f16()), rhs)
        ures = memref.cast(T.memref(T.f16()), res)
        gpu.host_register(ulhs)
        gpu.host_register(urhs)
        gpu.host_register(ures)

        print_lhs_as_memref_32(lhs)
        print_rhs_as_memref_32(rhs)

        @gpu.launch(grid_size=[1, 1, 1], block_size=[32, 1, 1])
        def kernel(bx, by, bz, tx, ty, tz, *grid_block_sizes):
            linalg.matmul(lhs, rhs, res)

        print_res_as_memref_32(res)

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        compute_linspace_val.emit()

        @func
        def printMemrefF32(x: T.memref(T.f32())):
            ...

        printMemrefF32_.append(printMemrefF32)

        print_lhs_as_memref_32.emit()
        print_rhs_as_memref_32.emit()
        print_res_as_memref_32.emit()
        main.emit()

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence(
            "main", [any_op_t()], [], arg_attrs=[{"transform.readonly": UnitAttr.get()}]
        )
        def main(module: any_op_t()):
            matmul = transform.match(module, ["linalg.matmul"])
            transform.nvgpu.rewrite_matmul_as_mma_sync(matmul)
            # clean up to simplify test below...
            all_loops = transform.match(
                module, interface=MatchInterfaceEnum.LoopLikeInterface
            )
            transform.apply_licm(all_loops)
            transform.apply_cse(module)

    assert ctx.module.operation.verify()
    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (d0 floordiv 4)>
    #map1 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8)>
    #map2 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>
    #map3 = affine_map<(d0) -> (d0 floordiv 4 + 8)>
    #map4 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 8)>
    #map5 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 9)>
    module {
      module attributes {transform.target_tag = "payload"} {
        func.func @compute_linspace_val(%arg0: index, %arg1: index, %arg2: index) -> f16 {
          %0 = arith.index_cast %arg0 : index to i32
          %1 = arith.index_cast %arg1 : index to i32
          %2 = arith.index_cast %arg2 : index to i32
          %3 = arith.muli %0, %2 : i32
          %4 = arith.addi %1, %3 : i32
          %5 = arith.sitofp %4 : i32 to f16
          %cst = arith.constant 6.400000e+01 : f16
          %6 = arith.divf %5, %cst : f16
          return %6 : f16
        }
        func.func private @printMemrefF32(memref<*xf32>)
        func.func @print_lhs_as_memref_32(%arg0: memref<16x16xf16>) {
          %c0 = arith.constant 0 : index
          %dim = memref.dim %arg0, %c0 : memref<16x16xf16>
          %c1 = arith.constant 1 : index
          %dim_0 = memref.dim %arg0, %c1 : memref<16x16xf16>
          %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xf32>
          scf.for %arg1 = %c0 to %dim step %c1 {
            scf.for %arg2 = %c0 to %dim_0 step %c1 {
              %0 = memref.load %arg0[%arg1, %arg2] : memref<16x16xf16>
              %1 = arith.extf %0 : f16 to f32
              memref.store %1, %alloc[%arg1, %arg2] : memref<?x?xf32>
            }
          }
          %cast = memref.cast %alloc : memref<?x?xf32> to memref<*xf32>
          call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
          memref.dealloc %alloc : memref<?x?xf32>
          return
        }
        func.func @print_rhs_as_memref_32(%arg0: memref<16x8xf16>) {
          %c0 = arith.constant 0 : index
          %dim = memref.dim %arg0, %c0 : memref<16x8xf16>
          %c1 = arith.constant 1 : index
          %dim_0 = memref.dim %arg0, %c1 : memref<16x8xf16>
          %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xf32>
          scf.for %arg1 = %c0 to %dim step %c1 {
            scf.for %arg2 = %c0 to %dim_0 step %c1 {
              %0 = memref.load %arg0[%arg1, %arg2] : memref<16x8xf16>
              %1 = arith.extf %0 : f16 to f32
              memref.store %1, %alloc[%arg1, %arg2] : memref<?x?xf32>
            }
          }
          %cast = memref.cast %alloc : memref<?x?xf32> to memref<*xf32>
          call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
          memref.dealloc %alloc : memref<?x?xf32>
          return
        }
        func.func @print_res_as_memref_32(%arg0: memref<16x8xf16>) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %dim = memref.dim %arg0, %c0 : memref<16x8xf16>
          %dim_0 = memref.dim %arg0, %c1 : memref<16x8xf16>
          %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xf32>
          scf.for %arg1 = %c0 to %dim step %c1 {
            scf.for %arg2 = %c0 to %dim_0 step %c1 {
              %0 = memref.load %arg0[%arg1, %arg2] : memref<16x8xf16>
              %1 = arith.extf %0 : f16 to f32
              memref.store %1, %alloc[%arg1, %arg2] : memref<?x?xf32>
            }
          }
          %cast = memref.cast %alloc : memref<?x?xf32> to memref<*xf32>
          call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
          memref.dealloc %alloc : memref<?x?xf32>
          return
        }
        func.func @main() {
          %alloc = memref.alloc() : memref<16x16xf16>
          %alloc_0 = memref.alloc() : memref<16x8xf16>
          %alloc_1 = memref.alloc() : memref<16x8xf16>
          %c0 = arith.constant 0 : index
          %dim = memref.dim %alloc_1, %c0 : memref<16x8xf16>
          %c1 = arith.constant 1 : index
          %dim_2 = memref.dim %alloc_1, %c1 : memref<16x8xf16>
          %dim_3 = memref.dim %alloc, %c1 : memref<16x16xf16>
          scf.for %arg0 = %c0 to %dim step %c1 {
            scf.for %arg1 = %c0 to %dim_3 step %c1 {
              %0 = func.call @compute_linspace_val(%arg0, %arg1, %dim_3) : (index, index, index) -> f16
              memref.store %0, %alloc[%arg0, %arg1] : memref<16x16xf16>
            }
          }
          scf.for %arg0 = %c0 to %dim_3 step %c1 {
            scf.for %arg1 = %c0 to %dim_2 step %c1 {
              %0 = func.call @compute_linspace_val(%arg0, %arg1, %dim_2) : (index, index, index) -> f16
              memref.store %0, %alloc_0[%arg0, %arg1] : memref<16x8xf16>
            }
          }
          scf.for %arg0 = %c0 to %dim step %c1 {
            scf.for %arg1 = %c0 to %dim_2 step %c1 {
              %0 = func.call @compute_linspace_val(%arg0, %arg1, %dim_2) : (index, index, index) -> f16
              memref.store %0, %alloc_1[%arg0, %arg1] : memref<16x8xf16>
            }
          }
          %cast = memref.cast %alloc : memref<16x16xf16> to memref<*xf16>
          %cast_4 = memref.cast %alloc_0 : memref<16x8xf16> to memref<*xf16>
          %cast_5 = memref.cast %alloc_1 : memref<16x8xf16> to memref<*xf16>
          gpu.host_register %cast : memref<*xf16>
          gpu.host_register %cast_4 : memref<*xf16>
          gpu.host_register %cast_5 : memref<*xf16>
          call @print_lhs_as_memref_32(%alloc) : (memref<16x16xf16>) -> ()
          call @print_rhs_as_memref_32(%alloc_0) : (memref<16x8xf16>) -> ()
          %c32 = arith.constant 32 : index
          gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c1, %arg11 = %c1) {
            %0 = gpu.thread_id  x
            %1 = affine.apply #map(%0)
            %2 = affine.apply #map1(%0)
            %3 = memref.load %alloc[%1, %2] : memref<16x16xf16>
            %4 = affine.apply #map2(%0)
            %5 = memref.load %alloc[%1, %4] : memref<16x16xf16>
            %6 = affine.apply #map3(%0)
            %7 = memref.load %alloc[%6, %2] : memref<16x16xf16>
            %8 = memref.load %alloc[%6, %4] : memref<16x16xf16>
            %9 = affine.apply #map4(%0)
            %10 = memref.load %alloc[%1, %9] : memref<16x16xf16>
            %11 = affine.apply #map5(%0)
            %12 = memref.load %alloc[%1, %11] : memref<16x16xf16>
            %13 = memref.load %alloc[%6, %9] : memref<16x16xf16>
            %14 = memref.load %alloc[%6, %11] : memref<16x16xf16>
            %15 = vector.splat %3 : vector<4x2xf16>
            %16 = vector.insert %3, %15 [0, 0] : f16 into vector<4x2xf16>
            %17 = vector.insert %5, %16 [0, 1] : f16 into vector<4x2xf16>
            %18 = vector.insert %7, %17 [1, 0] : f16 into vector<4x2xf16>
            %19 = vector.insert %8, %18 [1, 1] : f16 into vector<4x2xf16>
            %20 = vector.insert %10, %19 [2, 0] : f16 into vector<4x2xf16>
            %21 = vector.insert %12, %20 [2, 1] : f16 into vector<4x2xf16>
            %22 = vector.insert %13, %21 [3, 0] : f16 into vector<4x2xf16>
            %23 = vector.insert %14, %22 [3, 1] : f16 into vector<4x2xf16>
            %24 = memref.load %alloc_0[%2, %1] : memref<16x8xf16>
            %25 = memref.load %alloc_0[%4, %1] : memref<16x8xf16>
            %26 = memref.load %alloc_0[%9, %1] : memref<16x8xf16>
            %27 = memref.load %alloc_0[%11, %1] : memref<16x8xf16>
            %28 = vector.splat %24 : vector<2x2xf16>
            %29 = vector.insert %24, %28 [0, 0] : f16 into vector<2x2xf16>
            %30 = vector.insert %25, %29 [0, 1] : f16 into vector<2x2xf16>
            %31 = vector.insert %26, %30 [1, 0] : f16 into vector<2x2xf16>
            %32 = vector.insert %27, %31 [1, 1] : f16 into vector<2x2xf16>
            %33 = memref.load %alloc_1[%1, %2] : memref<16x8xf16>
            %34 = memref.load %alloc_1[%1, %4] : memref<16x8xf16>
            %35 = memref.load %alloc_1[%6, %2] : memref<16x8xf16>
            %36 = memref.load %alloc_1[%6, %4] : memref<16x8xf16>
            %37 = vector.splat %33 : vector<2x2xf16>
            %38 = vector.insert %33, %37 [0, 0] : f16 into vector<2x2xf16>
            %39 = vector.insert %34, %38 [0, 1] : f16 into vector<2x2xf16>
            %40 = vector.insert %35, %39 [1, 0] : f16 into vector<2x2xf16>
            %41 = vector.insert %36, %40 [1, 1] : f16 into vector<2x2xf16>
            %42 = nvgpu.mma.sync(%23, %32, %41) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
            %43 = vector.extract %42[0, 0] : f16 from vector<2x2xf16>
            %44 = vector.extract %42[0, 1] : f16 from vector<2x2xf16>
            %45 = vector.extract %42[1, 0] : f16 from vector<2x2xf16>
            %46 = vector.extract %42[1, 1] : f16 from vector<2x2xf16>
            memref.store %43, %alloc_1[%1, %2] : memref<16x8xf16>
            memref.store %44, %alloc_1[%1, %4] : memref<16x8xf16>
            memref.store %45, %alloc_1[%6, %2] : memref<16x8xf16>
            memref.store %46, %alloc_1[%6, %4] : memref<16x8xf16>
            gpu.terminator
          }
          call @print_res_as_memref_32(%alloc_1) : (memref<16x8xf16>) -> ()
          return
        }
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly}) {
          %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          transform.nvgpu.rewrite_matmul_as_mma_sync %0 : (!transform.any_op) -> ()
          %1 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
          transform.apply_licm to %1 : !transform.any_op
          transform.apply_cse to %arg0 : !transform.any_op
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, mod)


CUDA_RUNTIME_LIB_PATH = Path(_mlir_libs.__file__).parent / f"libmlir_cuda_runtime.so"

# based on https://github.com/llvm/llvm-project/blob/9cc2122bf5a81f7063c2a32b2cb78c8d615578a1/mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir#L6
@pytest.mark.skipif(not CUDA_RUNTIME_LIB_PATH.exists(), reason="no cuda library")
def test_transform_mma_sync_matmul_f16_f16_accum_run(ctx: MLIRContext, capfd):
    range_ = scf.range_

    M, K, N = 16, 16, 8

    # forward reference...
    # TODO(max): figure out closures...
    printMemrefF32_ = []

    @func
    def compute_linspace_val(ridx: T.index(), cidx: T.index(), stride_cidx: T.index()):
        r = arith.index_cast(ridx, to=T.i32())
        c = arith.index_cast(cidx, to=T.i32())
        stride_c = arith.index_cast(stride_cidx, to=T.i32())
        v2 = r * stride_c
        v3 = c + v2
        v4 = arith.sitofp(T.f16(), v3)
        factor = arith.constant(64.0, T.f16())
        v5 = arith.divf(v4, factor)
        return v5

    # TODO(max): this looks like generics...
    @func
    @canonicalize(using=scf.canonicalizer)
    def print_lhs_as_memref_32(lhs: T.memref(M, K, T.f16())):
        M = memref.dim(lhs, 0)
        K = memref.dim(lhs, 1)
        tmp_alloc = memref.alloc(M, K, T.f32())
        for m in range_(0, M):
            for k in range_(0, K):
                f16 = lhs[m, k]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[m, k] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def print_rhs_as_memref_32(rhs: T.memref(K, N, T.f16())):
        K = memref.dim(rhs, 0)
        N = memref.dim(rhs, 1)
        tmp_alloc = memref.alloc(K, N, T.f32())
        for k in range_(0, K):
            for n in range_(0, N):
                f16 = rhs[k, n]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[k, n] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def print_res_as_memref_32(res: T.memref(M, N, T.f16())):
        c0 = arith.constant(0, index=True)
        c1 = arith.constant(1, index=True)
        M = memref.dim(res, c0)
        N = memref.dim(res, c1)
        tmp_alloc = memref.alloc(M, N, T.f32())
        for m in range_(0, M):
            for n in range_(0, N):
                f16 = res[m, n]
                f32 = arith.extf(T.f32(), f16)
                tmp_alloc[m, n] = f32

        casted = memref.cast(T.memref(T.f32()), tmp_alloc)
        printMemrefF32_[0](casted)
        memref.dealloc(tmp_alloc)

    @func
    @canonicalize(using=scf.canonicalizer)
    def main():
        lhs = memref.alloc(M, K, T.f16())
        rhs = memref.alloc(K, N, T.f16())
        res = memref.alloc(M, N, T.f16())

        M_ = memref.dim(res, 0)
        N_ = memref.dim(res, 1)
        K_ = memref.dim(lhs, 1)

        _f1 = arith.constant(1.0e00, T.f16())
        _f0 = arith.constant(0.0e00, T.f16())
        _c32 = arith.constant(32, T.index())

        # Initialize the lhs matrix with a linspace function.
        for r in range_(0, M_):
            for c in range_(0, K_):
                idx = compute_linspace_val(r, c, K_)
                lhs[r, c] = idx

        # Initialize the rhs matrix with a linspace function.
        for r in range_(0, K_):
            for c in range_(0, N_):
                idx = compute_linspace_val(r, c, N_)
                rhs[r, c] = idx

        # Initialize the res matrix with a linspace function.
        for r in range_(0, M_):
            for c in range_(0, N_):
                idx = compute_linspace_val(r, c, N_)
                res[r, c] = idx

        ulhs = memref.cast(T.memref(T.f16()), lhs)
        urhs = memref.cast(T.memref(T.f16()), rhs)
        ures = memref.cast(T.memref(T.f16()), res)
        gpu.host_register(ulhs)
        gpu.host_register(urhs)
        gpu.host_register(ures)

        print_lhs_as_memref_32(lhs)
        print_rhs_as_memref_32(rhs)

        @gpu.launch(grid_size=[1, 1, 1], block_size=[32, 1, 1])
        def kernel(bx, by, bz, tx, ty, tz, *grid_block_sizes):
            linalg.matmul(lhs, rhs, res)

        print_res_as_memref_32(res)

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        compute_linspace_val.emit()

        @func
        def printMemrefF32(x: T.memref(T.f32())):
            ...

        printMemrefF32_.append(printMemrefF32)

        print_lhs_as_memref_32.emit()
        print_rhs_as_memref_32.emit()
        print_res_as_memref_32.emit()
        main.emit()

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence(
            "main", [any_op_t()], [], arg_attrs=[{"transform.readonly": UnitAttr.get()}]
        )
        def main(module: any_op_t()):
            matmul = transform.match(module, ["linalg.matmul"])
            transform.nvgpu.rewrite_matmul_as_mma_sync(matmul)
            # clean up to simplify test below...
            all_loops = transform.match(
                module, interface=MatchInterfaceEnum.LoopLikeInterface
            )
            transform.apply_licm(all_loops)
            transform.apply_cse(module)

    assert ctx.module.operation.verify()
    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )

    backend = LLVMJITBackend([CUDA_RUNTIME_LIB_PATH])
    compiled_module = backend.compile(
        find_ops(
            mod.operation,
            lambda x: "transform.target_tag" in x.attributes
            and x.attributes["transform.target_tag"].value == "payload",
            single=True,
        ),
        # the pipeline today https://github.com/llvm/llvm-project/blob/095367a521fc9ff714e1779e507bdd91d4fe9c7d/mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp#L122
        Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            **{
                "cubin-chip": "sm_80",
                "cubin-features": "+ptx76",
                "cubin-format": "fatbin",
            },
        ),
    )

    backend.load(compiled_module).main_capi_wrapper()
    correct = dedent(
        """\
        Unranked Memref base@ = rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data = 
        [[0,   0.015625,   0.03125,   0.046875,   0.0625,   0.078125,   0.09375,   0.109375,   0.125,   0.140625,   0.15625,   0.171875,   0.1875,   0.203125,   0.21875,   0.234375], 
         [0.25,   0.265625,   0.28125,   0.296875,   0.3125,   0.328125,   0.34375,   0.359375,   0.375,   0.390625,   0.40625,   0.421875,   0.4375,   0.453125,   0.46875,   0.484375], 
         [0.5,   0.515625,   0.53125,   0.546875,   0.5625,   0.578125,   0.59375,   0.609375,   0.625,   0.640625,   0.65625,   0.671875,   0.6875,   0.703125,   0.71875,   0.734375], 
         [0.75,   0.765625,   0.78125,   0.796875,   0.8125,   0.828125,   0.84375,   0.859375,   0.875,   0.890625,   0.90625,   0.921875,   0.9375,   0.953125,   0.96875,   0.984375], 
         [1,   1.01562,   1.03125,   1.04688,   1.0625,   1.07812,   1.09375,   1.10938,   1.125,   1.14062,   1.15625,   1.17188,   1.1875,   1.20312,   1.21875,   1.23438], 
         [1.25,   1.26562,   1.28125,   1.29688,   1.3125,   1.32812,   1.34375,   1.35938,   1.375,   1.39062,   1.40625,   1.42188,   1.4375,   1.45312,   1.46875,   1.48438], 
         [1.5,   1.51562,   1.53125,   1.54688,   1.5625,   1.57812,   1.59375,   1.60938,   1.625,   1.64062,   1.65625,   1.67188,   1.6875,   1.70312,   1.71875,   1.73438], 
         [1.75,   1.76562,   1.78125,   1.79688,   1.8125,   1.82812,   1.84375,   1.85938,   1.875,   1.89062,   1.90625,   1.92188,   1.9375,   1.95312,   1.96875,   1.98438], 
         [2,   2.01562,   2.03125,   2.04688,   2.0625,   2.07812,   2.09375,   2.10938,   2.125,   2.14062,   2.15625,   2.17188,   2.1875,   2.20312,   2.21875,   2.23438], 
         [2.25,   2.26562,   2.28125,   2.29688,   2.3125,   2.32812,   2.34375,   2.35938,   2.375,   2.39062,   2.40625,   2.42188,   2.4375,   2.45312,   2.46875,   2.48438], 
         [2.5,   2.51562,   2.53125,   2.54688,   2.5625,   2.57812,   2.59375,   2.60938,   2.625,   2.64062,   2.65625,   2.67188,   2.6875,   2.70312,   2.71875,   2.73438], 
         [2.75,   2.76562,   2.78125,   2.79688,   2.8125,   2.82812,   2.84375,   2.85938,   2.875,   2.89062,   2.90625,   2.92188,   2.9375,   2.95312,   2.96875,   2.98438], 
         [3,   3.01562,   3.03125,   3.04688,   3.0625,   3.07812,   3.09375,   3.10938,   3.125,   3.14062,   3.15625,   3.17188,   3.1875,   3.20312,   3.21875,   3.23438], 
         [3.25,   3.26562,   3.28125,   3.29688,   3.3125,   3.32812,   3.34375,   3.35938,   3.375,   3.39062,   3.40625,   3.42188,   3.4375,   3.45312,   3.46875,   3.48438], 
         [3.5,   3.51562,   3.53125,   3.54688,   3.5625,   3.57812,   3.59375,   3.60938,   3.625,   3.64062,   3.65625,   3.67188,   3.6875,   3.70312,   3.71875,   3.73438], 
         [3.75,   3.76562,   3.78125,   3.79688,   3.8125,   3.82812,   3.84375,   3.85938,   3.875,   3.89062,   3.90625,   3.92188,   3.9375,   3.95312,   3.96875,   3.98438]]
        Unranked Memref base@ = rank = 2 offset = 0 sizes = [16, 8] strides = [8, 1] data = 
        [[0,   0.015625,   0.03125,   0.046875,   0.0625,   0.078125,   0.09375,   0.109375], 
         [0.125,   0.140625,   0.15625,   0.171875,   0.1875,   0.203125,   0.21875,   0.234375], 
         [0.25,   0.265625,   0.28125,   0.296875,   0.3125,   0.328125,   0.34375,   0.359375], 
         [0.375,   0.390625,   0.40625,   0.421875,   0.4375,   0.453125,   0.46875,   0.484375], 
         [0.5,   0.515625,   0.53125,   0.546875,   0.5625,   0.578125,   0.59375,   0.609375], 
         [0.625,   0.640625,   0.65625,   0.671875,   0.6875,   0.703125,   0.71875,   0.734375], 
         [0.75,   0.765625,   0.78125,   0.796875,   0.8125,   0.828125,   0.84375,   0.859375], 
         [0.875,   0.890625,   0.90625,   0.921875,   0.9375,   0.953125,   0.96875,   0.984375], 
         [1,   1.01562,   1.03125,   1.04688,   1.0625,   1.07812,   1.09375,   1.10938], 
         [1.125,   1.14062,   1.15625,   1.17188,   1.1875,   1.20312,   1.21875,   1.23438], 
         [1.25,   1.26562,   1.28125,   1.29688,   1.3125,   1.32812,   1.34375,   1.35938], 
         [1.375,   1.39062,   1.40625,   1.42188,   1.4375,   1.45312,   1.46875,   1.48438], 
         [1.5,   1.51562,   1.53125,   1.54688,   1.5625,   1.57812,   1.59375,   1.60938], 
         [1.625,   1.64062,   1.65625,   1.67188,   1.6875,   1.70312,   1.71875,   1.73438], 
         [1.75,   1.76562,   1.78125,   1.79688,   1.8125,   1.82812,   1.84375,   1.85938], 
         [1.875,   1.89062,   1.90625,   1.92188,   1.9375,   1.95312,   1.96875,   1.98438]]
        Unranked Memref base@ = rank = 2 offset = 0 sizes = [16, 8] strides = [8, 1] data = 
        [[2.42188,   2.4668,   2.51172,   2.55664,   2.60156,   2.64648,   2.69141,   2.73633], 
         [6.29688,   6.40625,   6.51172,   6.61719,   6.72656,   6.83594,   6.94141,   7.04688], 
         [10.1719,   10.3438,   10.5156,   10.6797,   10.8516,   11.0234,   11.1875,   11.3594], 
         [14.0469,   14.2812,   14.5156,   14.7422,   14.9766,   15.2109,   15.4375,   15.6719], 
         [17.9219,   18.2188,   18.5156,   18.8125,   19.0938,   19.3906,   19.6875,   19.9844], 
         [21.7969,   22.1562,   22.5156,   22.875,   23.2188,   23.5781,   23.9375,   24.2969], 
         [25.6719,   26.0938,   26.5156,   26.9375,   27.3438,   27.7656,   28.1875,   28.6094], 
         [29.5469,   30.0312,   30.5156,   31,   31.4688,   31.9531,   32.4375,   32.9375], 
         [33.4375,   33.9688,   34.5,   35.0625,   35.5938,   36.1562,   36.6875,   37.25], 
         [37.3125,   37.9062,   38.5,   39.125,   39.7188,   40.3438,   40.9375,   41.5625], 
         [41.1875,   41.8438,   42.5,   43.1875,   43.8438,   44.5312,   45.1875,   45.875], 
         [45.0625,   45.7812,   46.5,   47.25,   47.9688,   48.7188,   49.4375,   50.1875], 
         [48.9375,   49.7188,   50.5,   51.3125,   52.0938,   52.9062,   53.6875,   54.5], 
         [52.8125,   53.6562,   54.5,   55.375,   56.2188,   57.0938,   57.9375,   58.8125], 
         [56.6875,   57.5938,   58.5,   59.4375,   60.3438,   61.2812,   62.1875,   63.125], 
         [60.5625,   61.5312,   62.5,   63.5,   64.5,   65.4375,   66.4375,   67.4375]]
    """
    )
    out, err = capfd.readouterr()
    filecheck(correct, re.sub(r"0x\w+", "", out))
