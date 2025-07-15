import re
import subprocess
from pathlib import Path
from textwrap import dedent

import mlir.extras.types as T
import pytest
from mlir.dialects import builtin
from mlir.dialects.memref import cast
from mlir.dialects.nvgpu import (
    TensorMapDescriptorType,
    TensorMapInterleaveKind,
    TensorMapL2PromoKind,
    TensorMapOOBKind,
    TensorMapSwizzleKind,
    tma_create_descriptor,
)
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.extras import named_sequence
from mlir.dialects.transform.structured import MatchInterfaceEnum
from mlir.ir import StringAttr, UnitAttr

from mlir import _mlir_libs
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import arith, gpu, linalg, memref, nvgpu, scf, transform
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.gpu import smem_space
from mlir.extras.dialects.ext.llvm import llvm_ptr_t
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    MLIRContext,
    filecheck,
    filecheck_with_comments,
    mlir_ctx as ctx,
)
from mlir.extras.util import find_ops

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
        tensor_map_2d = TensorMapDescriptorType.get(
            T.memref(32, 32, T.f32(), memory_space=3),
            TensorMapSwizzleKind.SWIZZLE_NONE,
            TensorMapL2PromoKind.L2PROMO_NONE,
            TensorMapOOBKind.OOB_NAN,
            TensorMapInterleaveKind.INTERLEAVE_NONE,
        )
        tensor_map_2d = tma_create_descriptor(
            tensor_map_2d, device_ptr_2d_unranked, [crd0, crd1]
        )

    create_tensor_map.emit()

    ctx.module.operation.verify()

    # CHECK:  func.func @create_tensor_map(%[[VAL_0:.*]]: memref<64x128xf32>) {
    # CHECK:    %[[VAL_1:.*]] = arith.constant 64 : index
    # CHECK:    %[[VAL_2:.*]] = arith.constant 128 : index
    # CHECK:    %[[VAL_3:.*]] = memref.cast %[[VAL_0]] : memref<64x128xf32> to memref<*xf32>
    # CHECK:    %[[VAL_4:.*]] = nvgpu.tma.create.descriptor %[[VAL_3]] box{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<*xf32> -> <tensor = memref<32x32xf32, 3>, swizzle = none, l2promo = none, oob = nan, interleave = none>
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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
        tmp_alloc = memref.alloc((M, K), T.f32())
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
        tmp_alloc = memref.alloc((K, N), T.f32())
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
        tmp_alloc = memref.alloc((M, N), T.f32())
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
        lhs = memref.alloc((M, K), T.f16())
        rhs = memref.alloc((K, N), T.f16())
        res = memref.alloc((M, N), T.f16())

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
        def printMemrefF32(x: T.memref(T.f32())): ...

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

    # CHECK-LABEL: #map = affine_map<()[s0] -> (s0 floordiv 4)>
    # CHECK:       #map1 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8)>
    # CHECK:       #map2 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 1)>
    # CHECK:       #map3 = affine_map<()[s0] -> (s0 floordiv 4 + 8)>
    # CHECK:       #map4 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 8)>
    # CHECK:       #map5 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 9)>

    # CHECK:  module attributes {transform.target_tag = "payload"} {
    # CHECK:    func.func @compute_linspace_val(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) -> f16 {
    # CHECK:      %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : index to i32
    # CHECK:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_1]] : index to i32
    # CHECK:      %[[VAL_5:.*]] = arith.index_cast %[[VAL_2]] : index to i32
    # CHECK:      %[[VAL_6:.*]] = arith.muli %[[VAL_3]], %[[VAL_5]] : i32
    # CHECK:      %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : i32
    # CHECK:      %[[VAL_8:.*]] = arith.sitofp %[[VAL_7]] : i32 to f16
    # CHECK:      %[[VAL_9:.*]] = arith.constant 6.400000e+01 : f16
    # CHECK:      %[[VAL_10:.*]] = arith.divf %[[VAL_8]], %[[VAL_9]] : f16
    # CHECK:      return %[[VAL_10]] : f16
    # CHECK:    }
    # CHECK:    func.func private @printMemrefF32(memref<*xf32>)
    # CHECK:    func.func @print_lhs_as_memref_32(%[[VAL_11:.*]]: memref<16x16xf16>) {
    # CHECK:      %[[VAL_12:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_13:.*]] = memref.dim %[[VAL_11]], %[[VAL_12]] : memref<16x16xf16>
    # CHECK:      %[[VAL_14:.*]] = arith.constant 1 : index
    # CHECK:      %[[VAL_15:.*]] = memref.dim %[[VAL_11]], %[[VAL_14]] : memref<16x16xf16>
    # CHECK:      %[[VAL_16:.*]] = memref.alloc(%[[VAL_13]], %[[VAL_15]]) : memref<?x?xf32>
    # CHECK:      scf.for %[[VAL_17:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_14]] {
    # CHECK:        scf.for %[[VAL_18:.*]] = %[[VAL_12]] to %[[VAL_15]] step %[[VAL_14]] {
    # CHECK:          %[[VAL_19:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<16x16xf16>
    # CHECK:          %[[VAL_20:.*]] = arith.extf %[[VAL_19]] : f16 to f32
    # CHECK:          memref.store %[[VAL_20]], %[[VAL_16]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<?x?xf32>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      %[[VAL_21:.*]] = memref.cast %[[VAL_16]] : memref<?x?xf32> to memref<*xf32>
    # CHECK:      call @printMemrefF32(%[[VAL_21]]) : (memref<*xf32>) -> ()
    # CHECK:      memref.dealloc %[[VAL_16]] : memref<?x?xf32>
    # CHECK:      return
    # CHECK:    }
    # CHECK:    func.func @print_rhs_as_memref_32(%[[VAL_22:.*]]: memref<16x8xf16>) {
    # CHECK:      %[[VAL_23:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_24:.*]] = memref.dim %[[VAL_22]], %[[VAL_23]] : memref<16x8xf16>
    # CHECK:      %[[VAL_25:.*]] = arith.constant 1 : index
    # CHECK:      %[[VAL_26:.*]] = memref.dim %[[VAL_22]], %[[VAL_25]] : memref<16x8xf16>
    # CHECK:      %[[VAL_27:.*]] = memref.alloc(%[[VAL_24]], %[[VAL_26]]) : memref<?x?xf32>
    # CHECK:      scf.for %[[VAL_28:.*]] = %[[VAL_23]] to %[[VAL_24]] step %[[VAL_25]] {
    # CHECK:        scf.for %[[VAL_29:.*]] = %[[VAL_23]] to %[[VAL_26]] step %[[VAL_25]] {
    # CHECK:          %[[VAL_30:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_28]], %[[VAL_29]]] : memref<16x8xf16>
    # CHECK:          %[[VAL_31:.*]] = arith.extf %[[VAL_30]] : f16 to f32
    # CHECK:          memref.store %[[VAL_31]], %[[VAL_27]]{{\[}}%[[VAL_28]], %[[VAL_29]]] : memref<?x?xf32>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      %[[VAL_32:.*]] = memref.cast %[[VAL_27]] : memref<?x?xf32> to memref<*xf32>
    # CHECK:      call @printMemrefF32(%[[VAL_32]]) : (memref<*xf32>) -> ()
    # CHECK:      memref.dealloc %[[VAL_27]] : memref<?x?xf32>
    # CHECK:      return
    # CHECK:    }
    # CHECK:    func.func @print_res_as_memref_32(%[[VAL_33:.*]]: memref<16x8xf16>) {
    # CHECK:      %[[VAL_34:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_35:.*]] = arith.constant 1 : index
    # CHECK:      %[[VAL_36:.*]] = memref.dim %[[VAL_33]], %[[VAL_34]] : memref<16x8xf16>
    # CHECK:      %[[VAL_37:.*]] = memref.dim %[[VAL_33]], %[[VAL_35]] : memref<16x8xf16>
    # CHECK:      %[[VAL_38:.*]] = memref.alloc(%[[VAL_36]], %[[VAL_37]]) : memref<?x?xf32>
    # CHECK:      scf.for %[[VAL_39:.*]] = %[[VAL_34]] to %[[VAL_36]] step %[[VAL_35]] {
    # CHECK:        scf.for %[[VAL_40:.*]] = %[[VAL_34]] to %[[VAL_37]] step %[[VAL_35]] {
    # CHECK:          %[[VAL_41:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_39]], %[[VAL_40]]] : memref<16x8xf16>
    # CHECK:          %[[VAL_42:.*]] = arith.extf %[[VAL_41]] : f16 to f32
    # CHECK:          memref.store %[[VAL_42]], %[[VAL_38]]{{\[}}%[[VAL_39]], %[[VAL_40]]] : memref<?x?xf32>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      %[[VAL_43:.*]] = memref.cast %[[VAL_38]] : memref<?x?xf32> to memref<*xf32>
    # CHECK:      call @printMemrefF32(%[[VAL_43]]) : (memref<*xf32>) -> ()
    # CHECK:      memref.dealloc %[[VAL_38]] : memref<?x?xf32>
    # CHECK:      return
    # CHECK:    }
    # CHECK:    func.func @main() {
    # CHECK:      %[[VAL_44:.*]] = memref.alloc() : memref<16x16xf16>
    # CHECK:      %[[VAL_45:.*]] = memref.alloc() : memref<16x8xf16>
    # CHECK:      %[[VAL_46:.*]] = memref.alloc() : memref<16x8xf16>
    # CHECK:      %[[VAL_47:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_48:.*]] = memref.dim %[[VAL_46]], %[[VAL_47]] : memref<16x8xf16>
    # CHECK:      %[[VAL_49:.*]] = arith.constant 1 : index
    # CHECK:      %[[VAL_50:.*]] = memref.dim %[[VAL_46]], %[[VAL_49]] : memref<16x8xf16>
    # CHECK:      %[[VAL_51:.*]] = memref.dim %[[VAL_44]], %[[VAL_49]] : memref<16x16xf16>
    # CHECK:      scf.for %[[VAL_52:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_49]] {
    # CHECK:        scf.for %[[VAL_53:.*]] = %[[VAL_47]] to %[[VAL_51]] step %[[VAL_49]] {
    # CHECK:          %[[VAL_54:.*]] = func.call @compute_linspace_val(%[[VAL_52]], %[[VAL_53]], %[[VAL_51]]) : (index, index, index) -> f16
    # CHECK:          memref.store %[[VAL_54]], %[[VAL_44]]{{\[}}%[[VAL_52]], %[[VAL_53]]] : memref<16x16xf16>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      scf.for %[[VAL_55:.*]] = %[[VAL_47]] to %[[VAL_51]] step %[[VAL_49]] {
    # CHECK:        scf.for %[[VAL_56:.*]] = %[[VAL_47]] to %[[VAL_50]] step %[[VAL_49]] {
    # CHECK:          %[[VAL_57:.*]] = func.call @compute_linspace_val(%[[VAL_55]], %[[VAL_56]], %[[VAL_50]]) : (index, index, index) -> f16
    # CHECK:          memref.store %[[VAL_57]], %[[VAL_45]]{{\[}}%[[VAL_55]], %[[VAL_56]]] : memref<16x8xf16>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      scf.for %[[VAL_58:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_49]] {
    # CHECK:        scf.for %[[VAL_59:.*]] = %[[VAL_47]] to %[[VAL_50]] step %[[VAL_49]] {
    # CHECK:          %[[VAL_60:.*]] = func.call @compute_linspace_val(%[[VAL_58]], %[[VAL_59]], %[[VAL_50]]) : (index, index, index) -> f16
    # CHECK:          memref.store %[[VAL_60]], %[[VAL_46]]{{\[}}%[[VAL_58]], %[[VAL_59]]] : memref<16x8xf16>
    # CHECK:        }
    # CHECK:      }
    # CHECK:      %[[VAL_61:.*]] = memref.cast %[[VAL_44]] : memref<16x16xf16> to memref<*xf16>
    # CHECK:      %[[VAL_62:.*]] = memref.cast %[[VAL_45]] : memref<16x8xf16> to memref<*xf16>
    # CHECK:      %[[VAL_63:.*]] = memref.cast %[[VAL_46]] : memref<16x8xf16> to memref<*xf16>
    # CHECK:      gpu.host_register %[[VAL_61]] : memref<*xf16>
    # CHECK:      gpu.host_register %[[VAL_62]] : memref<*xf16>
    # CHECK:      gpu.host_register %[[VAL_63]] : memref<*xf16>
    # CHECK:      call @print_lhs_as_memref_32(%[[VAL_44]]) : (memref<16x16xf16>) -> ()
    # CHECK:      call @print_rhs_as_memref_32(%[[VAL_45]]) : (memref<16x8xf16>) -> ()
    # CHECK:      %[[VAL_64:.*]] = arith.constant 32 : index
    # CHECK:      gpu.launch blocks(%[[VAL_65:.*]], %[[VAL_66:.*]], %[[VAL_67:.*]]) in (%[[VAL_68:.*]] = %[[VAL_49]], %[[VAL_69:.*]] = %[[VAL_49]], %[[VAL_70:.*]] = %[[VAL_49]]) threads(%[[VAL_71:.*]], %[[VAL_72:.*]], %[[VAL_73:.*]]) in (%[[VAL_74:.*]] = %[[VAL_64]], %[[VAL_75:.*]] = %[[VAL_49]], %[[VAL_76:.*]] = %[[VAL_49]]) {
    # CHECK:        %[[VAL_77:.*]] = gpu.thread_id  x
    # CHECK:        %[[VAL_78:.*]] = affine.apply #map(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_79:.*]] = affine.apply #map1(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_80:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_78]], %[[VAL_79]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_81:.*]] = affine.apply #map2(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_82:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_78]], %[[VAL_81]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_83:.*]] = affine.apply #map3(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_84:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_83]], %[[VAL_79]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_85:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_83]], %[[VAL_81]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_86:.*]] = affine.apply #map4(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_87:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_78]], %[[VAL_86]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_88:.*]] = affine.apply #map5(){{\[}}%[[VAL_77]]]
    # CHECK:        %[[VAL_89:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_78]], %[[VAL_88]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_90:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_83]], %[[VAL_86]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_91:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_83]], %[[VAL_88]]] : memref<16x16xf16>
    # CHECK:        %[[VAL_92:.*]] = vector.broadcast %[[VAL_80]] : f16 to vector<4x2xf16>
    # CHECK:        %[[VAL_93:.*]] = vector.insert %[[VAL_80]], %[[VAL_92]] [0, 0] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_94:.*]] = vector.insert %[[VAL_82]], %[[VAL_93]] [0, 1] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_95:.*]] = vector.insert %[[VAL_84]], %[[VAL_94]] [1, 0] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_96:.*]] = vector.insert %[[VAL_85]], %[[VAL_95]] [1, 1] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_97:.*]] = vector.insert %[[VAL_87]], %[[VAL_96]] [2, 0] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_98:.*]] = vector.insert %[[VAL_89]], %[[VAL_97]] [2, 1] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_99:.*]] = vector.insert %[[VAL_90]], %[[VAL_98]] [3, 0] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_100:.*]] = vector.insert %[[VAL_91]], %[[VAL_99]] [3, 1] : f16 into vector<4x2xf16>
    # CHECK:        %[[VAL_101:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_79]], %[[VAL_78]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_102:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_81]], %[[VAL_78]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_103:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_86]], %[[VAL_78]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_104:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_88]], %[[VAL_78]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_105:.*]] = vector.broadcast %[[VAL_101]] : f16 to vector<2x2xf16>
    # CHECK:        %[[VAL_106:.*]] = vector.insert %[[VAL_101]], %[[VAL_105]] [0, 0] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_107:.*]] = vector.insert %[[VAL_102]], %[[VAL_106]] [0, 1] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_108:.*]] = vector.insert %[[VAL_103]], %[[VAL_107]] [1, 0] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_109:.*]] = vector.insert %[[VAL_104]], %[[VAL_108]] [1, 1] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_110:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_78]], %[[VAL_79]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_111:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_78]], %[[VAL_81]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_112:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_83]], %[[VAL_79]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_113:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_83]], %[[VAL_81]]] : memref<16x8xf16>
    # CHECK:        %[[VAL_114:.*]] = vector.broadcast %[[VAL_110]] : f16 to vector<2x2xf16>
    # CHECK:        %[[VAL_115:.*]] = vector.insert %[[VAL_110]], %[[VAL_114]] [0, 0] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_116:.*]] = vector.insert %[[VAL_111]], %[[VAL_115]] [0, 1] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_117:.*]] = vector.insert %[[VAL_112]], %[[VAL_116]] [1, 0] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_118:.*]] = vector.insert %[[VAL_113]], %[[VAL_117]] [1, 1] : f16 into vector<2x2xf16>
    # CHECK:        %[[VAL_119:.*]] = nvgpu.mma.sync(%[[VAL_100]], %[[VAL_109]], %[[VAL_118]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    # CHECK:        %[[VAL_120:.*]] = vector.extract %[[VAL_119]][0, 0] : f16 from vector<2x2xf16>
    # CHECK:        %[[VAL_121:.*]] = vector.extract %[[VAL_119]][0, 1] : f16 from vector<2x2xf16>
    # CHECK:        %[[VAL_122:.*]] = vector.extract %[[VAL_119]][1, 0] : f16 from vector<2x2xf16>
    # CHECK:        %[[VAL_123:.*]] = vector.extract %[[VAL_119]][1, 1] : f16 from vector<2x2xf16>
    # CHECK:        memref.store %[[VAL_120]], %[[VAL_46]]{{\[}}%[[VAL_78]], %[[VAL_79]]] : memref<16x8xf16>
    # CHECK:        memref.store %[[VAL_121]], %[[VAL_46]]{{\[}}%[[VAL_78]], %[[VAL_81]]] : memref<16x8xf16>
    # CHECK:        memref.store %[[VAL_122]], %[[VAL_46]]{{\[}}%[[VAL_83]], %[[VAL_79]]] : memref<16x8xf16>
    # CHECK:        memref.store %[[VAL_123]], %[[VAL_46]]{{\[}}%[[VAL_83]], %[[VAL_81]]] : memref<16x8xf16>
    # CHECK:        gpu.terminator
    # CHECK:      }
    # CHECK:      call @print_res_as_memref_32(%[[VAL_46]]) : (memref<16x8xf16>) -> ()
    # CHECK:      return
    # CHECK:    }
    # CHECK:  }
    # CHECK:  module attributes {transform.with_named_sequence} {
    # CHECK:    transform.named_sequence @main(%[[VAL_124:.*]]: !transform.any_op {transform.readonly}) {
    # CHECK:      %[[VAL_125:.*]] = transform.structured.match ops{["linalg.matmul"]} in %[[VAL_124]] : (!transform.any_op) -> !transform.any_op
    # CHECK:      transform.nvgpu.rewrite_matmul_as_mma_sync %[[VAL_125]] : (!transform.any_op) -> ()
    # CHECK:      %[[VAL_126:.*]] = transform.structured.match interface{LoopLikeInterface} in %[[VAL_124]] : (!transform.any_op) -> !transform.any_op
    # CHECK:      transform.apply_licm to %[[VAL_126]] : !transform.any_op
    # CHECK:      transform.apply_cse to %[[VAL_124]] : !transform.any_op
    # CHECK:      transform.yield
    # CHECK:    }
    # CHECK:  }

    print(mod)
    filecheck_with_comments(mod)


CUDA_RUNTIME_LIB_PATH = Path(_mlir_libs.__file__).parent / f"libmlir_cuda_runtime.so"

NVIDIA_GPU = False
try:
    subprocess.check_output("nvidia-smi")
    NVIDIA_GPU = True
except Exception:
    print("No Nvidia GPU in system!")


# based on https://github.com/llvm/llvm-project/blob/9cc2122bf5a81f7063c2a32b2cb78c8d615578a1/mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir#L6
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
        tmp_alloc = memref.alloc((M, K), T.f32())
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
        tmp_alloc = memref.alloc((K, N), T.f32())
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
        tmp_alloc = memref.alloc((M, N), T.f32())
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
        lhs = memref.alloc((M, K), T.f16())
        rhs = memref.alloc((K, N), T.f16())
        res = memref.alloc((M, N), T.f16())

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
        def printMemrefF32(x: T.memref(T.f32())): ...

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

    if not NVIDIA_GPU:
        return

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


def test_tma(ctx: MLIRContext):
    M = K = N = 64

    @gpu.func
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def sgemm_tensor_core(
        A: T.memref(M, K, T.f16()),
        B: T.memref(K, N, T.f16()),
        C: T.memref(M, N, T.f32()),
        a_tma: llvm_ptr_t(),
        b_tma: llvm_ptr_t(),
    ):
        a_tma = builtin.unrealized_conversion_cast(
            [
                nvgpu.TensorMapDescriptorType.get(
                    T.memref(128, 64, T.f16(), memory_space=smem_space()),
                    swizzle=int(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B),
                    l2promo=int(nvgpu.TensorMapL2PromoKind.L2PROMO_NONE),
                    oob_fill=int(nvgpu.TensorMapOOBKind.OOB_ZERO),
                    interleave=int(nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE),
                )
            ],
            [a_tma],
        )
        b_tma = builtin.unrealized_conversion_cast(
            [
                nvgpu.TensorMapDescriptorType.get(
                    T.memref(64, 64, T.f16(), memory_space=smem_space()),
                    swizzle=int(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B),
                    l2promo=int(nvgpu.TensorMapL2PromoKind.L2PROMO_NONE),
                    oob_fill=int(nvgpu.TensorMapOOBKind.OOB_ZERO),
                    interleave=int(nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE),
                )
            ],
            [b_tma],
        )
        tid = gpu.thread_id()
        is_thread_0 = tid == 0

        mbarrier = nvgpu.mbarrier_create()
        nvgpu.mbarrier_init(mbarrier, 1, 0, predicate=is_thread_0)
        nvgpu.tma_prefetch_descriptor(a_tma)
        nvgpu.tma_prefetch_descriptor(b_tma)

        base = gpu.dynamic_shared_memory()

        shift = 0
        A_shared = memref.view(base, (M, K), dtype=T.f16(), shift=shift)
        shift += A_shared.n_elements
        B_shared = memref.view(base, (K, N), dtype=T.f16(), shift=shift)
        shift += B_shared.n_elements

        a = memref.view(base, (128, 64), dtype=T.f16(), shift=shift)
        shift += a.n_elements
        b1 = memref.view(base, (64, 64), dtype=T.f16(), shift=shift)
        shift += b1.n_elements
        b2 = memref.view(base, (64, 64), dtype=T.f16(), shift=shift)

        ta_count = a.n_elements + b1.n_elements + b2.n_elements
        nvgpu.mbarrier_arrive_expect_tx(mbarrier, ta_count, 0, predicate=is_thread_0)

        nvgpu.tma_async_load(
            a,
            mbarrier,
            a_tma,
            coordinates=[0, 0],
            mbar_id=0,
            predicate=is_thread_0,
        )
        nvgpu.tma_async_load(
            b1,
            mbarrier,
            b_tma,
            coordinates=[0, 0],
            mbar_id=0,
            predicate=is_thread_0,
        )
        nvgpu.tma_async_load(
            b2,
            mbarrier,
            b_tma,
            coordinates=[64, 0],
            mbar_id=0,
            predicate=is_thread_0,
        )
        nvgpu.mbarrier_try_wait_parity(mbarrier, mbar_id=0)

        accum = nvgpu.warpgroup_mma_init_accumulator(
            nvgpu.warpgroup_accumulator_t(M, N, T.f32())
        )
        lhs = nvgpu.warpgroup_generate_descriptor(
            nvgpu.warpgroup_descriptor(M, K, T.f16()), A_shared, a_tma
        )
        rhs = nvgpu.warpgroup_generate_descriptor(
            nvgpu.warpgroup_descriptor(K, N, T.f16()), B_shared, b_tma
        )
        acc = nvgpu.warpgroup_mma(accum, lhs, rhs, transpose_b=True)
        nvgpu.warpgroup_mma_store(acc, C)

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        sgemm_tensor_core.emit()

    # CHECK:  gpu.module @matmul [#nvvm.target]  {
    # CHECK:    gpu.func @sgemm_tensor_core(%[[VAL_0:.*]]: memref<64x64xf16>, %[[VAL_1:.*]]: memref<64x64xf16>, %[[VAL_2:.*]]: memref<64x64xf32>, %[[VAL_3:.*]]: !llvm.ptr, %[[VAL_4:.*]]: !llvm.ptr) kernel {
    # CHECK:      %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_3]] : !llvm.ptr to !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    # CHECK:      %[[VAL_6:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : !llvm.ptr to !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    # CHECK:      %[[VAL_7:.*]] = gpu.block_dim  x
    # CHECK:      %[[VAL_8:.*]] = gpu.block_dim  y
    # CHECK:      %[[VAL_9:.*]] = arith.muli %[[VAL_7]], %[[VAL_8]] : index
    # CHECK:      %[[VAL_10:.*]] = gpu.thread_id  z
    # CHECK:      %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_10]] : index
    # CHECK:      %[[VAL_12:.*]] = gpu.block_dim  x
    # CHECK:      %[[VAL_13:.*]] = gpu.thread_id  y
    # CHECK:      %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : index
    # CHECK:      %[[VAL_15:.*]] = arith.addi %[[VAL_11]], %[[VAL_14]] : index
    # CHECK:      %[[VAL_16:.*]] = gpu.thread_id  x
    # CHECK:      %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : index
    # CHECK:      %[[VAL_18:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_18]] : index
    # CHECK:      %[[VAL_20:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_21:.*]] = arith.constant 1 : index
    # CHECK:      %[[VAL_22:.*]] = arith.constant 0 : index
    # CHECK:      nvgpu.mbarrier.init %[[VAL_20]]{{\[}}%[[VAL_22]]], %[[VAL_21]], predicate = %[[VAL_19]] : <memorySpace = #gpu.address_space<workgroup>>
    # CHECK:      nvgpu.tma.prefetch.descriptor %[[VAL_5]] : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    # CHECK:      nvgpu.tma.prefetch.descriptor %[[VAL_6]] : <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    # CHECK:      %[[VAL_23:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_24:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_25:.*]] = memref.view %[[VAL_23]]{{\[}}%[[VAL_24]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_26:.*]] = arith.constant 8192 : index
    # CHECK:      %[[VAL_27:.*]] = memref.view %[[VAL_23]]{{\[}}%[[VAL_26]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_28:.*]] = arith.constant 16384 : index
    # CHECK:      %[[VAL_29:.*]] = memref.view %[[VAL_23]]{{\[}}%[[VAL_28]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_30:.*]] = arith.constant 32768 : index
    # CHECK:      %[[VAL_31:.*]] = memref.view %[[VAL_23]]{{\[}}%[[VAL_30]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_32:.*]] = arith.constant 40960 : index
    # CHECK:      %[[VAL_33:.*]] = memref.view %[[VAL_23]]{{\[}}%[[VAL_32]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_34:.*]] = arith.constant 16384 : index
    # CHECK:      %[[VAL_35:.*]] = arith.constant 0 : index
    # CHECK:      nvgpu.mbarrier.arrive.expect_tx %[[VAL_20]]{{\[}}%[[VAL_35]]], %[[VAL_34]], predicate = %[[VAL_19]] : <memorySpace = #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_36:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_37:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_38:.*]] = arith.constant 0 : index
    # CHECK:      nvgpu.tma.async.load %[[VAL_5]]{{\[}}%[[VAL_36]], %[[VAL_37]]], %[[VAL_20]]{{\[}}%[[VAL_38]]] to %[[VAL_29]], predicate = %[[VAL_19]] : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<128x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_39:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_40:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_41:.*]] = arith.constant 0 : index
    # CHECK:      nvgpu.tma.async.load %[[VAL_6]]{{\[}}%[[VAL_39]], %[[VAL_40]]], %[[VAL_20]]{{\[}}%[[VAL_41]]] to %[[VAL_31]], predicate = %[[VAL_19]] : <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_42:.*]] = arith.constant 64 : index
    # CHECK:      %[[VAL_43:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_44:.*]] = arith.constant 0 : index
    # CHECK:      nvgpu.tma.async.load %[[VAL_6]]{{\[}}%[[VAL_42]], %[[VAL_43]]], %[[VAL_20]]{{\[}}%[[VAL_44]]] to %[[VAL_33]], predicate = %[[VAL_19]] : <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_45:.*]] = arith.constant 10000000 : index
    # CHECK:      %[[VAL_46:.*]] = arith.constant 0 : index
    # CHECK:      %[[VAL_47:.*]] = arith.constant false
    # CHECK:      nvgpu.mbarrier.try_wait.parity %[[VAL_20]]{{\[}}%[[VAL_46]]], %[[VAL_47]], %[[VAL_45]] : <memorySpace = #gpu.address_space<workgroup>>
    # CHECK:      %[[VAL_48:.*]] = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<64x64xf32>>
    # CHECK:      %[[VAL_49:.*]] = nvgpu.warpgroup.generate.descriptor %[[VAL_25]], %[[VAL_5]] : memref<64x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>>
    # CHECK:      %[[VAL_50:.*]] = nvgpu.warpgroup.generate.descriptor %[[VAL_27]], %[[VAL_6]] : memref<64x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>>
    # CHECK:      %[[VAL_51:.*]] = nvgpu.warpgroup.mma %[[VAL_49]], %[[VAL_50]], %[[VAL_48]] {transposeB} : <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x64xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<64x64xf32>> -> <fragmented = vector<64x64xf32>>
    # CHECK:      nvgpu.warpgroup.mma.store %[[VAL_51]], %[[VAL_2]] : <fragmented = vector<64x64xf32>> to memref<64x64xf32>
    # CHECK:      gpu.return
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)
