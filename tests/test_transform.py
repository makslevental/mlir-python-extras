from textwrap import dedent

import pytest
from mlir.dialects import linalg as linalg_dialect, arith as arith_dialect
from mlir.dialects import pdl
from mlir.dialects.builtin import module
from mlir.dialects.gpu import MappingId
from mlir.dialects.transform import (
    get_parent_op,
    apply_cse,
    apply_licm,
    any_op_t,
)
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.structured import MatchInterfaceEnum
from mlir.ir import UnitAttr, StringAttr

from mlir.extras import types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import linalg, arith, tensor
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.gpu import block_attr, thread_attr
from mlir.extras.dialects.ext.scf import (
    range_,
    canonicalizer,
)
from mlir.extras.dialects.ext.tensor import pad
from mlir.extras.dialects.ext import transform
from mlir.extras.dialects.ext.transform import (
    match,
    tile_to_scf_for,
    tile_to_scf_forall,
    split_handle,
    include,
    transform_op_t,
    transform_any_op_t,
    get_producer_of_operand,
    get_consumers_of_result,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic_unroll(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def loop_unroll_op():
        for i in range_(0, 42, 5):
            v = i + i

    loop_unroll_op.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target):
            m = match(target, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="scf.for")
            loop_unroll(loop, 4)

    correct = dedent(
        """\
    module {
      func.func @loop_unroll_op() {
        %c0 = arith.constant 0 : index
        %c42 = arith.constant 42 : index
        %c5 = arith.constant 5 : index
        scf.for %arg0 = %c0 to %c42 step %c5 {
          %0 = arith.addi %arg0, %arg0 : index
        }
        return
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !pdl.operation
          transform.loop.unroll %1 {factor = 4 : i64} : !pdl.operation
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(ctx.module, Pipeline().transform_interpreter())

    correct = dedent(
        """\
    module {
      func.func @loop_unroll_op() {
        %c0 = arith.constant 0 : index
        %c42 = arith.constant 42 : index
        %c5 = arith.constant 5 : index
        %c40 = arith.constant 40 : index
        %c20 = arith.constant 20 : index
        scf.for %arg0 = %c0 to %c40 step %c20 {
          %1 = arith.addi %arg0, %arg0 : index
          %c1 = arith.constant 1 : index
          %2 = arith.muli %c5, %c1 : index
          %3 = arith.addi %arg0, %2 : index
          %4 = arith.addi %3, %3 : index
          %c2 = arith.constant 2 : index
          %5 = arith.muli %c5, %c2 : index
          %6 = arith.addi %arg0, %5 : index
          %7 = arith.addi %6, %6 : index
          %c3 = arith.constant 3 : index
          %8 = arith.muli %c5, %c3 : index
          %9 = arith.addi %arg0, %8 : index
          %10 = arith.addi %9, %9 : index
        }
        %0 = arith.addi %c40, %c40 : index
        return
      }
    }
    """
    )
    filecheck(correct, mod)


def test_basic_tile(ctx):
    @func
    @canonicalize(using=canonicalizer)
    def pad_tensor_3_4(input_tensor: T.tensor(4, 16, T.f32()), pad_value: T.f32()):
        @pad(input_tensor, [3, 4], [5, 3])
        def pad_(i: T.index(), j: T.index()):
            return pad_value

        return pad_

    pad_tensor_3_4.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target):
            m = match(target, ["tensor.pad"])
            tiled_linalg_op, loops = tile_to_scf_for(m, sizes=[2, 3])

    correct = dedent(
        """\
    module {
      func.func @pad_tensor_3_4(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<12x23xf32> {
        %padded = tensor.pad %arg0 low[3, 4] high[5, 3] {
        ^bb0(%arg2: index, %arg3: index):
          tensor.yield %arg1 : f32
        } : tensor<4x16xf32> to tensor<12x23xf32>
        return %padded : tensor<12x23xf32>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0[2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter().canonicalize(),
    )
    correct = dedent(
        """\
    #map = affine_map<(d0) -> (-d0 + 23, 3)>
    #map1 = affine_map<(d0) -> (-d0 + 3, 0)>
    #map2 = affine_map<(d0) -> (0, d0 - 3)>
    #map3 = affine_map<(d0) -> (4, d0)>
    #map4 = affine_map<(d0) -> (0, d0 - 1)>
    #map5 = affine_map<(d0, d1) -> (d0 - d1)>
    #map6 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
    #map7 = affine_map<(d0) -> (-d0 + 4, 0)>
    #map8 = affine_map<(d0) -> (0, d0 - 4)>
    #map9 = affine_map<(d0) -> (16, d0)>
    #map10 = affine_map<(d0, d1) -> (0, d0 + d1 - 4)>
    #map11 = affine_map<(d0, d1, d2, d3) -> (-d0 + d1 - d2 + d3)>
    module {
      func.func @pad_tensor_3_4(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<12x23xf32> {
        %c3 = arith.constant 3 : index
        %c23 = arith.constant 23 : index
        %c2 = arith.constant 2 : index
        %c12 = arith.constant 12 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<12x23xf32>
        %1 = scf.for %arg2 = %c0 to %c12 step %c2 iter_args(%arg3 = %0) -> (tensor<12x23xf32>) {
          %2 = scf.for %arg4 = %c0 to %c23 step %c3 iter_args(%arg5 = %arg3) -> (tensor<12x23xf32>) {
            %3 = affine.min #map(%arg4)
            %4 = affine.max #map1(%arg2)
            %5 = affine.max #map2(%arg2)
            %6 = affine.min #map3(%5)
            %7 = affine.max #map4(%arg2)
            %8 = affine.min #map3(%7)
            %9 = affine.apply #map5(%8, %6)
            %10 = arith.cmpi eq, %9, %c0 : index
            %11 = affine.apply #map6(%4, %8, %6)
            %12 = affine.max #map7(%arg4)
            %13 = affine.max #map8(%arg4)
            %14 = affine.min #map9(%13)
            %15 = affine.max #map10(%3, %arg4)
            %16 = affine.min #map9(%15)
            %17 = affine.apply #map5(%16, %14)
            %18 = arith.cmpi eq, %17, %c0 : index
            %19 = arith.ori %18, %10 : i1
            %20 = affine.apply #map11(%12, %3, %16, %14)
            %21 = scf.if %19 -> (tensor<2x?xf32>) {
              %generated = tensor.generate %3 {
              ^bb0(%arg6: index, %arg7: index):
                tensor.yield %arg1 : f32
              } : tensor<2x?xf32>
              scf.yield %generated : tensor<2x?xf32>
            } else {
              %extracted_slice = tensor.extract_slice %arg0[%6, %14] [%9, %17] [1, 1] : tensor<4x16xf32> to tensor<?x?xf32>
              %padded = tensor.pad %extracted_slice low[%4, %12] high[%11, %20] {
              ^bb0(%arg6: index, %arg7: index):
                tensor.yield %arg1 : f32
              } : tensor<?x?xf32> to tensor<2x?xf32>
              scf.yield %padded : tensor<2x?xf32>
            }
            %inserted_slice = tensor.insert_slice %21 into %arg5[%arg2, %arg4] [2, %3] [1, 1] : tensor<2x?xf32> into tensor<12x23xf32>
            scf.yield %inserted_slice : tensor<12x23xf32>
          }
          scf.yield %2 : tensor<12x23xf32>
        }
        return %1 : tensor<12x23xf32>
      }
    }
    """
    )
    filecheck(correct, mod)


def test_linalg_tile(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        arg0: T.tensor(4, 16, T.f32()),
        arg1: T.tensor(16, 8, T.f32()),
        out: T.tensor(4, 8, T.f32()),
    ):
        return linalg.matmul(arg0, arg1, out)

    matmul.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target):
            m = match(target, ["linalg.matmul"])
            tiled_linalg_op, loops = tile_to_scf_for(m, sizes=[2, 3])

    correct = dedent(
        """\
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0[2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter().canonicalize(),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (-d0 + 8, 3)>
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %c3 = arith.constant 3 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %0 = scf.for %arg3 = %c0 to %c4 step %c2 iter_args(%arg4 = %arg2) -> (tensor<4x8xf32>) {
          %1 = scf.for %arg5 = %c0 to %c8 step %c3 iter_args(%arg6 = %arg4) -> (tensor<4x8xf32>) {
            %2 = affine.min #map(%arg5)
            %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [2, 16] [1, 1] : tensor<4x16xf32> to tensor<2x16xf32>
            %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [16, %2] [1, 1] : tensor<16x8xf32> to tensor<16x?xf32>
            %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [2, %2] [1, 1] : tensor<4x8xf32> to tensor<2x?xf32>
            %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice, %extracted_slice_0 : tensor<2x16xf32>, tensor<16x?xf32>) outs(%extracted_slice_1 : tensor<2x?xf32>) -> tensor<2x?xf32>
            %inserted_slice = tensor.insert_slice %3 into %arg6[%arg3, %arg5] [2, %2] [1, 1] : tensor<2x?xf32> into tensor<4x8xf32>
            scf.yield %inserted_slice : tensor<4x8xf32>
          }
          scf.yield %1 : tensor<4x8xf32>
        }
        return %0 : tensor<4x8xf32>
      }
    }
    """
    )
    filecheck(correct, mod)


def test_simple_matmul_tile_foreach_thread(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        arg0: T.tensor(4, 16, T.f32()),
        arg1: T.tensor(16, 8, T.f32()),
        out: T.tensor(4, 8, T.f32()),
    ):
        return linalg.matmul(arg0, arg1, out)

    matmul.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target):
            m = match(target, ["linalg.matmul"])
            tiled_linalg_op, loops = tile_to_scf_forall(m, tile_sizes=[2, 3])

    correct = dedent(
        """\
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter().canonicalize(),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (d0 * -3 + 8, 3)>
    #map1 = affine_map<(d0) -> (d0 * 2)>
    #map2 = affine_map<(d0) -> (d0 * 3)>
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = scf.forall (%arg3, %arg4) in (2, 3) shared_outs(%arg5 = %arg2) -> (tensor<4x8xf32>) {
          %1 = affine.min #map(%arg4)
          %2 = affine.apply #map1(%arg3)
          %3 = affine.apply #map2(%arg4)
          %4 = affine.apply #map1(%arg3)
          %5 = affine.apply #map2(%arg4)
          %extracted_slice = tensor.extract_slice %arg0[%2, 0] [2, 16] [1, 1] : tensor<4x16xf32> to tensor<2x16xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [16, %1] [1, 1] : tensor<16x8xf32> to tensor<16x?xf32>
          %extracted_slice_1 = tensor.extract_slice %arg5[%4, %5] [2, %1] [1, 1] : tensor<4x8xf32> to tensor<2x?xf32>
          %6 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice, %extracted_slice_0 : tensor<2x16xf32>, tensor<16x?xf32>) outs(%extracted_slice_1 : tensor<2x?xf32>) -> tensor<2x?xf32>
          %7 = affine.apply #map1(%arg3)
          %8 = affine.apply #map2(%arg4)
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %6 into %arg5[%7, %8] [2, %1] [1, 1] : tensor<2x?xf32> into tensor<4x8xf32>
          }
        }
        return %0 : tensor<4x8xf32>
      }
    }
    """
    )

    filecheck(correct, mod)


def test_common_extension_sugar(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def select_cmp_eq_select(arg0: T.i64(), arg1: T.i64()):
        a = arg0 == arg1
        b = arith_dialect.select(a, arg0, arg1)
        return b

    select_cmp_eq_select.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target):
            m = match(target, ["func.func"])

            @apply_patterns(m)
            def pats():
                transform.apply_patterns.canonicalization()

    correct = dedent(
        """\
    module {
      func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
        %0 = arith.cmpi eq, %arg0, %arg1 : i64
        %1 = arith.select %0, %arg0, %arg1 : i64
        return %1 : i64
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          transform.apply_patterns to %0 {
            transform.apply_patterns.canonicalization
          } : !transform.any_op
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter().canonicalize(),
    )

    correct = dedent(
        """\
    module {
      func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
        return %arg1 : i64
      }
    }
    """
    )

    filecheck(correct, mod)


def test_apply_cse(ctx: MLIRContext):
    M, N, K = 3, 5, 3

    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        A: T.tensor(M, N, T.f32()),
        B: T.tensor(N, K, T.f32()),
        C: T.tensor(M, K, T.f32()),
    ):
        return linalg.matmul(A, B, C)

    matmul.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(variant_op):
            matmul = match(variant_op, ["linalg.matmul"])

            forall_op, tiled_generic = tile_to_scf_forall(
                matmul, tile_sizes=[2], mapping=[block_attr(MappingId.DimX)]
            )

            top_func = match(variant_op, ["func.func"])

            @apply_patterns(top_func)
            def pats():
                transform.apply_patterns.canonicalization()

            top_func = match(variant_op, ["func.func"])
            apply_cse(top_func)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func @matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
        %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<3x5xf32>, tensor<5x3xf32>) outs(%arg2 : tensor<3x3xf32>) -> tensor<3x3xf32>
        return %0 : tensor<3x3xf32>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [2](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          transform.apply_patterns to %1 {
            transform.apply_patterns.canonicalization
          } : !transform.any_op
          %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          transform.apply_cse to %2 : !transform.any_op
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter().canonicalize(),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (d0 * -2 + 3, 2)>
    #map1 = affine_map<(d0) -> (d0 * 2)>
    module {
      func.func @matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
        %0 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %arg2) -> (tensor<3x3xf32>) {
          %1 = affine.min #map(%arg3)
          %2 = affine.apply #map1(%arg3)
          %extracted_slice = tensor.extract_slice %arg0[%2, 0] [%1, 5] [1, 1] : tensor<3x5xf32> to tensor<?x5xf32>
          %extracted_slice_0 = tensor.extract_slice %arg4[%2, 0] [%1, 3] [1, 1] : tensor<3x3xf32> to tensor<?x3xf32>
          %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice, %arg1 : tensor<?x5xf32>, tensor<5x3xf32>) outs(%extracted_slice_0 : tensor<?x3xf32>) -> tensor<?x3xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %3 into %arg4[%2, 0] [%1, 3] [1, 1] : tensor<?x3xf32> into tensor<3x3xf32>
          }
        } {mapping = [#gpu.block<x>]}
        return %0 : tensor<3x3xf32>
      }
    }
    """
    )
    filecheck(correct, mod)


def test_two_schedules(ctx: MLIRContext):
    N, H, W = 1, 66, 66
    C_i, C_o = 1, 3
    K = 3

    @func
    @canonicalize(using=canonicalizer)
    def conv_2d_nhwc_hwcf(
        input: T.tensor(N, C_i, H, W, T.f32()),
        kernel: T.tensor(C_o, C_i, K, K, T.f32()),
        output: T.tensor(N, C_o, H - 2, W - 2, T.f32()),
    ):
        return linalg.conv_2d_nchw_fchw(input, kernel, output)

    conv_2d_nhwc_hwcf.emit()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("tile_outer", [any_op_t()], [])
        def tile_outer(target):
            m = match(target, ["linalg.conv_2d_nchw_fchw"])
            tiled = tile_to_scf_forall(
                m,
                tile_sizes=[0, 1, 8, 8],
                mapping=[
                    block_attr(MappingId.DimX),
                    block_attr(MappingId.DimY),
                    block_attr(MappingId.DimZ),
                ],
            )

        @named_sequence("tile_inner", [any_op_t()], [])
        def tile_inner(target):
            m = match(target, ["linalg.conv_2d_nchw_fchw"])
            tiled = tile_to_scf_forall(
                m,
                tile_sizes=[0, 1, 1, 1],
                mapping=[
                    thread_attr(MappingId.DimX),
                    thread_attr(MappingId.DimY),
                    thread_attr(MappingId.DimZ),
                ],
            )

    correct = dedent(
        """\
    module {
      func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x1x66x66xf32>, %arg1: tensor<3x1x3x3xf32>, %arg2: tensor<1x3x64x64xf32>) -> tensor<1x3x64x64xf32> {
        %0 = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<1x1x66x66xf32>, tensor<3x1x3x3xf32>) outs(%arg2 : tensor<1x3x64x64xf32>) -> tensor<1x3x64x64xf32>
        return %0 : tensor<1x3x64x64xf32>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @tile_outer(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [0, 1, 8, 8](mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          transform.yield 
        }
        transform.named_sequence @tile_inner(%arg0: !transform.any_op) {
          %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!transform.any_op) -> !transform.any_op
          %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [0, 1, 1, 1](mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          transform.yield 
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    mod = run_pipeline(
        ctx.module,
        Pipeline()
        .transform_interpreter(entry_point="tile_outer")
        .transform_interpreter(entry_point="tile_inner")
        .canonicalize(),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (d0 * 8)>
    module {
      func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x1x66x66xf32>, %arg1: tensor<3x1x3x3xf32>, %arg2: tensor<1x3x64x64xf32>) -> tensor<1x3x64x64xf32> {
        %0 = scf.forall (%arg3, %arg4, %arg5) in (3, 8, 8) shared_outs(%arg6 = %arg2) -> (tensor<1x3x64x64xf32>) {
          %1 = affine.apply #map(%arg4)
          %2 = affine.apply #map(%arg5)
          %3 = affine.apply #map(%arg4)
          %4 = affine.apply #map(%arg5)
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %1, %2] [1, 1, 10, 10] [1, 1, 1, 1] : tensor<1x1x66x66xf32> to tensor<1x1x10x10xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<3x1x3x3xf32> to tensor<1x1x3x3xf32>
          %extracted_slice_1 = tensor.extract_slice %arg6[0, %arg3, %3, %4] [1, 1, 8, 8] [1, 1, 1, 1] : tensor<1x3x64x64xf32> to tensor<1x1x8x8xf32>
          %5 = scf.forall (%arg7, %arg8, %arg9) in (1, 8, 8) shared_outs(%arg10 = %extracted_slice_1) -> (tensor<1x1x8x8xf32>) {
            %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0, %arg8, %arg9] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x1x10x10xf32> to tensor<1x1x3x3xf32>
            %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg7, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x1x3x3xf32> to tensor<1x1x3x3xf32>
            %extracted_slice_4 = tensor.extract_slice %arg10[0, %arg7, %arg8, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x8x8xf32> to tensor<1x1x1x1xf32>
            %8 = linalg.conv_2d_nchw_fchw ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) outs(%extracted_slice_4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %8 into %arg10[0, %arg7, %arg8, %arg9] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x1x8x8xf32>
            }
          } {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
          %6 = affine.apply #map(%arg4)
          %7 = affine.apply #map(%arg5)
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %5 into %arg6[0, %arg3, %6, %7] [1, 1, 8, 8] [1, 1, 1, 1] : tensor<1x1x8x8xf32> into tensor<1x3x64x64xf32>
          }
        } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}
        return %0 : tensor<1x3x64x64xf32>
      }
    }
    """
    )

    filecheck(correct, mod)


# based off of https://github.com/nod-ai/iree-amd-aie/blob/89361beb07f4846e65d3a171503b96dcc9267fed/tests/samples/matmul_fill_spec_pack.mlir
def test_matmul_schedule(ctx: MLIRContext):
    M, K, N = 16, 256, 256

    @func
    def matmul_i8_i8(
        A: T.tensor(M, K, T.i8()),
        B: T.tensor(K, N, T.i8()),
    ):
        empty = tensor.empty((M, N), T.i8())
        filled = linalg_dialect.fill(arith.constant(0), outs=[empty])
        return linalg.matmul(A, B, filled)

    @module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_i8_i8.emit(force=True)

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence(
            "cleanup",
            [any_op_t()],
            [],
            arg_attrs=[{"transform.readonly": UnitAttr.get()}],
        )
        def cleanup(target: any_op_t()):
            top_func = match(target, ["func.func"])

            @apply_patterns(top_func)
            def pats():
                transform.apply_patterns.linalg.tiling_canonicalization()
                # transform.apply_patterns.iree.fold_fill_into_pad
                transform.apply_patterns.scf.for_loop_canonicalization()
                transform.apply_patterns.canonicalization()

            all_loops = match(target, interface=MatchInterfaceEnum.LoopLikeInterface)
            apply_licm(all_loops)
            apply_cse(top_func)

        @named_sequence(
            "main", [any_op_t()], [], arg_attrs=[{"transform.readonly": UnitAttr.get()}]
        )
        def main(variant_op: any_op_t()):
            ops = match(variant_op, ops=["linalg.fill", "linalg.matmul"])
            fill, matmul = split_handle(ops)
            # First level tile to forall with tile_sizes [16, 64].
            tiled_matmul, (forall,) = tile_to_scf_forall(
                matmul,
                [16, 64],
                mapping=[
                    thread_attr(MappingId.DimY),
                    thread_attr(MappingId.DimX),
                ],
            )
            # Fuse fill operation into the loop
            transform.structured.fuse_into_containing_op(fill, forall)
            # Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
            packed = transform.structured.pack(tiled_matmul, packed_sizes=[16, 64, 64])

            # Transpose B matrix from [K N n k] to [K N k n]
            pack_producer_b0 = get_producer_of_operand(packed, 1)
            packed_b0, pack_b0, empty_unpack_b0 = transform.structured.pack_transpose(
                pack_producer_b0, packed, inner_perm=[1, 0]
            )

            # Run canonicalization to fold fill with pack and unpack operations.
            include("cleanup", [variant_op])

            # TODO(max): can't bufferize right now because missing what's the rule/wisdom on upstreaming transform extension patterns into upstream? can this go?
            # https://github.com/openxla/iree/blob/010710b6b13f6385e514d53eeb1de1daf7d683d3/compiler/src/iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.cpp#L126
            # https://github.com/openxla/iree/blob/dc4e59ec0c513872e320836142e79707edf28b1d/compiler/src/iree/compiler/Codegen/Interfaces/BufferizationInterfaces.cpp#L648
            # # Bufferize to shared memory allocation
            pack_producer_a0 = get_producer_of_operand(packed_b0, 0)
            pack_producer_c0 = get_producer_of_operand(packed_b0, 2)
            buffer_a0, new_a0 = transform.structured.bufferize_to_allocation(
                pack_b0,
                memory_space="shared",
                bufferize_destination_only=True,
                emit_dealloc=True,
            )
            buffer_b0, new_b0 = transform.structured.bufferize_to_allocation(
                pack_producer_a0,
                memory_space="shared",
                bufferize_destination_only=True,
                emit_dealloc=True,
            )
            buffer_c0, new_c0 = transform.structured.bufferize_to_allocation(
                pack_producer_c0,
                memory_space="shared",
                bufferize_destination_only=True,
                emit_dealloc=True,
            )

            # Second level tile to forall with tile_sizes [1, 1].
            tiled_matmul_1, (forall_1,) = tile_to_scf_forall(
                matmul,
                [1, 1],
                mapping=[
                    thread_attr(MappingId.DimY),
                    thread_attr(MappingId.DimX),
                ],
            )
            # Find the fill operation to fuse.
            # TODO(ravishankarm): Find a better way to find the fill operation.
            fused_fill_1 = get_producer_of_operand(forall_1, 0)
            transform.structured.fuse_into_containing_op(fused_fill_1, forall_1)

            # Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
            packed_2 = transform.structured.pack(
                tiled_matmul_1, packed_sizes=[0, 0, 0, 4, 8, 8]
            )

            # Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
            pack_producer_a = get_producer_of_operand(packed_2, 0)
            packed_a, pack_a, empty_unpack_a = transform.structured.pack_transpose(
                pack_producer_a, packed_2, outer_perm=[0, 1, 3, 2]
            )

            # Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
            pack_producer_b = get_producer_of_operand(packed_a, 1)
            packed_b, pack_b, empty_unpack_b = transform.structured.pack_transpose(
                pack_producer_b, packed_a, inner_perm=[1, 0], outer_perm=[0, 1, 3, 2]
            )

            # Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
            unpack = get_consumers_of_result(transform_any_op_t(), packed_b, 0)
            packed_c, pack_c, unpack_c = transform.structured.pack_transpose(
                unpack, packed_b, outer_perm=[0, 1, 3, 2]
            )

            # Fold fill operation with pack and unpack.
            include("cleanup", [variant_op])

            # Bufferize to local memory allocation
            buffer_a, new_a = transform.structured.bufferize_to_allocation(
                pack_a,
                memory_space="local",
                bufferize_destination_only=True,
            )
            buffer_b, new_b = transform.structured.bufferize_to_allocation(
                pack_b,
                memory_space="local",
                bufferize_destination_only=True,
            )

            # Earlier handle for pack operation is now defunct. Find it again.
            fused_pack_fill = get_producer_of_operand(packed_c, 2)
            buffer_c, new_c = transform.structured.bufferize_to_allocation(
                fused_pack_fill,
                memory_space="local",
                bufferize_destination_only=True,
            )

            # Tile reduction dimension.
            tiled_reduction, loop = tile_to_scf_for(packed_c, sizes=[0, 0, 1])

            # Clean up.
            include("cleanup", [variant_op])

    correct = """\
        module {
          module attributes {transform.target_tag = "payload"} {
            func.func @matmul_i8_i8(%arg0: tensor<16x256xi8>, %arg1: tensor<256x256xi8>) -> tensor<16x256xi8> {
              %0 = tensor.empty() : tensor<16x256xi8>
              %c0_i32 = arith.constant 0 : i32
              %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<16x256xi8>) -> tensor<16x256xi8>
              %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<16x256xi8>, tensor<256x256xi8>) outs(%1 : tensor<16x256xi8>) -> tensor<16x256xi8>
              return %2 : tensor<16x256xi8>
            }
          }
          module attributes {transform.with_named_sequence} {
            transform.named_sequence @cleanup(%arg0: !transform.any_op {transform.readonly}) {
              %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
              transform.apply_patterns to %0 {
                transform.apply_patterns.linalg.tiling_canonicalization
                transform.apply_patterns.scf.for_loop_canonicalization
                transform.apply_patterns.canonicalization
              } : !transform.any_op
              %1 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
              transform.apply_licm to %1 : !transform.any_op
              transform.apply_cse to %0 : !transform.any_op
              transform.yield 
            }
            transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly}) {
              %0 = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
              %1:2 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              %tiled_op, %forall_op = transform.structured.tile_using_forall %1#1 tile_sizes [16, 64](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
              %2 = transform.structured.pack %tiled_op packed_sizes = [16, 64, 64] : (!transform.any_op) -> !transform.any_op
              %3 = transform.get_producer_of_operand %2[1] : (!transform.any_op) -> !transform.any_op
              %packed_op, %pack_op, %un_pack_op = transform.structured.pack_transpose %3 with_compute_op(%2) inner_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
              transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
              %4 = transform.get_producer_of_operand %packed_op[0] : (!transform.any_op) -> !transform.any_op
              %5 = transform.get_producer_of_operand %packed_op[2] : (!transform.any_op) -> !transform.any_op
              %allocated_buffer, %new_ops = transform.structured.bufferize_to_allocation %pack_op {bufferize_destination_only, emit_dealloc, memory_space = "shared"} : !transform.any_op
              %allocated_buffer_0, %new_ops_1 = transform.structured.bufferize_to_allocation %4 {bufferize_destination_only, emit_dealloc, memory_space = "shared"} : !transform.any_op
              %allocated_buffer_2, %new_ops_3 = transform.structured.bufferize_to_allocation %5 {bufferize_destination_only, emit_dealloc, memory_space = "shared"} : !transform.any_op
              %tiled_op_4, %forall_op_5 = transform.structured.tile_using_forall %1#1 tile_sizes [1, 1](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              %6 = transform.get_producer_of_operand %forall_op_5[0] : (!transform.any_op) -> !transform.any_op
              %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %6 into %forall_op_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
              %7 = transform.structured.pack %tiled_op_4 packed_sizes = [0, 0, 0, 4, 8, 8] : (!transform.any_op) -> !transform.any_op
              %8 = transform.get_producer_of_operand %7[0] : (!transform.any_op) -> !transform.any_op
              %packed_op_8, %pack_op_9, %un_pack_op_10 = transform.structured.pack_transpose %8 with_compute_op(%7) outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
              %9 = transform.get_producer_of_operand %packed_op_8[1] : (!transform.any_op) -> !transform.any_op
              %packed_op_11, %pack_op_12, %un_pack_op_13 = transform.structured.pack_transpose %9 with_compute_op(%packed_op_8) outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
              %10 = transform.get_consumers_of_result %packed_op_11[0] : (!transform.any_op) -> !transform.any_op
              %packed_op_14, %pack_op_15, %un_pack_op_16 = transform.structured.pack_transpose %10 with_compute_op(%packed_op_11) outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
              transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
              %allocated_buffer_17, %new_ops_18 = transform.structured.bufferize_to_allocation %pack_op_9 {bufferize_destination_only, memory_space = "local"} : !transform.any_op
              %allocated_buffer_19, %new_ops_20 = transform.structured.bufferize_to_allocation %pack_op_12 {bufferize_destination_only, memory_space = "local"} : !transform.any_op
              %11 = transform.get_producer_of_operand %packed_op_14[2] : (!transform.any_op) -> !transform.any_op
              %allocated_buffer_21, %new_ops_22 = transform.structured.bufferize_to_allocation %11 {bufferize_destination_only, memory_space = "local"} : !transform.any_op
              %tiled_linalg_op, %loops = transform.structured.tile_using_for %packed_op_14[0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
              transform.yield 
            }
          }
        }
    """
    filecheck(correct, ctx.module)


# based off of https://github.com/nod-ai/iree-amd-aie/blob/89361beb07f4846e65d3a171503b96dcc9267fed/tests/samples/matmul_fill_spec_pack.mlir
def test_matmul_schedule_run(ctx: MLIRContext):
    M, K, N = 16, 256, 256

    @func
    def matmul_i8_i8(
        A: T.tensor(M, K, T.i8()),
        B: T.tensor(K, N, T.i8()),
    ):
        empty = tensor.empty((M, N), T.i8())
        filled = linalg_dialect.fill(arith.constant(0), outs=[empty])
        return linalg.matmul(A, B, filled)

    @module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_i8_i8.emit(force=True)

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence(
            "cleanup",
            [any_op_t()],
            [],
            arg_attrs=[{"transform.readonly": UnitAttr.get()}],
        )
        def cleanup(target: any_op_t()):
            top_func = match(target, ["func.func"])

            @apply_patterns(top_func)
            def pats():
                transform.apply_patterns.linalg.tiling_canonicalization()
                # transform.apply_patterns.iree.fold_fill_into_pad
                transform.apply_patterns.scf.for_loop_canonicalization()
                transform.apply_patterns.canonicalization()

            all_loops = match(target, interface=MatchInterfaceEnum.LoopLikeInterface)
            apply_licm(all_loops)
            apply_cse(top_func)

        @named_sequence(
            "main", [any_op_t()], [], arg_attrs=[{"transform.readonly": UnitAttr.get()}]
        )
        def main(variant_op: any_op_t()):
            ops = match(variant_op, ops=["linalg.fill", "linalg.matmul"])
            fill, matmul = split_handle(ops)
            # First level tile to forall with tile_sizes [16, 64].
            tiled_matmul, (forall,) = tile_to_scf_forall(
                matmul,
                [16, 64],
                mapping=[
                    thread_attr(MappingId.DimY),
                    thread_attr(MappingId.DimX),
                ],
            )
            # Fuse fill operation into the loop
            transform.structured.fuse_into_containing_op(fill, forall)
            # Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
            packed = transform.structured.pack(tiled_matmul, packed_sizes=[16, 64, 64])

            # Transpose B matrix from [K N n k] to [K N k n]
            pack_producer_b0 = get_producer_of_operand(packed, 1)
            packed_b0, pack_b0, empty_unpack_b0 = transform.structured.pack_transpose(
                pack_producer_b0, packed, inner_perm=[1, 0]
            )

            # Run canonicalization to fold fill with pack and unpack operations.
            include("cleanup", [variant_op])

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )
    correct = """\
        #map = affine_map<(d0) -> (d0 * 16)>
        #map1 = affine_map<(d0) -> (d0 * 64)>
        #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
        #map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
        #map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
        module {
          module attributes {transform.target_tag = "payload"} {
            func.func @matmul_i8_i8(%arg0: tensor<16x256xi8>, %arg1: tensor<256x256xi8>) -> tensor<16x256xi8> {
              %c0_i32 = arith.constant 0 : i32
              %0 = tensor.empty() : tensor<16x256xi8>
              %1 = tensor.empty() : tensor<1x4x16x64xi8>
              %2 = tensor.empty() : tensor<4x1x64x64xi8>
              %3 = tensor.empty() : tensor<1x1x16x64xi8>
              %4 = scf.forall (%arg2, %arg3) in (1, 4) shared_outs(%arg4 = %0) -> (tensor<16x256xi8>) {
                %5 = affine.apply #map(%arg2)
                %6 = affine.apply #map1(%arg3)
                %extracted_slice = tensor.extract_slice %arg0[%5, 0] [16, 256] [1, 1] : tensor<16x256xi8> to tensor<16x256xi8>
                %extracted_slice_0 = tensor.extract_slice %arg1[0, %6] [256, 64] [1, 1] : tensor<256x256xi8> to tensor<256x64xi8>
                %extracted_slice_1 = tensor.extract_slice %arg4[%5, %6] [16, 64] [1, 1] : tensor<16x256xi8> to tensor<16x64xi8>
                %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %1 : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
                %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %2 : tensor<256x64xi8> -> tensor<4x1x64x64xi8>
                %7 = linalg.fill ins(%c0_i32 : i32) outs(%3 : tensor<1x1x16x64xi8>) -> tensor<1x1x16x64xi8>
                %8 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_2 : tensor<1x4x16x64xi8>, tensor<4x1x64x64xi8>) outs(%7 : tensor<1x1x16x64xi8>) {
                ^bb0(%in: i8, %in_3: i8, %out: i8):
                  %9 = arith.muli %in, %in_3 : i8
                  %10 = arith.addi %out, %9 : i8
                  linalg.yield %10 : i8
                } -> tensor<1x1x16x64xi8>
                %unpack = tensor.unpack %8 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %extracted_slice_1 : tensor<1x1x16x64xi8> -> tensor<16x64xi8>
                scf.forall.in_parallel {
                  tensor.parallel_insert_slice %unpack into %arg4[%5, %6] [16, 64] [1, 1] : tensor<16x64xi8> into tensor<16x256xi8>
                }
              } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
              return %4 : tensor<16x256xi8>
            }
          }
          module attributes {transform.with_named_sequence} {
            transform.named_sequence @cleanup(%arg0: !transform.any_op {transform.readonly}) {
              %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
              transform.apply_patterns to %0 {
                transform.apply_patterns.linalg.tiling_canonicalization
                transform.apply_patterns.scf.for_loop_canonicalization
                transform.apply_patterns.canonicalization
              } : !transform.any_op
              %1 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
              transform.apply_licm to %1 : !transform.any_op
              transform.apply_cse to %0 : !transform.any_op
              transform.yield 
            }
            transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly}) {
              %0 = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
              %1:2 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              %tiled_op, %forall_op = transform.structured.tile_using_forall %1#1 tile_sizes [16, 64](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
              %2 = transform.structured.pack %tiled_op packed_sizes = [16, 64, 64] : (!transform.any_op) -> !transform.any_op
              %3 = transform.get_producer_of_operand %2[1] : (!transform.any_op) -> !transform.any_op
              %packed_op, %pack_op, %un_pack_op = transform.structured.pack_transpose %3 with_compute_op(%2) inner_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
              transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
              transform.yield 
            }
          }
        }
    """
    filecheck(correct, mod)


def test_tensor_pack_schedule_lower_pack_run(ctx: MLIRContext):
    @func
    def tensor_pack(
        src: T.tensor(129, 47, 16, 16, T.f32()),
        dst: T.tensor(17, 2, 16, 16, 32, 8, T.f32()),
    ):
        return tensor.pack(
            src,
            dst,
            inner_dims_pos=[1, 0],
            inner_tiles=[32, 8],
            padding_value=arith.constant(0.0),
        )

    @module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        tensor_pack.emit(force=True)

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence("main", [any_op_t()], [])
        def main(variant_op: any_op_t()):
            packed = match(
                variant_op,
                ops=["tensor.pack"],
                matched_op=transform_op_t("tensor.pack"),
            )
            lowered_pack = transform.structured.lower_pack(packed)

    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )

    correct = dedent(
        """\
        module {
          module attributes {transform.target_tag = "payload"} {
            func.func @tensor_pack(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<17x2x16x16x32x8xf32>) -> tensor<17x2x16x16x32x8xf32> {
              %cst = arith.constant 0.000000e+00 : f32
              %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[7, 17, 0, 0] {
              ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
                tensor.yield %cst : f32
              } : tensor<129x47x16x16xf32> to tensor<136x64x16x16xf32>
              %expanded = tensor.expand_shape %padded [[0, 1], [2, 3], [4], [5]] : tensor<136x64x16x16xf32> into tensor<17x8x2x32x16x16xf32>
              %transposed = linalg.transpose ins(%expanded : tensor<17x8x2x32x16x16xf32>) outs(%arg1 : tensor<17x2x16x16x32x8xf32>) permutation = [0, 2, 4, 5, 3, 1] 
              return %transposed : tensor<17x2x16x16x32x8xf32>
            }
          }
          module attributes {transform.with_named_sequence} {
            transform.named_sequence @main(%arg0: !transform.any_op) {
              %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.pack">
              %pad_op, %expand_shape_op, %transpose_op = transform.structured.lower_pack %0 : (!transform.op<"tensor.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
              transform.yield 
            }
          }
        }
    """
    )
