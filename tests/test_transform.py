from textwrap import dedent

import pytest
from mlir.dialects.gpu import MappingId

from mlir.utils import types as T
from mlir.utils.ast.canonicalize import canonicalize
from mlir.utils.dialects import linalg, arith
from mlir.utils.dialects.ext.func import func
from mlir.utils.dialects.ext.gpu import block_attr, thread_attr
from mlir.utils.dialects.ext.scf import (
    range_,
    canonicalizer,
)
from mlir.utils.dialects.ext.tensor import pad
from mlir.utils.dialects.ext.transform import (
    sequence,
    unroll,
    get_parent_for,
    match,
    tile,
    tile_to_scf_forall,
    apply_patterns,
)
from mlir.utils.dialects.transform import apply_patterns_canonicalization, apply_cse
from mlir.utils.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from mlir.utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic_unroll(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def loop_unroll_op():
        for i in range_(0, 42, 5):
            v = i + i

    loop_unroll_op.emit()

    @sequence(target_tag="basic")
    def basic(target):
        m = match(target, ["arith.addi"])
        loop = get_parent_for(m)
        unroll(loop, 4)

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
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !pdl.operation
        transform.loop.unroll %1 {factor = 4 : i64} : !pdl.operation
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule"),
    )

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
    filecheck(correct, ctx.module)


def test_basic_tile(ctx):
    @func
    @canonicalize(using=canonicalizer)
    def pad_tensor_3_4(input_tensor: T.tensor(4, 16, T.f32), pad_value: T.f32):
        @pad(input_tensor, [3, 4], [5, 3])
        def pad_(i: T.index, j: T.index):
            return pad_value

        return pad_

    pad_tensor_3_4.emit()

    @sequence(target_tag="basic")
    def basic(target):
        m = match(target, ["tensor.pad"])
        tiled_linalg_op, loops = tile(m, sizes=[2, 3])

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
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %tiled_linalg_op, %loops:2 = transform.structured.tile %0[2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule")
        .canonicalize(),
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
        %c23 = arith.constant 23 : index
        %c12 = arith.constant 12 : index
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
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
            %21 = scf.if %19 -> (tensor<?x?xf32>) {
              %generated = tensor.generate %3 {
              ^bb0(%arg6: index, %arg7: index):
                tensor.yield %arg1 : f32
              } : tensor<2x?xf32>
              %cast_0 = tensor.cast %generated : tensor<2x?xf32> to tensor<?x?xf32>
              scf.yield %cast_0 : tensor<?x?xf32>
            } else {
              %extracted_slice = tensor.extract_slice %arg0[%6, %14] [%9, %17] [1, 1] : tensor<4x16xf32> to tensor<?x?xf32>
              %padded = tensor.pad %extracted_slice low[%4, %12] high[%11, %20] {
              ^bb0(%arg6: index, %arg7: index):
                tensor.yield %arg1 : f32
              } : tensor<?x?xf32> to tensor<?x?xf32>
              scf.yield %padded : tensor<?x?xf32>
            }
            %cast = tensor.cast %21 : tensor<?x?xf32> to tensor<2x?xf32>
            %inserted_slice = tensor.insert_slice %cast into %arg5[%arg2, %arg4] [2, %3] [1, 1] : tensor<2x?xf32> into tensor<12x23xf32>
            scf.yield %inserted_slice : tensor<12x23xf32>
          }
          scf.yield %2 : tensor<12x23xf32>
        }
        return %1 : tensor<12x23xf32>
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_linalg_tile(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        arg0: T.tensor(4, 16, T.f32),
        arg1: T.tensor(16, 8, T.f32),
        out: T.tensor(4, 8, T.f32),
    ):
        return linalg.matmul(arg0, arg1, out)

    matmul.emit()

    @sequence(target_tag="basic")
    def basic(target):
        m = match(target, ["linalg.matmul"])
        tiled_linalg_op, loops = tile(m, sizes=[2, 3])

    correct = dedent(
        """\
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %tiled_linalg_op, %loops:2 = transform.structured.tile %0[2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule")
        .canonicalize(),
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (-d0 + 8, 3)>
    module {
      func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %c8 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
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
    filecheck(correct, ctx.module)


def test_simple_matmul_tile_foreach_thread(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        arg0: T.tensor(4, 16, T.f32),
        arg1: T.tensor(16, 8, T.f32),
        out: T.tensor(4, 8, T.f32),
    ):
        return linalg.matmul(arg0, arg1, out)

    matmul.emit()

    @sequence(target_tag="basic")
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
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   num_threads [] tile_sizes [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule")
        .canonicalize(),
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

    filecheck(correct, ctx.module)


def test_common_extension_sugar(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def select_cmp_eq_select(arg0: T.i64, arg1: T.i64):
        a = arg0 == arg1
        b = arith.select(a, arg0, arg1)
        return b

    select_cmp_eq_select.emit()

    @sequence(target_tag="basic")
    def basic(target):
        m = match(target, ["func.func"])

        @apply_patterns(m)
        def pats():
            apply_patterns_canonicalization()

    correct = dedent(
        """\
    module {
      func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
        %0 = arith.cmpi eq, %arg0, %arg1 : i64
        %1 = arith.select %0, %arg0, %arg1 : i64
        return %1 : i64
      }
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        apply_patterns to %0 {
          transform.apply_patterns.canonicalization
        } : !transform.any_op
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule")
        .canonicalize(),
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

    filecheck(correct, ctx.module)


def test_apply_cse(ctx: MLIRContext):
    M, N, K = 3, 5, 3

    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        A: T.tensor(M, N, T.f32),
        B: T.tensor(N, K, T.f32),
        C: T.tensor(M, K, T.f32),
    ):
        return linalg.matmul(A, B, C)

    matmul.emit()

    @sequence(target_tag="basic")
    def basic(variant_op):
        matmul = match(variant_op, ["linalg.matmul"])

        forall_op, tiled_generic = tile_to_scf_forall(
            matmul, tile_sizes=[2], mapping=[block_attr(MappingId.DimX)]
        )

        top_func = match(variant_op, ["func.func"])

        @apply_patterns(top_func)
        def pats():
            apply_patterns_canonicalization()

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
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   num_threads [] tile_sizes [2](mapping = [#gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        apply_patterns to %1 {
          transform.apply_patterns.canonicalization
        } : !transform.any_op
        %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        apply_cse to %2 : !transform.any_op
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .transform_dialect_interpreter()
        .transform_dialect_erase_schedule()
        .canonicalize(),
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
    filecheck(correct, ctx.module)


def test_two_schedules(ctx: MLIRContext):
    N, H, W = 1, 66, 66
    C_i, C_o = 1, 3
    K = 3

    @func
    @canonicalize(using=canonicalizer)
    def conv_2d_nhwc_hwcf(
        input: T.tensor(N, C_i, H, W, T.f32),
        kernel: T.tensor(C_o, C_i, K, K, T.f32),
        output: T.tensor(N, C_o, H - 2, W - 2, T.f32),
    ):
        return linalg.conv_2d_nchw_fchw(input, kernel, output)

    conv_2d_nhwc_hwcf.emit()

    @sequence(target_tag="tile_outer")
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

    @sequence(target_tag="tile_inner")
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
      transform.sequence  failures(propagate) attributes {transform.target_tag = "tile_outer"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   num_threads [] tile_sizes [0, 1, 8, 8](mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
      transform.sequence  failures(propagate) attributes {transform.target_tag = "tile_inner"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   num_threads [] tile_sizes [0, 1, 1, 1](mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .transform_dialect_interpreter(debug_transform_root_tag="tile_outer")
        .transform_dialect_interpreter(debug_transform_root_tag="tile_inner")
        .transform_dialect_erase_schedule()
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

    filecheck(correct, ctx.module)
