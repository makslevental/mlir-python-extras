import mlir.extras.types as T
import pytest
from mlir.dialects.builtin import module
from mlir.dialects.func import return_
from mlir.dialects.memref import alloca_scope, alloca_scope_return
from mlir.dialects.scf import yield_ as scf_yield
from mlir.dialects.tensor import rank, yield_ as tensor_yield
from mlir.extras.types import tensor

from mlir.extras.dialects.ext import linalg, memref
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.cf import br, cond_br
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.memref import alloca_scope
from mlir.extras.dialects.ext.scf import execute_region
from mlir.extras.dialects.ext.tensor import S, generate

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)
from mlir.extras.util import bb

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_simple_region_op(ctx: MLIRContext):
    @execute_region([])
    def demo_region():
        one = constant(1)
        scf_yield([])

    ctx.module.operation.verify()

    # CHECK:  scf.execute_region {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    scf.yield
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_no_args_decorator(ctx: MLIRContext):
    @alloca_scope([])
    def demo_scope1():
        one = constant(1)
        alloca_scope_return([])

    @alloca_scope([])
    def demo_scope2():
        one = constant(2)
        alloca_scope_return([])

    ctx.module.operation.verify()

    # CHECK:  memref.alloca_scope  {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  }
    # CHECK:  memref.alloca_scope  {
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_func(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    demo_fun1()
    ctx.module.operation.verify()

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @demo_fun1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_block_args(ctx: MLIRContext):
    one = constant(1, T.index())
    two = constant(2, T.index())

    @generate(tensor(S, 3, S, T.f32()), dynamic_extents=[one, two])
    def demo_fun1(i: T.index(), j: T.index(), k: T.index()):
        one = constant(1.0)
        tensor_yield(one)

    r = rank(demo_fun1)

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.generate %[[VAL_0]], %[[VAL_1]] {
    # CHECK:  ^bb0(%[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    tensor.yield %[[VAL_6]] : f32
    # CHECK:  } : tensor<?x3x?xf32>
    # CHECK:  %[[VAL_7:.*]] = tensor.rank %[[VAL_8:.*]] : tensor<?x3x?xf32>

    filecheck_with_comments(ctx.module)


def test_empty_results_list(ctx: MLIRContext):
    one = constant(1, T.index())
    two = constant(2, T.index())

    @func
    def demo_fun1():
        mem1 = memref.alloc((10, 10), T.f32())
        mem2 = memref.alloc((10, 10), T.f32())
        x = linalg.copy(mem1, mem2)

    demo_fun1.emit()

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
    # CHECK:  func.func @demo_fun1() {
    # CHECK:    %[[VAL_2:.*]] = memref.alloc() : memref<10x10xf32>
    # CHECK:    %[[VAL_3:.*]] = memref.alloc() : memref<10x10xf32>
    # CHECK:    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_2]] : memref<10x10xf32>) outs(%[[VAL_3]] : memref<10x10xf32>)
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([one])
        with bb():
            two = constant(2)
            return one

    foo1()
    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_2:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_bbs_multiple(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([one])
        with bb() as (b1, _):
            two = constant(2)
            return_([two])
        with bb() as (b2, _):
            two = constant(3)
            return one

    foo1.emit()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    return %[[VAL_1]] : i32
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_br(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            return_([])
        with bb() as (b2, _):
            three = constant(3)
            br(b1)
        with bb() as (b3, _):
            four = constant(4)

    foo1.emit()
    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    return
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    cf.br ^bb1
    # CHECK:  ^bb3:
    # CHECK:    %[[VAL_3:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_br_loop(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            x = br()
        with bb(x) as (b2, _):
            three = constant(3)
            br(b1)
        with bb() as (b3, _):
            four = constant(4)

    foo1.emit()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    cf.br ^bb2
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    cf.br ^bb1
    # CHECK:  ^bb3:
    # CHECK:    %[[VAL_3:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_br_loop_arg(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            x = br(two)
        with bb(x) as (b2, (arg,)):
            three = constant(3)
            five = three + arg
            br(b1)
        with bb() as (b3, _):
            four = constant(4)

    foo1.emit()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    cf.br ^bb2(%[[VAL_1]] : i32)
    # CHECK:  ^bb2(%[[VAL_2:.*]]: i32):
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3 : i32
    # CHECK:    %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : i32
    # CHECK:    cf.br ^bb1
    # CHECK:  ^bb3:
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_br_two_preds(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            x = br(two)
        with bb() as (b2, _):
            four = constant(4)
            y = br(four)
        with bb(x, y) as (b3, [arg]):
            three = constant(3)
            five = three + arg

    foo1.emit()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    cf.br ^bb3(%[[VAL_1]] : i32)
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_2:.*]] = arith.constant 4 : i32
    # CHECK:    cf.br ^bb3(%[[VAL_2]] : i32)
    # CHECK:  ^bb3(%[[VAL_3:.*]]: i32):
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3 : i32
    # CHECK:    %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_cond_br(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            three = constant(3)
            cond = two < three
            x = cond_br(cond)
        with x.true as (b2, _):
            four = constant(4)
            return_([])
        with x.false as (b3, _):
            five = constant(5)

    foo1.emit()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    %[[VAL_3:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_2]] : i32
    # CHECK:    cf.cond_br %[[VAL_3]], ^bb2, ^bb3
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  ^bb3:
    # CHECK:    %[[VAL_5:.*]] = arith.constant 5 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_bbs_cond_br_operands(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            three = constant(3)
            cond = two < three
            x = cond_br(
                cond, true_dest_operands=[two, three], false_dest_operands=[two, three]
            )
        with x.true as (b2, _):
            four = constant(4)
            return_([])
        with x.false as (b3, _):
            five = constant(5)

    foo1.emit()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    %[[VAL_3:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_2]] : i32
    # CHECK:    cf.cond_br %[[VAL_3]], ^bb2(%[[VAL_1]], %[[VAL_2]] : i32, i32), ^bb3(%[[VAL_1]], %[[VAL_2]] : i32, i32)
    # CHECK:  ^bb2(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
    # CHECK:    %[[VAL_6:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  ^bb3(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
    # CHECK:    %[[VAL_9:.*]] = arith.constant 5 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


@func(emit=False, sym_visibility="private")
def matmul_i16_i16(
    A: "T.memref(64, 32, T.i16())",
    B: "T.memref(32, 64, T.i16())",
    C: "T.memref(64, 64, T.i16())",
):
    linalg.matmul(A, B, C)


def test_defer_emit_1(ctx: MLIRContext):
    matmul_i16_i16.emit(decl=True)

    @module
    def mod():
        matmul_i16_i16.emit(force=True)

    # CHECK:  func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    # CHECK:  module {
    # CHECK:    func.func private @matmul_i16_i16(%[[VAL_0:.*]]: memref<64x32xi16>, %[[VAL_1:.*]]: memref<32x64xi16>, %[[VAL_2:.*]]: memref<64x64xi16>) {
    # CHECK:      linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<64x32xi16>, memref<32x64xi16>) outs(%[[VAL_2]] : memref<64x64xi16>)
    # CHECK:      return
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_defer_emit_2(ctx: MLIRContext):
    matmul_i16_i16.emit(force=True)

    @module
    def mod():
        matmul_i16_i16.emit(decl=True)

    # CHECK:  func.func private @matmul_i16_i16(%[[VAL_0:.*]]: memref<64x32xi16>, %[[VAL_1:.*]]: memref<32x64xi16>, %[[VAL_2:.*]]: memref<64x64xi16>) {
    # CHECK:    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<64x32xi16>, memref<32x64xi16>) outs(%[[VAL_2]] : memref<64x64xi16>)
    # CHECK:    return
    # CHECK:  }
    # CHECK:  module {
    # CHECK:    func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    # CHECK:  }

    filecheck_with_comments(ctx.module)


M, K, N = 64, 32, 64


@func(emit=False, sym_visibility="private")
def matmul_i16_i16(
    A: "T.memref(M, K, T.i16())",
    B: "T.memref(K, N, T.i16())",
    C: "T.memref(M, N, T.i16())",
):
    linalg.matmul(A, B, C)


def test_defer_emit_3(ctx: MLIRContext):
    matmul_i16_i16.emit(force=True)

    @module
    def mod():
        matmul_i16_i16.emit(decl=True)

    # CHECK:  func.func private @matmul_i16_i16(%[[VAL_0:.*]]: memref<64x32xi16>, %[[VAL_1:.*]]: memref<32x64xi16>, %[[VAL_2:.*]]: memref<64x64xi16>) {
    # CHECK:    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<64x32xi16>, memref<32x64xi16>) outs(%[[VAL_2]] : memref<64x64xi16>)
    # CHECK:    return
    # CHECK:  }
    # CHECK:  module {
    # CHECK:    func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_successor_ctx_manager(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([])
        with bb() as (b1, _):
            two = constant(2)
            three = constant(3)
            cond = two < three
            x = cond_br(cond)
        with x.true as (b2, _):
            four = constant(4)
            return_([])
        with x.false as (b3, _):
            five = constant(5)

    foo1()

    # CHECK:  func.func @foo1() {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  ^bb1:
    # CHECK:    %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:    %[[VAL_2:.*]] = arith.constant 3 : i32
    # CHECK:    %[[VAL_3:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_2]] : i32
    # CHECK:    cf.cond_br %[[VAL_3]], ^bb2, ^bb3
    # CHECK:  ^bb2:
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4 : i32
    # CHECK:    return
    # CHECK:  ^bb3:
    # CHECK:    %[[VAL_5:.*]] = arith.constant 5 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)
