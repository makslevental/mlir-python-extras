from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.dialects.func import return_
from mlir.dialects.memref import alloca_scope, alloca_scope_return
from mlir.dialects.scf import yield_ as scf_yield
from mlir.dialects.tensor import rank, yield_ as tensor_yield
from mlir.dialects.builtin import module
from mlir.extras.dialects.ext import linalg, memref
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.cf import br, cond_br
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.memref import alloca_scope
from mlir.extras.dialects.ext.scf import execute_region
from mlir.extras.dialects.ext.tensor import S, generate
from mlir.extras.util import bb

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir.extras.types import tensor

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_simple_region_op(ctx: MLIRContext):
    @execute_region([])
    def demo_region():
        one = constant(1)
        scf_yield([])

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      scf.execute_region {
        %c1_i32 = arith.constant 1 : i32
        scf.yield
      }
    }
    """
        ),
        ctx.module,
    )


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
    filecheck(
        dedent(
            """\
    module {
      memref.alloca_scope  {
        %c1_i32 = arith.constant 1 : i32
      }
      memref.alloca_scope  {
        %c2_i32 = arith.constant 2 : i32
      }
    }
    """
        ),
        ctx.module,
    )


def test_func(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    demo_fun1()
    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      func.func @demo_fun1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @demo_fun1() : () -> i32
    }
    """
        ),
        ctx.module,
    )


def test_block_args(ctx: MLIRContext):
    one = constant(1, T.index())
    two = constant(2, T.index())

    @generate(tensor(S, 3, S, T.f32()), dynamic_extents=[one, two])
    def demo_fun1(i: T.index(), j: T.index(), k: T.index()):
        one = constant(1.0)
        tensor_yield(one)

    r = rank(demo_fun1)

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %generated = tensor.generate %c1, %c2 {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %cst = arith.constant 1.000000e+00 : f32
        tensor.yield %cst : f32
      } : tensor<?x3x?xf32>
      %rank = tensor.rank %generated : tensor<?x3x?xf32>
    }
    """
        ),
        ctx.module,
    )


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
    filecheck(
        dedent(
            """\
    module {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      func.func @demo_fun1() {
        %alloc = memref.alloc() : memref<10x10xf32>
        %alloc_0 = memref.alloc() : memref<10x10xf32>
        linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc : memref<10x10xf32>) outs(%alloc_0 : memref<10x10xf32>)
        return
      }
    }
    """
        ),
        ctx.module,
    )


def test_bbs(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return_([one])
        with bb():
            two = constant(2)
            return one

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        return %c2_i32 : i32
      ^bb2:  // no predecessors
        %c3_i32 = arith.constant 3 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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
    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // pred: ^bb2
        %c2_i32 = arith.constant 2 : i32
        return
      ^bb2:  // no predecessors
        %c3_i32 = arith.constant 3 : i32
        cf.br ^bb1
      ^bb3:  // no predecessors
        %c4_i32 = arith.constant 4 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // pred: ^bb2
        %c2_i32 = arith.constant 2 : i32
        cf.br ^bb2
      ^bb2:  // pred: ^bb1
        %c3_i32 = arith.constant 3 : i32
        cf.br ^bb1
      ^bb3:  // no predecessors
        %c4_i32 = arith.constant 4 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // pred: ^bb2
        %c2_i32 = arith.constant 2 : i32
        cf.br ^bb2(%c2_i32 : i32)
      ^bb2(%0: i32):  // pred: ^bb1
        %c3_i32 = arith.constant 3 : i32
        %1 = arith.addi %c3_i32, %0 : i32
        cf.br ^bb1
      ^bb3:  // no predecessors
        %c4_i32 = arith.constant 4 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        cf.br ^bb3(%c2_i32 : i32)
      ^bb2:  // no predecessors
        %c4_i32 = arith.constant 4 : i32
        cf.br ^bb3(%c4_i32 : i32)
      ^bb3(%0: i32):  // 2 preds: ^bb1, ^bb2
        %c3_i32 = arith.constant 3 : i32
        %1 = arith.addi %c3_i32, %0 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %0 = arith.cmpi slt, %c2_i32, %c3_i32 : i32
        cf.cond_br %0, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %c4_i32 = arith.constant 4 : i32
        return
      ^bb3:  // pred: ^bb1
        %c5_i32_0 = arith.constant 5 : i32
        return
      }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %0 = arith.cmpi slt, %c2_i32, %c3_i32 : i32
        cf.cond_br %0, ^bb2(%c2_i32, %c3_i32 : i32, i32), ^bb3(%c2_i32, %c3_i32 : i32, i32)
      ^bb2(%1: i32, %2: i32):  // pred: ^bb1
        %c4_i32 = arith.constant 4 : i32
        return
      ^bb3(%3: i32, %4: i32):  // pred: ^bb1
        %c5_i32 = arith.constant 5 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
      module {
        func.func private @matmul_i16_i16(%arg0: memref<64x32xi16>, %arg1: memref<32x64xi16>, %arg2: memref<64x64xi16>) {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<64x32xi16>, memref<32x64xi16>) outs(%arg2 : memref<64x64xi16>)
          return
        }
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_defer_emit_2(ctx: MLIRContext):

    matmul_i16_i16.emit(force=True)

    @module
    def mod():
        matmul_i16_i16.emit(decl=True)

    correct = dedent(
        """\
    module {
      func.func private @matmul_i16_i16(%arg0: memref<64x32xi16>, %arg1: memref<32x64xi16>, %arg2: memref<64x64xi16>) {
        linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<64x32xi16>, memref<32x64xi16>) outs(%arg2 : memref<64x64xi16>)
        return
      }
      module {
        func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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

    correct = dedent(
        """\
    module {
      func.func private @matmul_i16_i16(%arg0: memref<64x32xi16>, %arg1: memref<32x64xi16>, %arg2: memref<64x64xi16>) {
        linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<64x32xi16>, memref<32x64xi16>) outs(%arg2 : memref<64x64xi16>)
        return
      }
      module {
        func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
      }
    }
    """
    )
    filecheck(correct, ctx.module)


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
    correct = dedent(
        """\
    module {
      func.func @foo1() {
        %c1_i32 = arith.constant 1 : i32
        return
      ^bb1:  // no predecessors
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %0 = arith.cmpi slt, %c2_i32, %c3_i32 : i32
        cf.cond_br %0, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %c4_i32 = arith.constant 4 : i32
        return
      ^bb3:  // pred: ^bb1
        %c5_i32_0 = arith.constant 5 : i32
        return
      }
    """
    )
    filecheck(correct, ctx.module)
