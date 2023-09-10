from textwrap import dedent

import pytest
from mlir.dialects._linalg_ops_gen import CopyOp

import mlir.utils.types as T

# this has to be above the next one
from mlir.utils.dialects.ext import linalg
from mlir.utils.dialects import linalg

from mlir.utils.dialects.ext import memref
from mlir.utils.dialects.ext.arith import constant
from mlir.utils.dialects.ext.func import func
from mlir.utils.dialects.ext.tensor import S
from mlir.utils.dialects.memref import alloca_scope, alloca_scope_return
from mlir.utils.dialects.scf import execute_region, yield_ as scf_yield
from mlir.utils.dialects.tensor import generate, yield_ as tensor_yield
from mlir.utils.dialects.tensor import rank

# noinspection PyUnresolvedReferences
from mlir.utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir.utils.types import tensor

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_simple_region_op(ctx: MLIRContext):
    @execute_region([])
    def demo_region():
        one = constant(1)
        scf_yield()

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
        alloca_scope_return()

    @alloca_scope
    def demo_scope2():
        one = constant(2)
        alloca_scope_return()

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
    one = constant(1, T.index)
    two = constant(2, T.index)

    @generate(tensor(S, 3, S, T.f32), dynamic_extents=[one, two])
    def demo_fun1(i: T.index, j: T.index, k: T.index):
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
    one = constant(1, T.index)
    two = constant(2, T.index)

    @func
    def demo_fun1():
        mem1 = memref.alloc((10, 10), T.f32)
        mem2 = memref.alloc((10, 10), T.f32)
        x = linalg.copy(mem1, mem2)
        assert isinstance(x, CopyOp)

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
