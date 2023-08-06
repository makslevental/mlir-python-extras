from textwrap import dedent

import pytest

import mlir_utils.types as T
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.ext.tensor import S, rank
from mlir_utils.dialects.memref import alloca_scope, return_
from mlir_utils.dialects.scf import execute_region, yield_ as scf_yield
from mlir_utils.dialects.tensor import generate, yield_ as tensor_yield

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import tensor_t

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
        %c1_i64 = arith.constant 1 : i64
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
        return_()

    @alloca_scope
    def demo_scope2():
        one = constant(2)
        return_()

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      memref.alloca_scope  {
        %c1_i64 = arith.constant 1 : i64
      }
      memref.alloca_scope  {
        %c2_i64 = arith.constant 2 : i64
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
      func.func @demo_fun1() -> i64 {
        %c1_i64 = arith.constant 1 : i64
        return %c1_i64 : i64
      }
      %0 = func.call @demo_fun1() : () -> i64
    }
    """
        ),
        ctx.module,
    )


def test_block_args(ctx: MLIRContext):
    one = constant(1, T.index_t)
    two = constant(2, T.index_t)

    @generate(tensor_t(S, 3, S, T.f64_t), dynamic_extents=[one, two])
    def demo_fun1(i: T.index_t, j: T.index_t, k: T.index_t):
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
        %cst = arith.constant 1.000000e+00 : f64
        tensor.yield %cst : f64
      } : tensor<?x3x?xf64>
      %rank = tensor.rank %generated : tensor<?x3x?xf64>
    }
    """
        ),
        ctx.module,
    )
