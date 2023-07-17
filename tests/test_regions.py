from textwrap import dedent

import pytest

from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.memref import alloca_scope, return_
from mlir_utils.dialects.scf import execute_region, yield_
from mlir_utils.dialects.util import constant

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_simple_region_op(ctx: MLIRContext):
    @execute_region([])
    def demo_region():
        one = constant(1)
        yield_()

    demo_region()

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

    demo_scope1()
    demo_scope2()

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
        return

    demo_fun1()
    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      func.func @demo_fun1() {
        %c1_i64 = arith.constant 1 : i64
        return
      }
    }
    """
        ),
        ctx.module,
    )
