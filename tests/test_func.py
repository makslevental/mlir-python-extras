import inspect
from textwrap import dedent

import pytest
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_emit(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()
    correct = dedent(
        """\
    module {
      func.func @demo_fun1() -> i64 {
        %c1_i64 = arith.constant 1 : i64
        return %c1_i64 : i64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    # print("wrapped foo", foo1)
    foo1.emit()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i64 {
        %c1_i64 = arith.constant 1 : i64
        return %c1_i64 : i64
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i64 {
        %c1_i64 = arith.constant 1 : i64
        return %c1_i64 : i64
      }
      %0 = func.call @foo1() : () -> i64
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    print()

    @func
    def foo1():
        one = constant(1)
        return one

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i64 {
        %c1_i64 = arith.constant 1 : i64
        return %c1_i64 : i64
      }
      %0 = func.call @foo1() : () -> i64
    }
    """
    )
    filecheck(correct, ctx.module)
