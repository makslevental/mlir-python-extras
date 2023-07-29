import inspect
from textwrap import dedent

import pytest

from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t

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
      func.func @demo_fun1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_declare_byte_rep(ctx: MLIRContext):
    def demo_fun1():
        ...

    assert demo_fun1.__code__.co_code == b"\x97\x00d\x00S\x00"


def test_declare(ctx: MLIRContext):
    @func
    def demo_fun1() -> i32_t:
        ...

    @func
    def demo_fun2() -> (i32_t, i32_t):
        ...

    @func
    def demo_fun3(x: i32_t) -> (i32_t, i32_t):
        ...

    @func
    def demo_fun4(x: i32_t, y: i32_t) -> (i32_t, i32_t):
        ...

    demo_fun1()
    demo_fun2()
    one = constant(1)
    demo_fun3(one)
    demo_fun4(one, one)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func private @demo_fun1() -> i32
      func.func private @demo_fun2() -> (i32, i32)
      func.func private @demo_fun3(i32) -> (i32, i32)
      func.func private @demo_fun4(i32, i32) -> (i32, i32)
      %0 = func.call @demo_fun1() : () -> i32
      %1:2 = func.call @demo_fun2() : () -> (i32, i32)
      %c1_i32 = arith.constant 1 : i32
      %2:2 = func.call @demo_fun3(%c1_i32) : (i32) -> (i32, i32)
      %3:2 = func.call @demo_fun4(%c1_i32, %c1_i32) : (i32, i32) -> (i32, i32)
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
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
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
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)
