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
    assert inspect.isfunction(demo_fun1.emit)
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
