from textwrap import dedent

import pytest

from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.scf import for_, range_, yield_

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_for_simple(ctx: MLIRContext):
    @for_(1, 2, 3)
    def forfoo(i):
        one = constant(1.0)
        return

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      scf.for %arg0 = %c1 to %c2 step %c3 {
        %cst = arith.constant 1.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_for_iter_args(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    @for_(1, 2, 3, iter_args=[one, two])
    def forfoo(i, *iter_args):
        assert isinstance(i, Scalar)
        assert repr(i) == "Scalar(%arg0, index)"
        assert len(iter_args) == 2 and all(isinstance(i, Scalar) for i in iter_args)
        assert repr(iter_args) == "(Scalar(%arg1, f64), Scalar(%arg2, f64))"
        one = constant(1.0)
        return one, one

    assert len(forfoo) == 2 and all(isinstance(i, Scalar) for i in forfoo)
    assert repr(forfoo) == "(Scalar(%0#0, f64), Scalar(%0#1, f64))"
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %0:2 = scf.for %arg0 = %c1 to %c2 step %c3 iter_args(%arg1 = %cst, %arg2 = %cst_0) -> (f64, f64) {
        %cst_1 = arith.constant 1.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_for_bare(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    _i = 0
    for i, (i1, i2) in range_(0, 10, iter_args=[one, two]):
        _i += 1
        assert isinstance(i, Scalar) and repr(i) == "Scalar(%arg0, index)"
        assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%arg1, f64)"
        assert isinstance(i2, Scalar) and repr(i2) == "Scalar(%arg2, f64)"
        three = constant(3.0)
        four = constant(4.0)
        yield_(three, four)
    assert _i == 1

    assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%0#0, f64)"
    assert isinstance(i2, Scalar) and repr(i2) == "Scalar(%0#1, f64)"

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0:2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst, %arg2 = %cst_0) -> (f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        %cst_2 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1, %cst_2 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)
