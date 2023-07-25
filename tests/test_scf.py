from textwrap import dedent

import pytest

from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.scf import (
    for_,
    range_,
    yield_,
    canonicalizer,
    stack_if,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import f64_t

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


def test_scf_canonicalizer(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, i1 in range_(0, 10, iter_args=[one]):
            _i += 1
            assert isinstance(i, Scalar) and repr(i) == "Scalar(%arg0, index)"
            assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%arg1, f64)"
            three = constant(3.0)
            yield three
        assert _i == 1

        assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%0, f64)"

    foo()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst) -> (f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_scf_canonicalizer_tuple(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def foo():
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
            yield three, four
        assert _i == 1

        assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%0#0, f64)"
        assert isinstance(i2, Scalar) and repr(i2) == "Scalar(%0#1, f64)"

    foo()

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


def test_if_replace_yield_2(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_1 = arith.constant 3.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_yield_3(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if res := stack_if(one < two, (f64_t,)):
            three = constant(3.0)
            yield three
        else:
            four = constant(4.0)
            yield four
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1 : f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_yield_4(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if res := stack_if(one < two, (f64_t, f64_t)):
            three = constant(3.0)
            yield three, three
        else:
            four = constant(4.0)
            yield four, four
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_yield_5(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if res := stack_if(one < two, (f64_t, f64_t, f64_t)):
            three = constant(3.0)
            yield three, three, three
        else:
            four = constant(4.0)
            yield four, four, four
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:3 = scf.if %0 -> (f64, f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_cond_2(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            yield three
        else:
            four = constant(4.0)
            yield four

        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1 : f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_cond_3(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            yield three, three
        else:
            four = constant(4.0)
            yield four, four
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_cond_4(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            yield three, three, three
        else:
            four = constant(4.0)
            yield four, four, four
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:3 = scf.if %0 -> (f64, f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_nested_no_else_no_yield(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_1 = arith.constant 3.000000e+00 : f64
        %1 = arith.cmpf olt, %cst, %cst_0 : f64
        scf.if %1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_nested_with_else_no_yield(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_1 = arith.constant 3.000000e+00 : f64
        %1 = arith.cmpf olt, %cst, %cst_0 : f64
        scf.if %1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        } else {
          %cst_2 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_else_with_nested_no_yields_yield_results(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            yield three
        else:
            five = constant(5.0)
            yield five
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        scf.if %2 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        }
        scf.yield %cst_1 : f64
      } else {
        %cst_1 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_else_with_nested_no_yields_yield_multiple_results(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            yield three, three
        else:
            five = constant(5.0)
            yield five, five
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_1 = arith.constant 3.000000e+00 : f64
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        scf.if %2 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        }
        scf.yield %cst_1, %cst_1 : f64, f64
      } else {
        %cst_1 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_1, %cst_1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_else_else_with_yields(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if one < two:
            three = constant(3.0)
        else:
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)

        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_1 = arith.constant 3.000000e+00 : f64
      } else {
        %1 = arith.cmpf olt, %cst, %cst_0 : f64
        scf.if %1 {
          %cst_1 = arith.constant 4.000000e+00 : f64
        } else {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_canonicalize_elif(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield
        else:
            six = constant(6.0)
            yield

        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_2 = arith.constant 4.000000e+00 : f64
      } else {
        %1 = arith.cmpf olt, %cst_0, %cst_1 : f64
        scf.if %1 {
          %cst_2 = arith.constant 5.000000e+00 : f64
        } else {
          %cst_2 = arith.constant 6.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_canonicalize_elif_elif(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
        elif two < three:
            five = constant(5.0)
        elif two < three:
            six = constant(6.0)
        else:
            seven = constant(7.0)

        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_2 = arith.constant 4.000000e+00 : f64
      } else {
        %1 = arith.cmpf olt, %cst_0, %cst_1 : f64
        scf.if %1 {
          %cst_2 = arith.constant 5.000000e+00 : f64
        } else {
          %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
          scf.if %2 {
            %cst_2 = arith.constant 6.000000e+00 : f64
          } else {
            %cst_2 = arith.constant 7.000000e+00 : f64
          }
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_else_nested_elif(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
        else:
            if two < three:
                six = constant(6.0)
            elif three < four:
                seven = constant(7.0)
            else:
                eight = constant(8.0)

        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_3 = arith.constant 5.000000e+00 : f64
      } else {
        %1 = arith.cmpf olt, %cst_0, %cst_1 : f64
        scf.if %1 {
          %cst_3 = arith.constant 6.000000e+00 : f64
        } else {
          %2 = arith.cmpf olt, %cst_1, %cst_2 : f64
          scf.if %2 {
            %cst_3 = arith.constant 7.000000e+00 : f64
          } else {
            %cst_3 = arith.constant 8.000000e+00 : f64
          }
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_yields_results(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        res: f64_t
        res1: f64_t
        if res := one < two:
            four = constant(4.0)
            yield four
        elif res1 := two < three:
            five = constant(5.0)
            yield five
        else:
            six = constant(6.0)
            yield six

        correct = dedent(
            """\
        module
        %1 = scf.if %0 -> (f64) {
          %cst_2 = arith.constant 4.000000e+00 : f64
          scf.yield %cst_2 : f64
        } else {
          %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
          %3 = scf.if %2 -> (f64) {
            %cst_2 = arith.constant 5.000000e+00 : f64
            scf.yield %cst_2 : f64
          } else {
            %cst_2 = arith.constant 6.000000e+00 : f64
            scf.yield %cst_2 : f64
          }
          scf.yield %3 : f64
        }
        """
        )
        filecheck(correct, "module\n" + str(res.owner))
        correct = """\
        module
        %3 = scf.if %2 -> (f64) {
          %cst_2 = arith.constant 5.000000e+00 : f64
          scf.yield %cst_2 : f64
        } else {
          %cst_2 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_2 : f64
        }
        """
        filecheck(correct, "module\n" + str(res1.owner))
        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_2 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_2 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_2 = arith.constant 5.000000e+00 : f64
          scf.yield %cst_2 : f64
        } else {
          %cst_2 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_2 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_elif_yields_results(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        res: (f64_t, f64_t)
        res1: (f64_t, f64_t)
        res2: (f64_t, f64_t)
        if res := one < two:
            five = constant(5.0)
            yield five, five
        elif res1 := two < three:
            six = constant(6.0)
            yield six, six
        elif res2 := three < four:
            seven = constant(7.0)
            yield seven, seven
        else:
            eight = constant(8.0)
            yield eight, eight

        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_3 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_3, %cst_3 : f64, f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3:2 = scf.if %2 -> (f64, f64) {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3, %cst_3 : f64, f64
        } else {
          %4 = arith.cmpf olt, %cst_1, %cst_2 : f64
          %5:2 = scf.if %4 -> (f64, f64) {
            %cst_3 = arith.constant 7.000000e+00 : f64
            scf.yield %cst_3, %cst_3 : f64, f64
          } else {
            %cst_3 = arith.constant 8.000000e+00 : f64
            scf.yield %cst_3, %cst_3 : f64, f64
          }
          scf.yield %5#0, %5#1 : f64, f64
        }
        scf.yield %3#0, %3#1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_elif_explicit_yields_results(ctx: MLIRContext):
    @canonicalize(with_=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if res := stack_if(one < two, (f64_t, f64_t)):
            five = constant(5.0)
            yield five, five
        elif res1 := stack_if(two < three, (f64_t, f64_t)):
            six = constant(6.0)
            yield six, six
        elif res2 := stack_if(three < four, (f64_t, f64_t)):
            seven = constant(7.0)
            yield seven, seven
        else:
            eight = constant(8.0)
            yield eight, eight

        return

    iffoo()
    ctx.module.operation.verify()

    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_3 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_3, %cst_3 : f64, f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3:2 = scf.if %2 -> (f64, f64) {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3, %cst_3 : f64, f64
        } else {
          %4 = arith.cmpf olt, %cst_1, %cst_2 : f64
          %5:2 = scf.if %4 -> (f64, f64) {
            %cst_3 = arith.constant 7.000000e+00 : f64
            scf.yield %cst_3, %cst_3 : f64, f64
          } else {
            %cst_3 = arith.constant 8.000000e+00 : f64
            scf.yield %cst_3, %cst_3 : f64, f64
          }
          scf.yield %5#0, %5#1 : f64, f64
        }
        scf.yield %3#0, %3#1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)
