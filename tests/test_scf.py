from textwrap import dedent

import pytest

import mlir_utils.types as T
from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.scf import (
    for_,
    range_,
    yield_,
    canonicalizer,
    if_,
    else_,
    if_ctx_manager,
    else_ctx_manager,
)
from mlir_utils.dialects.memref import alloca_scope, return_

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
    assert repr(forfoo) == "[Scalar(%0#0, f64), Scalar(%0#1, f64)]"
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


def test_if_region_op_no_results_single_branch(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    @if_(one < two, results=[])
    def iffoo():
        three = constant(3.0)
        return

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
        %cst_1 = arith.constant 3.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_region_op_no_results_else_branch(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(2.0)

    @if_(one < two, results=[])
    def iffoo():
        three = constant(3.0)

    @else_(iffoo)
    def iffoo_else():
        four = constant(4.0)

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
        %cst_1 = arith.constant 4.000000e+00 : f64
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
        res1, res2 = yield_(three, four)
    assert _i == 1

    assert isinstance(res1, Scalar) and repr(res1) == "Scalar(%0#0, f64)"
    assert isinstance(res2, Scalar) and repr(res2) == "Scalar(%0#1, f64)"

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


def test_scf_canonicalizer_with_implicit_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def foo():
        _i = 0
        for i in range_(0, 10):
            _i += 1
            assert isinstance(i, Scalar) and repr(i) == "Scalar(%arg0, index)"
            three = constant(3.0)
        assert _i == 1

    foo()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %cst = arith.constant 3.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_scf_canonicalizer_with_explicit_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, i1 in range_(0, 10, iter_args=[one]):
            _i += 1
            assert isinstance(i, Scalar) and repr(i) == "Scalar(%arg0, index)"
            assert isinstance(i1, Scalar) and repr(i1) == "Scalar(%arg1, f64)"
            three = constant(3.0)
            res = yield three
        assert _i == 1

        assert isinstance(res, Scalar) and repr(res) == "Scalar(%0, f64)"

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
    @canonicalize(using=canonicalizer)
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
            res1, res2 = yield three, four
        assert _i == 1

        assert isinstance(res1, Scalar) and repr(res1) == "Scalar(%0#0, f64)"
        assert isinstance(res2, Scalar) and repr(res2) == "Scalar(%0#1, f64)"

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


def test_if_ctx_manager(ctx: MLIRContext):
    # fmt: off
    # @formatter:off
    one = constant(1.0)
    two = constant(2.0)
    with if_ctx_manager(one < two, results=[T._placeholder_opaque_t()]) as if_op:  # if
        three = constant(3.0)
        res = yield_(three)
    with else_ctx_manager(if_op) as _:  # else
        with if_ctx_manager(one < two, results=[T._placeholder_opaque_t()]) as if_op:  # if
            three = constant(4.0)
            res = yield_(three)
        with else_ctx_manager(if_op) as _:  # else
            three = constant(5.0)
            res = yield_(three)
        res = yield_(res)
    # fmt: on
    # @formatter:on

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
        %2 = arith.cmpf olt, %cst, %cst_0 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_1 = arith.constant 4.000000e+00 : f64
          scf.yield %cst_1 : f64
        } else {
          %cst_1 = arith.constant 5.000000e+00 : f64
          scf.yield %cst_1 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_yield_2(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
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


def test_if_explicit_yield_with_for(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
                five = constant(5.0)
        print("hello")
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %cst_3 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_explicit_yield_with_for_in_else(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        else:
            for i in range_(0, 10):
                four = constant(4.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 4.000000e+00 : f64
          %cst_2 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_explicit_yield_with_for_in_both(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
                five = constant(5.0)
        else:
            for i in range_(0, 10):
                four = constant(6.0)
                five = constant(7.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %cst_3 = arith.constant 5.000000e+00 : f64
        }
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 6.000000e+00 : f64
          %cst_2 = arith.constant 7.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_replace_yield_3(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        else:
            four = constant(4.0)
            res = yield four
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        else:
            four = constant(4.0)
            res1, res2 = yield four, four
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2, res3 = yield three, three, three
        else:
            four = constant(4.0)
            res1, res2, res3 = yield four, four, four
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        else:
            four = constant(4.0)
            res = yield four

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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        else:
            four = constant(4.0)
            res1, res2 = yield four, four
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2, res3 = yield three, three, three
        else:
            four = constant(4.0)
            res1, res2, res3 = yield four, four, four
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
    @canonicalize(using=canonicalizer)
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
    @canonicalize(using=canonicalizer)
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            res = yield three
        else:
            five = constant(5.0)
            res = yield five
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            res1, res2 = yield three, three
        else:
            five = constant(5.0)
            res1, res2 = yield five, five
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


def test_if_with_else_else_with_yields_explicit2(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
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


def test_if_with_else_else_with_yields_explicit2_first_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if one < two:
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)
        else:
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
        %1 = arith.cmpf olt, %cst, %cst_0 : f64
        scf.if %1 {
          %cst_1 = arith.constant 4.000000e+00 : f64
        } else {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      } else {
        %cst_1 = arith.constant 3.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_else_else_with_no_yields(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
        else:
            if two < three:
                five = constant(5.0)
            else:
                six = constant(6.0)

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
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
        else:
            if two < three:
                five = constant(5.0)
            else:
                if two < three:
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


def test_if_with_else_nested_elif_first_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            if two < three:
                six = constant(6.0)
            else:
                if three < four:
                    seven = constant(7.0)
                else:
                    eight = constant(8.0)
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
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      scf.if %0 {
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
      } else {
        %cst_3 = arith.constant 5.000000e+00 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_results_long(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            res = yield four
        else:
            if two < three:
                five = constant(5.0)
                res1 = yield five
            else:
                if two < three:
                    six = constant(6.0)
                    res2 = yield six
                else:
                    if two < three:
                        seven = constant(7.0)
                        res3 = yield seven
                    else:
                        if two < three:
                            eight = constant(8.0)
                            res4 = yield eight
                        else:
                            if two < three:
                                nine = constant(9.0)
                                res5 = yield nine
                            else:
                                ten = constant(10.0)
                                res6 = yield ten
                            res = yield res6
                        res = yield res
                    res = yield res
                res = yield res
            res = yield res

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
          %4 = arith.cmpf olt, %cst_0, %cst_1 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_2 = arith.constant 6.000000e+00 : f64
            scf.yield %cst_2 : f64
          } else {
            %6 = arith.cmpf olt, %cst_0, %cst_1 : f64
            %7 = scf.if %6 -> (f64) {
              %cst_2 = arith.constant 7.000000e+00 : f64
              scf.yield %cst_2 : f64
            } else {
              %8 = arith.cmpf olt, %cst_0, %cst_1 : f64
              %9 = scf.if %8 -> (f64) {
                %cst_2 = arith.constant 8.000000e+00 : f64
                scf.yield %cst_2 : f64
              } else {
                %10 = arith.cmpf olt, %cst_0, %cst_1 : f64
                %11 = scf.if %10 -> (f64) {
                  %cst_2 = arith.constant 9.000000e+00 : f64
                  scf.yield %cst_2 : f64
                } else {
                  %cst_2 = arith.constant 1.000000e+01 : f64
                  scf.yield %cst_2 : f64
                }
                scf.yield %11 : f64
              }
              scf.yield %9 : f64
            }
            scf.yield %7 : f64
          }
          scf.yield %5 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_yields_results(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            res = yield four
        else:
            if two < three:
                five = constant(5.0)
                res1 = yield five
            else:
                six = constant(6.0)
                res2 = yield six
            res = yield res2

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


def test_if_with_elif_yields_results_nested(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            if two < three:
                six = constant(6.0)
                res1 = yield six
            else:
                eight = constant(8.0)
                res2 = yield eight
            res3 = yield res1
        else:
            if two < three:
                five = constant(5.0)
                res4 = yield five
            else:
                six = constant(6.0)
                res5 = yield six
            res5 = yield res5

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
      %1 = scf.if %0 -> (f64) {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3 : f64
        } else {
          %cst_3 = arith.constant 8.000000e+00 : f64
          scf.yield %cst_3 : f64
        }
        scf.yield %3 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_3 = arith.constant 5.000000e+00 : f64
          scf.yield %cst_3 : f64
        } else {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_yields_results_nested_second(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            res4 = yield five
        else:
            if two < three:
                if two < three:
                    six = constant(6.0)
                    res1 = yield six
                else:
                    eight = constant(8.0)
                    res2 = yield eight
                res3 = yield res1
            else:
                six = constant(6.0)
                res5 = yield six
            res5 = yield res5

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
      %1 = scf.if %0 -> (f64) {
        %cst_3 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_3 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %4 = arith.cmpf olt, %cst_0, %cst_1 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_3 = arith.constant 6.000000e+00 : f64
            scf.yield %cst_3 : f64
          } else {
            %cst_3 = arith.constant 8.000000e+00 : f64
            scf.yield %cst_3 : f64
          }
          scf.yield %5 : f64
        } else {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_yields_results_nested_last(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            res4 = yield five
        else:
            if two < three:
                six = constant(6.0)
                res5 = yield six
            else:
                if two < three:
                    six = constant(6.0)
                    res1 = yield six
                else:
                    eight = constant(8.0)
                    res2 = yield eight
                res3 = yield res1
            res3 = yield res3

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
      %1 = scf.if %0 -> (f64) {
        %cst_3 = arith.constant 5.000000e+00 : f64
        scf.yield %cst_3 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_3 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_3 : f64
        } else {
          %4 = arith.cmpf olt, %cst_0, %cst_1 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_3 = arith.constant 6.000000e+00 : f64
            scf.yield %cst_3 : f64
          } else {
            %cst_3 = arith.constant 8.000000e+00 : f64
            scf.yield %cst_3 : f64
          }
          scf.yield %5 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_elif_elif_yields_results(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            res1, res2 = yield five, five
        else:
            if two < three:
                six = constant(6.0)
                res3, res4 = yield six, six
            else:
                if three < four:
                    seven = constant(7.0)
                    res5, res6 = yield seven, seven
                else:
                    eight = constant(8.0)
                    res7, res8 = yield eight, eight
                res7, res8 = yield res7, res8
            res7, res8 = yield res7, res8

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


def test_with_for_if_replace_yield_2_first_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
            yield
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_second_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        else:
            for i in range_(0, 10):
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
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 4.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_both_branches(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
        else:
            for i in range_(0, 10):
                four = constant(5.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
        }
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_both_branches_one_nested_if(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
                if one < two:
                    three = constant(5.0)
        else:
            for i in range_(0, 10):
                four = constant(5.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %1 = arith.cmpf olt, %cst, %cst_0 : f64
          scf.if %1 {
            %cst_3 = arith.constant 5.000000e+00 : f64
          }
        }
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_both_branches_two_nested_if(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
                if one < two:
                    three = constant(5.0)
                if one < two:
                    three = constant(5.0)
        else:
            for i in range_(0, 10):
                four = constant(5.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %1 = arith.cmpf olt, %cst, %cst_0 : f64
          scf.if %1 {
            %cst_3 = arith.constant 5.000000e+00 : f64
          }
          %2 = arith.cmpf olt, %cst, %cst_0 : f64
          scf.if %2 {
            %cst_3 = arith.constant 5.000000e+00 : f64
          }
        }
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_both_branches_nested_else_if(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i in range_(0, 10):
                four = constant(4.0)
                if one < two:
                    three = constant(5.0)
                else:
                    three = constant(6.0)
        else:
            for i in range_(0, 10):
                four = constant(5.0)
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %1 = arith.cmpf olt, %cst, %cst_0 : f64
          scf.if %1 {
            %cst_3 = arith.constant 5.000000e+00 : f64
          } else {
            %cst_3 = arith.constant 6.000000e+00 : f64
          }
        }
      } else {
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c10 step %c1 {
          %cst_1 = arith.constant 5.000000e+00 : f64
        }
      }
    }

    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_first_branch_with_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i, it in range_(0, 10, iter_args=[three]):
                four = constant(4.0)
                res = yield four
            res = yield res
        else:
            four = constant(4.0)
            res = yield four

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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        %2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst_1) -> (f64) {
          %cst_2 = arith.constant 4.000000e+00 : f64
          scf.yield %cst_2 : f64
        }
        scf.yield %2 : f64
      } else {
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_with_for_if_replace_yield_2_both_branches_one_nested_if_with_yield(
    ctx: MLIRContext,
):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i, it in range_(0, 10, iter_args=[three]):
                four = constant(4.0)
                if one < two:
                    five = constant(5.0)
                    res = yield five
                else:
                    six = constant(6.0)
                    res = yield six
                res = yield res
            res = yield res
        else:
            seven = constant(7.0)
            for i, it in range_(0, 10, iter_args=[seven]):
                res = yield it
            res = yield res
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
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        %2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst_1) -> (f64) {
          %cst_2 = arith.constant 4.000000e+00 : f64
          %3 = arith.cmpf olt, %cst, %cst_0 : f64
          %4 = scf.if %3 -> (f64) {
            %cst_3 = arith.constant 5.000000e+00 : f64
            scf.yield %cst_3 : f64
          } else {
            %cst_3 = arith.constant 6.000000e+00 : f64
            scf.yield %cst_3 : f64
          }
          scf.yield %4 : f64
        }
        scf.yield %2 : f64
      } else {
        %cst_1 = arith.constant 7.000000e+00 : f64
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        %2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst_1) -> (f64) {
          scf.yield %arg1 : f64
        }
        scf.yield %2 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_if_with_nested_region(
    ctx: MLIRContext,
):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:

            @alloca_scope([one.type])
            def demo_scope1():
                one = constant(1.0)
                return_([one])

            res = yield demo_scope1
        else:
            seven = constant(7.0)
            yield seven
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
        %2 = memref.alloca_scope  -> (f64) {
          %cst_2 = arith.constant 1.000000e+00 : f64
          memref.alloca_scope.return %cst_2 : f64
        }
        scf.yield %2 : f64
      } else {
        %cst_1 = arith.constant 7.000000e+00 : f64
        scf.yield %cst_1 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_elif_1(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            four = constant(4.0)
            res2 = yield four
        elif one < two:
            five = constant(5.0)
            res3 = yield five
        else:
            six = constant(6.0)
            res4 = yield six

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
        %cst_1 = arith.constant 4.000000e+00 : f64
        scf.yield %cst_1 : f64
      } else {
        %2 = arith.cmpf olt, %cst, %cst_0 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_1 = arith.constant 5.000000e+00 : f64
          scf.yield %cst_1 : f64
        } else {
          %cst_1 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_1 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_elif_nested_first_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            if three < four:
                seven = constant(7.0)
                res2 = yield seven
            elif four < five:
                eight = constant(8.0)
                res3 = yield eight
            else:
                nine = constant(9.0)
                res4 = yield nine
        elif two < three:
            six = constant(6.0)
            res1 = yield six
        else:
            ten = constant(10.0)
            res5 = yield ten

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
      %cst_3 = arith.constant 5.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %2 = arith.cmpf olt, %cst_1, %cst_2 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_4 = arith.constant 7.000000e+00 : f64
          scf.yield %cst_4 : f64
        } else {
          %4 = arith.cmpf olt, %cst_2, %cst_3 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_4 = arith.constant 8.000000e+00 : f64
            scf.yield %cst_4 : f64
          } else {
            %cst_4 = arith.constant 9.000000e+00 : f64
            scf.yield %cst_4 : f64
          }
          scf.yield %5 : f64
        }
        scf.yield %3 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_4 = arith.constant 6.000000e+00 : f64
          scf.yield %cst_4 : f64
        } else {
          %cst_4 = arith.constant 1.000000e+01 : f64
          scf.yield %cst_4 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_elif_nested_second_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)
            res1 = yield six
        elif two < three:
            if three < four:
                seven = constant(7.0)
                res2 = yield seven
            elif four < five:
                eight = constant(8.0)
                res3 = yield eight
            else:
                nine = constant(9.0)
                res4 = yield nine
        else:
            ten = constant(10.0)
            res5 = yield ten

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
      %cst_3 = arith.constant 5.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_4 = arith.constant 6.000000e+00 : f64
        scf.yield %cst_4 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %4 = arith.cmpf olt, %cst_1, %cst_2 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_4 = arith.constant 7.000000e+00 : f64
            scf.yield %cst_4 : f64
          } else {
            %6 = arith.cmpf olt, %cst_2, %cst_3 : f64
            %7 = scf.if %6 -> (f64) {
              %cst_4 = arith.constant 8.000000e+00 : f64
              scf.yield %cst_4 : f64
            } else {
              %cst_4 = arith.constant 9.000000e+00 : f64
              scf.yield %cst_4 : f64
            }
            scf.yield %7 : f64
          }
          scf.yield %5 : f64
        } else {
          %cst_4 = arith.constant 1.000000e+01 : f64
          scf.yield %cst_4 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_elif_nested_else_branch(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)
            res1 = yield six
        elif two < three:
            ten = constant(10.0)
            res5 = yield ten
        else:
            if three < four:
                seven = constant(7.0)
                res2 = yield seven
            elif four < five:
                eight = constant(8.0)
                res3 = yield eight
            else:
                nine = constant(9.0)
                res4 = yield nine

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
      %cst_3 = arith.constant 5.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1 = scf.if %0 -> (f64) {
        %cst_4 = arith.constant 6.000000e+00 : f64
        scf.yield %cst_4 : f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3 = scf.if %2 -> (f64) {
          %cst_4 = arith.constant 1.000000e+01 : f64
          scf.yield %cst_4 : f64
        } else {
          %4 = arith.cmpf olt, %cst_1, %cst_2 : f64
          %5 = scf.if %4 -> (f64) {
            %cst_4 = arith.constant 7.000000e+00 : f64
            scf.yield %cst_4 : f64
          } else {
            %6 = arith.cmpf olt, %cst_2, %cst_3 : f64
            %7 = scf.if %6 -> (f64) {
              %cst_4 = arith.constant 8.000000e+00 : f64
              scf.yield %cst_4 : f64
            } else {
              %cst_4 = arith.constant 9.000000e+00 : f64
              scf.yield %cst_4 : f64
            }
            scf.yield %7 : f64
          }
          scf.yield %5 : f64
        }
        scf.yield %3 : f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_elif_nested_else_branch_multiple_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)
            res1, res2 = yield six, six
        elif two < three:
            ten = constant(10.0)
            res3, res4 = yield ten, ten
        else:
            if three < four:
                seven = constant(7.0)
                res5, res6 = yield seven, seven
            elif four < five:
                eight = constant(8.0)
                res7, res8 = yield eight, eight
            else:
                nine = constant(9.0)
                res9, res10 = yield nine, nine

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
      %cst_3 = arith.constant 5.000000e+00 : f64
      %0 = arith.cmpf olt, %cst, %cst_0 : f64
      %1:2 = scf.if %0 -> (f64, f64) {
        %cst_4 = arith.constant 6.000000e+00 : f64
        scf.yield %cst_4, %cst_4 : f64, f64
      } else {
        %2 = arith.cmpf olt, %cst_0, %cst_1 : f64
        %3:2 = scf.if %2 -> (f64, f64) {
          %cst_4 = arith.constant 1.000000e+01 : f64
          scf.yield %cst_4, %cst_4 : f64, f64
        } else {
          %4 = arith.cmpf olt, %cst_1, %cst_2 : f64
          %5:2 = scf.if %4 -> (f64, f64) {
            %cst_4 = arith.constant 7.000000e+00 : f64
            scf.yield %cst_4, %cst_4 : f64, f64
          } else {
            %6 = arith.cmpf olt, %cst_2, %cst_3 : f64
            %7:2 = scf.if %6 -> (f64, f64) {
              %cst_4 = arith.constant 8.000000e+00 : f64
              scf.yield %cst_4, %cst_4 : f64, f64
            } else {
              %cst_4 = arith.constant 9.000000e+00 : f64
              scf.yield %cst_4, %cst_4 : f64, f64
            }
            scf.yield %7#0, %7#1 : f64, f64
          }
          scf.yield %5#0, %5#1 : f64, f64
        }
        scf.yield %3#0, %3#1 : f64, f64
      }
    }
    """
    )
    filecheck(correct, ctx.module)
