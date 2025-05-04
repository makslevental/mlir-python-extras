from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import scf
from mlir.extras.dialects.ext import tensor
from mlir.extras.dialects.ext.arith import constant, Scalar
from mlir.extras.dialects.ext.scf import (
    for__,
    range_,
    yield_,
    canonicalizer,
    if_,
    else_,
    if_ctx_manager,
    else_ctx_manager,
    forall_,
    parange_,
    forall,
    parange,
    reduce,
    while__,
    while___,
    placeholder_opaque_t,
    another_reduce,
)
from mlir.extras.dialects.ext.tensor import empty, Tensor
from mlir.dialects.memref import alloca_scope_return
from mlir.extras.dialects.ext.memref import alloca_scope

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_for_simple(ctx: MLIRContext):
    @for__(1, 2, 3)
    def forfoo(i):
        one = constant(1.0)
        return

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
    # CHECK:  scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_for_iter_args(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    @for__(1, 2, 3, iter_args=[one, two])
    def forfoo(i, *iter_args):
        assert isinstance(i, Scalar)
        assert repr(i) == "Scalar(<block argument> of type 'index' at index: 0)"
        assert len(iter_args) == 2 and all(isinstance(i, Scalar) for i in iter_args)
        assert (
            repr(iter_args)
            == "(Scalar(<block argument> of type 'f32' at index: 1), Scalar(<block argument> of type 'f32' at index: 2))"
        )
        one = constant(1.0)
        return one, one

    assert len(forfoo) == 2 and all(isinstance(i, Scalar) for i in forfoo)
    assert (str(forfoo[0]), str(forfoo[1])) == (
        "Scalar(%0#0, f32)",
        "Scalar(%0#1, f32)",
    )
    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_5:.*]]:2 = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_1]]) -> (f32, f32) {
    # CHECK:    %[[VAL_9:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_9]], %[[VAL_9]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_if_region_op_no_results_single_branch(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    @if_(one < two, results=[])
    def iffoo():
        three = constant(3.0)
        return

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_for_bare(ctx: MLIRContext):
    one = constant(1.0)
    two = constant(1.0)

    _i = 0
    for i, (i1, i2), _ in range_(0, 10, iter_args=[one, two]):
        _i += 1
        assert (
            isinstance(i, Scalar)
            and repr(i) == "Scalar(<block argument> of type 'index' at index: 0)"
        )
        assert (
            isinstance(i1, Scalar)
            and repr(i1) == "Scalar(<block argument> of type 'f32' at index: 1)"
        )
        assert (
            isinstance(i2, Scalar)
            and repr(i2) == "Scalar(<block argument> of type 'f32' at index: 2)"
        )
        three = constant(3.0)
        four = constant(4.0)
        res1, res2 = yield_(three, four)
    assert _i == 1

    assert isinstance(res1, Scalar) and str(res1) == "Scalar(%0#0, f32)"
    assert isinstance(res2, Scalar) and str(res2) == "Scalar(%0#1, f32)"

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_5:.*]]:2 = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_1]]) -> (f32, f32) {
    # CHECK:    %[[VAL_9:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_10:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_9]], %[[VAL_10]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_mixed_start_stop_for(ctx: MLIRContext):
    ten = constant(10)

    for i in range_(0, ten):
        three = constant(3.0)
        four = constant(4.0)
        yield_()

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 10 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:  scf.for %[[VAL_3:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_2]]  : i32 {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_scf_canonicalizer_with_implicit_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def foo():
        _i = 0
        for i in range_(0, 10):
            _i += 1
            assert isinstance(i, Scalar) and str(i) == "Scalar(%arg0, index)"
            three = constant(3.0)
        assert _i == 1

    foo()

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_scf_canonicalizer_with_explicit_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, i1, _ in range_(0, 10, iter_args=[one]):
            _i += 1
            assert isinstance(i, Scalar) and str(i) == "Scalar(%arg0, index)"
            assert isinstance(i1, Scalar) and str(i1) == "Scalar(%arg1, f32)"
            three = constant(3.0)
            res = yield three
        assert _i == 1

        assert isinstance(res, Scalar) and str(res) == "Scalar(%0, f32)"

    foo()

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_5:.*]] = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_7:.*]] = %[[VAL_0]]) -> (f32) {
    # CHECK:    %[[VAL_8:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_8]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_scf_canonicalizer_tuple(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, (i1, i2), _ in range_(0, 10, iter_args=[one, two]):
            _i += 1
            assert isinstance(i, Scalar) and str(i) == "Scalar(%arg0, index)"
            assert isinstance(i1, Scalar) and str(i1) == "Scalar(%arg1, f32)"
            assert isinstance(i2, Scalar) and str(i2) == "Scalar(%arg2, f32)"
            three = constant(3.0)
            four = constant(4.0)
            res1, res2 = yield three, four
        assert _i == 1

        assert isinstance(res1, Scalar) and str(res1) == "Scalar(%0#0, f32)"
        assert isinstance(res2, Scalar) and str(res2) == "Scalar(%0#1, f32)"

    foo()

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_5:.*]]:2 = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_1]]) -> (f32, f32) {
    # CHECK:    %[[VAL_9:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_10:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_9]], %[[VAL_10]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_if_ctx_manager(ctx: MLIRContext):
    # fmt: off
    # @formatter:off
    one = constant(1.0)
    two = constant(2.0)
    with if_ctx_manager(one < two, results=[placeholder_opaque_t()]) as if_op:  # if
        three = constant(3.0)
        res = yield_(three)
    with else_ctx_manager(if_op) as _:  # else
        with if_ctx_manager(one < two, results=[placeholder_opaque_t()]) as if_op:  # if
            three = constant(4.0)
            res = yield_(three)
        with else_ctx_manager(if_op) as _:  # else
            three = constant(5.0)
            res = yield_(three)
        res = yield_(res)
    # fmt: on
    # @formatter:on

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (f32) {
    # CHECK:      %[[VAL_7:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_7]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_8]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_9:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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
        return

    iffoo()
    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_10:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_11:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_12:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_12]] {
    # CHECK:      %[[VAL_14:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      %[[VAL_15:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (f32, f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]], %[[VAL_4]] : f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]], %[[VAL_5]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]]:3 = scf.if %[[VAL_2]] -> (f32, f32, f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]], %[[VAL_4]], %[[VAL_4]] : f32, f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]], %[[VAL_5]], %[[VAL_5]] : f32, f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (f32, f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]], %[[VAL_4]] : f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]], %[[VAL_5]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]]:3 = scf.if %[[VAL_2]] -> (f32, f32, f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]], %[[VAL_4]], %[[VAL_4]] : f32, f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]], %[[VAL_5]], %[[VAL_5]] : f32, f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    scf.if %[[VAL_4]] {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    scf.if %[[VAL_4]] {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_4]] : f32
    # CHECK:    scf.if %[[VAL_5]] {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_4]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (f32, f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_4]] : f32
    # CHECK:    scf.if %[[VAL_5]] {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_4]], %[[VAL_4]] : f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]], %[[VAL_7]] : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    scf.if %[[VAL_4]] {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    scf.if %[[VAL_3]] {
    # CHECK:      %[[VAL_4:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_6:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    scf.if %[[VAL_4]] {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_3]] {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    scf.if %[[VAL_5]] {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_7:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_3]] {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    scf.if %[[VAL_5]] {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:      scf.if %[[VAL_7]] {
    # CHECK:        %[[VAL_8:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_9:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_4]] {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    scf.if %[[VAL_5]] {
    # CHECK:      %[[VAL_6:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:      scf.if %[[VAL_7]] {
    # CHECK:        %[[VAL_8:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_9:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_10:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (f32) {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (f32) {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_8]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_9:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:      %[[VAL_10:.*]] = scf.if %[[VAL_9]] -> (f32) {
    # CHECK:        %[[VAL_11:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_11]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_12:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:        %[[VAL_13:.*]] = scf.if %[[VAL_12]] -> (f32) {
    # CHECK:          %[[VAL_14:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_14]] : f32
    # CHECK:        } else {
    # CHECK:          %[[VAL_15:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:          %[[VAL_16:.*]] = scf.if %[[VAL_15]] -> (f32) {
    # CHECK:            %[[VAL_17:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:            scf.yield %[[VAL_17]] : f32
    # CHECK:          } else {
    # CHECK:            %[[VAL_18:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:            %[[VAL_19:.*]] = scf.if %[[VAL_18]] -> (f32) {
    # CHECK:              %[[VAL_20:.*]] = arith.constant 9.000000e+00 : f32
    # CHECK:              scf.yield %[[VAL_20]] : f32
    # CHECK:            } else {
    # CHECK:              %[[VAL_21:.*]] = arith.constant 1.000000e+01 : f32
    # CHECK:              scf.yield %[[VAL_21]] : f32
    # CHECK:            }
    # CHECK:            scf.yield %[[VAL_22:.*]] : f32
    # CHECK:          }
    # CHECK:          scf.yield %[[VAL_23:.*]] : f32
    # CHECK:        }
    # CHECK:        scf.yield %[[VAL_24:.*]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_25:.*]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_26:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (f32) {
    # CHECK:    %[[VAL_5:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_5]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (f32) {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_8]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_9]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_10:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (f32) {
    # CHECK:    %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (f32) {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_8]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_9]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_10:.*]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_12:.*]] = scf.if %[[VAL_11]] -> (f32) {
    # CHECK:      %[[VAL_13:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_13]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_14:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_14]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_15:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (f32) {
    # CHECK:    %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_6]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_8:.*]] = scf.if %[[VAL_7]] -> (f32) {
    # CHECK:      %[[VAL_9:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:      %[[VAL_10:.*]] = scf.if %[[VAL_9]] -> (f32) {
    # CHECK:        %[[VAL_11:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_11]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_12]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_13:.*]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_14:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_14]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_15:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (f32) {
    # CHECK:    %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_6]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_8:.*]] = scf.if %[[VAL_7]] -> (f32) {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_9]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_10:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:      %[[VAL_11:.*]] = scf.if %[[VAL_10]] -> (f32) {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_12]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_13:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_13]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_14:.*]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_15:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_5:.*]]:2 = scf.if %[[VAL_4]] -> (f32, f32) {
    # CHECK:    %[[VAL_6:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_6]], %[[VAL_6]] : f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_8:.*]]:2 = scf.if %[[VAL_7]] -> (f32, f32) {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_9]], %[[VAL_9]] : f32, f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_10:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:      %[[VAL_11:.*]]:2 = scf.if %[[VAL_10]] -> (f32, f32) {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_12]], %[[VAL_12]] : f32, f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_13:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_13]], %[[VAL_13]] : f32, f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : f32, f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_15:.*]]#0, %[[VAL_15]]#1 : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_9:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_10:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_11:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] {
    # CHECK:      %[[VAL_13:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:      scf.if %[[VAL_9]] {
    # CHECK:        %[[VAL_10:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_11:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_12:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_13:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_13]] {
    # CHECK:      %[[VAL_15:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:      scf.if %[[VAL_9]] {
    # CHECK:        %[[VAL_10:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      }
    # CHECK:      %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:      scf.if %[[VAL_11]] {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_13:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_14:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_15:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_15]] {
    # CHECK:      %[[VAL_17:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_4:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_9:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:      scf.if %[[VAL_9]] {
    # CHECK:        %[[VAL_10:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_11:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_12:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_13:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_14:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_14]] {
    # CHECK:      %[[VAL_16:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_with_for_if_replace_yield_2_first_branch_with_yield(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i, it, _ in range_(0, 10, iter_args=[three]):
                four = constant(4.0)
                res = yield four
            res = yield res
        else:
            four = constant(4.0)
            res = yield four

        return

    iffoo()
    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] iter_args(%[[VAL_10:.*]] = %[[VAL_4]]) -> (f32) {
    # CHECK:      %[[VAL_11:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_11]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_12:.*]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_13:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_13]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_with_for_if_replace_yield_2_both_branches_one_nested_if_with_yield(
    ctx: MLIRContext,
):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            for i, it, _ in range_(0, 10, iter_args=[three]):
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
            for i, it, _ in range_(0, 10, iter_args=[seven]):
                res = yield it
            res = yield res
        return

    iffoo()
    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_6:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] iter_args(%[[VAL_10:.*]] = %[[VAL_4]]) -> (f32) {
    # CHECK:      %[[VAL_11:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:      %[[VAL_12:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:      %[[VAL_13:.*]] = scf.if %[[VAL_12]] -> (f32) {
    # CHECK:        %[[VAL_14:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_14]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_15:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_15]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_16:.*]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_17:.*]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_18:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:    %[[VAL_19:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_20:.*]] = arith.constant 10 : index
    # CHECK:    %[[VAL_21:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_19]] to %[[VAL_20]] step %[[VAL_21]] iter_args(%[[VAL_24:.*]] = %[[VAL_18]]) -> (f32) {
    # CHECK:      scf.yield %[[VAL_24]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_25:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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
                alloca_scope_return([one])

            res = yield demo_scope1
        else:
            seven = constant(7.0)
            yield seven
        return

    iffoo()
    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = memref.alloca_scope  -> (f32) {
    # CHECK:      %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:      memref.alloca_scope.return %[[VAL_5]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_6:.*]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_3:.*]] = scf.if %[[VAL_2]] -> (f32) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_4]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:    %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (f32) {
    # CHECK:      %[[VAL_7:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_7]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_8:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_8]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_9:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:  %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (f32) {
    # CHECK:    %[[VAL_7:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:    %[[VAL_8:.*]] = scf.if %[[VAL_7]] -> (f32) {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_9]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_10:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:      %[[VAL_11:.*]] = scf.if %[[VAL_10]] -> (f32) {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_12]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_13:.*]] = arith.constant 9.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_13]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_14:.*]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_15:.*]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_16:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_17:.*]] = scf.if %[[VAL_16]] -> (f32) {
    # CHECK:      %[[VAL_18:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:      scf.yield %[[VAL_18]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_19:.*]] = arith.constant 1.000000e+01 : f32
    # CHECK:      scf.yield %[[VAL_19]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_20:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:  %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (f32) {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_8:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_9:.*]] = scf.if %[[VAL_8]] -> (f32) {
    # CHECK:      %[[VAL_10:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:      %[[VAL_11:.*]] = scf.if %[[VAL_10]] -> (f32) {
    # CHECK:        %[[VAL_12:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_12]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_13:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:        %[[VAL_14:.*]] = scf.if %[[VAL_13]] -> (f32) {
    # CHECK:          %[[VAL_15:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_15]] : f32
    # CHECK:        } else {
    # CHECK:          %[[VAL_16:.*]] = arith.constant 9.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_16]] : f32
    # CHECK:        }
    # CHECK:        scf.yield %[[VAL_17:.*]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_18:.*]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_19:.*]] = arith.constant 1.000000e+01 : f32
    # CHECK:      scf.yield %[[VAL_19]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_20:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:  %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (f32) {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]] : f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_8:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_9:.*]] = scf.if %[[VAL_8]] -> (f32) {
    # CHECK:      %[[VAL_10:.*]] = arith.constant 1.000000e+01 : f32
    # CHECK:      scf.yield %[[VAL_10]] : f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:      %[[VAL_12:.*]] = scf.if %[[VAL_11]] -> (f32) {
    # CHECK:        %[[VAL_13:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_13]] : f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_14:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:        %[[VAL_15:.*]] = scf.if %[[VAL_14]] -> (f32) {
    # CHECK:          %[[VAL_16:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_16]] : f32
    # CHECK:        } else {
    # CHECK:          %[[VAL_17:.*]] = arith.constant 9.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_17]] : f32
    # CHECK:        }
    # CHECK:        scf.yield %[[VAL_18:.*]] : f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_19:.*]] : f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_20:.*]] : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f32
    # CHECK:  %[[VAL_4:.*]] = arith.constant 5.000000e+00 : f32
    # CHECK:  %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
    # CHECK:  %[[VAL_6:.*]]:2 = scf.if %[[VAL_5]] -> (f32, f32) {
    # CHECK:    %[[VAL_7:.*]] = arith.constant 6.000000e+00 : f32
    # CHECK:    scf.yield %[[VAL_7]], %[[VAL_7]] : f32, f32
    # CHECK:  } else {
    # CHECK:    %[[VAL_8:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_2]] : f32
    # CHECK:    %[[VAL_9:.*]]:2 = scf.if %[[VAL_8]] -> (f32, f32) {
    # CHECK:      %[[VAL_10:.*]] = arith.constant 1.000000e+01 : f32
    # CHECK:      scf.yield %[[VAL_10]], %[[VAL_10]] : f32, f32
    # CHECK:    } else {
    # CHECK:      %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_2]], %[[VAL_3]] : f32
    # CHECK:      %[[VAL_12:.*]]:2 = scf.if %[[VAL_11]] -> (f32, f32) {
    # CHECK:        %[[VAL_13:.*]] = arith.constant 7.000000e+00 : f32
    # CHECK:        scf.yield %[[VAL_13]], %[[VAL_13]] : f32, f32
    # CHECK:      } else {
    # CHECK:        %[[VAL_14:.*]] = arith.cmpf olt, %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:        %[[VAL_15:.*]]:2 = scf.if %[[VAL_14]] -> (f32, f32) {
    # CHECK:          %[[VAL_16:.*]] = arith.constant 8.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_16]], %[[VAL_16]] : f32, f32
    # CHECK:        } else {
    # CHECK:          %[[VAL_17:.*]] = arith.constant 9.000000e+00 : f32
    # CHECK:          scf.yield %[[VAL_17]], %[[VAL_17]] : f32, f32
    # CHECK:        }
    # CHECK:        scf.yield %[[VAL_18:.*]]#0, %[[VAL_18]]#1 : f32, f32
    # CHECK:      }
    # CHECK:      scf.yield %[[VAL_19:.*]]#0, %[[VAL_19]]#1 : f32, f32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_20:.*]]#0, %[[VAL_20]]#1 : f32, f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_forall_1(ctx: MLIRContext):
    @forall_([1], [2], [3])
    def forfoo(ivs):
        one = constant(1.0)

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
    # CHECK:  scf.forall (%[[VAL_3:.*]]) = (1) to (2) step (3) {
    # CHECK:    %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_forall_3(ctx: MLIRContext):
    @forall_([1, 1], [2, 2], [3, 3])
    def forfoo(iv1, iv2):
        one = constant(1.0)

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  scf.forall (%[[VAL_6:.*]], %[[VAL_7:.*]]) = (1, 1) to (2, 2) step (3, 3) {
    # CHECK:    %[[VAL_8:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_forall_insert_slice(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    @forall_([1, 1], [2, 2], [3, 3], shared_outs=[ten])
    def forfoo(i, j, shared_outs):
        one = constant(1.0)

        return lambda: tensor.parallel_insert_slice(
            ten,
            shared_outs,
            offsets=[i, j],
            static_sizes=[10, 10],
            static_strides=[1, 1],
        )

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (1, 1) to (2, 2) step (3, 3) shared_outs(%[[VAL_10:.*]] = %[[VAL_0]]) -> (tensor<10x10xi32>) {
    # CHECK:    %[[VAL_11:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    scf.forall.in_parallel {
    # CHECK:      tensor.parallel_insert_slice %[[VAL_0]] into %[[VAL_10]]{{\[}}%[[VAL_8]], %[[VAL_9]]] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_forall_insert_slice_no_region(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    @forall_([1, 1], [2, 2], [3, 3], shared_outs=[ten])
    def forfoo(i, j, shared_outs):
        one = constant(1.0)

        return lambda: tensor.parallel_insert_slice(
            ten,
            shared_outs,
            offsets=[i, j],
            static_sizes=[10, 10],
            static_strides=[1, 1],
        )

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (1, 1) to (2, 2) step (3, 3) shared_outs(%[[VAL_10:.*]] = %[[VAL_0]]) -> (tensor<10x10xi32>) {
    # CHECK:    %[[VAL_11:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    scf.forall.in_parallel {
    # CHECK:      tensor.parallel_insert_slice %[[VAL_0]] into %[[VAL_10]]{{\[}}%[[VAL_8]], %[[VAL_9]]] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


@pytest.mark.xfail
def test_parange_no_inits(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    @parange_([1, 1], [2, 2], [3, 3], inits=[])
    def forfoo(i, j):
        one = constant(1.0)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x10xi32>
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c3_2 = arith.constant 3 : index
      scf.parallel (%arg0, %arg1) = (%c1, %c1_0) to (%c2, %c2_1) step (%c3, %c3_2) {
        %cst = arith.constant 1.000000e+00 : f32
        scf.yield
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_forall_insert_slice_no_region_with_for(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    for i, j, shared_outs in forall([1, 1], [2, 2], [3, 3], shared_outs=[ten]):
        one = constant(1.0)

        scf.parallel_insert_slice(
            ten,
            shared_outs,
            offsets=[i, j],
            static_sizes=[10, 10],
            static_strides=[1, 1],
        )

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (1, 1) to (2, 2) step (3, 3) shared_outs(%[[VAL_10:.*]] = %[[VAL_0]]) -> (tensor<10x10xi32>) {
    # CHECK:    %[[VAL_11:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    scf.forall.in_parallel {
    # CHECK:      tensor.parallel_insert_slice %[[VAL_0]] into %[[VAL_10]]{{\[}}%[[VAL_8]], %[[VAL_9]]] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


@pytest.mark.xfail
def test_parange_no_inits_with_for(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    for i, j in parange([1, 1], [2, 2], [3, 3], inits=[]):
        one = constant(1.0)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x10xi32>
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c3_2 = arith.constant 3 : index
      scf.parallel (%arg0, %arg1) = (%c1, %c1_0) to (%c2, %c2_1) step (%c3, %c3_2) {
        %cst = arith.constant 1.000000e+00 : f32
        scf.yield
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_parange_inits_with_for(ctx: MLIRContext):
    ten = empty(10, 10, T.i32())

    for i, j in parange([1, 1], [2, 2], [3, 3], inits=[ten]):
        one = constant(1.0)
        twenty = empty(10, 10, T.i32())

        @reduce(twenty)
        def res(lhs: Tensor, rhs: Tensor):
            assert isinstance(lhs, Tensor)
            assert isinstance(rhs, Tensor)
            return lhs + rhs

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]] = scf.parallel (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (%[[VAL_1]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_5]], %[[VAL_6]]) init (%[[VAL_0]]) -> tensor<10x10xi32> {
    # CHECK:    %[[VAL_10:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    %[[VAL_11:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:    scf.reduce(%[[VAL_11]] : tensor<10x10xi32>) {
    # CHECK:    ^bb0(%[[VAL_12:.*]]: tensor<10x10xi32>, %[[VAL_13:.*]]: tensor<10x10xi32>):
    # CHECK:      %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : tensor<10x10xi32>
    # CHECK:      scf.reduce.return %[[VAL_14]] : tensor<10x10xi32>
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_parange_inits_with_for_with_two_reduce(ctx: MLIRContext):
    one = constant(1, index=True)

    for i, j in parange([1, 1], [2, 2], [3, 3], inits=[one, one]):

        @reduce(i, j, num_reductions=2)
        def res1(lhs: T.index(), rhs: T.index()):
            return lhs + rhs

        @another_reduce(res1)
        def res1(lhs: T.index(), rhs: T.index()):
            return lhs + rhs

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_7:.*]]:2 = scf.parallel (%[[VAL_8:.*]], %[[VAL_9:.*]]) = (%[[VAL_1]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_5]], %[[VAL_6]]) init (%[[VAL_0]], %[[VAL_0]]) -> (index, index) {
    # CHECK:    scf.reduce(%[[VAL_8]], %[[VAL_9]] : index, index) {
    # CHECK:    ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
    # CHECK:      %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : index
    # CHECK:      scf.reduce.return %[[VAL_12]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
    # CHECK:      %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
    # CHECK:      scf.reduce.return %[[VAL_15]] : index
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_parange_inits_with_for_with_three_reduce(ctx: MLIRContext):
    one = constant(1, index=True)

    for i, j, k in parange([1, 1, 1], [2, 2, 2], [3, 3, 3], inits=[one, one, one]):

        @reduce(i, j, k, num_reductions=3)
        def res1(lhs: T.index(), rhs: T.index()):
            return lhs + rhs

        @another_reduce(res1)
        def res1(lhs: T.index(), rhs: T.index()):
            return lhs + rhs

        @another_reduce(res1)
        def res2(lhs: T.index(), rhs: T.index()):
            return lhs + rhs

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_5:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_8:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
    # CHECK:  %[[VAL_10:.*]]:3 = scf.parallel (%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) = (%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) to (%[[VAL_4]], %[[VAL_5]], %[[VAL_6]]) step (%[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) init (%[[VAL_0]], %[[VAL_0]], %[[VAL_0]]) -> (index, index, index) {
    # CHECK:    scf.reduce(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : index, index, index) {
    # CHECK:    ^bb0(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
    # CHECK:      %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : index
    # CHECK:      scf.reduce.return %[[VAL_16]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_17:.*]]: index, %[[VAL_18:.*]]: index):
    # CHECK:      %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : index
    # CHECK:      scf.reduce.return %[[VAL_19]] : index
    # CHECK:    }, {
    # CHECK:    ^bb0(%[[VAL_20:.*]]: index, %[[VAL_21:.*]]: index):
    # CHECK:      %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : index
    # CHECK:      scf.reduce.return %[[VAL_22]] : index
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_2(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    w = while__(one < two)
    while inits := next(w, False):
        r = yield_(*inits)

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_0]], %[[VAL_4:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.condition(%[[VAL_5]]) %[[VAL_3]], %[[VAL_4]] : i32, i32
    # CHECK:  } do {
    # CHECK:  ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
    # CHECK:    scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        while inits := one < two:
            r = yield inits

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_0]], %[[VAL_4:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.condition(%[[VAL_5]]) %[[VAL_3]], %[[VAL_4]] : i32, i32
    # CHECK:  } do {
    # CHECK:  ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
    # CHECK:    scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_2(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    def foo():
        cond = one < two
        while inits := while___(cond):
            r = yield_(inits)

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_0]], %[[VAL_4:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.condition(%[[VAL_5]]) %[[VAL_3]], %[[VAL_4]] : i32, i32
    # CHECK:  } do {
    # CHECK:  ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
    # CHECK:    scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_with_if(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        if one < two:
            while inits := one < two:
                r = yield inits

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]]:2 = scf.while (%[[VAL_4:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_6]]) %[[VAL_4]], %[[VAL_5]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_with_elif(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        if one < two:
            while inits := one < two:
                r = yield inits
        elif one < two:
            while inits := one < two:
                r = yield inits
        else:
            while inits := one < two:
                r = yield inits

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  scf.if %[[VAL_2]] {
    # CHECK:    %[[VAL_3:.*]]:2 = scf.while (%[[VAL_4:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_6]]) %[[VAL_4]], %[[VAL_5]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:  } else {
    # CHECK:    %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.if %[[VAL_9]] {
    # CHECK:      %[[VAL_10:.*]]:2 = scf.while (%[[VAL_11:.*]] = %[[VAL_0]], %[[VAL_12:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:        %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:        scf.condition(%[[VAL_13]]) %[[VAL_11]], %[[VAL_12]] : i32, i32
    # CHECK:      } do {
    # CHECK:      ^bb0(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32):
    # CHECK:        scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:      }
    # CHECK:    } else {
    # CHECK:      %[[VAL_16:.*]]:2 = scf.while (%[[VAL_17:.*]] = %[[VAL_0]], %[[VAL_18:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:        %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:        scf.condition(%[[VAL_19]]) %[[VAL_17]], %[[VAL_18]] : i32, i32
    # CHECK:      } do {
    # CHECK:      ^bb0(%[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32):
    # CHECK:        scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:      }
    # CHECK:    }
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_with_if_with_results(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        if one < two:
            while inits := one < two:
                r1, r2 = yield inits
            r1, r2 = yield r1, r2
        else:
            while inits := one < two:
                r1, r2 = yield inits
            r1, r2 = yield r1, r2

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (i32, i32) {
    # CHECK:    %[[VAL_4:.*]]:2 = scf.while (%[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_7]]) %[[VAL_5]], %[[VAL_6]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_10:.*]]#0, %[[VAL_10]]#1 : i32, i32
    # CHECK:  } else {
    # CHECK:    %[[VAL_11:.*]]:2 = scf.while (%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_14]]) %[[VAL_12]], %[[VAL_13]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_17:.*]]#0, %[[VAL_17]]#1 : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_with_if_with_results_2(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        if one < two:
            while inits := one < two:
                r = yield inits
            r1, r2 = yield r
        else:
            while inits := one < two:
                r = yield inits
            r1, r2 = yield r

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (i32, i32) {
    # CHECK:    %[[VAL_4:.*]]:2 = scf.while (%[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_7]]) %[[VAL_5]], %[[VAL_6]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_10:.*]]#0, %[[VAL_10]]#1 : i32, i32
    # CHECK:  } else {
    # CHECK:    %[[VAL_11:.*]]:2 = scf.while (%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_14]]) %[[VAL_12]], %[[VAL_13]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_17:.*]]#0, %[[VAL_17]]#1 : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_with_if_with_results_3(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        if one < two:
            while inits := one < two:
                r1, r2 = yield inits
            r = yield r1, r2
        else:
            while inits := one < two:
                r1, r2 = yield inits
            r = yield r1, r2

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]]:2 = scf.if %[[VAL_2]] -> (i32, i32) {
    # CHECK:    %[[VAL_4:.*]]:2 = scf.while (%[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_7]]) %[[VAL_5]], %[[VAL_6]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_10:.*]]#0, %[[VAL_10]]#1 : i32, i32
    # CHECK:  } else {
    # CHECK:    %[[VAL_11:.*]]:2 = scf.while (%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:      %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:      scf.condition(%[[VAL_14]]) %[[VAL_12]], %[[VAL_13]] : i32, i32
    # CHECK:    } do {
    # CHECK:    ^bb0(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
    # CHECK:      scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_17:.*]]#0, %[[VAL_17]]#1 : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_while_canonicalize_nested_if(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    @canonicalize(using=canonicalizer)
    def foo():
        while inits := one < two:
            if one < two:
                three = constant(3)
            yield inits

    foo()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_0]], %[[VAL_4:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
    # CHECK:    %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.condition(%[[VAL_5]]) %[[VAL_3]], %[[VAL_4]] : i32, i32
    # CHECK:  } do {
    # CHECK:  ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
    # CHECK:    %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:    scf.if %[[VAL_8]] {
    # CHECK:      %[[VAL_9:.*]] = arith.constant 3 : i32
    # CHECK:    }
    # CHECK:    scf.yield %[[VAL_0]], %[[VAL_1]] : i32, i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)
