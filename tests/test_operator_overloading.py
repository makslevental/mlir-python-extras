from textwrap import dedent

import pytest

import mlir.utils.types as T
from mlir.utils.dialects.ext.arith import constant, Scalar
from mlir.utils.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir.utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_arithmetic(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)
    one + two
    one - two
    one / two
    one // two
    one % two

    one = constant(1.0)
    two = constant(2.0)
    one + two
    one - two
    one / two
    try:
        one // two
    except ValueError as e:
        assert str(e) == "floordiv not supported for lhs=Scalar(%cst, f32)"
    one % two

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.addi %c1_i32, %c2_i32 : i32
      %1 = arith.subi %c1_i32, %c2_i32 : i32
      %2 = arith.divsi %c1_i32, %c2_i32 : i32
      %3 = arith.floordivsi %c1_i32, %c2_i32 : i32
      %4 = arith.remsi %c1_i32, %c2_i32 : i32
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %5 = arith.addf %cst, %cst_0 : f32
      %6 = arith.subf %cst, %cst_0 : f32
      %7 = arith.divf %cst, %cst_0 : f32
      %8 = arith.remf %cst, %cst_0 : f32
    }
    """
        ),
        ctx.module,
    )


def test_tensor_arithmetic(ctx: MLIRContext):
    one = constant(1)
    assert isinstance(one, Scalar)
    two = constant(2)
    assert isinstance(two, Scalar)
    three = one + two
    assert isinstance(three, Scalar)

    ten1 = empty((10, 10, 10), T.f32)
    assert isinstance(ten1, Tensor)
    ten2 = empty((10, 10, 10), T.f32)
    assert isinstance(ten2, Tensor)
    ten3 = ten1 + ten2
    assert isinstance(ten3, Tensor)

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.addi %c1_i32, %c2_i32 : i32
      %1 = tensor.empty() : tensor<10x10x10xf32>
      %2 = tensor.empty() : tensor<10x10x10xf32>
      %3 = arith.addf %1, %2 : tensor<10x10x10xf32>
    }
    """
        ),
        ctx.module,
    )


def test_r_arithmetic(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)
    one - two
    two - one

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.subi %c1_i32, %c2_i32 : i32
      %1 = arith.subi %c2_i32, %c1_i32 : i32
    }
    """
        ),
        ctx.module,
    )


def test_arith_cmp(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two
    one & two
    one | two
    assert one._ne(two)
    assert not one._eq(two)

    one = constant(1.0)
    two = constant(2.0)
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two
    assert one._ne(two)
    assert not one._eq(two)

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.cmpi ult, %c1_i32, %c2_i32 : i32
      %1 = arith.cmpi ule, %c1_i32, %c2_i32 : i32
      %2 = arith.cmpi ugt, %c1_i32, %c2_i32 : i32
      %3 = arith.cmpi uge, %c1_i32, %c2_i32 : i32
      %4 = arith.cmpi eq, %c1_i32, %c2_i32 : i32
      %5 = arith.cmpi ne, %c1_i32, %c2_i32 : i32
      %6 = arith.andi %c1_i32, %c2_i32 : i32
      %7 = arith.ori %c1_i32, %c2_i32 : i32
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %8 = arith.cmpf olt, %cst, %cst_0 : f32
      %9 = arith.cmpf ole, %cst, %cst_0 : f32
      %10 = arith.cmpf ogt, %cst, %cst_0 : f32
      %11 = arith.cmpf oge, %cst, %cst_0 : f32
      %12 = arith.cmpf oeq, %cst, %cst_0 : f32
      %13 = arith.cmpf one, %cst, %cst_0 : f32
    }
    """
        ),
        ctx.module,
    )


def test_scalar_promotion(ctx: MLIRContext):
    one = constant(1)
    one + 2
    one - 2
    one / 2
    one // 2
    one % 2

    one = constant(1.0)
    one + 2.0
    one - 2.0
    one / 2.0
    one % 2.0

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.addi %c1_i32, %c2_i32 : i32
      %c2_i32_0 = arith.constant 2 : i32
      %1 = arith.subi %c1_i32, %c2_i32_0 : i32
      %c2_i32_1 = arith.constant 2 : i32
      %2 = arith.divsi %c1_i32, %c2_i32_1 : i32
      %c2_i32_2 = arith.constant 2 : i32
      %3 = arith.floordivsi %c1_i32, %c2_i32_2 : i32
      %c2_i32_3 = arith.constant 2 : i32
      %4 = arith.remsi %c1_i32, %c2_i32_3 : i32
      %cst = arith.constant 1.000000e+00 : f32
      %cst_4 = arith.constant 2.000000e+00 : f32
      %5 = arith.addf %cst, %cst_4 : f32
      %cst_5 = arith.constant 2.000000e+00 : f32
      %6 = arith.subf %cst, %cst_5 : f32
      %cst_6 = arith.constant 2.000000e+00 : f32
      %7 = arith.divf %cst, %cst_6 : f32
      %cst_7 = arith.constant 2.000000e+00 : f32
      %8 = arith.remf %cst, %cst_7 : f32
    }
    """
    )
    filecheck(correct, ctx.module)
