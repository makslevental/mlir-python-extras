from textwrap import dedent

import pytest

from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import f64_t, index_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_tensor_arithmetic(ctx: MLIRContext):
    one = constant(1)
    assert isinstance(one, Scalar)
    two = constant(2)
    assert isinstance(two, Scalar)
    three = one + two
    assert isinstance(three, Scalar)

    ten1 = empty((10, 10, 10), f64_t)
    assert isinstance(ten1, Tensor)
    ten2 = empty((10, 10, 10), f64_t)
    assert isinstance(ten2, Tensor)
    ten3 = ten1 + ten2
    assert isinstance(ten3, Tensor)

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %0 = arith.addi %c1_i64, %c2_i64 : i64
      %1 = tensor.empty() : tensor<10x10x10xf64>
      %2 = tensor.empty() : tensor<10x10x10xf64>
      %3 = arith.addf %1, %2 : tensor<10x10x10xf64>
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
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %0 = arith.subi %c1_i64, %c2_i64 : i64
      %1 = arith.subi %c2_i64, %c1_i64 : i64
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
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %0 = arith.cmpi ult, %c1_i64, %c2_i64 : i64
      %1 = arith.cmpi ule, %c1_i64, %c2_i64 : i64
      %2 = arith.cmpi ugt, %c1_i64, %c2_i64 : i64
      %3 = arith.cmpi uge, %c1_i64, %c2_i64 : i64
      %4 = arith.cmpi eq, %c1_i64, %c2_i64 : i64
      %5 = arith.cmpi ne, %c1_i64, %c2_i64 : i64
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %6 = arith.cmpf olt, %cst, %cst_0 : f64
      %7 = arith.cmpf ole, %cst, %cst_0 : f64
      %8 = arith.cmpf ogt, %cst, %cst_0 : f64
      %9 = arith.cmpf oge, %cst, %cst_0 : f64
      %10 = arith.cmpf oeq, %cst, %cst_0 : f64
      %11 = arith.cmpf one, %cst, %cst_0 : f64
    }
    """
        ),
        ctx.module,
    )
