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
    print()
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
