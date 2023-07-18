from textwrap import dedent

import pytest

from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import f64, index

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_tensor_arithmetic(ctx: MLIRContext):
    print()
    one = constant(1, index)
    assert isinstance(one, Scalar)
    two = constant(2, index)
    assert isinstance(two, Scalar)
    three = one + two
    assert isinstance(three, Scalar)

    ten1 = empty((10, 10, 10), f64)
    assert isinstance(ten1, Tensor)
    ten2 = empty((10, 10, 10), f64)
    assert isinstance(ten2, Tensor)
    ten3 = ten1 + ten2
    assert isinstance(ten3, Tensor)

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %0 = arith.addi %c1, %c2 : index
      %1 = tensor.empty() : tensor<10x10x10xf64>
      %2 = tensor.empty() : tensor<10x10x10xf64>
      %3 = arith.addf %1, %2 : tensor<10x10x10xf64>
    }
    """
        ),
        ctx.module,
    )
