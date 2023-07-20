import pytest

from mlir_utils.dialects.ext.tensor import S, empty
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.util import register_value_caster

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import f64_t, RankedTensorType

from mlir.ir import OpResult

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_caster_registration(ctx: MLIRContext):
    sizes = S, 3, S
    ten = empty(sizes, f64_t)
    assert repr(ten) == "Tensor(%0, tensor<?x3x?xf64>)"

    def dummy_caster(val):
        return val

    register_value_caster(RankedTensorType.static_typeid, dummy_caster)
    ten = empty(sizes, f64_t)
    assert repr(ten) == "Tensor(%1, tensor<?x3x?xf64>)"

    register_value_caster(RankedTensorType.static_typeid, dummy_caster, 0)
    ten = empty(sizes, f64_t)
    assert repr(ten) != "Tensor(%1, tensor<?x3x?xf64>)"
    assert isinstance(ten, OpResult)

    one = constant(1)
    assert repr(one) == "Scalar(%3, i64)"
