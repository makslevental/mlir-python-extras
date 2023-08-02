import pytest
from mlir.ir import OpResult, RankedTensorType

from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.tensor import S, empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import (
    f32_t,
    cmp16_t,
    i8_t,
    i16_t,
    i32_t,
    f16_t,
    cmp32_t,
    cmp64_t,
)
from mlir_utils.util import register_value_caster

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_caster_registration(ctx: MLIRContext):
    sizes = S, 3, S
    ten = empty(sizes, f32_t)
    assert repr(ten) == "Tensor(%0, tensor<?x3x?xf32>)"

    def dummy_caster(val):
        return val

    register_value_caster(RankedTensorType.static_typeid)(dummy_caster)
    ten = empty(sizes, f32_t)
    assert repr(ten) == "Tensor(%1, tensor<?x3x?xf32>)"

    register_value_caster(RankedTensorType.static_typeid, 0)(dummy_caster)
    ten = empty(sizes, f32_t)
    assert repr(ten) != "Tensor(%1, tensor<?x3x?xf32>)"
    assert isinstance(ten, OpResult)

    one = constant(1)
    assert repr(one) == "Scalar(%3, i32)"


def test_scalar_register_value_caster_decorator(ctx: MLIRContext):
    assert isinstance(constant(1, type=i8_t), Scalar)
    assert isinstance(constant(1, type=i16_t), Scalar)
    assert isinstance(constant(1, type=i32_t), Scalar)
    assert isinstance(constant(1, type=i32_t), Scalar)

    assert isinstance(constant(1, index=True), Scalar)

    assert isinstance(constant(1, type=f16_t), Scalar)
    assert isinstance(constant(1, type=f32_t), Scalar)
    assert isinstance(constant(1, type=f32_t), Scalar)

    assert isinstance(constant(1, type=cmp16_t), Scalar)
    assert isinstance(constant(1, type=cmp32_t), Scalar)
    assert isinstance(constant(1, type=cmp64_t), Scalar)

    ctx.module.operation.verify()
