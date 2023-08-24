import pytest
from mlir.ir import OpResult, RankedTensorType

import mlir_utils.types as T
from mlir_utils.dialects.ext.arith import constant, Scalar
from mlir_utils.dialects.ext.tensor import S, empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.meta import register_value_caster

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_caster_registration(ctx: MLIRContext):
    sizes = S, 3, S
    ten = empty(sizes, T.f32)
    assert repr(ten) == "Tensor(%0, tensor<?x3x?xf32>)"

    def dummy_caster(val):
        return val

    register_value_caster(RankedTensorType.static_typeid)(dummy_caster)
    ten = empty(sizes, T.f32)
    assert repr(ten) == "Tensor(%1, tensor<?x3x?xf32>)"

    register_value_caster(RankedTensorType.static_typeid, 0)(dummy_caster)
    ten = empty(sizes, T.f32)
    assert repr(ten) != "Tensor(%1, tensor<?x3x?xf32>)"
    assert isinstance(ten, OpResult)

    one = constant(1)
    assert repr(one) == "Scalar(%3, i32)"


def test_scalar_register_value_caster_decorator(ctx: MLIRContext):
    assert isinstance(constant(1, type=T.i8), Scalar)
    assert isinstance(constant(1, type=T.i16), Scalar)
    assert isinstance(constant(1, type=T.i32), Scalar)
    assert isinstance(constant(1, type=T.i32), Scalar)

    assert isinstance(constant(1, index=True), Scalar)

    assert isinstance(constant(1, type=T.f16), Scalar)
    assert isinstance(constant(1, type=T.f32), Scalar)
    assert isinstance(constant(1, type=T.f32), Scalar)

    assert isinstance(constant(1, type=T.cmp16), Scalar)
    assert isinstance(constant(1, type=T.cmp32), Scalar)
    assert isinstance(constant(1, type=T.cmp64), Scalar)

    ctx.module.operation.verify()
