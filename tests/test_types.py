import pytest

import mlir_utils.types as T
from mlir_utils.dialects.ext.tensor import S

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import tensor_t, memref_t, vector_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_shaped_types(ctx: MLIRContext):
    t = tensor_t(S, 3, S, T.f64_t)
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor_t(T.f64_t)
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"
    t = tensor_t(S, 3, S, element_type=T.f64_t)
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor_t(element_type=T.f64_t)
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"

    m = memref_t(S, 3, S, T.f64_t)
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref_t(T.f64_t)
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"
    m = memref_t(S, 3, S, element_type=T.f64_t)
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref_t(element_type=T.f64_t)
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"

    v = vector_t(3, 3, 3, T.f64_t)
    assert repr(v) == "VectorType(vector<3x3x3xf64>)"
