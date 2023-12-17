import pytest

import mlir.extras.types as T
from mlir.extras.dialects.ext.tensor import S

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir.extras.types import tensor, memref, vector

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_shaped_types(ctx: MLIRContext):
    t = tensor(S, 3, S, T.f64)
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor(T.f64)
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"
    t = tensor(S, 3, S, element_type=T.f64)
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor(element_type=T.f64)
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"

    m = memref(S, 3, S, T.f64)
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref(T.f64)
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"
    m = memref(S, 3, S, element_type=T.f64)
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref(element_type=T.f64)
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"

    v = vector(3, 3, 3, T.f64)
    assert repr(v) == "VectorType(vector<3x3x3xf64>)"
