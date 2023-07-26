import numpy as np
import pytest

from mlir_utils.dialects.ext.tensor import Tensor

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    arr = np.random.randint(0, 10, (10, 10))
    ten = Tensor(arr)
    assert np.array_equal(arr, ten.literal_value)
