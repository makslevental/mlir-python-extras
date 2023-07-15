from textwrap import dedent

import pytest

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("mlir_ctx")


def test_smoke(mlir_ctx: MLIRContext):
    correct = dedent(
        """\
    module {
    }
    """
    )
    filecheck(correct, mlir_ctx.module)
