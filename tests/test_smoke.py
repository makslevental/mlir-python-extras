import os
from pathlib import Path
from textwrap import dedent

import pytest

import mlir_utils.dialects
from mlir_utils._configuration.generate_trampolines import (
    generate_trampolines,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from util import skip_jax_not_installed, skip_torch_mlir_not_installed

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_smoke(ctx: MLIRContext):
    correct = dedent(
        """\
    module {
    }
    """
    )
    filecheck(correct, ctx.module)


# skip if jax *is* installed because jax doesn't generate almost any of the upstream dialects
@pytest.mark.skipif(lambda: not skip_jax_not_installed(), reason="jax installed")
def test_dialect_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import (
        arith,
        bufferization,
        builtin,
        cf,
        complex,
        func,
        gpu,
        linalg,
        math,
        memref,
        ml_program,
        pdl,
        scf,
        shape,
        sparse_tensor,
        tensor,
        tosa,
        transform,
        vector,
    )


@pytest.mark.skipif(skip_jax_not_installed(), reason="jax not installed")
def test_dialect_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import (
        builtin,
        chlo,
        func,
        mhlo,
        ml_program,
        sparse_tensor,
        stablehlo,
    )


@pytest.mark.skipif(skip_torch_mlir_not_installed(), reason="torch_mlir not installed")
def test_torch_dialect_trampolines_smoke():
    try:
        modu = __import__("mlir_utils.dialects.torch", fromlist=["*"])
        os.remove(modu.__file__)
    except ModuleNotFoundError:
        pass
    generate_trampolines(
        "torch_mlir.dialects.torch", Path(mlir_utils.dialects.__path__[0]), "torch"
    )
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import torch


@pytest.mark.skipif(skip_jax_not_installed(), reason="jax not installed")
def test_jax_trampolines_smoke():
    for mod in ["chlo", "mhlo", "stablehlo"]:
        try:
            modu = __import__(f"mlir_utils.dialects.{mod}", fromlist=["*"])
            os.remove(modu.__file__)
        except ModuleNotFoundError:
            pass
        generate_trampolines(
            f"jaxlib.mlir.dialects.{mod}", Path(mlir_utils.dialects.__path__[0]), mod
        )
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import (
        builtin,
        chlo,
        func,
        mhlo,
        ml_program,
        sparse_tensor,
        stablehlo,
    )
