from pathlib import Path
from textwrap import dedent

import pytest

import mlir_utils.dialects
from mlir_utils.dialects.generate_trampolines import generate_trampoline

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


def test_torch_dialect_trampolines_smoke():
    from torch_mlir.dialects import torch

    generate_trampoline(torch, Path(mlir_utils.dialects.__path__[0]) / "torch.py")
    # noinspection PyUnresolvedReferences
    from mlir_utils.dialects import torch
