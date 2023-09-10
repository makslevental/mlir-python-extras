from pathlib import Path
from textwrap import dedent

import pytest

from util import skip_jax_not_installed, skip_torch_mlir_not_installed


def test_smoke():
    from mlir.utils.context import mlir_mod_ctx
    from mlir.utils.testing import filecheck

    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        correct = dedent(
            """\
        module {
        }
        """
        )

    filecheck(correct, ctx.module)


def test_dialect_trampolines_smoke():
    from mlir.utils._configuration.generate_trampolines import (
        generate_all_upstream_trampolines,
    )

    generate_all_upstream_trampolines()
    # noinspection PyUnresolvedReferences
    from mlir.utils.dialects import (
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


@pytest.mark.skipif(skip_torch_mlir_not_installed(), reason="torch_mlir not installed")
def test_torch_dialect_trampolines_smoke():
    import torch_mlir.utils.dialects

    from torch_mlir.utils._configuration.generate_trampolines import (
        generate_trampolines,
    )

    generate_trampolines(
        "torch_mlir.dialects.torch",
        Path(torch_mlir.utils.dialects.__path__[0]),
        "torch",
    )
    # noinspection PyUnresolvedReferences
    from torch_mlir.utils.dialects import torch


@pytest.mark.skipif(skip_jax_not_installed(), reason="jax not installed")
def test_jax_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from jaxlib.mlir.utils.dialects import (
        arith,
        builtin,
        chlo,
        func,
        math,
        memref,
        mhlo,
        ml_program,
        scf,
        sparse_tensor,
        stablehlo,
        vector,
    )
