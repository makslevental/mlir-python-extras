import pytest

from util import jax_not_installed, mlir_bindings_not_installed


@pytest.mark.skipif(jax_not_installed(), reason="jax not installed")
def test_jax_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from jaxlib.mlir.dialects import (
        arith,
        builtin,
        chlo,
        func,
        math,
        memref,
        mhlo,
        scf,
        sparse_tensor,
        stablehlo,
        vector,
    )
