import pytest

from util import jax_not_installed, mlir_bindings_not_installed


@pytest.mark.skipif(
    mlir_bindings_not_installed(), reason="mlir python bindings not installed"
)
def test_dialect_trampolines_smoke():
    from mlir.utils._configuration.generate_trampolines import (
        generate_all_upstream_trampolines,
    )

    generate_all_upstream_trampolines()


@pytest.mark.skipif(jax_not_installed(), reason="jax not installed")
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
        scf,
        sparse_tensor,
        stablehlo,
        vector,
    )
