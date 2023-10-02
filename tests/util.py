def skip_torch_mlir_not_installed():
    try:
        from torch_mlir.dialects import torch

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def skip_jax_not_installed():
    try:
        from jaxlib import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def mlir_bindings_installed():
    try:
        import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def aie_bindings_installed():
    try:
        import aie

        # don't skip
        return False

    except ImportError:
        # skip
        return True
