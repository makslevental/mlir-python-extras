import mlir.extras.types as T
import pytest

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import arith
from mlir.extras.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    MLIRContext,
    filecheck_with_comments,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_arith_constant_canonicalizer(ctx: MLIRContext):
    @func(emit=True)
    @canonicalize(using=arith.canonicalizer)
    def foo():
        # CHECK: %c0_i32 = arith.constant 0 : i32
        row_m: T.i32() = 0
        # CHECK: %cst = arith.constant 0.000000e+00 : f32
        row_l: T.f32() = 0.0

    filecheck_with_comments(ctx.module)
