from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects.ext import linalg, memref, tensor

# noinspection PyUnresolvedReferences
from mlir.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    x = memref.alloc(10, 10, T.i32())
    linalg.fill(5, x)
    linalg.fill_rng_2d(0.0, 10.0, 1, x)

    x = tensor.empty(10, 10, T.i32())
    y = linalg.fill_rng_2d(0.0, 10.0, 1, x)
    z = linalg.fill(5, x)

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c5_i32 = arith.constant 5 : i32
      linalg.fill ins(%c5_i32 : i32) outs(%alloc : memref<10x10xi32>)
      %cst = arith.constant 0.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+01 : f64
      %c1_i32 = arith.constant 1 : i32
      linalg.fill_rng_2d ins(%cst, %cst_0, %c1_i32 : f64, f64, i32) outs(%alloc : memref<10x10xi32>)
      %0 = tensor.empty() : tensor<10x10xi32>
      %cst_1 = arith.constant 0.000000e+00 : f64
      %cst_2 = arith.constant 1.000000e+01 : f64
      %c1_i32_3 = arith.constant 1 : i32
      %1 = linalg.fill_rng_2d ins(%cst_1, %cst_2, %c1_i32_3 : f64, f64, i32) outs(%0 : tensor<10x10xi32>) -> tensor<10x10xi32>
      %c5_i32_4 = arith.constant 5 : i32
      %2 = linalg.fill ins(%c5_i32_4 : i32) outs(%0 : tensor<10x10xi32>) -> tensor<10x10xi32>
    }
    """
    )
    filecheck(correct, ctx.module)
