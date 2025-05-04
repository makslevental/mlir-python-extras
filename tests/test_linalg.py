from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects.ext import linalg, memref, tensor

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    MLIRContext,
    filecheck,
    filecheck_with_comments,
    mlir_ctx as ctx,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    x = memref.alloc((10, 10), T.i32())
    linalg.fill(5, x)
    linalg.fill_rng_2d(0.0, 10.0, 1, x)

    x = tensor.empty(10, 10, T.i32())
    y = linalg.fill_rng_2d(0.0, 10.0, 1, x)
    z = linalg.fill(5, x)

    # CHECK:  %[[VAL_0:.*]] = memref.alloc() : memref<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 5 : i32
    # CHECK:  linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_0]] : memref<10x10xi32>)
    # CHECK:  %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1.000000e+01 : f64
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i32
    # CHECK:  linalg.fill_rng_2d ins(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : f64, f64, i32) outs(%[[VAL_0]] : memref<10x10xi32>)
    # CHECK:  %[[VAL_5:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f64
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1.000000e+01 : f64
    # CHECK:  %[[VAL_8:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_9:.*]] = linalg.fill_rng_2d ins(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : f64, f64, i32) outs(%[[VAL_5]] : tensor<10x10xi32>) -> tensor<10x10xi32>
    # CHECK:  %[[VAL_10:.*]] = arith.constant 5 : i32
    # CHECK:  %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_10]] : i32) outs(%[[VAL_5]] : tensor<10x10xi32>) -> tensor<10x10xi32>

    filecheck_with_comments(ctx.module)
