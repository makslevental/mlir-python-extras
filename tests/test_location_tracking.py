import sys
from os import sep
from pathlib import Path
from textwrap import dedent

import pytest

import mlir.utils.types as T
from mlir.utils.ast.canonicalize import canonicalize
from mlir.utils.dialects.ext.arith import constant
from mlir.utils.dialects.ext.scf import canonicalizer
from mlir.utils.dialects.ext.tensor import S
from mlir.utils.dialects.tensor import generate, yield_ as tensor_yield, rank

# noinspection PyUnresolvedReferences
from mlir.utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")

THIS_DIR = str(Path(__file__).parent.absolute())


def get_asm(operation):
    return operation.get_asm(enable_debug_info=True, pretty_debug_info=True).replace(
        THIS_DIR, "THIS_DIR"
    )


@pytest.mark.skipif(sys.version_info.minor != 12, reason="only check latest")
def test_if_replace_yield_5(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0, T.f32)
        two = constant(2.0, T.f32)
        if one < two:
            three = constant(3.0, T.f32)
            res1, res2, res3 = yield three, three, three
        else:
            four = constant(4.0, T.f32)
            res1, res2, res3 = yield four, four, four
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        r"""
module {
  %cst = arith.constant 1.000000e+00 : f32 THIS_DIR/test_location_tracking.py:34:10
  %cst_0 = arith.constant 2.000000e+00 : f32 THIS_DIR/test_location_tracking.py:35:10
  %0 = arith.cmpf olt, %cst, %cst_0 : f32 THIS_DIR/test_location_tracking.py:36:7
  %1:3 = scf.if %0 -> (f32, f32, f32) {
    %cst_1 = arith.constant 3.000000e+00 : f32 THIS_DIR/test_location_tracking.py:37:16
    scf.yield %cst_1, %cst_1, %cst_1 : f32, f32, f32 THIS_DIR/test_location_tracking.py:38:27
  } else {
    %cst_1 = arith.constant 4.000000e+00 : f32 THIS_DIR/test_location_tracking.py:40:15
    scf.yield %cst_1, %cst_1, %cst_1 : f32, f32, f32 THIS_DIR/test_location_tracking.py:41:27
  } [unknown]
} [unknown]
#loc = [unknown]
#loc1 = THIS_DIR/test_location_tracking.py:34:10
#loc2 = THIS_DIR/test_location_tracking.py:35:10
#loc3 = THIS_DIR/test_location_tracking.py:36:7
#loc4 = THIS_DIR/test_location_tracking.py:37:16
#loc5 = THIS_DIR/test_location_tracking.py:38:27
#loc6 = THIS_DIR/test_location_tracking.py:40:15
#loc7 = THIS_DIR/test_location_tracking.py:41:27
    """
    ).replace("/", sep)

    asm = get_asm(ctx.module.operation)
    filecheck(correct, asm)


@pytest.mark.skipif(sys.version_info.minor != 12, reason="only check latest")
def test_block_args(ctx: MLIRContext):
    one = constant(1, T.index)
    two = constant(2, T.index)

    @generate(T.tensor(S, 3, S, T.f32), dynamic_extents=[one, two])
    def demo_fun1(i: T.index, j: T.index, k: T.index):
        one = constant(1.0)
        tensor_yield(one)

    r = rank(demo_fun1)

    ctx.module.operation.verify()

    correct = dedent(
        r"""
#loc3 = THIS_DIR/test_location_tracking.py:80:5
module {
  %c1 = arith.constant 1 : index THIS_DIR/test_location_tracking.py:77:10
  %c2 = arith.constant 2 : index THIS_DIR/test_location_tracking.py:78:10
  %generated = tensor.generate %c1, %c2 {
  ^bb0(%arg0: index THIS_DIR/test_location_tracking.py:80:5, %arg1: index THIS_DIR/test_location_tracking.py:80:5, %arg2: index THIS_DIR/test_location_tracking.py:80:5):
    %cst = arith.constant 1.000000e+00 : f32 THIS_DIR/test_location_tracking.py:82:14
    tensor.yield %cst : f32 THIS_DIR/test_location_tracking.py:83:8
  } : tensor<?x3x?xf32> THIS_DIR/test_location_tracking.py:80:5
  %rank = tensor.rank %generated : tensor<?x3x?xf32> THIS_DIR/test_location_tracking.py:85:8
} [unknown]
#loc = [unknown]
#loc1 = THIS_DIR/test_location_tracking.py:77:10
#loc2 = THIS_DIR/test_location_tracking.py:78:10
#loc4 = THIS_DIR/test_location_tracking.py:82:14
#loc5 = THIS_DIR/test_location_tracking.py:83:8
#loc6 = THIS_DIR/test_location_tracking.py:85:8
    """
    ).replace("/", sep)
    asm = get_asm(ctx.module.operation)
    filecheck(correct, asm)
