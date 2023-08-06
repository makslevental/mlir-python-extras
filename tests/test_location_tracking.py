from os import sep
from pathlib import Path
from textwrap import dedent

import pytest

import mlir_utils.types as T
from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.scf import (
    canonicalizer,
    stack_if,
)
from mlir_utils.dialects.ext.tensor import S
from mlir_utils.dialects.tensor import generate, yield_ as tensor_yield, rank

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")

THIS_DIR = str(Path(__file__).parent.absolute())


def get_asm(operation):
    return operation.get_asm(enable_debug_info=True, pretty_debug_info=True).replace(
        THIS_DIR, "THIS_DIR"
    )


def test_if_replace_yield_5(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if res := stack_if(one < two, (T.f64_t, T.f64_t, T.f64_t)):
            three = constant(3.0)
            yield three, three, three
        else:
            four = constant(4.0)
            yield four, four, four
        return

    iffoo()
    ctx.module.operation.verify()
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant 1.000000e+00 : f64 THIS_DIR{sep}test_location_tracking.py:35:10
      %cst_0 = arith.constant 2.000000e+00 : f64 THIS_DIR{sep}test_location_tracking.py:36:10
      %0 = arith.cmpf olt, %cst, %cst_0 : f64 THIS_DIR{sep}test_location_tracking.py:37:23
      %1:3 = scf.if %0 -> (f64, f64, f64) {{
        %cst_1 = arith.constant 3.000000e+00 : f64 THIS_DIR{sep}test_location_tracking.py:38:16
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64 THIS_DIR{sep}test_location_tracking.py:39:8
      }} else {{
        %cst_1 = arith.constant 4.000000e+00 : f64 THIS_DIR{sep}test_location_tracking.py:41:24
        scf.yield %cst_1, %cst_1, %cst_1 : f64, f64, f64 THIS_DIR{sep}test_location_tracking.py:42:8
      }} THIS_DIR{sep}test_location_tracking.py:37:14
    }} [unknown]
    #loc = [unknown]
    #loc1 = THIS_DIR{sep}test_location_tracking.py:35:10
    #loc2 = THIS_DIR{sep}test_location_tracking.py:36:10
    #loc3 = THIS_DIR{sep}test_location_tracking.py:37:23
    #loc4 = THIS_DIR{sep}test_location_tracking.py:37:14
    #loc5 = THIS_DIR{sep}test_location_tracking.py:38:16
    #loc6 = THIS_DIR{sep}test_location_tracking.py:39:8
    #loc7 = THIS_DIR{sep}test_location_tracking.py:41:24
    #loc8 = THIS_DIR{sep}test_location_tracking.py:42:8
    """
    )
    asm = get_asm(ctx.module.operation)
    filecheck(correct, asm)


def test_block_args(ctx: MLIRContext):
    one = constant(1, T.index_t)
    two = constant(2, T.index_t)

    @generate(T.tensor_t(S, 3, S, T.f64_t), dynamic_extents=[one, two])
    def demo_fun1(i: T.index_t, j: T.index_t, k: T.index_t):
        one = constant(1.0)
        tensor_yield(one)

    r = rank(demo_fun1)

    ctx.module.operation.verify()

    correct = dedent(
        f"""\
    #loc3 = THIS_DIR{sep}test_location_tracking.py:80:5
    module {{
      %c1 = arith.constant 1 : index THIS_DIR{sep}test_location_tracking.py:77:10
      %c2 = arith.constant 2 : index THIS_DIR{sep}test_location_tracking.py:78:10
      %generated = tensor.generate %c1, %c2 {{
      ^bb0(%arg0: index THIS_DIR{sep}test_location_tracking.py:80:5, %arg1: index THIS_DIR{sep}test_location_tracking.py:80:5, %arg2: index THIS_DIR{sep}test_location_tracking.py:80:5):
        %cst = arith.constant 1.000000e+00 : f64 THIS_DIR{sep}test_location_tracking.py:82:14
        tensor.yield %cst : f64 THIS_DIR{sep}test_location_tracking.py:83:8
      }} : tensor<?x3x?xf64> THIS_DIR{sep}test_location_tracking.py:80:5
      %rank = tensor.rank %generated : tensor<?x3x?xf64> THIS_DIR{sep}test_location_tracking.py:85:8
    }} [unknown]
    #loc = [unknown]
    #loc1 = THIS_DIR{sep}test_location_tracking.py:77:10
    #loc2 = THIS_DIR{sep}test_location_tracking.py:78:10
    #loc4 = THIS_DIR{sep}test_location_tracking.py:82:14
    #loc5 = THIS_DIR{sep}test_location_tracking.py:83:8
    #loc6 = THIS_DIR{sep}test_location_tracking.py:85:8
    """
    )
    asm = get_asm(ctx.module.operation)
    filecheck(correct, asm)
