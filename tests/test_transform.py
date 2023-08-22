from textwrap import dedent

import pytest

from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.ext.scf import (
    range_,
    canonicalizer,
)
from mlir_utils.dialects.ext.transform import (
    sequence,
    unroll,
    get_parent_for,
    structured_match,
)
from mlir_utils.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic_unroll(ctx: MLIRContext):
    @func
    @canonicalize(using=canonicalizer)
    def loop_unroll_op():
        for i in range_(0, 42, 5):
            v = i + i

    loop_unroll_op.emit()

    @sequence(target_tag="basic")
    def basic(target):
        m = structured_match(target, ["arith.addi"])
        loop = get_parent_for(m)
        unroll(loop, 4)

    correct = dedent(
        """\
    module {
      func.func @loop_unroll_op() {
        %c0 = arith.constant 0 : index
        %c42 = arith.constant 42 : index
        %c5 = arith.constant 5 : index
        scf.for %arg0 = %c0 to %c42 step %c5 {
          %0 = arith.addi %arg0, %arg0 : index
        }
        return
      }
      transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
      ^bb0(%arg0: !pdl.operation):
        %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!pdl.operation) -> !transform.any_op
        %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !pdl.operation
        transform.loop.unroll %1 {factor = 4 : i64} : !pdl.operation
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    run_pipeline(
        ctx.module,
        Pipeline()
        .add_pass("test-transform-dialect-interpreter")
        .add_pass("test-transform-dialect-erase-schedule"),
    )

    correct = dedent(
        """\
    module {
      func.func @loop_unroll_op() {
        %c0 = arith.constant 0 : index
        %c42 = arith.constant 42 : index
        %c5 = arith.constant 5 : index
        %c40 = arith.constant 40 : index
        %c20 = arith.constant 20 : index
        scf.for %arg0 = %c0 to %c40 step %c20 {
          %1 = arith.addi %arg0, %arg0 : index
          %c1 = arith.constant 1 : index
          %2 = arith.muli %c5, %c1 : index
          %3 = arith.addi %arg0, %2 : index
          %4 = arith.addi %3, %3 : index
          %c2 = arith.constant 2 : index
          %5 = arith.muli %c5, %c2 : index
          %6 = arith.addi %arg0, %5 : index
          %7 = arith.addi %6, %6 : index
          %c3 = arith.constant 3 : index
          %8 = arith.muli %c5, %c3 : index
          %9 = arith.addi %arg0, %8 : index
          %10 = arith.addi %9, %9 : index
        }
        %0 = arith.addi %c40, %c40 : index
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)
