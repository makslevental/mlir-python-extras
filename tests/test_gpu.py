from textwrap import dedent

import pytest

import mlir_utils.types as T
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.gpu import thread_attr as thread
from mlir_utils.dialects.ext.memref import alloc
from mlir_utils.dialects.ext.memref import load, store
from mlir_utils.dialects.ext.scf import forall, in_parallel_
from mlir_utils.dialects.gpu import host_register
from mlir_utils.dialects.math import fma
from mlir_utils.dialects.memref import cast

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic(ctx: MLIRContext):
    unranked_memref_f32 = T.memref(element_type=T.f32)
    mem = cast(unranked_memref_f32, alloc((10, 10), element_type=T.f32))
    host_register(mem)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xf32>
      %cast = memref.cast %alloc : memref<10x10xf32> to memref<*xf32>
      gpu.host_register %cast : memref<*xf32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_forall_insert_slice_no_region_with_for_with_gpu_mapping(ctx: MLIRContext):
    x = alloc((10, 10), T.f32)
    y = alloc((10, 10), T.f32)
    alpha = constant(1, T.f32)

    for i, j in forall(
        [1, 1],
        [2, 2],
        [3, 3],
        device_mapping=[thread("x"), thread("y")],
    ):
        a = load(x, (i, j))
        b = load(y, (i, j))
        c = fma(alpha, a, b)
        store(c, y, (i, j))

        in_parallel_()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xf32>
      %alloc_0 = memref.alloc() : memref<10x10xf32>
      %cst = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2_2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c3_3 = arith.constant 3 : index
      scf.forall (%arg0, %arg1) = (1, 1) to (2, 2) step (3, 3) {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<10x10xf32>
        %1 = memref.load %alloc_0[%arg0, %arg1] : memref<10x10xf32>
        %2 = math.fma %cst, %0, %1 : f32
        memref.store %2, %alloc_0[%arg0, %arg1] : memref<10x10xf32>
      } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
    }
    """
    )
    filecheck(correct, ctx.module)
