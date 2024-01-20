import re
from textwrap import dedent

import pytest
from mlir.ir import MLIRError, Type

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext.arith import Scalar, constant
from mlir.extras.dialects.ext.memref import (
    alloc,
    alloca,
    S,
    alloca_scope,
    alloca_scope_return,
)
from mlir.extras.dialects.ext.scf import (
    range_,
    yield_,
    canonicalizer,
)
from mlir.dialects.memref import subview

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_simple_literal_indexing(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())

    w = mem[2, 4, 6, 8]
    assert isinstance(w, Scalar)

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c8 = arith.constant 8 : index
      %0 = memref.load %alloc[%c2, %c4, %c6, %c8] : memref<10x22x333x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_simple_literal_indexing_alloca(ctx: MLIRContext):
    @alloca_scope([])
    def demo_scope2():
        mem = alloca((10, 22, 333, 4444), T.i32())

        w = mem[2, 4, 6, 8]
        assert isinstance(w, Scalar)
        alloca_scope_return([])

    correct = dedent(
        """\
    module {
      memref.alloca_scope  {
        %alloca = memref.alloca() : memref<10x22x333x4444xi32>
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c6 = arith.constant 6 : index
        %c8 = arith.constant 8 : index
        %0 = memref.load %alloca[%c2, %c4, %c6, %c8] : memref<10x22x333x4444xi32>
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())

    w = mem[...]
    assert w == mem
    w = mem[:]
    assert w == mem
    w = mem[:, :]
    assert w == mem
    w = mem[:, :, :]
    assert w == mem
    w = mem[:, :, :, :]
    assert w == mem
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
    } 
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_1(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())
    w = mem[1, ...]
    w = mem[1, :, ...]
    w = mem[1, :, :, ...]

    try:
        w = mem[1, :, :, :, :]
    except IndexError as e:
        assert (
            str(e)
            == "Too many indices for shaped type with rank: 5 non-None/Ellipsis indices for dim 4."
        )

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_0 = arith.constant 1 : index
      %subview_1 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_2 = arith.constant 1 : index
      %subview_3 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_4 = arith.constant 1 : index
    }
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_2(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())
    w = mem[1, :]
    w = mem[1, :, :]
    w = mem[1, :, :, :]
    w = mem[:, 1]
    w = mem[:, :, 1]
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_0 = arith.constant 1 : index
      %subview_1 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_2 = arith.constant 1 : index
      %subview_3 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556744>>
      %c1_4 = arith.constant 1 : index
      %subview_5 = memref.subview %alloc[0, 1, 0, 0] [10, 1, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 1479852>>
      %c1_6 = arith.constant 1 : index
      %subview_7 = memref.subview %alloc[0, 0, 1, 0] [10, 22, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x1x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 4444>>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_3(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())

    w = mem[:, :, :, 1]
    w = mem[:, 1, :, 1]
    w = mem[1, :, :, 1]
    w = mem[1, 1, :, :]
    w = mem[:, :, 1, 1]
    w = mem[:, 1, 1, :]
    w = mem[1, :, 1, :]
    w = mem[1, 1, :, 1]
    w = mem[1, :, 1, 1]
    w = mem[:, 1, 1, 1]
    w = mem[1, 1, 1, :]

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[0, 0, 0, 1] [10, 22, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 1>>
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %subview_2 = memref.subview %alloc[0, 1, 0, 1] [10, 1, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x333x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 1479853>>
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %subview_5 = memref.subview %alloc[1, 0, 0, 1] [1, 22, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 32556745>>
      %c1_6 = arith.constant 1 : index
      %c1_7 = arith.constant 1 : index
      %subview_8 = memref.subview %alloc[1, 1, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x333x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 34036596>>
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %subview_11 = memref.subview %alloc[0, 0, 1, 1] [10, 22, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x1x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 4445>>
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %subview_14 = memref.subview %alloc[0, 1, 1, 0] [10, 1, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x1x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 1484296>>
      %c1_15 = arith.constant 1 : index
      %c1_16 = arith.constant 1 : index
      %subview_17 = memref.subview %alloc[1, 0, 1, 0] [1, 22, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x1x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 32561188>>
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c1_20 = arith.constant 1 : index
      %subview_21 = memref.subview %alloc[1, 1, 0, 1] [1, 1, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x333x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 34036597>>
      %c1_22 = arith.constant 1 : index
      %c1_23 = arith.constant 1 : index
      %c1_24 = arith.constant 1 : index
      %subview_25 = memref.subview %alloc[1, 0, 1, 1] [1, 22, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x1x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 32561189>>
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c1_28 = arith.constant 1 : index
      %subview_29 = memref.subview %alloc[0, 1, 1, 1] [10, 1, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x1x1xi32, strided<[32556744, 1479852, 4444, 1], offset: 1484297>>
      %c1_30 = arith.constant 1 : index
      %c1_31 = arith.constant 1 : index
      %c1_32 = arith.constant 1 : index
      %subview_33 = memref.subview %alloc[1, 1, 1, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x1x4444xi32, strided<[32556744, 1479852, 4444, 1], offset: 34041040>>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_none_indices(ctx: MLIRContext):
    mem = alloc((10, 22, 333, 4444), T.i32())
    w = mem[None]
    w = mem[:, None]
    w = mem[None, None]
    w = mem[:, :, None]
    w = mem[:, :, :, None]
    w = mem[:, :, :, :, None]
    w = mem[..., None]
    w = mem[:, None, :, :, None]
    w = mem[:, None, None, :, None]
    w = mem[:, None, None, None, None]
    w = mem[None, None, None, None, None]
    try:
        w = mem[None, None, None, None, None, None]
        print(w.owner)
    except IndexError as e:
        assert str(e) == "pop index out of range"
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2], [3], [4]] : memref<10x22x333x4444xi32> into memref<1x10x22x333x4444xi32>
      %subview = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_0 = memref.expand_shape %subview [[0, 1], [2], [3], [4]] : memref<10x22x333x4444xi32> into memref<10x1x22x333x4444xi32>
      %subview_1 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1, 2], [3], [4], [5]] : memref<10x22x333x4444xi32> into memref<1x10x1x22x333x4444xi32>
      %subview_3 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_4 = memref.expand_shape %subview_3 [[0], [1, 2], [3], [4]] : memref<10x22x333x4444xi32> into memref<10x22x1x333x4444xi32>
      %subview_5 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_6 = memref.expand_shape %subview_5 [[0], [1], [2, 3], [4]] : memref<10x22x333x4444xi32> into memref<10x22x333x1x4444xi32>
      %subview_7 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_8 = memref.expand_shape %subview_7 [[0], [1], [2], [3, 4]] : memref<10x22x333x4444xi32> into memref<10x22x333x4444x1xi32>
      %subview_9 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_10 = memref.expand_shape %subview_9 [[0], [1], [2], [3, 4]] : memref<10x22x333x4444xi32> into memref<10x22x333x4444x1xi32>
      %subview_11 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_12 = memref.expand_shape %subview_11 [[0, 1], [2], [3], [4, 5]] : memref<10x22x333x4444xi32> into memref<10x1x22x333x4444x1xi32>
      %subview_13 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_14 = memref.expand_shape %subview_13 [[0, 1], [2, 3], [4], [5, 6]] : memref<10x22x333x4444xi32> into memref<10x1x22x1x333x4444x1xi32>
      %subview_15 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_16 = memref.expand_shape %subview_15 [[0, 1], [2, 3], [4, 5], [6, 7]] : memref<10x22x333x4444xi32> into memref<10x1x22x1x333x1x4444x1xi32>
      %subview_17 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
      %expand_shape_18 = memref.expand_shape %subview_17 [[0, 1, 2], [3, 4], [5, 6], [7, 8]] : memref<10x22x333x4444xi32> into memref<1x10x1x22x1x333x1x4444x1xi32>
      %subview_19 = memref.subview %alloc[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices(ctx: MLIRContext):
    mem = alloc((7, 22, 333, 4444), T.i32())
    w = mem[:, 0:22:2]
    w = mem[:, 0:22:2, 0:330:30]
    w = mem[:, 0:22:2, 0:330:30, 0:4400:400]
    w = mem[:, :, 100:200:5, 1000:2000:50]
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<7x22x333x4444xi32>
      %subview = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32>
      %subview_0 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32>
      %subview_1 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32>
      %subview_2 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<[32556744, 1479852, 22220, 50], offset: 445400>>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices_insertion(ctx: MLIRContext):
    mem = alloc((7, 22, 333, 4444), T.i32())
    w = mem[:, 0:22:2]
    mem[:, 0:22:2] = w
    w = mem[:, 0:22:2, 0:330:30]
    mem[:, 0:22:2, 0:330:30] = w
    w = mem[:, 0:22:2, 0:330:30, 0:4400:400]
    mem[:, 0:22:2, 0:330:30, 0:4400:400] = w
    w = mem[:, :, 100:200:5, 1000:2000:50]
    mem[:, :, 100:200:5, 1000:2000:50] = w

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<7x22x333x4444xi32>
      %subview = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32>
      %subview_0 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32>
      memref.copy %subview, %subview_0 : memref<7x11x333x4444xi32> to memref<7x11x333x4444xi32>
      %subview_1 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32>
      %subview_2 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32>
      memref.copy %subview_1, %subview_2 : memref<7x11x11x4444xi32> to memref<7x11x11x4444xi32>
      %subview_3 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32>
      %subview_4 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32>
      memref.copy %subview_3, %subview_4 : memref<7x11x11x11xi32> to memref<7x11x11x11xi32>
      %subview_5 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<[32556744, 1479852, 22220, 50], offset: 445400>>
      %subview_6 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<[32556744, 1479852, 22220, 50], offset: 445400>>
      memref.copy %subview_5, %subview_6 : memref<7x22x20x20xi32, strided<[32556744, 1479852, 22220, 50], offset: 445400>> to memref<7x22x20x20xi32, strided<[32556744, 1479852, 22220, 50], offset: 445400>>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_move_slice(ctx: MLIRContext):
    mem = alloc((8, 8), T.i32())
    w = mem[0:4, 0:4]
    mem[4:8, 4:8] = w

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<8x8xi32>
      %subview = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<8x8xi32> to memref<4x4xi32>
      %subview_0 = memref.subview %alloc[4, 4] [4, 4] [1, 1] : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: 36>>
      memref.copy %subview, %subview_0 : memref<4x4xi32> to memref<4x4xi32, strided<[8, 1], offset: 36>>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_for_loops(ctx: MLIRContext):
    mem = alloc((10, 10), T.i32())
    for i, it_mem in range_(0, 10, iter_args=[mem]):
        it_mem[i, i] = it_mem[i, i] + it_mem[i, i]
        res = yield_(it_mem)

    assert repr(res) == "MemRef(%0, memref<10x10xi32>)"
    assert res.owner.name == "scf.for"
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %alloc) -> (memref<10x10xi32>) {
        %1 = memref.load %arg1[%arg0, %arg0] : memref<10x10xi32>
        %2 = memref.load %arg1[%arg0, %arg0] : memref<10x10xi32>
        %3 = arith.addi %1, %2 : i32
        memref.store %3, %arg1[%arg0, %arg0] : memref<10x10xi32>
        scf.yield %arg1 : memref<10x10xi32>
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_for_loops_canonicalizer(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def tenfoo():
        mem = alloc((10, 10), T.i32())
        for i, it_mem in range_(0, 10, iter_args=[mem]):
            it_mem[i, i] = it_mem[i, i] + it_mem[i, i]
            res = yield it_mem

        assert repr(res) == "MemRef(%0, memref<10x10xi32>)"
        assert res.owner.name == "scf.for"

    tenfoo()

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %alloc) -> (memref<10x10xi32>) {
        %1 = memref.load %arg1[%arg0, %arg0] : memref<10x10xi32>
        %2 = memref.load %arg1[%arg0, %arg0] : memref<10x10xi32>
        %3 = arith.addi %1, %2 : i32
        memref.store %3, %arg1[%arg0, %arg0] : memref<10x10xi32>
        scf.yield %arg1 : memref<10x10xi32>
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_subview_mixed_offsets(ctx: MLIRContext):
    def tenfoo():
        mem = alloc((10, 10), T.i32())
        i, j = constant(0, index=True), constant(0, index=True)
        v = subview(
            T.memref(5, 5, T.i32()),
            mem,
            offsets=[i, j],
            sizes=[],
            strides=[],
            static_offsets=[S, S],
            static_sizes=[5, 5],
            static_strides=[1, 1],
        )
        try:
            v.owner.verify()
        except MLIRError as e:
            diag = str(e.error_diagnostics[0]).strip()
            correct_type = re.findall(r"'memref<(.*)>'", diag)
            assert len(correct_type) == 1
            correct_type = Type.parse(f"memref<{correct_type[0]}>")
            v.owner.erase()
            v = subview(
                correct_type,
                mem,
                offsets=[i, j],
                sizes=[],
                strides=[],
                static_offsets=[S, S],
                static_sizes=[5, 5],
                static_strides=[1, 1],
            )

    tenfoo()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %subview = memref.subview %alloc[%c0, %c0_0] [5, 5] [1, 1] : memref<10x10xi32> to memref<5x5xi32, strided<[10, 1], offset: ?>>
    }
    """
    )

    filecheck(correct, ctx.module)
