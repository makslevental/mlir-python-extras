import platform
import re
from textwrap import dedent

import numpy as np
import pytest

import mlir.extras.types as T
from mlir.dialects.memref import subview
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import memref
from mlir.extras.dialects.ext.arith import Scalar, constant
from mlir.extras.dialects.ext.memref import (
    alloc,
    alloca,
    alloca_scope,
    alloca_scope_return,
    global_,
)
from mlir.extras.dialects.ext.scf import (
    range_,
    yield_,
    canonicalizer,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir.ir import MLIRError, Type

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def get_np_view_offset(np_view):
    return np_view.ctypes.data - np_view.base.ctypes.data


def test_simple_literal_indexing(ctx: MLIRContext):
    mem = alloc(10, 22, 333, 4444, T.i32())

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


def test_simple_slicing(ctx: MLIRContext):
    mem = alloc(10, T.i32())

    w = mem[5:]
    w = mem[:5]

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10xi32>
      %subview = memref.subview %alloc[5] [5] [1] : memref<10xi32> to memref<5xi32, strided<[1], offset: 5>>
      %subview_0 = memref.subview %alloc[0] [5] [1] : memref<10xi32> to memref<5xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_simple_literal_indexing_alloca(ctx: MLIRContext):
    @alloca_scope([])
    def demo_scope2():
        mem = alloca(10, 22, 333, 4444, T.i32())

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
    mem = alloc(10, 22, 333, 4444, T.i32())

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
    sizes = (10, 22, 333, 4444)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[1:2, ...]
    golden_w_2 = golden_mem[1:2, :, ...]
    golden_w_3 = golden_mem[1:2, :, :, ...]

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_3_strides = (np.array(golden_w_3.strides) // dtype_size_in_bytes).tolist()

    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes
    golden_w_3_offset = get_np_view_offset(golden_w_3) // dtype_size_in_bytes

    mem = alloc(*sizes, T.i32())
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
        f"""\
    module {{
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_1_strides}, offset: {golden_w_1_offset}>>
      %c1_0 = arith.constant 1 : index
      %subview_1 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_2_strides}, offset: {golden_w_2_offset}>>
      %c1_2 = arith.constant 1 : index
      %subview_3 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_3_strides}, offset: {golden_w_3_offset}>>
      %c1_4 = arith.constant 1 : index
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_2(ctx: MLIRContext):
    sizes = (10, 22, 333, 4444)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[1:2, :]
    golden_w_2 = golden_mem[1:2, :, :]
    golden_w_3 = golden_mem[1:2, :, :, :]
    golden_w_4 = golden_mem[:, 1:2]
    golden_w_5 = golden_mem[:, :, 1:2]

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_3_strides = (np.array(golden_w_3.strides) // dtype_size_in_bytes).tolist()
    golden_w_4_strides = (np.array(golden_w_4.strides) // dtype_size_in_bytes).tolist()
    golden_w_5_strides = (np.array(golden_w_5.strides) // dtype_size_in_bytes).tolist()

    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes
    golden_w_3_offset = get_np_view_offset(golden_w_3) // dtype_size_in_bytes
    golden_w_4_offset = get_np_view_offset(golden_w_4) // dtype_size_in_bytes
    golden_w_5_offset = get_np_view_offset(golden_w_5) // dtype_size_in_bytes

    mem = alloc(*sizes, T.i32())
    w = mem[1, :]
    w = mem[1, :, :]
    w = mem[1, :, :, :]
    w = mem[:, 1]
    w = mem[:, :, 1]
    correct = dedent(
        f"""\
    module {{
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_1_strides}, offset: {golden_w_1_offset}>>
      %c1_0 = arith.constant 1 : index
      %subview_2 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_2_strides}, offset: {golden_w_2_offset}>>
      %c1_2 = arith.constant 1 : index
      %subview_3 = memref.subview %alloc[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x4444xi32, strided<{golden_w_3_strides}, offset: {golden_w_3_offset}>>
      %c1_4 = arith.constant 1 : index
      %subview_5 = memref.subview %alloc[0, 1, 0, 0] [10, 1, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x333x4444xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
      %c1_6 = arith.constant 1 : index
      %subview_7 = memref.subview %alloc[0, 0, 1, 0] [10, 22, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x1x4444xi32, strided<{golden_w_5_strides}, offset: {golden_w_5_offset}>>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_3(ctx: MLIRContext):
    sizes = (10, 22, 333, 4444)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[:, :, :, 1:2]
    golden_w_2 = golden_mem[:, 1:2, :, 1:2]
    golden_w_3 = golden_mem[1:2, :, :, 1:2]
    golden_w_4 = golden_mem[1:2, 1:2, :, :]
    golden_w_5 = golden_mem[:, :, 1:2, 1:2]
    golden_w_6 = golden_mem[:, 1:2, 1:2, :]
    golden_w_7 = golden_mem[1:2, :, 1:2, :]
    golden_w_8 = golden_mem[1:2, 1:2, :, 1:2]
    golden_w_9 = golden_mem[1:2, :, 1:2, 1:2]
    golden_w_10 = golden_mem[:, 1:2, 1:2, 1:2]
    golden_w_11 = golden_mem[1:2, 1:2, 1:2, :]

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_3_strides = (np.array(golden_w_3.strides) // dtype_size_in_bytes).tolist()
    golden_w_4_strides = (np.array(golden_w_4.strides) // dtype_size_in_bytes).tolist()
    golden_w_5_strides = (np.array(golden_w_5.strides) // dtype_size_in_bytes).tolist()
    golden_w_6_strides = (np.array(golden_w_6.strides) // dtype_size_in_bytes).tolist()
    golden_w_7_strides = (np.array(golden_w_7.strides) // dtype_size_in_bytes).tolist()
    golden_w_8_strides = (np.array(golden_w_8.strides) // dtype_size_in_bytes).tolist()
    golden_w_9_strides = (np.array(golden_w_9.strides) // dtype_size_in_bytes).tolist()
    golden_w_10_strides = (
        np.array(golden_w_10.strides) // dtype_size_in_bytes
    ).tolist()
    golden_w_11_strides = (
        np.array(golden_w_11.strides) // dtype_size_in_bytes
    ).tolist()

    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes
    golden_w_3_offset = get_np_view_offset(golden_w_3) // dtype_size_in_bytes
    golden_w_4_offset = get_np_view_offset(golden_w_4) // dtype_size_in_bytes
    golden_w_5_offset = get_np_view_offset(golden_w_5) // dtype_size_in_bytes
    golden_w_6_offset = get_np_view_offset(golden_w_6) // dtype_size_in_bytes
    golden_w_7_offset = get_np_view_offset(golden_w_7) // dtype_size_in_bytes
    golden_w_8_offset = get_np_view_offset(golden_w_8) // dtype_size_in_bytes
    golden_w_9_offset = get_np_view_offset(golden_w_9) // dtype_size_in_bytes
    golden_w_10_offset = get_np_view_offset(golden_w_10) // dtype_size_in_bytes
    golden_w_11_offset = get_np_view_offset(golden_w_11) // dtype_size_in_bytes

    mem = alloc(*sizes, T.i32())

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
        f"""\
    module {{
      %alloc = memref.alloc() : memref<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %subview = memref.subview %alloc[0, 0, 0, 1] [10, 22, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x333x1xi32, strided<{golden_w_1_strides}, offset: {golden_w_1_offset}>>
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %subview_2 = memref.subview %alloc[0, 1, 0, 1] [10, 1, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x333x1xi32, strided<{golden_w_2_strides}, offset: {golden_w_2_offset}>>
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %subview_3 = memref.subview %alloc[1, 0, 0, 1] [1, 22, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x333x1xi32, strided<{golden_w_3_strides}, offset: {golden_w_3_offset}>>
      %c1_6 = arith.constant 1 : index
      %c1_7 = arith.constant 1 : index
      %subview_8 = memref.subview %alloc[1, 1, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x333x4444xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %subview_11 = memref.subview %alloc[0, 0, 1, 1] [10, 22, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x22x1x1xi32, strided<{golden_w_5_strides}, offset: {golden_w_5_offset}>>
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %subview_14 = memref.subview %alloc[0, 1, 1, 0] [10, 1, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x1x4444xi32, strided<{golden_w_6_strides}, offset: {golden_w_6_offset}>>
      %c1_15 = arith.constant 1 : index
      %c1_16 = arith.constant 1 : index
      %subview_17 = memref.subview %alloc[1, 0, 1, 0] [1, 22, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x1x4444xi32, strided<{golden_w_7_strides}, offset: {golden_w_7_offset}>>
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c1_20 = arith.constant 1 : index
      %subview_21 = memref.subview %alloc[1, 1, 0, 1] [1, 1, 333, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x333x1xi32, strided<{golden_w_8_strides}, offset: {golden_w_8_offset}>>
      %c1_22 = arith.constant 1 : index
      %c1_23 = arith.constant 1 : index
      %c1_24 = arith.constant 1 : index
      %subview_25 = memref.subview %alloc[1, 0, 1, 1] [1, 22, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x22x1x1xi32, strided<{golden_w_9_strides}, offset: {golden_w_9_offset}>>
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c1_28 = arith.constant 1 : index
      %subview_29 = memref.subview %alloc[0, 1, 1, 1] [10, 1, 1, 1] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<10x1x1x1xi32, strided<{golden_w_10_strides}, offset: {golden_w_10_offset}>>
      %c1_30 = arith.constant 1 : index
      %c1_31 = arith.constant 1 : index
      %c1_32 = arith.constant 1 : index
      %subview_33 = memref.subview %alloc[1, 1, 1, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : memref<10x22x333x4444xi32> to memref<1x1x1x4444xi32, strided<{golden_w_11_strides}, offset: {golden_w_11_offset}>>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_none_indices(ctx: MLIRContext):
    mem = alloc(10, 22, 333, 4444, T.i32())
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
    sizes = (7, 22, 333, 4444)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[:, 0:22:2]
    golden_w_2 = golden_mem[:, 0:22:2, 0:330:30]
    golden_w_3 = golden_mem[:, 0:22:2, 0:330:30, 0:4400:400]
    golden_w_4 = golden_mem[:, :, 100:200:5, 1000:2000:50]

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_3_strides = (np.array(golden_w_3.strides) // dtype_size_in_bytes).tolist()
    golden_w_4_strides = (np.array(golden_w_4.strides) // dtype_size_in_bytes).tolist()

    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes
    golden_w_3_offset = get_np_view_offset(golden_w_3) // dtype_size_in_bytes
    golden_w_4_offset = get_np_view_offset(golden_w_4) // dtype_size_in_bytes

    assert golden_w_1_offset == golden_w_2_offset == golden_w_3_offset == 0

    mem = alloc(*sizes, T.i32())
    w = mem[:, 0:22:2]
    w = mem[:, 0:22:2, 0:330:30]
    w = mem[:, 0:22:2, 0:330:30, 0:4400:400]
    w = mem[:, :, 100:200:5, 1000:2000:50]
    correct = dedent(
        f"""\
    module {{
      %alloc = memref.alloc() : memref<7x22x333x4444xi32>
      %subview = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32, strided<{golden_w_1_strides}>>
      %subview_0 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32, strided<{golden_w_2_strides}>>
      %subview_1 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32, strided<{golden_w_3_strides}>>
      %subview_2 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices_insertion(ctx: MLIRContext):
    sizes = (7, 22, 333, 4444)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[:, 0:22:2]
    golden_mem[:, 0:22:2] = golden_w_1
    golden_w_2 = golden_mem[:, 0:22:2, 0:330:30]
    golden_mem[:, 0:22:2, 0:330:30] = golden_w_2
    golden_w_3 = golden_mem[:, 0:22:2, 0:330:30, 0:4400:400]
    golden_mem[:, 0:22:2, 0:330:30, 0:4400:400] = golden_w_3
    golden_w_4 = golden_mem[:, :, 100:200:5, 1000:2000:50]
    golden_mem[:, :, 100:200:5, 1000:2000:50] = golden_w_4

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_3_strides = (np.array(golden_w_3.strides) // dtype_size_in_bytes).tolist()
    golden_w_4_strides = (np.array(golden_w_4.strides) // dtype_size_in_bytes).tolist()

    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes
    golden_w_3_offset = get_np_view_offset(golden_w_3) // dtype_size_in_bytes
    golden_w_4_offset = get_np_view_offset(golden_w_4) // dtype_size_in_bytes

    assert golden_w_1_offset == golden_w_2_offset == golden_w_3_offset == 0

    mem = alloc(*sizes, T.i32())
    w = mem[:, 0:22:2]
    mem[:, 0:22:2] = w
    w = mem[:, 0:22:2, 0:330:30]
    mem[:, 0:22:2, 0:330:30] = w
    w = mem[:, 0:22:2, 0:330:30, 0:4400:400]
    mem[:, 0:22:2, 0:330:30, 0:4400:400] = w
    w = mem[:, :, 100:200:5, 1000:2000:50]
    mem[:, :, 100:200:5, 1000:2000:50] = w

    correct = dedent(
        f"""\
    module {{
      %alloc = memref.alloc() : memref<7x22x333x4444xi32>
      %subview = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32, strided<{golden_w_1_strides}>>
      %subview_0 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : memref<7x22x333x4444xi32> to memref<7x11x333x4444xi32, strided<{golden_w_1_strides}>>
      memref.copy %subview, %subview_0 : memref<7x11x333x4444xi32, strided<{golden_w_1_strides}>> to memref<7x11x333x4444xi32, strided<{golden_w_1_strides}>>
      %subview_1 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32, strided<{golden_w_2_strides}>>
      %subview_2 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : memref<7x22x333x4444xi32> to memref<7x11x11x4444xi32, strided<{golden_w_2_strides}>>
      memref.copy %subview_1, %subview_2 : memref<7x11x11x4444xi32, strided<{golden_w_2_strides}>> to memref<7x11x11x4444xi32, strided<{golden_w_2_strides}>>
      %subview_3 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32, strided<{golden_w_3_strides}>>
      %subview_4 = memref.subview %alloc[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : memref<7x22x333x4444xi32> to memref<7x11x11x11xi32, strided<{golden_w_3_strides}>>
      memref.copy %subview_3, %subview_4 : memref<7x11x11x11xi32, strided<{golden_w_3_strides}>> to memref<7x11x11x11xi32, strided<{golden_w_3_strides}>>
      %subview_5 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
      %subview_6 = memref.subview %alloc[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : memref<7x22x333x4444xi32> to memref<7x22x20x20xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
      memref.copy %subview_5, %subview_6 : memref<7x22x20x20xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>> to memref<7x22x20x20xi32, strided<{golden_w_4_strides}, offset: {golden_w_4_offset}>>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_move_slice(ctx: MLIRContext):
    sizes = (8, 8)
    dtype_size_in_bytes = np.int32().dtype.itemsize
    golden_mem = np.zeros(sizes, dtype=np.int32)
    golden_w_1 = golden_mem[0:4, 0:4]
    golden_w_2 = golden_mem[4:8, 4:8]
    golden_w_2[:, :] = golden_w_1

    golden_w_1_strides = (np.array(golden_w_1.strides) // dtype_size_in_bytes).tolist()
    golden_w_1_offset = get_np_view_offset(golden_w_1) // dtype_size_in_bytes
    assert golden_w_1_offset == 0
    golden_w_2_strides = (np.array(golden_w_2.strides) // dtype_size_in_bytes).tolist()
    golden_w_2_offset = get_np_view_offset(golden_w_2) // dtype_size_in_bytes

    mem = alloc(*sizes, T.i32())
    w = mem[0:4, 0:4]
    mem[4:8, 4:8] = w

    correct = dedent(
        f"""\
    module {{
      %alloc = memref.alloc() : memref<8x8xi32>
      %subview = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<8x8xi32> to memref<4x4xi32, strided<{golden_w_1_strides}>>
      %subview_0 = memref.subview %alloc[4, 4] [4, 4] [1, 1] : memref<8x8xi32> to memref<4x4xi32, strided<{golden_w_2_strides}, offset: {golden_w_2_offset}>>
      memref.copy %subview, %subview_0 : memref<4x4xi32, strided<{golden_w_1_strides}>> to memref<4x4xi32, strided<{golden_w_2_strides}, offset: {golden_w_2_offset}>>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_for_loops(ctx: MLIRContext):
    mem = alloc(10, 10, T.i32())
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
        mem = alloc(10, 10, T.i32())
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
        mem = alloc(10, 10, T.i32())
        i, j = constant(0, index=True), constant(0, index=True)
        v = subview(
            mem,
            offsets=[i, j],
            sizes=[5, 5],
            strides=[1, 1],
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
                mem,
                offsets=[i, j],
                sizes=[5, 5],
                strides=[1, 1],
                result_type=correct_type,
            )

    tenfoo()
    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %subview = memref.subview %alloc[0, 0] [5, 5] [1, 1] : memref<10x10xi32> to memref<5x5xi32, strided<[10, 1]>>
    }
    """
    )

    filecheck(correct, ctx.module)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="On windows int64 is inferred to be i64 ",
)
def test_memref_global_windows(ctx: MLIRContext):
    k = 32
    weight1 = global_(np.ones((k,), dtype=np.int32))
    weight2 = global_(np.ones((k,), dtype=np.int64))
    weight3 = global_(np.ones((k,), dtype=np.float32))
    weight4 = global_(np.ones((k,), dtype=np.float64))
    weight5 = memref.global_(np.ones((k,), dtype=np.int16))
    weight6 = memref.global_(np.ones((k,), dtype=np.float16))
    print(ctx.module)

    correct = dedent(
        """\
    module {
      memref.global "private" constant @weight1 : memref<32xi32> = dense<1>
      memref.global "private" constant @weight2 : memref<32xi64> = dense<1>
      memref.global "private" constant @weight3 : memref<32xf32> = dense<1.000000e+00>
      memref.global "private" constant @weight4 : memref<32xf64> = dense<1.000000e+00>
      memref.global "private" constant @weight5 : memref<32xi16> = dense<1>
      memref.global "private" constant @weight6 : memref<32xf16> = dense<1.000000e+00>
    }
    """
    )

    filecheck(correct, ctx.module)


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="On linux/mac int64 is inferred to be index (through np.longlong)",
)
def test_memref_global_non_windows(ctx: MLIRContext):
    k = 32
    weight1 = global_(np.ones((k,), dtype=np.int32))
    weight2 = global_(np.ones((k,), dtype=np.int64))
    weight3 = global_(np.ones((k,), dtype=np.float32))
    weight4 = global_(np.ones((k,), dtype=np.float64))
    weight5 = memref.global_(np.ones((k,), dtype=np.int16))
    weight6 = memref.global_(np.ones((k,), dtype=np.float16))
    print(ctx.module)

    correct = dedent(
        """\
    module {
      memref.global "private" constant @weight1 : memref<32xi32> = dense<1>
      memref.global "private" constant @weight2 : memref<32xindex> = dense<1>
      memref.global "private" constant @weight3 : memref<32xf32> = dense<1.000000e+00>
      memref.global "private" constant @weight4 : memref<32xf64> = dense<1.000000e+00>
      memref.global "private" constant @weight5 : memref<32xi16> = dense<1>
      memref.global "private" constant @weight6 : memref<32xf16> = dense<1.000000e+00>
    }
    """
    )

    filecheck(correct, ctx.module)


def test_memref_view(ctx: MLIRContext):
    m, k, n = 16, 16, 16
    dtype = T.f32()
    byte_width_dtype = dtype.width // 8
    ab_buffer = alloc((m * k + k * n) * byte_width_dtype, T.i8())
    a_buffer = memref.view(ab_buffer, (m, k), dtype=dtype)
    b_buffer = memref.view(ab_buffer, (k, n), dtype=dtype, shift=m * k)

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<2048xi8>
      %c0 = arith.constant 0 : index
      %view = memref.view %alloc[%c0][] : memref<2048xi8> to memref<16x16xf32>
      %c1024 = arith.constant 1024 : index
      %view_0 = memref.view %alloc[%c1024][] : memref<2048xi8> to memref<16x16xf32>
    }
    """
    )
    filecheck(correct, ctx.module)
