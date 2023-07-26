from textwrap import dedent

import numpy as np
import pytest
from mlir_utils.ast.canonicalize import canonicalize

from mlir_utils.dialects.ext.arith import Scalar
from mlir_utils.dialects.ext.scf import (
    range_,
    yield_,
    canonicalizer,
)
from mlir_utils.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    arr = np.random.randint(0, 10, (10, 10))
    ten = Tensor(arr)
    assert np.array_equal(arr, ten.literal_value)


def test_simple_literal_indexing(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)

    w = ten[0]
    w = ten[2, 4]
    w = ten[2, 4, 6]
    w = ten[2, 4, 6, 8]
    assert isinstance(w, Scalar)

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %c0 = arith.constant 0 : index
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %extracted_slice_0 = tensor.extract_slice %0[2, 4, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
      %c2_1 = arith.constant 2 : index
      %c4_2 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %extracted_slice_3 = tensor.extract_slice %0[2, 4, 6, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>
      %c2_4 = arith.constant 2 : index
      %c4_5 = arith.constant 4 : index
      %c6_6 = arith.constant 6 : index
      %c8 = arith.constant 8 : index
      %extracted = tensor.extract %0[%c2_4, %c4_5, %c6_6, %c8] : tensor<10x22x333x4444xi32>
    } 
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)

    w = ten[...]
    assert w == ten
    w = ten[:]
    assert w == ten
    w = ten[:, :]
    assert w == ten
    w = ten[:, :, :]
    assert w == ten
    w = ten[:, :, :, :]
    assert w == ten
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
    } 
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_1(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)
    w = ten[1, ...]
    w = ten[1, :, ...]
    w = ten[1, :, :, ...]

    try:
        w = ten[1, :, :, :, :]
    except IndexError as e:
        assert (
            str(e)
            == "Too many indices for tensor: 5 non-None/Ellipsis indices for dim 4."
        )

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %extracted_slice = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_0 = arith.constant 1 : index
      %extracted_slice_1 = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_2 = arith.constant 1 : index
      %extracted_slice_3 = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_4 = arith.constant 1 : index
    }
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_2(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)
    w = ten[1, :]
    w = ten[1, :, :]
    w = ten[1, :, :, :]
    w = ten[:, 1]
    w = ten[:, :, 1]
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %extracted_slice = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_0 = arith.constant 1 : index
      %extracted_slice_1 = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_2 = arith.constant 1 : index
      %extracted_slice_3 = tensor.extract_slice %0[1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
      %c1_4 = arith.constant 1 : index
      %extracted_slice_5 = tensor.extract_slice %0[0, 1, 0, 0] [10, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x4444xi32>
      %c1_6 = arith.constant 1 : index
      %extracted_slice_7 = tensor.extract_slice %0[0, 0, 1, 0] [10, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_3(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)

    w = ten[:, :, :, 1]
    w = ten[:, 1, :, 1]
    w = ten[1, :, :, 1]
    w = ten[1, 1, :, :]
    w = ten[:, :, 1, 1]
    w = ten[:, 1, 1, :]
    w = ten[1, :, 1, :]
    w = ten[1, 1, :, 1]
    w = ten[1, :, 1, 1]
    w = ten[:, 1, 1, 1]
    w = ten[1, 1, 1, :]

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %c1 = arith.constant 1 : index
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, 1] [10, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x1xi32>
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %extracted_slice_2 = tensor.extract_slice %0[0, 1, 0, 1] [10, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x1xi32>
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %extracted_slice_5 = tensor.extract_slice %0[1, 0, 0, 1] [1, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x1xi32>
      %c1_6 = arith.constant 1 : index
      %c1_7 = arith.constant 1 : index
      %extracted_slice_8 = tensor.extract_slice %0[1, 1, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %extracted_slice_11 = tensor.extract_slice %0[0, 0, 1, 1] [10, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x1xi32>
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %extracted_slice_14 = tensor.extract_slice %0[0, 1, 1, 0] [10, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x4444xi32>
      %c1_15 = arith.constant 1 : index
      %c1_16 = arith.constant 1 : index
      %extracted_slice_17 = tensor.extract_slice %0[1, 0, 1, 0] [1, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x4444xi32>
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c1_20 = arith.constant 1 : index
      %extracted_slice_21 = tensor.extract_slice %0[1, 1, 0, 1] [1, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x1xi32>
      %c1_22 = arith.constant 1 : index
      %c1_23 = arith.constant 1 : index
      %c1_24 = arith.constant 1 : index
      %extracted_slice_25 = tensor.extract_slice %0[1, 0, 1, 1] [1, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x1xi32>
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c1_28 = arith.constant 1 : index
      %extracted_slice_29 = tensor.extract_slice %0[0, 1, 1, 1] [10, 1, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x1xi32>
      %c1_30 = arith.constant 1 : index
      %c1_31 = arith.constant 1 : index
      %c1_32 = arith.constant 1 : index
      %extracted_slice_33 = tensor.extract_slice %0[1, 1, 1, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_none_indices(ctx: MLIRContext):
    ten = empty((10, 22, 333, 4444), i32_t)
    w = ten[None]
    w = ten[:, None]
    w = ten[None, None]
    w = ten[:, :, None]
    w = ten[:, :, :, None]
    w = ten[:, :, :, :, None]
    w = ten[..., None]
    w = ten[:, None, :, :, None]
    w = ten[:, None, None, :, None]
    w = ten[:, None, None, None, None]
    w = ten[None, None, None, None, None]
    try:
        w = ten[None, None, None, None, None, None]
        print(w.owner)
    except IndexError as e:
        assert str(e) == "pop index out of range"
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %expanded = tensor.expand_shape %0 [[0, 1], [2], [3], [4]] : tensor<10x22x333x4444xi32> into tensor<1x10x22x333x4444xi32>
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_0 = tensor.expand_shape %extracted_slice [[0, 1], [2], [3], [4]] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444xi32>
      %extracted_slice_1 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_2 = tensor.expand_shape %extracted_slice_1 [[0, 1, 2], [3], [4], [5]] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x333x4444xi32>
      %extracted_slice_3 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_4 = tensor.expand_shape %extracted_slice_3 [[0], [1, 2], [3], [4]] : tensor<10x22x333x4444xi32> into tensor<10x22x1x333x4444xi32>
      %extracted_slice_5 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_6 = tensor.expand_shape %extracted_slice_5 [[0], [1], [2, 3], [4]] : tensor<10x22x333x4444xi32> into tensor<10x22x333x1x4444xi32>
      %extracted_slice_7 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_8 = tensor.expand_shape %extracted_slice_7 [[0], [1], [2], [3, 4]] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
      %extracted_slice_9 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0], [1], [2], [3, 4]] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
      %extracted_slice_11 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_12 = tensor.expand_shape %extracted_slice_11 [[0, 1], [2], [3], [4, 5]] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444x1xi32>
      %extracted_slice_13 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_14 = tensor.expand_shape %extracted_slice_13 [[0, 1], [2, 3], [4], [5, 6]] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x4444x1xi32>
      %extracted_slice_15 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_16 = tensor.expand_shape %extracted_slice_15 [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x1x4444x1xi32>
      %extracted_slice_17 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
      %expanded_18 = tensor.expand_shape %extracted_slice_17 [[0, 1, 2], [3, 4], [5, 6], [7, 8]] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x1x333x1x4444x1xi32>
      %extracted_slice_19 = tensor.extract_slice %0[0, 0, 0, 0] [10, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices(ctx: MLIRContext):
    ten = empty((7, 22, 333, 4444), i32_t)
    w = ten[:, 0:22:2]
    w = ten[:, 0:22:2, 0:330:30]
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    w = ten[:, :, 100:200:5, 1000:2000:50]
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<7x22x333x4444xi32>
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
      %extracted_slice_0 = tensor.extract_slice %0[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
      %extracted_slice_1 = tensor.extract_slice %0[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
      %extracted_slice_2 = tensor.extract_slice %0[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices_insertion(ctx: MLIRContext):
    ten = empty((7, 22, 333, 4444), i32_t)
    w = ten[:, 0:22:2]
    ten[:, 0:22:2] = w
    w = ten[:, 0:22:2, 0:330:30]
    ten[:, 0:22:2, 0:330:30] = w
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    ten[:, 0:22:2, 0:330:30, 0:4400:400] = w
    w = ten[:, :, 100:200:5, 1000:2000:50]
    ten[:, :, 100:200:5, 1000:2000:50] = w

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<7x22x333x4444xi32>
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
      %inserted_slice = tensor.insert_slice %extracted_slice into %0[0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x11x333x4444xi32> into tensor<7x22x333x4444xi32>
      %extracted_slice_0 = tensor.extract_slice %inserted_slice[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
      %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %inserted_slice[0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x11x11x4444xi32> into tensor<7x22x333x4444xi32>
      %extracted_slice_2 = tensor.extract_slice %inserted_slice_1[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
      %inserted_slice_3 = tensor.insert_slice %extracted_slice_2 into %inserted_slice_1[0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x11x11x11xi32> into tensor<7x22x333x4444xi32>
      %extracted_slice_4 = tensor.extract_slice %inserted_slice_3[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>
      %inserted_slice_5 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_3[0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x20x20xi32> into tensor<7x22x333x4444xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_move_slice(ctx: MLIRContext):
    ten = empty((8, 8), i32_t)
    w = ten[0:4, 0:4]
    ten[4:8, 4:8] = w

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<8x8xi32>
      %extracted_slice = tensor.extract_slice %0[0, 0] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      %inserted_slice = tensor.insert_slice %extracted_slice into %0[4, 4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_fold_1(ctx: MLIRContext):
    ten_arr = np.random.randint(0, 10, (10, 10)).astype(np.int32)
    x_arr = ten_arr + ten_arr
    y_arr = x_arr * x_arr
    z_arr = y_arr - x_arr

    ten = Tensor(ten_arr, fold=True)
    x = ten + ten
    y = x * x
    z = y - x
    assert np.array_equal(z_arr, z.literal_value)
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant dense<{ten_arr.tolist()}> : tensor<10x10xi32>
      %cst_0 = arith.constant dense<{x_arr.tolist()}> : tensor<10x10xi32>
      %cst_1 = arith.constant dense<{y_arr.tolist()}> : tensor<10x10xi32>
      %cst_2 = arith.constant dense<{z_arr.tolist()}> : tensor<10x10xi32>
    }}
    """
    )
    filecheck(correct, ctx.module)


def test_for_loops(ctx: MLIRContext):
    ten = empty((7, 22, 333, 4444), i32_t)
    for i, r1 in range_(0, 10, iter_args=[ten]):
        y = r1 + r1
        yield_(y)

    assert repr(r1) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
    assert r1.owner.name == "scf.for"
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<7x22x333x4444xi32>
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %1 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0) -> (tensor<7x22x333x4444xi32>) {
        %2 = arith.addi %arg1, %arg1 : tensor<7x22x333x4444xi32>
        scf.yield %2 : tensor<7x22x333x4444xi32>
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_for_loops_canonicalizer(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def tenfoo():
        ten = empty((7, 22, 333, 4444), i32_t)
        for i, r1 in range_(0, 10, iter_args=[ten]):
            y = r1 + r1
            yield y

        assert repr(r1) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
        assert r1.owner.name == "scf.for"

    tenfoo()

    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<7x22x333x4444xi32>
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %1 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0) -> (tensor<7x22x333x4444xi32>) {
        %2 = arith.addi %arg1, %arg1 : tensor<7x22x333x4444xi32>
        scf.yield %2 : tensor<7x22x333x4444xi32>
      }
    }
    """
    )

    filecheck(correct, ctx.module)
