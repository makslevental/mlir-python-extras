import platform
from textwrap import dedent

import numpy as np
import pytest

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext import arith
from mlir.extras.dialects.ext.arith import Scalar
from mlir.extras.dialects.ext.scf import (
    range_,
    yield_,
    canonicalizer,
)
from mlir.extras.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)
import mlir.extras.types as T

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    arr = np.random.randint(0, 10, (10, 10))
    ten = Tensor(arr)
    assert np.array_equal(arr, ten.literal_value)


def test_simple_literal_indexing(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

    w = ten[0]
    w = ten[2, 4]
    w = ten[2, 4, 6]
    w = ten[2, 4, 6, 8]
    assert isinstance(w, Scalar)

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_0]][2, 4, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_8:.*]] = arith.constant 6 : index
    # CHECK:  %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_0]][2, 4, 6, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>
    # CHECK:  %[[VAL_10:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_11:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_12:.*]] = arith.constant 6 : index
    # CHECK:  %[[VAL_13:.*]] = arith.constant 8 : index
    # CHECK:  %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]]] : tensor<10x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

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

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_1(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
    w = ten[1, ...]
    w = ten[1, :, ...]
    w = ten[1, :, :, ...]

    try:
        w = ten[1, :, :, :, :]
    except IndexError as e:
        assert (
            str(e)
            == "Too many indices for shaped type with rank: 5 non-None/Ellipsis indices for dim 4."
        )

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_2(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
    w = ten[1, :]
    w = ten[1, :, :]
    w = ten[1, :, :, :]
    w = ten[:, 1]
    w = ten[:, :, 1]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 0, 0] [10, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x4444xi32>
    # CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1, 0] [10, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_3(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

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

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 1] [10, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x1xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 0, 1] [10, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x1xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 1] [1, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x1xi32>
    # CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_11:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
    # CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_13:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1, 1] [10, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x1xi32>
    # CHECK:  %[[VAL_15:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 1, 0] [10, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x4444xi32>
    # CHECK:  %[[VAL_18:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_19:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 1, 0] [1, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x4444xi32>
    # CHECK:  %[[VAL_21:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_22:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_23:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_24:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 0, 1] [1, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x1xi32>
    # CHECK:  %[[VAL_25:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_26:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_27:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_28:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 1, 1] [1, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x1xi32>
    # CHECK:  %[[VAL_29:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_30:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_31:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_32:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 1, 1] [10, 1, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x1xi32>
    # CHECK:  %[[VAL_33:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_34:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_35:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_36:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 1, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>

    filecheck_with_comments(ctx.module)


def test_none_indices(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
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

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 10, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x22x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4]] output_shape [10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2], [3], [4], [5]] output_shape [1, 10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x333x4444xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1, 2], [3], [4]] output_shape [10, 22, 1, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x1x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3], [4]] output_shape [10, 22, 333, 1, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x333x1x4444xi32>
    # CHECK:  %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
    # CHECK:  %[[VAL_7:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
    # CHECK:  %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4, 5]] output_shape [10, 1, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444x1xi32>
    # CHECK:  %[[VAL_9:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3], [4], [5, 6]] output_shape [10, 1, 22, 1, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x4444x1xi32>
    # CHECK:  %[[VAL_10:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3], [4, 5], [6, 7]] output_shape [10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x1x4444x1xi32>
    # CHECK:  %[[VAL_11:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2], [3, 4], [5, 6], [7, 8]] output_shape [1, 10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x1x333x1x4444x1xi32>

    filecheck_with_comments(ctx.module)


def test_nontrivial_slices(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
    w = ten[:, 0:22:2]
    w = ten[:, 0:22:2, 0:330:30]
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    w = ten[:, :, 100:200:5, 1000:2000:50]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>

    filecheck_with_comments(ctx.module)


def test_nontrivial_slices_insertion(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
    w = ten[:, 0:22:2]
    ten[:, 0:22:2] = w
    w = ten[:, 0:22:2, 0:330:30]
    ten[:, 0:22:2, 0:330:30] = w
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    ten[:, 0:22:2, 0:330:30, 0:4400:400] = w
    w = ten[:, :, 100:200:5, 1000:2000:50]
    ten[:, :, 100:200:5, 1000:2000:50] = w

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x11x333x4444xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.insert_slice %[[VAL_3]] into %[[VAL_2]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x11x11x4444xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
    # CHECK:  %[[VAL_6:.*]] = tensor.insert_slice %[[VAL_5]] into %[[VAL_4]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x11x11x11xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_6]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>
    # CHECK:  %[[VAL_8:.*]] = tensor.insert_slice %[[VAL_7]] into %[[VAL_6]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x20x20xi32> into tensor<7x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_move_slice(ctx: MLIRContext):
    ten = empty(8, 8, T.i32())
    w = ten[0:4, 0:4]
    ten[4:8, 4:8] = w

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<8x8xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_0]][4, 4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>

    filecheck_with_comments(ctx.module)


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
    ten = empty(7, 22, 333, 4444, T.i32())
    for i, r1, _ in range_(0, 10, iter_args=[ten]):
        y = r1 + r1
        res = yield_(y)

    assert str(res) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
    assert res.owner.name == "scf.for"

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x333x4444xi32>) {
    # CHECK:    %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : tensor<7x22x333x4444xi32>
    # CHECK:    scf.yield %[[VAL_7]] : tensor<7x22x333x4444xi32>
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_for_loops_canonicalizer(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def tenfoo():
        ten = empty(7, 22, 333, 4444, T.i32())
        for i, r1, _ in range_(0, 10, iter_args=[ten]):
            y = r1 + r1
            res = yield y

        assert str(res) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
        assert res.owner.name == "scf.for"

    tenfoo()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x333x4444xi32>) {
    # CHECK:    %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : tensor<7x22x333x4444xi32>
    # CHECK:    scf.yield %[[VAL_7]] : tensor<7x22x333x4444xi32>
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_promotion_int_arr(ctx: MLIRContext):
    ten_arr = np.random.randint(0, 10, (10, 10)).astype(np.int32)
    ten = Tensor(ten_arr)
    other = np.random.randint(0, 10, (10, 10)).astype(np.int32)

    x = ten + other
    y = ten - other
    z = ten / other
    w = ten // other
    v = ten % other

    ctx.module.operation.verify()
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant dense<{ten_arr.tolist()}> : tensor<10x10xi32>
      %cst_0 = arith.constant dense<{other.tolist()}> : tensor<10x10xi32>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xi32>
      %cst_1 = arith.constant dense<{Tensor(y.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xi32>
      %cst_2 = arith.constant dense<{Tensor(z.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xi32>
      %cst_3 = arith.constant dense<{Tensor(w.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xi32>
      %cst_4 = arith.constant dense<{Tensor(v.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xi32>
    }}
    """
    )
    filecheck(correct, ctx.module)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="windows has index here whereas linux/mac has i64",
)
def test_promotion_python_constant(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (10, 10)).astype(int)
    ten = Tensor(ten_arr_int)

    x = ten + 1
    y = ten - 1
    z = ten / 1
    w = ten // 1
    v = ten % 1

    ten_arr_float = np.random.randint(0, 10, (10, 10)).astype(float)
    ten = Tensor(ten_arr_float)
    xx = ten + 1.0
    yy = ten - 1.0
    zz = ten / 1.0
    vv = ten % 1.0

    ctx.module.operation.verify()
    # windows in CI...
    bits = np.dtype(int).itemsize * 8
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<10x10xi{bits}>
      %cst_0 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xi{bits}>
      %cst_1 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xi{bits}>
      %cst_2 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xi{bits}>
      %cst_3 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xi{bits}>
      %cst_4 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xi{bits}>
      %cst_5 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<10x10xf64>
      %cst_6 = arith.constant dense<1.0> : tensor<10x10xf64>
      %5 = arith.addf %cst_5, %cst_6 : tensor<10x10xf64>
      %cst_7 = arith.constant dense<1.0> : tensor<10x10xf64>
      %6 = arith.subf %cst_5, %cst_7 : tensor<10x10xf64>
      %cst_8 = arith.constant dense<1.0> : tensor<10x10xf64>
      %7 = arith.divf %cst_5, %cst_8 : tensor<10x10xf64>
      %cst_9 = arith.constant dense<1.0> : tensor<10x10xf64>
      %8 = arith.remf %cst_5, %cst_9 : tensor<10x10xf64>
    }}
    """
    )
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="windows has index here whereas linux/mac has i64",
)
def test_promotion_python_constant_win(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (10, 10)).astype(int)
    ten = Tensor(ten_arr_int)

    x = ten + 1
    y = ten - 1
    z = ten / 1
    w = ten // 1
    v = ten % 1

    ten_arr_float = np.random.randint(0, 10, (10, 10)).astype(float)
    ten = Tensor(ten_arr_float)
    xx = ten + 1.0
    yy = ten - 1.0
    zz = ten / 1.0
    vv = ten % 1.0

    ctx.module.operation.verify()
    # windows in CI...
    bits = np.dtype(int).itemsize * 8
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<10x10xindex>
      %cst_0 = arith.constant dense<1> : tensor<10x10xindex>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xindex>
      %cst_1 = arith.constant dense<1> : tensor<10x10xindex>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xindex>
      %cst_2 = arith.constant dense<1> : tensor<10x10xindex>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xindex>
      %cst_3 = arith.constant dense<1> : tensor<10x10xindex>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xindex>
      %cst_4 = arith.constant dense<1> : tensor<10x10xindex>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xindex>
      %cst_5 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<10x10xf64>
      %cst_6 = arith.constant dense<1.0> : tensor<10x10xf64>
      %5 = arith.addf %cst_5, %cst_6 : tensor<10x10xf64>
      %cst_7 = arith.constant dense<1.0> : tensor<10x10xf64>
      %6 = arith.subf %cst_5, %cst_7 : tensor<10x10xf64>
      %cst_8 = arith.constant dense<1.0> : tensor<10x10xf64>
      %7 = arith.divf %cst_5, %cst_8 : tensor<10x10xf64>
      %cst_9 = arith.constant dense<1.0> : tensor<10x10xf64>
      %8 = arith.remf %cst_5, %cst_9 : tensor<10x10xf64>
    }}
    """
    )
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


def test_promotion_arith(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (2, 2)).astype(np.int32)
    ten = Tensor(ten_arr_int)
    one = arith.constant(1, type=T.i32())
    x = ten + one

    ten_arr_float = np.random.randint(0, 10, (3, 3)).astype(np.float32)
    ten = Tensor(ten_arr_float)
    one = arith.constant(1.0, type=T.f32())
    x = ten + one

    ctx.module.operation.verify()
    correct = dedent(
        f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<2x2xi32>
      %c1_i32 = arith.constant 1 : i32
      %splat = tensor.splat %c1_i32 : tensor<2x2xi32>
      %0 = arith.addi %cst, %splat : tensor<2x2xi32>
      %cst_0 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<3x3xf32>
      %cst_1 = arith.constant 1.0 : f32
      %splat_2 = tensor.splat %cst_1 : tensor<3x3xf32>
      %1 = arith.addf %cst_0, %splat_2 : tensor<3x3xf32>
    }}
    """
    )
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


def test_tensor_arithmetic(ctx: MLIRContext):
    one = arith.constant(1)
    assert isinstance(one, Scalar)
    two = arith.constant(2)
    assert isinstance(two, Scalar)
    three = one + two
    assert isinstance(three, Scalar)

    ten1 = empty(10, 10, 10, T.f32())
    assert isinstance(ten1, Tensor)
    ten2 = empty(10, 10, 10, T.f32())
    assert isinstance(ten2, Tensor)
    ten3 = ten1 + ten2
    assert isinstance(ten3, Tensor)

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]] = tensor.empty() : tensor<10x10x10xf32>
    # CHECK:  %[[VAL_4:.*]] = tensor.empty() : tensor<10x10x10xf32>
    # CHECK:  %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : tensor<10x10x10xf32>

    filecheck_with_comments(ctx.module)
