import platform
from textwrap import dedent

import numpy as np
import pytest

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects.ext.arith import Scalar, constant
from mlir.extras.dialects.ext.scf import (
    range_,
    yield_,
    canonicalizer,
)
from mlir.extras.dialects.ext.tensor import Tensor, empty

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
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
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
    } 
    """
    )
    filecheck(correct, ctx.module)


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
    ten = empty(10, 22, 333, 4444, T.i32())
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
    correct = dedent(
        """\
    module {
      %0 = tensor.empty() : tensor<10x22x333x4444xi32>
      %expanded = tensor.expand_shape %0 [[0, 1], [2], [3], [4]] output_shape [1, 10, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x22x333x4444xi32>
      %expanded_0 = tensor.expand_shape %0 [[0, 1], [2], [3], [4]] output_shape [10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444xi32>
      %expanded_1 = tensor.expand_shape %0 [[0, 1, 2], [3], [4], [5]] output_shape [1, 10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x333x4444xi32>
      %expanded_2 = tensor.expand_shape %0 [[0], [1, 2], [3], [4]] output_shape [10, 22, 1, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x1x333x4444xi32>
      %expanded_3 = tensor.expand_shape %0 [[0], [1], [2, 3], [4]] output_shape [10, 22, 333, 1, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x333x1x4444xi32>
      %expanded_4 = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
      %expanded_5 = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
      %expanded_6 = tensor.expand_shape %0 [[0, 1], [2], [3], [4, 5]] output_shape [10, 1, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444x1xi32>
      %expanded_7 = tensor.expand_shape %0 [[0, 1], [2, 3], [4], [5, 6]] output_shape [10, 1, 22, 1, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x4444x1xi32>
      %expanded_8 = tensor.expand_shape %0 [[0, 1], [2, 3], [4, 5], [6, 7]] output_shape [10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x1x4444x1xi32>
      %expanded_9 = tensor.expand_shape %0 [[0, 1, 2], [3, 4], [5, 6], [7, 8]] output_shape [1, 10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x1x333x1x4444x1xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_nontrivial_slices(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
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
    ten = empty(7, 22, 333, 4444, T.i32())
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
    ten = empty(8, 8, T.i32())
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
    ten = empty(7, 22, 333, 4444, T.i32())
    for i, r1, _ in range_(0, 10, iter_args=[ten]):
        y = r1 + r1
        res = yield_(y)

    assert str(res) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
    assert res.owner.name == "scf.for"
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
        ten = empty(7, 22, 333, 4444, T.i32())
        for i, r1, _ in range_(0, 10, iter_args=[ten]):
            y = r1 + r1
            res = yield y

        assert str(res) == "Tensor(%1, tensor<7x22x333x4444xi32>)"
        assert res.owner.name == "scf.for"

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
    one = constant(1, type=T.i32())
    x = ten + one

    ten_arr_float = np.random.randint(0, 10, (3, 3)).astype(np.float32)
    ten = Tensor(ten_arr_float)
    one = constant(1.0, type=T.f32())
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
