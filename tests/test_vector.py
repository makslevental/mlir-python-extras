from textwrap import dedent

import numpy as np
import pytest
from mlir.dialects import builtin
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import (
    any_op_t,
)
from mlir.dialects.transform.extras import apply_patterns, named_sequence
from mlir.dialects.transform.vector import (
    VectorContractLowering,
    VectorMultiReductionLowering,
    VectorTransferSplit,
    VectorTransposeLowering,
)
from mlir.ir import (
    StringAttr,
    UnitAttr,
    ShapedType,
    AffineMap,
    AffineConstantExpr,
)

from mlir.extras import types as T
from mlir.extras.context import ExplicitlyManagedModule

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
from mlir.extras.dialects.ext import arith, linalg, memref, transform, vector, scf, func
from mlir.extras.dialects.ext.vector import outer, shuffle, load
from mlir.extras.dialects.ext.transform import (
    get_parent_op,
    match,
    tile_to_scf_for,
    transform_any_op_t,
)
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
from mlir.extras.util import find_ops

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


# based on /home/mlevental/dev_projects/llvm-project/mlir/test/Dialect/LLVM/transform-e2e.mlir
def test_e2e(ctx: MLIRContext):
    backend = LLVMJITBackend()
    module = ExplicitlyManagedModule()

    scale = 16
    M, K, N = 2 * scale, 4 * scale, 6 * scale

    @func.func
    def smol_matmul(
        A: T.tensor(M, K, T.f32()),
        B: T.tensor(K, N, T.f32()),
        C: T.tensor(M, N, T.f32()),
    ):
        return linalg.matmul(A, B, C)

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        smol_matmul.emit(force=True)

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence("main", [any_op_t()], [])
        def main(module_op: any_op_t()):
            matmul = match(module_op, ops=["linalg.matmul"])
            tiled_matmul, (_, _, inner_loop) = tile_to_scf_for(matmul, sizes=[2, 2, 2])
            transform.structured.vectorize_children_and_apply_patterns(
                get_parent_op(
                    transform_any_op_t(), tiled_matmul, isolated_from_above=True
                )
            )
            new_mod = transform.bufferization.one_shot_bufferize(
                module_op,
                function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
                bufferize_function_boundaries=True,
            )

            func_op = match(new_mod, ops=["func.func"])

            @apply_patterns(func_op)
            def pats():
                transform.apply_patterns.vector.lower_contraction(
                    lowering_strategy=VectorContractLowering.OuterProduct
                )
                transform.apply_patterns.vector.transfer_permutation_patterns()
                transform.apply_patterns.vector.lower_multi_reduction(
                    lowering_strategy=VectorMultiReductionLowering.InnerParallel
                )
                transform.apply_patterns.vector.split_transfer_full_partial(
                    split_transfer_strategy=VectorTransferSplit.LinalgCopy
                )
                transform.apply_patterns.vector.transfer_to_scf(
                    max_transfer_rank=1, full_unroll=True
                )
                transform.apply_patterns.vector.lower_transfer(max_transfer_rank=1)
                transform.apply_patterns.vector.lower_shape_cast()
                transform.apply_patterns.vector.lower_transpose(
                    lowering_strategy=VectorTransposeLowering.Shuffle1D
                )

    module = module.finish()

    vectorized_module = run_pipeline(
        module,
        pipeline=Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )

    print(vectorized_module)

    compiled_module = backend.compile(
        find_ops(
            vectorized_module.operation,
            lambda x: "transform.target_tag" in x.attributes
            and x.attributes["transform.target_tag"].value == "payload",
            single=True,
        ),
        kernel_name=smol_matmul.__name__,
        pipeline=Pipeline().lower_to_llvm(),
    )

    A = np.random.randint(0, 10, (M, K)).astype(np.float32)
    B = np.random.randint(0, 10, (K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    backend.load(compiled_module).smol_matmul_capi_wrapper(A, B, C)
    assert np.allclose(A @ B, C)


def test_e2e_sugar(ctx: MLIRContext):
    backend = LLVMJITBackend()

    scale = 16
    M, K, N = 2 * scale, 4 * scale, 6 * scale
    v2f32 = T.vector(2, T.f32())

    @func.func(emit=True)
    def smol_matmul(
        A: T.memref(M, K, T.f32()),
        B: T.memref(K, N, T.f32()),
        C: T.memref(M, N, T.f32()),
    ):
        cst = arith.constant(np.full([4], 0.0, np.float32), T.vector(4, T.f32()))
        cst_0 = arith.constant(
            np.full([2, 2], 0.0, np.float32), T.vector(2, 2, T.f32())
        )
        for i, C, v0 in scf.range_(0, M, 2, iter_args=[C]):
            for j, C, v1 in scf.range_(0, N, 2, iter_args=[C]):
                for k, C, v2 in scf.range_(0, K, 2, iter_args=[C]):
                    cst[0::1] = A @ load(v2f32) @ [i, k]
                    cst[2::1] = A @ load(v2f32) @ [i + 1, k]
                    cst_0[0] = C @ load(v2f32) @ [i, j]
                    cst_0[1] = C @ load(v2f32) @ [i + 1, j]
                    cst = cst @ shuffle(mask=[0, 2, 1, 3]) @ cst

                    v19 = cst[0:2:1] @ outer(cst_0) @ (B @ load(v2f32) @ [k, j])
                    v20 = cst[2:4:1] @ outer(v19) @ (B @ load(v2f32) @ [k + 1, j])
                    C[i, j] = v20[0]
                    C[i + 1, j] = v20[1]

                    scf.yield_(C)
                scf.yield_(v2)
            scf.yield_(v1)

    compiled_module = backend.compile(
        ctx.module,
        kernel_name=smol_matmul.__name__,
        pipeline=Pipeline().lower_to_llvm(),
    )

    A = np.random.randint(0, 10, (M, K)).astype(np.float32)
    B = np.random.randint(0, 10, (K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    backend.load(compiled_module).smol_matmul_capi_wrapper(A, B, C)
    assert np.allclose(A @ B, C)


def test_np_constructor(ctx: MLIRContext):
    M, K, N = 2, 4, 6
    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    vec = arith.constant(A, vector=True)
    assert (
        repr(vec)
        == f"Vector(%cst = arith.constant dense<{vec.literal_value.tolist()}> : vector<2x4xi32>)"
    )


def test_vector_wrappers(ctx: MLIRContext):
    M, K, N = 2, 4, 6
    mem = memref.alloc((M, K, N), T.i32())
    vec = vector.transfer_read(
        T.vector(M, K, T.i32()), mem, [0, 0, 0], padding=5, in_bounds=[True, True]
    )
    e_vec = vector.extract(vec, [0])
    vector.transfer_write(e_vec, mem, [0, 0, 0], in_bounds=[True])

    b = vector.broadcast(T.vector(10, T.i32()), 5)
    r = vector.reduction(vector.CombiningKind.ADD, b)

    b = vector.broadcast(T.vector(4, 8, 16, 32, T.i32()), 5)
    acc = vector.broadcast(T.vector(4, 16, T.i32()), 0)
    r = vector.multi_reduction(vector.CombiningKind.ADD, b, acc, [1, 3])

    b = vector.broadcast(T.vector(4, 8, 16, T.i32()), 5)
    e = vector.extract_strided_slice(b, [0, 2], [2, 4], [1, 1])

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<2x4x6xi32>
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c5_i32 = arith.constant 5 : i32
      %0 = vector.transfer_read %alloc[%c0, %c0_0, %c0_1], %c5_i32 {in_bounds = [true, true]} : memref<2x4x6xi32>, vector<2x4xi32>
      %1 = vector.extract %0[0] : vector<4xi32> from vector<2x4xi32>
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      vector.transfer_write %1, %alloc[%c0_2, %c0_3, %c0_4] {in_bounds = [true]} : vector<4xi32>, memref<2x4x6xi32>
      %c5_i32_5 = arith.constant 5 : i32
      %2 = vector.broadcast %c5_i32_5 : i32 to vector<10xi32>
      %3 = vector.reduction <add>, %2 : vector<10xi32> into i32
      %c5_i32_6 = arith.constant 5 : i32
      %4 = vector.broadcast %c5_i32_6 : i32 to vector<4x8x16x32xi32>
      %c0_i32 = arith.constant 0 : i32
      %5 = vector.broadcast %c0_i32 : i32 to vector<4x16xi32>
      %6 = vector.multi_reduction <add>, %4, %5 [1, 3] : vector<4x8x16x32xi32> to vector<4x16xi32>
      %c5_i32_7 = arith.constant 5 : i32
      %7 = vector.broadcast %c5_i32_7 : i32 to vector<4x8x16xi32>
      %8 = vector.extract_strided_slice %7 {offsets = [0, 2], sizes = [2, 4], strides = [1, 1]} : vector<4x8x16xi32> to vector<2x4x16xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


# Illustrates an 8x8 Sparse Matrix x Vector implemented with only operations
# of the vector dialect (and some std/scf). Essentially, this example performs
# the following multiplication:
#
#     0  1  2  3  4  5  6  7
#   +------------------------+
# 0 | 1  0  2  0  0  1  0  1 |   | 1 |   | 21 |
# 1 | 1  8  0  0  3  0  1  0 |   | 2 |   | 39 |
# 2 | 0  0  1  0  0  2  6  2 |   | 3 |   | 73 |
# 3 | 0  3  0  1  0  1  0  1 | x | 4 | = | 24 |
# 4 | 5  0  0  1  1  1  0  0 |   | 5 |   | 20 |
# 5 | 0  3  0  0  2  1  2  0 |   | 6 |   | 36 |
# 6 | 4  0  7  0  1  0  1  0 |   | 7 |   | 37 |
# 7 | 0  3  0  2  0  0  1  1 |   | 8 |   | 29 |
#   +------------------------+
#
# The sparse storage scheme used is an extended column scheme (also referred
# to as jagged diagonal, which is essentially a vector friendly variant of
# the general sparse row-wise scheme (also called compressed row storage),
# using fixed length vectors and no explicit pointer indexing into the
# value array to find the rows.
#
# The extended column storage for the matrix shown above is as follows.
#
#      VALUE           INDEX
#   +---------+     +---------+
# 0 | 1 2 1 1 |     | 0 2 5 7 |
# 1 | 1 8 3 1 |     | 0 1 4 6 |
# 2 | 1 2 6 2 |     | 2 5 6 7 |
# 3 | 3 1 1 1 |     | 1 3 5 7 |
# 4 | 5 1 1 1 |     | 0 3 4 5 |
# 5 | 3 2 1 2 |     | 1 4 5 6 |
# 6 | 4 7 1 1 |     | 0 2 4 6 |
# 7 | 3 2 1 1 |     | 1 3 6 7 |
#   +---------+     +---------+
#
# This example illustrates an effective SAXPY version that operates
# on the transposed jagged diagonal storage to obtain higher vector
# lengths. Another example in this directory illustrates a DOT
# version of the operation.


def aligned(a, alignment=16):
    if (a.ctypes.data % alignment) == 0:
        return a
    assert alignment % a.itemsize == 0
    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs : ofs + a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert aa.ctypes.data % alignment == 0
    return aa


def test_memref_of_vector_linalg_generic_2(ctx: MLIRContext):
    @func.func
    def spmv8x8(
        AVAL: T.memref(4, T.vector(8, T.f32())),
        AIDX: T.memref(4, T.vector(8, T.i32())),
        X: T.memref(ShapedType.get_dynamic_size(), T.f32()),
        B: T.memref(1, T.vector(8, T.f32())),
    ):
        c0 = arith.constant(0, T.index())
        cst = arith.constant(0.0, T.f32())
        v0 = vector.constant_mask(result=T.vector(8, T.bool()), mask_dim_sizes=[8])
        v1 = vector.broadcast(vector=T.vector(8, T.f32()), source=cst)
        id_map_1 = AffineMap.get_identity(1)
        c2 = AffineConstantExpr.get(0)
        map1 = AffineMap.get(1, 0, [c2])

        @linalg.generic(
            [AVAL, AIDX],
            [B],
            [id_map_1, id_map_1, map1],
            [linalg.IteratorType.reduction],
        )
        def result(aval, aidx, b):
            v6 = vector.gather(
                result=T.vector(8, T.f32()),
                base=X,
                indices=[c0],
                index_vec=aidx,
                mask=v0,
                pass_thru=v1,
            )
            return vector.fma(lhs=aval, rhs=v6, acc=b)

    spmv8x8.emit()
    assert ctx.module.operation.verify()

    backend = LLVMJITBackend()
    compiled_module = backend.compile(
        ctx.module, kernel_name="spmv8x8", pipeline=Pipeline().lower_to_llvm()
    )

    sparse_A = [
        [1, 0, 2, 0, 0, 1, 0, 1],
        [1, 8, 0, 0, 3, 0, 1, 0],
        [0, 0, 1, 0, 0, 2, 6, 2],
        [0, 3, 0, 1, 0, 1, 0, 1],
        [5, 0, 0, 1, 1, 1, 0, 0],
        [0, 3, 0, 0, 2, 1, 2, 0],
        [4, 0, 7, 0, 1, 0, 1, 0],
        [0, 3, 0, 2, 0, 0, 1, 1],
    ]
    AVAL = (
        np.array([[c for i, c in enumerate(r) if c > 0] for r in sparse_A])
        .astype(np.float32)
        .T.copy()
    )
    AIDX = (
        np.array([[i for i, c in enumerate(r) if c > 0] for r in sparse_A])
        .astype(np.int32)
        .T.copy()
    )
    X = np.arange(1, 9).astype(np.float32).T.copy()
    B = np.zeros_like(X).astype(np.float32).T.copy()
    AVAL, AIDX, X, B = map(lambda x: aligned(x, 64), [AVAL, AIDX, X, B])

    backend.load(compiled_module).spmv8x8_capi_wrapper(AVAL, AIDX, X, B)

    assert np.allclose(B, [21, 39, 73, 24, 20, 36, 37, 29])

    assert np.allclose(B, np.array(sparse_A) @ X)


def test_memref_of_vector_linalg_generic_3(ctx: MLIRContext):
    @func.func
    def mv8x8(
        A: T.memref(8, T.vector(8, T.f32())),
        X: T.memref(8, T.f32()),
        B: T.memref(1, T.vector(8, T.f32())),
    ):
        id_map_1 = AffineMap.get_identity(1)
        ac0 = AffineConstantExpr.get(0)
        map2 = AffineMap.get(1, 0, [ac0])

        @linalg.generic(
            [A, X],
            [B],
            [id_map_1, id_map_1, map2],
            [linalg.IteratorType.reduction],
        )
        def result(a, x, b):
            x = vector.broadcast(T.vector(8, T.f32()), x)
            b = vector.fma(lhs=a, rhs=x, acc=b)
            return b

    mv8x8.emit()
    assert ctx.module.operation.verify()

    backend = LLVMJITBackend()
    compiled_module = backend.compile(
        ctx.module, kernel_name="mv8x8", pipeline=Pipeline().lower_to_llvm()
    )

    A = (
        np.array(
            [
                [1, 0, 2, 0, 0, 1, 0, 1],
                [1, 8, 0, 0, 3, 0, 1, 0],
                [0, 0, 1, 0, 0, 2, 6, 2],
                [0, 3, 0, 1, 0, 1, 0, 1],
                [5, 0, 0, 1, 1, 1, 0, 0],
                [0, 3, 0, 0, 2, 1, 2, 0],
                [4, 0, 7, 0, 1, 0, 1, 0],
                [0, 3, 0, 2, 0, 0, 1, 1],
            ]
        )
        .astype(np.float32)
        .T.copy()
    )
    X = np.arange(1, 9).astype(np.float32)
    B = np.zeros_like(X).astype(np.float32)
    A, X, B = map(lambda x: aligned(x, 64), [A, X, B])

    backend.load(compiled_module).mv8x8_capi_wrapper(A, X, B)

    assert np.allclose(B, [21, 39, 73, 24, 20, 36, 37, 29])
    assert np.allclose(B, A.T @ X)
