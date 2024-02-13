import sys

import numpy as np
import pytest

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import mlir.extras.dialects.ext.memref
from mlir.dialects import builtin
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
from mlir.extras import types as T
from mlir.extras.context import ExplicitlyManagedModule, RAIIMLIRContext
from mlir.extras.dialects.ext import linalg, transform, arith, vector
from mlir.dialects.bufferization import LayoutMapOption
from mlir.extras.dialects.ext.func import func
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
from mlir.ir import StringAttr, UnitAttr

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


# based on /home/mlevental/dev_projects/llvm-project/mlir/test/Dialect/LLVM/transform-e2e.mlir
def test_e2e(ctx: MLIRContext):
    ctx = RAIIMLIRContext()
    backend = LLVMJITBackend()
    module = ExplicitlyManagedModule()

    M, K, N = 2, 4, 6

    @func
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

    # https://github.com/makslevental/llvm-project/blob/f6643263631bcb0d191ef923963ac1a5ca9ac5fd/mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp#L44
    lower_to_llvm = (
        Pipeline()
        .Func(
            Pipeline()
            # Blanket-convert any remaining high-level vector ops to loops if any remain.
            .convert_vector_to_scf()
            # Blanket-convert any remaining linalg ops to loops if any remain.
            .convert_linalg_to_loops()
        )
        # Blanket-convert any remaining affine ops if any remain.
        .lower_affine()
        # Convert SCF to CF (always needed).
        .convert_scf_to_cf()
        # Sprinkle some cleanups.
        .canonicalize()
        .cse()
        # Convert vector to LLVM (always needed).
        .convert_vector_to_llvm()
        # Convert Math to LLVM (always needed).
        .Func(Pipeline().convert_math_to_llvm())
        # Expand complicated MemRef operations before lowering them.
        .expand_strided_metadata()
        # The expansion may create affine expressions. Get rid of them.
        .lower_affine()
        # Convert MemRef to LLVM (always needed).
        .finalize_memref_to_llvm()
        # Convert Func to LLVM (always needed).
        .convert_func_to_llvm()
        # Convert Index to LLVM (always needed).
        .convert_index_to_llvm()
        # Convert remaining unrealized_casts (always needed).
        .reconcile_unrealized_casts()
    )

    compiled_module = backend.compile(
        find_ops(
            vectorized_module.operation,
            lambda x: "transform.target_tag" in x.attributes
            and x.attributes["transform.target_tag"].value == "payload",
            single=True,
        ),
        kernel_name=smol_matmul.__name__,
        pipeline=lower_to_llvm,
    )
    print(compiled_module)

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
