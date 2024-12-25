import platform

import numpy as np
from textwrap import dedent

import pytest

from mlir.extras.runtime.passes import Pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext, backend

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")
pytest.mark.usefixtures("backend")


@pytest.mark.skipif(
    platform.machine() == "aarch64", reason="https://github.com/numba/numba/issues/9109"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows can't load runner extras"
)
def test_simple_parfor(ctx: MLIRContext, backend: LLVMJITBackend):
    module = ctx.module.parse(
        dedent(
            """\
    func.func @main(%a : memref<?x?xf32>) -> () attributes { llvm.emit_c_interface } {
        %i1 = arith.constant 0 : index
        %i2 = arith.constant 1 : index
        %bound1 = memref.dim %a, %i1 : memref<?x?xf32>
        %bound2 = memref.dim %a, %i2 : memref<?x?xf32>
        %value = arith.constant 20.0 : f32
        %initial = arith.constant 0 : index
        %step = arith.constant 1: index
        scf.parallel (%iv1, %iv2, %iv3, %iv4) = (%initial, %initial, %initial, %initial) to (%bound1, %bound2, %bound1, %bound2) step (%step, %step, %step, %step) {
            vector.print %iv1 :index
            vector.print %iv2 :index
            vector.print %iv3 :index
            vector.print %iv4 :index
            %e = arith.addi %iv3, %iv4 : index
            %d = arith.addi %e, %iv1 : index
            %b = arith.index_cast %d : index to i32
            %c = arith.sitofp %b : i32 to f32
            %t = arith.addf %value, %c : f32
            memref.store %t, %a[%iv1, %iv2] : memref<?x?xf32>
        }
        return
    }    
    """
        )
    )
    module = backend.compile(
        module,
        kernel_name="main",
        pipeline=Pipeline()
        .bufferize()
        .async_parallel_for(num_workers=2)
        .async_to_async_runtime()
        .async_runtime_ref_counting()
        .async_runtime_ref_counting_opt()
        .arith_expand()
        .convert_async_to_llvm()
        .convert_scf_to_cf()
        .convert_vector_to_llvm()
        .convert_arith_to_llvm()
        .finalize_memref_to_llvm()
        .convert_func_to_llvm()
        .convert_cf_to_llvm()
        .reconcile_unrealized_casts(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.random.randint(0, 10, (10, 10)).astype(np.float32)
    # invoker.main(A)
