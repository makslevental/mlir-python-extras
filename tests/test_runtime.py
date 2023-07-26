import pytest

from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.func import func, declare
from mlir_utils.dialects.ext.scf import (
    range_,
    canonicalizer,
)
from mlir_utils.dialects.ext.tensor import empty
from mlir_utils.runtime.passes import Pipeline
from mlir_utils.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t, memref_t, f32_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@pytest.fixture
def backend() -> LLVMJITBackend:
    return LLVMJITBackend()


def test_tensor_additions(ctx: MLIRContext, backend: LLVMJITBackend):
    # TODO(max): ValueError: foo requires closure of length 0, not 1
    print_memref_32 = declare("printMemrefF32", [memref_t(element_type=f32_t)])

    @func
    @canonicalize(using=canonicalizer)
    def foo():
        ten = empty((7, 22, 333, 4444), i32_t)
        for i, r1 in range_(0, 10, iter_args=[ten]):
            y = r1 + r1
            yield y

        return r1

    foo.emit()

    print(ctx.module)
    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline().convert_elementwise_to_linalg().bufferize()
        # .lower_to_llvm()
        .materialize(),
    )
    print(module)
