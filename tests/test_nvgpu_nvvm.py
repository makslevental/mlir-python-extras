from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.nvgpu import tensormap_descriptor
from mlir.dialects.memref import cast
from mlir.dialects.nvgpu import tma_create_descriptor

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic(ctx: MLIRContext):
    @func
    def create_tensor_map(
        device_ptr_2d: T.memref(64, 128, element_type=T.f32()),
    ):
        crd0 = constant(64, index=True)
        crd1 = constant(128, index=True)
        device_ptr_2d_unranked = cast(T.memref(element_type=T.f32()), device_ptr_2d)
        tensor_map_2d = tensormap_descriptor(T.memref(32, 32, T.f32(), memory_space=3))
        tensor_map_2d = tma_create_descriptor(
            tensor_map_2d, device_ptr_2d_unranked, [crd0, crd1]
        )

    create_tensor_map.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func @create_tensor_map(%arg0: memref<64x128xf32>) {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %cast = memref.cast %arg0 : memref<64x128xf32> to memref<*xf32>
        %0 = nvgpu.tma.create.descriptor %cast box[%c64, %c128] : memref<*xf32> -> <tensor = memref<32x32xf32, 3>, swizzle = none, l2promo = none, oob = nan, interleave = none>
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)
