from textwrap import dedent

import aie
import aie.mlir.utils.types as T
import pytest
from aie.mlir.ir import Value
from aie.mlir.utils.context import mlir_mod_ctx
from aie.mlir.utils.dialects.aie import tile, buffer, CoreOp, end
from aie.mlir.utils.dialects.ext.arith import constant
from aie.mlir.utils.dialects.ext.memref import load
from aie.mlir.utils.dialects.memref import store
from aie.mlir.utils.meta import region_op
# noinspection PyUnresolvedReferences
from aie.mlir.utils.testing import filecheck, MLIRContext
from aie.mlir.utils.util import get_user_code_loc

from util import (
    skip_jax_not_installed,
    mlir_bindings_installed,
    aie_bindings_installed,
)


@pytest.mark.skipif(
    mlir_bindings_installed(), reason="mlir python bindings not installed"
)
def test_smoke():
    from mlir.utils.context import mlir_mod_ctx
    from mlir.utils.testing import filecheck

    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        correct = dedent(
            """\
        module {
        }
        """
        )

    filecheck(correct, ctx.module)


@pytest.mark.skipif(
    mlir_bindings_installed(), reason="mlir python bindings not installed"
)
def test_dialect_trampolines_smoke():
    from mlir.utils._configuration.generate_trampolines import (
        generate_all_upstream_trampolines,
    )

    generate_all_upstream_trampolines()
    # noinspection PyUnresolvedReferences
    from mlir.utils.dialects import (
        arith,
        bufferization,
        builtin,
        cf,
        complex,
        func,
        gpu,
        linalg,
        math,
        memref,
        ml_program,
        pdl,
        scf,
        shape,
        sparse_tensor,
        tensor,
        tosa,
        transform,
        vector,
    )


@pytest.mark.skipif(skip_jax_not_installed(), reason="jax not installed")
def test_jax_trampolines_smoke():
    # noinspection PyUnresolvedReferences
    from jaxlib.mlir.utils.dialects import (
        arith,
        builtin,
        chlo,
        func,
        math,
        memref,
        mhlo,
        ml_program,
        scf,
        sparse_tensor,
        stablehlo,
        vector,
    )


@pytest.fixture
def ctx() -> MLIRContext:
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        aie.dialects.aie.register_dialect(ctx.context)
        yield ctx


# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def core(tile: Value, *, stack_size=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return CoreOp(T.index, tile, stackSize=stack_size, loc=loc, ip=ip)


core = region_op(core, terminator=lambda *args: end())


def test_basic(ctx: MLIRContext):
    tile13 = tile(T.index, 1, 3)

    @core(tile13)
    def demo_fun1():
        one = constant(1)

    correct = dedent(
        """\
    module {
      %0 = AIE.tile(1, 3)
      %1 = AIE.core(%0) {
        %c1_i32 = arith.constant 1 : i32
        AIE.end
      }
    }
    """
    )
    filecheck(correct, ctx.module)


@pytest.mark.skipif(aie_bindings_installed(), reason="aie bindings not installed")
def test01_memory_read_write(ctx: MLIRContext):
    tile13 = tile(T.index, 1, 3)
    buf13_0 = buffer(T.memref(256, T.i32), tile13)

    @core(tile13)
    def core13():
        val1 = constant(7)
        idx1 = constant(7, index=True)
        two = val1 + val1
        store(two, buf13_0, [idx1])
        val2 = constant(8)
        idx2 = constant(5, index=True)
        store(val2, buf13_0, [idx2])
        val3 = load(buf13_0, [idx1])
        idx3 = constant(9, index=True)
        store(val3, buf13_0, [idx3])

    correct = dedent(
        """\
    module {
      %0 = AIE.tile(1, 3)
      %1 = AIE.buffer(%0) : memref<256xi32>
      %2 = AIE.core(%0) {
        %c7_i32 = arith.constant 7 : i32
        %c7 = arith.constant 7 : index
        %3 = arith.addi %c7_i32, %c7_i32 : i32
        memref.store %3, %1[%c7] : memref<256xi32>
        %c8_i32 = arith.constant 8 : i32
        %c5 = arith.constant 5 : index
        memref.store %c8_i32, %1[%c5] : memref<256xi32>
        %4 = memref.load %1[%c7] : memref<256xi32>
        %c9 = arith.constant 9 : index
        memref.store %4, %1[%c9] : memref<256xi32>
        AIE.end
      }
    }
    """
    )
    filecheck(correct, ctx.module)
