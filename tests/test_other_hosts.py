import re
from enum import IntEnum
from textwrap import dedent
import pytest

from util import (
    jax_not_installed,
    mlir_bindings_not_installed,
    aie_bindings_not_installed,
)


@pytest.mark.skipif(
    mlir_bindings_not_installed(), reason="mlir python bindings not installed"
)
def test_dialect_trampolines_smoke():
    from mlir.utils._configuration.generate_trampolines import (
        generate_all_upstream_trampolines,
    )

    generate_all_upstream_trampolines()


@pytest.mark.skipif(jax_not_installed(), reason="jax not installed")
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


if aie_bindings_not_installed():
    pytest.skip("aie not install", allow_module_level=True)


import aie.mlir.utils.types as T
from aie.mlir.ir import (
    AttrBuilder,
    IntegerAttr,
    IntegerType,
    MemRefType,
    UnrankedMemRefType,
    OpView,
    Operation,
    Type,
    TypeAttr,
    Value,
    register_attribute_builder,
)
from aie.mlir.utils.dialects.ext.cf import br
from aie.mlir.utils.runtime.passes import run_pipeline, Pipeline
from aie.mlir.utils.context import mlir_mod_ctx
from aie.mlir.utils.dialects.aie import (
    CoreOp,
    ObjectFifoCreateOp,
    ObjectFifoLinkOp,
    ObjectFifoAcquireOp,
    ObjectFifoReleaseOp,
    ObjectFifoSubviewAccessOp,
    TileOp,
    buffer,
    device,
    end,
)
from aie.mlir.utils.dialects.ext.arith import constant
from aie.mlir.utils.dialects.ext.memref import load
from aie.mlir.utils.dialects.memref import store
from aie.mlir.utils.dialects.ext.func import func
from aie.mlir.utils.meta import region_op, maybe_cast, bb

# noinspection PyUnresolvedReferences
from aie.mlir.utils.testing import filecheck, MLIRContext
from aie.mlir.utils.util import get_user_code_loc, get_result_or_results

from aie.mlir.utils.ast.canonicalize import canonicalize
from aie.mlir.utils.dialects.ext.scf import range_, canonicalizer
import aie


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


def tile(col, row, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result = T.index
    return maybe_cast(get_result_or_results(TileOp(result, col, row, loc=loc, ip=ip)))


def link(fifo_ins: list[str], fifo_outs: list[str], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    for i, f in enumerate(fifo_ins):
        if isinstance(f, (OpView, Operation)):
            f = f.sym_name.value
        fifo_ins[i] = AttrBuilder.get("SymbolRefAttr")(f, context=loc.context)
    for i, f in enumerate(fifo_outs):
        if isinstance(f, (OpView, Operation)):
            f = f.sym_name.value
        fifo_outs[i] = AttrBuilder.get("SymbolRefAttr")(f, context=loc.context)
    return maybe_cast(
        get_result_or_results(ObjectFifoLinkOp(fifo_ins, fifo_outs, loc=loc, ip=ip))
    )


def acquire(subview: Type, port, obj_fifo_name, size, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(obj_fifo_name, (OpView, Operation)):
        obj_fifo_name = obj_fifo_name.sym_name.value
    return maybe_cast(
        get_result_or_results(
            ObjectFifoAcquireOp(subview, port, obj_fifo_name, size, loc=loc, ip=ip)
        )
    )


def object_fifo(
    sym_name,
    producer_tile: Value,
    consumer_tiles: list[Value],
    elem_number,
    elem_type,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(elem_type, (MemRefType, UnrankedMemRefType)):
        elem_type = object_fifo_t(elem_type)
    return maybe_cast(
        get_result_or_results(
            ObjectFifoCreateOp(
                sym_name,
                producer_tile,
                consumer_tiles,
                elem_number,
                elem_type,
                loc=loc,
                ip=ip,
            )
        )
    )


def release(port, obj_fifo_name, size, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(obj_fifo_name, (OpView, Operation)):
        obj_fifo_name = obj_fifo_name.sym_name.value
    return maybe_cast(
        get_result_or_results(
            ObjectFifoReleaseOp(port, obj_fifo_name, size, loc=loc, ip=ip)
        )
    )


def subview_access(output: Type, subview: Value, index, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            ObjectFifoSubviewAccessOp(output, subview, index, loc=loc, ip=ip)
        )
    )


# !AIE.objectFifoSubview<memref<16xi32>>
def subview_t(*args, **kwargs):
    m = T.memref(*args, **kwargs)
    return Type.parse(f"!AIE.objectFifoSubview<{m}>")


# !AIE.objectFifo<memref<256xi32>>
def object_fifo_t(m):
    return Type.parse(f"!AIE.objectFifo<{m}>")


class AIEDevice(IntEnum):
    xcvc1902 = 1
    xcve2302 = 2
    xcve2802 = 3

    def __str__(self):
        if self is AIEDevice.xcvc1902:
            return "xcvc1902"
        if self is AIEDevice.xcve2302:
            return "xcve2302"
        if self is AIEDevice.xcve2802:
            return "xcve2802"
        raise ValueError("Unknown AIEDevice enum entry.")


class ObjectFifoPort(IntEnum):
    Produce = 0
    Consume = 1

    def __str__(self):
        if self is ObjectFifoPort.Produce:
            return "Produce"
        if self is ObjectFifoPort.Consume:
            return "Consume"
        raise ValueError("Unknown ObjectFifoPort enum entry.")


@register_attribute_builder("AIEDevice")
def aie_device(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("AIE_ObjectFifo_Depth")
def aie_object_fifo_depth(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("ObjectFifoPort")
def object_fifo_port(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("anonymous_459")
def object_fifo_type_attr(x, context):
    return TypeAttr.get(x)


def test_basic(ctx: MLIRContext):
    tile13 = tile(1, 3)

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


def test01_memory_read_write(ctx: MLIRContext):
    tile13 = tile(1, 3)
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


def test_code_region(ctx: MLIRContext):
    @device(AIEDevice.xcvc1902)
    def in_device():
        @func
        def test_func(x: T.memref(8, 8, T.i32)) -> T.i32:
            ...

        s = tile(0, 2)
        m = tile(1, 2)
        t = tile(3, 3)
        of0 = object_fifo("of0", s, [m], 2, T.memref(256, T.i32))
        of1 = object_fifo("of1", m, [t], 2, T.memref(8, 8, T.i32))
        link([of0], [of1])

        @core(t)
        @canonicalize(using=canonicalizer)
        def inner():
            for i in range_(0, 10):
                subview = acquire(
                    subview_t(8, 8, T.i32), ObjectFifoPort.Consume, of1, 1
                )
                elem0 = subview_access(T.memref(8, 8, T.i32), subview, 0)
                res = test_func(elem0)
                release(ObjectFifoPort.Consume, of1, 1)

    correct = dedent(
        """\
    module {
      AIE.device(xcvc1902) {
        func.func private @test_func(memref<8x8xi32>) -> i32
        %0 = AIE.tile(0, 2)
        %1 = AIE.tile(1, 2)
        %2 = AIE.tile(3, 3)
        AIE.objectFifo @of0(%0, {%1}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
        AIE.objectFifo @of1(%1, {%2}, 2 : i32) : !AIE.objectFifo<memref<8x8xi32>>
        AIE.objectFifo.link [@of0] -> [@of1]()
        %3 = AIE.core(%2) {
          %c0 = arith.constant 0 : index
          %c10 = arith.constant 10 : index
          %c1 = arith.constant 1 : index
          scf.for %arg0 = %c0 to %c10 step %c1 {
            %4 = AIE.objectFifo.acquire @of1(Consume, 1) : !AIE.objectFifoSubview<memref<8x8xi32>>
            %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<8x8xi32>> -> memref<8x8xi32>
            %6 = func.call @test_func(%5) : (memref<8x8xi32>) -> i32
            AIE.objectFifo.release @of1(Consume, 1)
          }
          AIE.end
        }
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_bbs(ctx: MLIRContext):
    tile83 = tile(8, 3)
    buf0 = buffer(T.memref(32, 32, T.i32), tile83)
    buf1 = buffer(T.memref(32, 32, T.i32), tile83)
    buf2 = buffer(T.memref(32, 32, T.i32), tile83)

    @core(tile83)
    @canonicalize(using=canonicalizer)
    def core83():
        br0 = br()
        with bb(br0):
            br1 = br()
        with bb(br1):
            for _ in range_(0, 64, 32):
                for i in range_(0, 32):
                    for j in range_(0, 32):
                        for k in range_(0, 32):
                            buf2[i, k] += buf0[i, j] * buf1[j, k]

    correct = dedent(
        """\
    module {
      %0 = AIE.tile(8, 3)
      %1 = AIE.buffer(%0) : memref<32x32xi32>
      %2 = AIE.buffer(%0) : memref<32x32xi32>
      %3 = AIE.buffer(%0) : memref<32x32xi32>
      %4 = AIE.core(%0) {
        cf.br ^bb1
      ^bb1:  // pred: ^bb0
        cf.br ^bb2
      ^bb2:  // pred: ^bb1
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        scf.for %arg0 = %c0 to %c64 step %c32 {
          %c1 = arith.constant 1 : index
          scf.for %arg1 = %c0 to %c32 step %c1 {
            scf.for %arg2 = %c0 to %c32 step %c1 {
              scf.for %arg3 = %c0 to %c32 step %c1 {
                %5 = memref.load %3[%arg1, %arg3] : memref<32x32xi32>
                %6 = memref.load %1[%arg1, %arg2] : memref<32x32xi32>
                %7 = memref.load %2[%arg2, %arg3] : memref<32x32xi32>
                %8 = arith.muli %6, %7 : i32
                %9 = arith.addi %5, %8 : i32
                memref.store %9, %3[%arg1, %arg3] : memref<32x32xi32>
              }
            }
          }
        }
        AIE.end
      }
    }
    """
    )

    run_pipeline(ctx.module, Pipeline().cse())
    filecheck(correct, ctx.module)
