import ctypes
import platform
import re
from textwrap import dedent

import mlir.extras.types as T
import numpy as np
import pytest
from mlir.dialects.arith import sitofp, index_cast
from mlir.dialects.memref import cast
from mlir.ir import UnitAttr, Module, StridedLayoutAttr, InsertionPoint, Context
from mlir.runtime import get_unranked_memref_descriptor, get_ranked_memref_descriptor

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContext, ExplicitlyManagedModule
from mlir.extras.dialects.ext import linalg
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext.memref import load, store, S
from mlir.extras.dialects.ext.scf import (
    canonicalizer,
    range_,
)
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.extras.runtime.refbackend import (
    LLVMJITBackend,
    convert_returns_from_ctype,
    refback_cb_attr,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
    backend,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")
pytest.mark.usefixtures("backend")


if platform.system().lower() == "emscripten":
    pytest.skip("runtime doesn't work on emscripten yet", allow_module_level=True)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows can't load runner extras"
)
def test_smoke(ctx: MLIRContext, backend: LLVMJITBackend, capfd):
    # TODO(max): ValueError: foo requires closure of length 0, not 1
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def printMemrefF32(x: unranked_memref_f32): ...

    @func
    @canonicalize(using=canonicalizer)
    def foo(x: unranked_memref_f32):
        printMemrefF32(x)
        return

    foo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    # CHECK:  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
    # CHECK:  llvm.func @foo(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
    # CHECK:    %[[VAL_2:.*]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0] : !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_3]][1] : !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_4]][0] : !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_4]][1] : !llvm.struct<(i64, ptr)>
    # CHECK:    llvm.call @printMemrefF32(%[[VAL_5]], %[[VAL_6]]) : (i64, !llvm.ptr) -> ()
    # CHECK:    llvm.return
    # CHECK:  }
    # CHECK:  llvm.func @_mlir_ciface_foo(%[[VAL_7:.*]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
    # CHECK:    %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr -> !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_8]][0] : !llvm.struct<(i64, ptr)>
    # CHECK:    %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_8]][1] : !llvm.struct<(i64, ptr)>
    # CHECK:    llvm.call @foo(%[[VAL_9]], %[[VAL_10]]) : (i64, !llvm.ptr) -> ()
    # CHECK:    llvm.return
    # CHECK:  }

    filecheck_with_comments(module)

    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))
    backend.load(module).foo(AA)
    correct = dedent(
        """\
    Unranked Memref base@ =  rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data = 
    [[1,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1]]
    """
    )
    out, err = capfd.readouterr()
    filecheck(correct, re.sub(r"0x\w+", "", out))


def test_munge_calling_conventions(ctx: MLIRContext, backend: LLVMJITBackend, capfd):
    ranked_memref_2x2_f32 = T.memref(2, 2, T.f32())
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def foo(x: ranked_memref_2x2_f32):
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def refbackend_consume_return_callback_first(x: unranked_memref_f32): ...

    @func
    def foo_wrapper(x: unranked_memref_f32):
        x = cast(ranked_memref_2x2_f32, x)
        y = foo(x)
        y = cast(unranked_memref_f32, y)
        refbackend_consume_return_callback_first(y)

    foo_wrapper.emit()

    # CHECK:  func.func @foo(%[[VAL_0:.*]]: memref<2x2xf32>) -> memref<2x2xf32> {
    # CHECK:    return %[[VAL_0]] : memref<2x2xf32>
    # CHECK:  }
    # CHECK:  func.func private @refbackend_consume_return_callback_first(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
    # CHECK:  func.func @foo_wrapper(%[[VAL_1:.*]]: memref<*xf32>) {
    # CHECK:    %[[VAL_2:.*]] = memref.cast %[[VAL_1]] : memref<*xf32> to memref<2x2xf32>
    # CHECK:    %[[VAL_3:.*]] = call @foo(%[[VAL_2]]) : (memref<2x2xf32>) -> memref<2x2xf32>
    # CHECK:    %[[VAL_4:.*]] = memref.cast %[[VAL_3]] : memref<2x2xf32> to memref<*xf32>
    # CHECK:    call @refbackend_consume_return_callback_first(%[[VAL_4]]) : (memref<*xf32>) -> ()
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_munge_calling_conventions_setup(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def foo(x: ranked_memref_4x4_f32):
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32): ...

    @func
    def foo_wrapper(x: unranked_memref_f32):
        x = cast(ranked_memref_4x4_f32, x)
        y = foo(x)
        y = cast(unranked_memref_f32, y)
        cb(y)

    foo_wrapper.emit()

    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))

    def callback(*args):
        if not len(args):
            print("FAIL")
            return
        results = convert_returns_from_ctype(args, invoker.ret_types)
        if not np.array_equal(results[0], A):
            print("FAIL")
        else:
            print("SUCCESS")

    module = backend.compile(
        ctx.module,
        kernel_name="foo_wrapper",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
        return_consumer=cb._func_op,
    )
    invoker = backend.load(module, consume_return_callback=callback)
    invoker.foo_wrapper(AA)
    out, err = capfd.readouterr()
    assert out.strip() == "SUCCESS"


def test_munge_calling_conventions_setup_mutate(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def foo(x: ranked_memref_4x4_f32):
        el = load(x, (0, 0))
        el = el * constant(2.0, T.f32())
        store(el, x, (0, 0))
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32): ...

    @func
    def foo_wrapper(x: unranked_memref_f32):
        x = cast(ranked_memref_4x4_f32, x)
        y = foo(x)
        y = cast(unranked_memref_f32, y)
        cb(y)

    foo_wrapper.emit()

    # CHECK:  func.func @foo(%[[VAL_0:.*]]: memref<4x4xf32>) -> memref<4x4xf32> {
    # CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<4x4xf32>
    # CHECK:    %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 0 : index
    # CHECK:    memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<4x4xf32>
    # CHECK:    return %[[VAL_0]] : memref<4x4xf32>
    # CHECK:  }
    # CHECK:  func.func private @cb(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
    # CHECK:  func.func @foo_wrapper(%[[VAL_8:.*]]: memref<*xf32>) {
    # CHECK:    %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<*xf32> to memref<4x4xf32>
    # CHECK:    %[[VAL_10:.*]] = call @foo(%[[VAL_9]]) : (memref<4x4xf32>) -> memref<4x4xf32>
    # CHECK:    %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<4x4xf32> to memref<*xf32>
    # CHECK:    call @cb(%[[VAL_11]]) : (memref<*xf32>) -> ()
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)

    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))

    def callback(*args):
        if not len(args):
            print("FAIL")
            return
        results = convert_returns_from_ctype(args, invoker.ret_types)
        A[0, 0] = 2
        if not np.array_equal(results[0], A):
            print("FAIL")
        else:
            print("SUCCESS")

    module = backend.compile(
        ctx.module,
        kernel_name="foo_wrapper",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
        return_consumer=cb._func_op,
    )
    invoker = backend.load(module, consume_return_callback=callback)
    invoker.foo_wrapper(AA)


def test_munge_calling_conventions_setup_auto_cb(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def foo(x: ranked_memref_4x4_f32):
        el = load(x, (0, 0))
        el = el * constant(2.0, T.f32())
        store(el, x, (0, 0))
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32): ...

    @func
    def foo_wrapper(x: unranked_memref_f32):
        x = cast(ranked_memref_4x4_f32, x)
        y = foo(x)
        y = cast(unranked_memref_f32, y)
        cb(y)

    foo_wrapper.emit()

    # CHECK:  func.func @foo(%[[VAL_0:.*]]: memref<4x4xf32>) -> memref<4x4xf32> {
    # CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<4x4xf32>
    # CHECK:    %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 0 : index
    # CHECK:    memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<4x4xf32>
    # CHECK:    return %[[VAL_0]] : memref<4x4xf32>
    # CHECK:  }
    # CHECK:  func.func private @cb(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
    # CHECK:  func.func @foo_wrapper(%[[VAL_8:.*]]: memref<*xf32>) {
    # CHECK:    %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<*xf32> to memref<4x4xf32>
    # CHECK:    %[[VAL_10:.*]] = call @foo(%[[VAL_9]]) : (memref<4x4xf32>) -> memref<4x4xf32>
    # CHECK:    %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<4x4xf32> to memref<*xf32>
    # CHECK:    call @cb(%[[VAL_11]]) : (memref<*xf32>) -> ()
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)

    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))

    module = backend.compile(
        ctx.module,
        kernel_name="foo_wrapper",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
        return_consumer=cb._func_op,
    )
    invoker = backend.load(module)
    results = invoker.foo_wrapper(AA)
    A[0, 0] = 2
    assert np.array_equal(results, A)


def test_munge_calling_conventions_setup_auto_cb_auto_wrapper(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())

    @func
    def foo(x: ranked_memref_4x4_f32):
        el = load(x, (0, 0))
        el = el * constant(2.0, T.f32())
        store(el, x, (0, 0))
        return x

    foo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    module.operation.verify()

    # CHECK:  func.func @foo(%[[VAL_0:.*]]: memref<4x4xf32>) -> memref<4x4xf32> attributes {llvm.emit_c_interface} {
    # CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_2:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<4x4xf32>
    # CHECK:    %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:    %[[VAL_5:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_7:.*]] = arith.constant 0 : index
    # CHECK:    memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<4x4xf32>
    # CHECK:    return %[[VAL_0]] : memref<4x4xf32>
    # CHECK:  }
    # CHECK:  func.func private @foo_return_consumer(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
    # CHECK:  func.func @foo_capi_wrapper(%[[VAL_8:.*]]: memref<4x4xf32>) attributes {llvm.emit_c_interface} {
    # CHECK:    %[[VAL_9:.*]] = call @foo(%[[VAL_8]]) : (memref<4x4xf32>) -> memref<4x4xf32>
    # CHECK:    %[[VAL_10:.*]] = memref.cast %[[VAL_9]] : memref<4x4xf32> to memref<*xf32>
    # CHECK:    call @foo_return_consumer(%[[VAL_10]]) : (memref<*xf32>) -> ()
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_munge_calling_conventions_setup_auto_cb_auto_wrapper_run(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())

    @func
    def foo(x: ranked_memref_4x4_f32):
        el = load(x, (0, 0))
        el = el * constant(2.0, T.f32())
        store(el, x, (0, 0))
        return x

    foo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))

    results = invoker.foo_capi_wrapper(AA)
    A[0, 0] = 2
    assert np.array_equal(results, A)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows can't load runner extras"
)
def test_munge_calling_conventions_setup_auto_cb_auto_wrapper_run_cast_np_array(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32 = T.memref(4, 4, T.f32())
    unranked_memref_f32 = T.memref(element_type=T.f32())

    @func
    def printMemrefF32(x: unranked_memref_f32): ...

    @func
    def foo(x: ranked_memref_4x4_f32):
        el = load(x, (0, 0))
        el = el * constant(2.0, T.f32())
        store(el, x, (0, 0))
        y = cast(unranked_memref_f32, x)
        printMemrefF32(y)
        return

    foo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="foo",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )
    invoker = backend.load(module)
    A = np.ones((4, 4)).astype(np.float32)
    invoker.foo(A)

    correct = dedent(
        """\
    Unranked Memref base@ =  rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data = 
    [[2,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1], 
     [1,   1,   1,   1]]
    """
    )
    out, err = capfd.readouterr()
    filecheck(correct, re.sub(r"0x\w+", "", out))


def test_setting_memref_diagonal(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(mem: ranked_memref_kxk_f32):
        for i, it_mem, _ in range_(0, K, iter_args=[mem]):
            it_mem[i, i] = it_mem[i, i] + it_mem[i, i] * sitofp(
                T.f32(), index_cast(T.i32(), i)
            )
            res = yield it_mem
        return res, res

    memfoo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.ones((K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))

    results, results1 = invoker.memfoo_capi_wrapper(AA)
    assert np.array_equal(np.diagonal(results), np.arange(1, K + 1))
    assert np.array_equal(np.diagonal(results1), np.arange(1, K + 1))


def test_setting_memref_diagonal_no_iter(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(mem: ranked_memref_kxk_f32):
        for i in range_(0, K):
            mem[i, i] = mem[i, i] + mem[i, i] * sitofp(T.f32(), index_cast(T.i32(), i))

    memfoo.emit()
    print(ctx.module)

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.ones((K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))

    invoker.memfoo_capi_wrapper(AA)
    assert np.array_equal(np.diagonal(A), np.arange(1, K + 1))


def test_memref_double_loop(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(mem: ranked_memref_kxk_f32):
        for i, it_mem, _ in range_(0, K, iter_args=[mem]):
            for j, it_mem, _ in range_(0, K, iter_args=[it_mem]):
                it_mem[i, j] = (
                    it_mem[i, j]
                    + it_mem[i, j]
                    + sitofp(T.f32(), index_cast(T.i32(), i))
                    + sitofp(T.f32(), index_cast(T.i32(), j))
                )
                res = yield it_mem
            res = yield res
        return res

    memfoo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.ones((K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))

    results = invoker.memfoo_capi_wrapper(AA)

    B = np.ones((K, K)).astype(np.float32)
    for i in range(K):
        for j in range(K):
            B[i, j] = B[i, j] + B[i, j] + i + j

    assert np.array_equal(results, B)


def test_memref_double_loop_no_iter(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(mem: ranked_memref_kxk_f32):
        for i in range_(0, K):
            for j in range_(0, K):
                mem[i, j] = (
                    mem[i, j]
                    + mem[i, j]
                    + sitofp(T.f32(), index_cast(T.i32(), i))
                    + sitofp(T.f32(), index_cast(T.i32(), j))
                )

    memfoo.emit()

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.ones((K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))

    invoker.memfoo_capi_wrapper(AA)

    B = np.ones((K, K)).astype(np.float32)
    for i in range(K):
        for j in range(K):
            B[i, j] = B[i, j] + B[i, j] + i + j

    assert np.array_equal(A, B)


def _memref_tiled_add(K, D, ctx: MLIRContext, backend: LLVMJITBackend):
    F = K // D
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())
    layout = StridedLayoutAttr.get(S, (K, 1))
    ranked_memref_dxd_f32 = T.memref(D, D, T.f32(), layout=layout)

    @func
    @canonicalize(using=canonicalizer)
    def tile(
        A: ranked_memref_dxd_f32, B: ranked_memref_dxd_f32, C: ranked_memref_dxd_f32
    ):
        for i in range_(0, D):
            for j in range_(0, D):
                C[i, j] = A[i, j] + B[i, j]

    tile.emit()

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(
        A: ranked_memref_kxk_f32, B: ranked_memref_kxk_f32, C: ranked_memref_kxk_f32
    ):
        for i in range_(0, F):
            for j in range_(0, F):
                l = lambda l: l * D
                r = lambda r: (r + 1) * D
                a, b, c = (
                    A[l(i) : r(i), l(j) : r(j)],
                    B[l(i) : r(i), l(j) : r(j)],
                    C[l(i) : r(i), l(j) : r(j)],
                )
                tile(a, b, c)

    memfoo.emit()
    module = run_pipeline(ctx.module, str(Pipeline().cse()))
    correct = dedent(
        f"""\
    module {{
      func.func @tile(%arg0: memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>, %arg1: memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>, %arg2: memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>) {{
        %c0 = arith.constant 0 : index
        %c{D} = arith.constant {D} : index
        %c1 = arith.constant 1 : index
        scf.for %arg3 = %c0 to %c{D} step %c1 {{
          scf.for %arg4 = %c0 to %c{D} step %c1 {{
            %0 = memref.load %arg0[%arg3, %arg4] : memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
            %1 = memref.load %arg1[%arg3, %arg4] : memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
            %2 = arith.addf %0, %1 : f32
            memref.store %2, %arg2[%arg3, %arg4] : memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
          }}
        }}
        return
      }}
      func.func @memfoo(%arg0: memref<{K}x{K}xf32>, %arg1: memref<{K}x{K}xf32>, %arg2: memref<{K}x{K}xf32>) {{
        %c0 = arith.constant 0 : index
        %c{F} = arith.constant {F} : index
        %c1 = arith.constant 1 : index
        scf.for %arg3 = %c0 to %c{F} step %c1 {{
          scf.for %arg4 = %c0 to %c{F} step %c1 {{
            %c{D} = arith.constant {D} : index
            %0 = arith.muli %arg3, %c{D} : index
            %1 = arith.addi %arg3, %c1 : index
            %2 = arith.muli %arg4, %c{D} : index
            %3 = arith.addi %arg4, %c1 : index
            %subview = memref.subview %arg0[%0, %2] [{D}, {D}] [1, 1] : memref<{K}x{K}xf32> to memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
            %subview_0 = memref.subview %arg1[%0, %2] [{D}, {D}] [1, 1] : memref<{K}x{K}xf32> to memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
            %subview_1 = memref.subview %arg2[%0, %2] [{D}, {D}] [1, 1] : memref<{K}x{K}xf32> to memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>
            func.call @tile(%subview, %subview_0, %subview_1) : (memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>, memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>, memref<{D}x{D}xf32, strided<[{K}, 1], offset: ?>>) -> ()
          }}
        }}
        return
      }}
    }}
    """
    )
    filecheck(correct, module)

    module = Module.parse(str(module))

    module = backend.compile(
        module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
    )
    invoker = backend.load(module)
    A = np.random.randint(0, 10, (K, K)).astype(np.float32)
    B = np.random.randint(0, 10, (K, K)).astype(np.float32)
    C = np.zeros((K, K)).astype(np.float32)

    invoker.memfoo(A, B, C)
    assert np.array_equal(A + B, C)


def test_memref_tiled_add_1(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 32
    D = 4
    _memref_tiled_add(K, D, ctx, backend)


def test_memref_tiled_add_2(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 64
    D = 4
    _memref_tiled_add(K, D, ctx, backend)


def test_memref_tiled_add_3(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 64
    D = 16
    _memref_tiled_add(K, D, ctx, backend)


def test_memref_tiled_add_4(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 128
    D = 16
    _memref_tiled_add(K, D, ctx, backend)


def test_memref_tiled_add_5(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 256
    D = 32
    _memref_tiled_add(K, D, ctx, backend)


def test_linalg(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 256
    D = 32
    F = K // D
    ranked_memref_kxk_f32 = T.memref(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def memfoo(
        A: ranked_memref_kxk_f32, B: ranked_memref_kxk_f32, C: ranked_memref_kxk_f32
    ):
        for i in range_(0, F):
            for j in range_(0, F):
                l = lambda l: l * D
                r = lambda r: (r + 1) * D
                a, b, c = (
                    A[l(i) : r(i), l(j) : r(j)],
                    B[l(i) : r(i), l(j) : r(j)],
                    C[l(i) : r(i), l(j) : r(j)],
                )
                linalg.add(a, b, c)

    memfoo.emit()

    module = run_pipeline(ctx.module, str(Pipeline().cse()))

    # CHECK:  func.func @memfoo(%[[VAL_0:.*]]: memref<256x256xf32>, %[[VAL_1:.*]]: memref<256x256xf32>, %[[VAL_2:.*]]: memref<256x256xf32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_4:.*]] = arith.constant 8 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:    scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] {
    # CHECK:      scf.for %[[VAL_7:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] {
    # CHECK:        %[[VAL_8:.*]] = arith.constant 32 : index
    # CHECK:        %[[VAL_9:.*]] = arith.muli %[[VAL_6]], %[[VAL_8]] : index
    # CHECK:        %[[VAL_10:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : index
    # CHECK:        %[[VAL_11:.*]] = arith.muli %[[VAL_7]], %[[VAL_8]] : index
    # CHECK:        %[[VAL_12:.*]] = arith.addi %[[VAL_7]], %[[VAL_5]] : index
    # CHECK:        %[[VAL_13:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_9]], %[[VAL_11]]] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
    # CHECK:        %[[VAL_14:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_11]]] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
    # CHECK:        %[[VAL_15:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_11]]] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
    # CHECK:        linalg.add ins(%[[VAL_13]], %[[VAL_14]] : memref<32x32xf32, strided<[256, 1], offset: ?>>, memref<32x32xf32, strided<[256, 1], offset: ?>>) outs(%[[VAL_15]] : memref<32x32xf32, strided<[256, 1], offset: ?>>)
    # CHECK:      }
    # CHECK:    }
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(module)

    module = backend.compile(
        module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().convert_linalg_to_loops().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=True,
        generate_return_consumer=True,
    )
    invoker = backend.load(module)
    A = np.random.randint(0, 10, (K, K)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))
    B = np.random.randint(0, 10, (K, K)).astype(np.float32)
    BB = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(B)))
    C = np.zeros((K, K)).astype(np.float32)
    CC = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(C)))

    invoker.memfoo_capi_wrapper(AA, BB, CC)
    assert np.array_equal(A + B, C)


def test_linalg_tensor(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 256
    D = 32
    F = K // D
    ranked_tensor_kxk_f32 = T.tensor(K, K, T.f32())

    @func
    @canonicalize(using=canonicalizer)
    def tenfoo(
        A: ranked_tensor_kxk_f32, B: ranked_tensor_kxk_f32, C: ranked_tensor_kxk_f32
    ):
        x = linalg.add(A, B, C)
        print(x)

    tenfoo.emit()

    # CHECK:  func.func @tenfoo(%[[VAL_0:.*]]: tensor<256x256xf32>, %[[VAL_1:.*]]: tensor<256x256xf32>, %[[VAL_2:.*]]: tensor<256x256xf32>) {
    # CHECK:    %[[VAL_3:.*]] = linalg.add ins(%[[VAL_0]], %[[VAL_1]] : tensor<256x256xf32>, tensor<256x256xf32>) outs(%[[VAL_2]] : tensor<256x256xf32>) -> tensor<256x256xf32>
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)

    # TODO(max)
    # module = backend.compile(
    #     ctx.module,
    #     kernel_name=tenfoo.__name__,
    #     pipeline=Pipeline()
    #     .bufferize()
    #     .convert_linalg_to_loops()
    #     .bufferize()
    #     .lower_to_llvm(),
    #     generate_kernel_wrapper=True,
    #     generate_return_consumer=True,
    # )
    # invoker = backend.load(module)
    # A = np.random.randint(0, 10, (K, K)).astype(np.float32)
    # AA = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))
    # B = np.random.randint(0, 10, (K, K)).astype(np.float32)
    # BB = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(B)))
    # C = np.zeros((K, K)).astype(np.float32)
    # CC = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(C)))
    #
    # invoker.memfoo_capi_wrapper(AA, BB, CC)
    # # assert np.array_equal(A + B, C)


def test_raii_context():
    def foo():
        ctx = RAIIMLIRContext()
        mod = Module.create()
        with InsertionPoint(mod.body):

            @func(emit=True)
            def foo(x: T.i32()):
                return x

        # CHECK:  func.func @foo(%[[VAL_0:.*]]: i32) -> i32 {
        # CHECK:    return %[[VAL_0]] : i32
        # CHECK:  }

        filecheck_with_comments(mod)

    foo()

    assert Context.current is None


def test_explicit_module():
    ctx = RAIIMLIRContext()
    mod = ExplicitlyManagedModule()

    @func(emit=True)
    def foo(x: T.i32()):
        return x

    mod.finish()

    # CHECK:  func.func @foo(%[[VAL_0:.*]]: i32) -> i32 {
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(mod)
