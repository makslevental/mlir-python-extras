import ctypes
import re
from textwrap import dedent

import numpy as np
import pytest
from mlir.ir import UnitAttr
from mlir.runtime import get_unranked_memref_descriptor, get_ranked_memref_descriptor

from mlir_utils.ast.canonicalize import canonicalize
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func
from mlir_utils.dialects.ext.memref import load, store
from mlir_utils.dialects.ext.scf import (
    canonicalizer,
    range_,
)
from mlir_utils.dialects.memref import cast
from mlir_utils.runtime.passes import Pipeline
from mlir_utils.runtime.refbackend import (
    LLVMJITBackend,
    convert_returns_from_ctype,
    refback_cb_attr,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext, backend
from mlir_utils.types import memref_t, f32_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")
pytest.mark.usefixtures("backend")


def test_smoke(ctx: MLIRContext, backend: LLVMJITBackend, capfd):
    # TODO(max): ValueError: foo requires closure of length 0, not 1
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def printMemrefF32(x: unranked_memref_f32_t):
        ...

    @func
    @canonicalize(using=canonicalizer)
    def foo(x: unranked_memref_f32_t):
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
    correct = dedent(
        """\
    module attributes {llvm.data_layout = ""} {
      llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
      llvm.func @foo(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
        llvm.call @printMemrefF32(%arg0, %arg1) : (i64, !llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr) attributes {llvm.emit_c_interface} {
        %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
        %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr)> 
        %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)> 
        llvm.call @foo(%1, %2) : (i64, !llvm.ptr) -> ()
        llvm.return
      }
    }
    """
    )
    filecheck(correct, ctx.module)

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
    ranked_memref_2x2_f32_t = memref_t(2, 2, f32_t)
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def foo(x: ranked_memref_2x2_f32_t):
        return x

    foo.emit()

    @func
    def refbackend_consume_return_callback_first(x: unranked_memref_f32_t):
        ...

    @func
    def foo_wrapper(x: unranked_memref_f32_t):
        x = cast(ranked_memref_2x2_f32_t, x)
        y = foo(x)
        y = cast(unranked_memref_f32_t, y)
        refbackend_consume_return_callback_first(y)

    foo_wrapper.emit()

    correct = dedent(
        """\
    module {
      func.func @foo(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
        return %arg0 : memref<2x2xf32>
      }
      func.func private @refbackend_consume_return_callback_first(memref<*xf32>)
      func.func @foo_wrapper(%arg0: memref<*xf32>) {
        %cast = memref.cast %arg0 : memref<*xf32> to memref<2x2xf32>
        %0 = call @foo(%cast) : (memref<2x2xf32>) -> memref<2x2xf32>
        %cast_0 = memref.cast %0 : memref<2x2xf32> to memref<*xf32>
        call @refbackend_consume_return_callback_first(%cast_0) : (memref<*xf32>) -> ()
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_munge_calling_conventions_setup(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32_t):
        ...

    @func
    def foo_wrapper(x: unranked_memref_f32_t):
        x = cast(ranked_memref_4x4_f32_t, x)
        y = foo(x)
        y = cast(unranked_memref_f32_t, y)
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
    )
    invoker = backend.load(module, consume_return_callback=callback)
    invoker.foo_wrapper(AA)
    out, err = capfd.readouterr()
    assert out.strip() == "SUCCESS"


def test_munge_calling_conventions_setup_mutate(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        el = load(x, (0, 0))
        el = el * constant(2.0, f32_t)
        store(el, x, (0, 0))
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32_t):
        ...

    @func
    def foo_wrapper(x: unranked_memref_f32_t):
        x = cast(ranked_memref_4x4_f32_t, x)
        y = foo(x)
        y = cast(unranked_memref_f32_t, y)
        cb(y)

    foo_wrapper.emit()

    correct = dedent(
        """\
    module {
      func.func @foo(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %arg0[%c0, %c0_0] : memref<4x4xf32>
        %cst = arith.constant 2.000000e+00 : f32
        %1 = arith.mulf %0, %cst : f32
        %c0_1 = arith.constant 0 : index
        %c0_2 = arith.constant 0 : index
        memref.store %1, %arg0[%c0_1, %c0_2] : memref<4x4xf32>
        return %arg0 : memref<4x4xf32>
      }
      func.func private @cb(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
      func.func @foo_wrapper(%arg0: memref<*xf32>) {
        %cast = memref.cast %arg0 : memref<*xf32> to memref<4x4xf32>
        %0 = call @foo(%cast) : (memref<4x4xf32>) -> memref<4x4xf32>
        %cast_0 = memref.cast %0 : memref<4x4xf32> to memref<*xf32>
        call @cb(%cast_0) : (memref<*xf32>) -> ()
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)

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
    )
    invoker = backend.load(module, consume_return_callback=callback)
    invoker.foo_wrapper(AA)


def test_munge_calling_conventions_setup_auto_cb(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        el = load(x, (0, 0))
        el = el * constant(2.0, f32_t)
        store(el, x, (0, 0))
        return x

    foo.emit()

    @func(
        func_attrs={
            "llvm.emit_c_interface": UnitAttr.get(),
            refback_cb_attr: UnitAttr.get(),
        }
    )
    def cb(x: unranked_memref_f32_t):
        ...

    @func
    def foo_wrapper(x: unranked_memref_f32_t):
        x = cast(ranked_memref_4x4_f32_t, x)
        y = foo(x)
        y = cast(unranked_memref_f32_t, y)
        cb(y)

    foo_wrapper.emit()

    correct = dedent(
        """\
    module {
      func.func @foo(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %arg0[%c0, %c0_0] : memref<4x4xf32>
        %cst = arith.constant 2.000000e+00 : f32
        %1 = arith.mulf %0, %cst : f32
        %c0_1 = arith.constant 0 : index
        %c0_2 = arith.constant 0 : index
        memref.store %1, %arg0[%c0_1, %c0_2] : memref<4x4xf32>
        return %arg0 : memref<4x4xf32>
      }
      func.func private @cb(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
      func.func @foo_wrapper(%arg0: memref<*xf32>) {
        %cast = memref.cast %arg0 : memref<*xf32> to memref<4x4xf32>
        %0 = call @foo(%cast) : (memref<4x4xf32>) -> memref<4x4xf32>
        %cast_0 = memref.cast %0 : memref<4x4xf32> to memref<*xf32>
        call @cb(%cast_0) : (memref<*xf32>) -> ()
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    A = np.ones((4, 4)).astype(np.float32)
    AA = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(A)))

    module = backend.compile(
        ctx.module,
        kernel_name="foo_wrapper",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )
    invoker = backend.load(module)
    results = invoker.foo_wrapper(AA)
    A[0, 0] = 2
    assert np.array_equal(results[0], A)


def test_munge_calling_conventions_setup_auto_cb_auto_wrapper(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        el = load(x, (0, 0))
        el = el * constant(2.0, f32_t)
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

    correct = dedent(
        """\
    module {
      func.func @foo(%arg0: memref<4x4xf32>) -> memref<4x4xf32> attributes {llvm.emit_c_interface} {
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %arg0[%c0, %c0_0] : memref<4x4xf32>
        %cst = arith.constant 2.000000e+00 : f32
        %1 = arith.mulf %0, %cst : f32
        %c0_1 = arith.constant 0 : index
        %c0_2 = arith.constant 0 : index
        memref.store %1, %arg0[%c0_1, %c0_2] : memref<4x4xf32>
        return %arg0 : memref<4x4xf32>
      }
      func.func private @foo_return_consumer(memref<*xf32>) attributes {llvm.emit_c_interface, refbackend_consume_return_callback}
      func.func @foo_capi_wrapper(%arg0: memref<4x4xf32>) attributes {llvm.emit_c_interface} {
        %0 = call @foo(%arg0) : (memref<4x4xf32>) -> memref<4x4xf32>
        %cast_0 = memref.cast %0 : memref<4x4xf32> to memref<*xf32>
        call @foo_return_consumer(%cast_0) : (memref<*xf32>) -> ()
        return
      }
    }
    """
    )
    filecheck(correct, module)


def test_munge_calling_conventions_setup_auto_cb_auto_wrapper_run(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        el = load(x, (0, 0))
        el = el * constant(2.0, f32_t)
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
    assert np.array_equal(results[0], A)


def test_munge_calling_conventions_setup_auto_cb_auto_wrapper_run_cast_np_array(
    ctx: MLIRContext, backend: LLVMJITBackend, capfd
):
    ranked_memref_4x4_f32_t = memref_t(4, 4, f32_t)
    unranked_memref_f32_t = memref_t(element_type=f32_t)

    @func
    def printMemrefF32(x: unranked_memref_f32_t):
        ...

    @func
    def foo(x: ranked_memref_4x4_f32_t):
        el = load(x, (0, 0))
        el = el * constant(2.0, f32_t)
        store(el, x, (0, 0))
        y = cast(unranked_memref_f32_t, x)
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


def test_matmul(ctx: MLIRContext, backend: LLVMJITBackend):
    M, N, K = 4, 16, 8

    @func
    @canonicalize(using=canonicalizer)
    def matmul(
        A: memref_t(M, N, f32_t),
        B: memref_t(N, K, f32_t),
        C: memref_t(M, K, f32_t),
    ):
        for i in range_(0, M):
            for j in range_(0, N):
                for k in range_(0, K):
                    C[i, k] += A[i, j] * B[j, k]
                    yield
                yield
            yield

    matmul.emit()

    correct = dedent(
        """\
    module {
      func.func @matmul(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        scf.for %arg3 = %c0 to %c4 step %c1 {
          %c0_0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_1 = arith.constant 1 : index
          scf.for %arg4 = %c0_0 to %c16 step %c1_1 {
            %c0_2 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1_3 = arith.constant 1 : index
            scf.for %arg5 = %c0_2 to %c8 step %c1_3 {
              %0 = memref.load %arg2[%arg3, %arg5] : memref<4x8xf32>
              %1 = memref.load %arg0[%arg3, %arg4] : memref<4x16xf32>
              %2 = memref.load %arg1[%arg4, %arg5] : memref<16x8xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = arith.addf %0, %3 : f32
              memref.store %4, %arg2[%arg3, %arg5] : memref<4x8xf32>
            }
          }
        }
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    module = backend.compile(
        ctx.module,
        kernel_name="matmul",
        pipeline=Pipeline().bufferize().lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)
    C = np.zeros((M, K)).astype(np.float32)
    backend.load(module).matmul(A, B, C)
    assert np.allclose(A @ B, C)
