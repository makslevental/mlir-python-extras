import inspect
import sys
import threading
from textwrap import dedent
from typing import TypeVar

import pytest

import mlir.extras.types as T

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import mlir_mod_ctx, RAIIMLIRContextModule
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import func
from mlir.extras.dialects.ext import linalg, arith, scf, memref

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_emit(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()
    correct = dedent(
        """\
    module {
      func.func @demo_fun1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_declare_byte_rep(ctx: MLIRContext):
    def demo_fun1(): ...

    if sys.version_info.minor == 13:
        assert demo_fun1.__code__.co_code == b"\x95\x00g\x00"
    elif sys.version_info.minor == 12:
        assert demo_fun1.__code__.co_code == b"\x97\x00y\x00"
    elif sys.version_info.minor == 11:
        assert demo_fun1.__code__.co_code == b"\x97\x00d\x00S\x00"
    elif sys.version_info.minor in {8, 9, 10}:
        assert demo_fun1.__code__.co_code == b"d\x00S\x00"
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


def test_declare(ctx: MLIRContext):
    @func
    def demo_fun1() -> T.i32(): ...

    @func
    def demo_fun2() -> (T.i32(), T.i32()): ...

    @func
    def demo_fun3(x: T.i32()) -> (T.i32(), T.i32()): ...

    @func
    def demo_fun4(x: T.i32(), y: T.i32()) -> (T.i32(), T.i32()): ...

    demo_fun1()
    demo_fun2()
    one = constant(1)
    demo_fun3(one)
    demo_fun4(one, one)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func private @demo_fun1() -> i32
      func.func private @demo_fun2() -> (i32, i32)
      func.func private @demo_fun3(i32) -> (i32, i32)
      func.func private @demo_fun4(i32, i32) -> (i32, i32)
      %0 = func.call @demo_fun1() : () -> i32
      %1:2 = func.call @demo_fun2() : () -> (i32, i32)
      %c1_i32 = arith.constant 1 : i32
      %2:2 = func.call @demo_fun3(%c1_i32) : (i32) -> (i32, i32)
      %3:2 = func.call @demo_fun4(%c1_i32, %c1_i32) : (i32, i32) -> (i32, i32)
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1.emit()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_no_context():
    @func
    def foo1():
        one = constant(1)
        return one

    with mlir_mod_ctx() as mod_ctx:
        foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, mod_ctx.module)


generics = M, K, N, dtype = list(map(TypeVar, ["M", "K", "N", "dtype"]))


@func(generics=list(map(TypeVar, ["M", "N"])))
def matmul_i32_i32(
    A: "T.memref(M, N, T.i32())",
    B: "T.memref(M, N, T.i32())",
    C: "T.memref(M, N, T.i32())",
):
    linalg.matmul(A, B, C)


def test_func_no_context_2(ctx: MLIRContext):
    matmul_i32_i32[16, 16].emit()
    correct = dedent(
        """\
    module {
      func.func @matmul_i32_i32(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) {
        linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<16x16xi32>, memref<16x16xi32>) outs(%arg2 : memref<16x16xi32>)
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_generics_just_args(ctx: MLIRContext):

    @func(generics=generics)
    def mat_product_kernel(
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1.0, dtype)

    mat_product_kernel[32, 32, 32, T.f32()].emit()
    correct = dedent(
        """\
    module {
      func.func @mat_product_kernel(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
        %cst = arith.constant 1.000000e+00 : f32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_generics_closure(ctx: MLIRContext):
    generics = M, K, N, dtype = list(map(TypeVar, ["M", "K", "N", "dtype"]))

    @func(generics=generics)
    def mat_product_kernel(
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1, dtype)

    mat_product_kernel[32, 32, 32, T.i32()].emit()
    correct = dedent(
        """\
    module {
      func.func @mat_product_kernel(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) {
        %c1_i32 = arith.constant 1 : i32
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_generics_with_canonicalizations(ctx: MLIRContext):

    generics = M, K, N, dtype = list(map(TypeVar, ["M", "K", "N", "dtype"]))

    @func(generics=generics)
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def mat_product_kernel(
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        x = arith.constant(1, index=True)
        y = arith.constant(1, index=True)
        one = arith.constant(1.0, type=dtype)
        tmp = arith.constant(0, type=dtype)
        for k, tmp, _ in scf.range_(K, iter_args=[tmp]):
            tmp += A[x, k] * B[k, y]
            tmp = yield tmp
        C[x, y] = tmp + one

    mat_product_kernel[32, 32, 32, T.f32()].emit()
    correct = dedent(
        """\
    module {
      func.func @mat_product_kernel(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        %cst = arith.constant 1.000000e+00 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        %0 = scf.for %arg3 = %c0 to %c32 step %c1_2 iter_args(%arg4 = %cst_1) -> (f32) {
          %2 = memref.load %arg0[%c1, %arg3] : memref<32x32xf32>
          %3 = memref.load %arg1[%arg3, %c1_0] : memref<32x32xf32>
          %4 = math.fma %2, %3, %arg4 : f32
          scf.yield %4 : f32
        }
        %1 = arith.addf %0, %cst : f32
        memref.store %1, %arg2[%c1, %c1_0] : memref<32x32xf32>
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_raii_mlir_context_module():
    tls = threading.local()
    tls.ctx = RAIIMLIRContextModule()

    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()
    correct = dedent(
        """\
    module {
      func.func @demo_fun1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, tls.ctx.module)
