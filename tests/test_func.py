import inspect
import sys
import threading
from typing import TypeVar

import mlir.extras.types as T
import pytest
from mlir.ir import FunctionType

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import mlir_mod_ctx, RAIIMLIRContextModule
from mlir.extras.dialects.ext import linalg, arith, scf
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

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

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  func.func private @demo_fun1() -> i32
    # CHECK:  func.func private @demo_fun2() -> (i32, i32)
    # CHECK:  func.func private @demo_fun3(i32) -> (i32, i32)
    # CHECK:  func.func private @demo_fun4(i32, i32) -> (i32, i32)
    # CHECK:  %[[VAL_0:.*]] = func.call @demo_fun1() : () -> i32
    # CHECK:  %[[VAL_1:.*]]:2 = func.call @demo_fun2() : () -> (i32, i32)
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_3:.*]]:2 = func.call @demo_fun3(%[[VAL_2]]) : (i32) -> (i32, i32)
    # CHECK:  %[[VAL_4:.*]]:2 = func.call @demo_fun4(%[[VAL_2]], %[[VAL_2]]) : (i32, i32) -> (i32, i32)

    filecheck_with_comments(ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1.emit()
    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_no_context():
    @func
    def foo1():
        one = constant(1)
        return one

    with mlir_mod_ctx() as mod_ctx:
        foo1()

        # CHECK:  func.func @foo1() -> i32 {
        # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
        # CHECK:    return %[[VAL_0]] : i32
        # CHECK:  }
        # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

        filecheck_with_comments(mod_ctx.module)


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

    # CHECK:  func.func @matmul_i32_i32(%[[VAL_0:.*]]: memref<16x16xi32>, %[[VAL_1:.*]]: memref<16x16xi32>, %[[VAL_2:.*]]: memref<16x16xi32>) {
    # CHECK:    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<16x16xi32>, memref<16x16xi32>) outs(%[[VAL_2]] : memref<16x16xi32>)
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_just_args(ctx: MLIRContext):
    @func(generics=generics)
    def mat_product_kernel(
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1.0, dtype)

    mat_product_kernel[32, 32, 32, T.f32()].emit()

    # CHECK:  func.func @mat_product_kernel(%[[VAL_0:.*]]: memref<32x32xf32>, %[[VAL_1:.*]]: memref<32x32xf32>, %[[VAL_2:.*]]: memref<32x32xf32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  func.func @mat_product_kernel(%[[VAL_0:.*]]: memref<32x32xi32>, %[[VAL_1:.*]]: memref<32x32xi32>, %[[VAL_2:.*]]: memref<32x32xi32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  func.func @mat_product_kernel(%[[VAL_0:.*]]: memref<32x32xf32>, %[[VAL_1:.*]]: memref<32x32xf32>, %[[VAL_2:.*]]: memref<32x32xf32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
    # CHECK:    %[[VAL_7:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_8:.*]] = arith.constant 32 : index
    # CHECK:    %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_12:.*]] = %[[VAL_6]]) -> (f32) {
    # CHECK:      %[[VAL_13:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_11]]] : memref<32x32xf32>
    # CHECK:      %[[VAL_14:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_11]], %[[VAL_4]]] : memref<32x32xf32>
    # CHECK:      %[[VAL_15:.*]] = math.fma %[[VAL_13]], %[[VAL_14]], %[[VAL_12]] : f32
    # CHECK:      scf.yield %[[VAL_15]] : f32
    # CHECK:    }
    # CHECK:    %[[VAL_16:.*]] = arith.addf %[[VAL_17:.*]], %[[VAL_5]] : f32
    # CHECK:    memref.store %[[VAL_16]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<32x32xf32>
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


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

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(tls.ctx.module)


def test_explicit_function_type(ctx: MLIRContext):
    input_types = [T.i32(), T.i32()]
    result_types = [T.i32()]
    func_type = FunctionType.get(input_types, result_types)

    @func(function_type=func_type)
    def demo_fun1(a, b):
        one = constant(1)
        return one

    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 {
    # CHECK:    %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_2]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)
