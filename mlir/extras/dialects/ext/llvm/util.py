import warnings

# noinspection PyUnresolvedReferences
from .....dialects.llvm import *
from .....ir import Type, F16Type, F32Type, F64Type, BF16Type, IntegerType

try:
    from llvm import intrinsic_is_overloaded, intrinsic_get_type, print_type_to_string
    from llvm import types_
    from llvm.context import context as llvm_context
except ImportError:
    warnings.warn(
        "llvm bindings not installed; call_intrinsic won't work without supplying return type explicitly"
    )


def mlir_type_to_llvm_type(mlir_type, llvm_ctx):
    if F16Type.isinstance(mlir_type):
        return types_.half_type_in_context(llvm_ctx)
    if F32Type.isinstance(mlir_type):
        return types_.float_type_in_context(llvm_ctx)
    if F64Type.isinstance(mlir_type):
        return types_.double_type_in_context(llvm_ctx)
    if BF16Type.isinstance(mlir_type):
        return types_.b_float_type_in_context(llvm_ctx)
    if IntegerType.isinstance(mlir_type):
        return types_.int_type_in_context(llvm_ctx, mlir_type.width)

    raise NotImplementedError(f"{mlir_type} is not supported")


def llvm_type_str_to_mlir_type(llvm_type: str):
    if llvm_type.startswith("<"):
        return Type.parse(f"vector{llvm_type}")
    if llvm_type == "float":
        return F32Type.get()
    raise NotImplementedError(f"{llvm_type} is not supported")


_call_intrinsic = call_intrinsic


def call_intrinsic(*args, **kwargs):
    intr_id = kwargs.pop("intr_id")
    intr_name = kwargs.pop("intr_name")
    mlir_ret_type = kwargs.pop("return_type", None)
    if mlir_ret_type:
        return _call_intrinsic(mlir_ret_type, intr_name, args, [], [])

    is_overloaded = kwargs.pop("is_overloaded", None)
    if is_overloaded is None:
        is_overloaded = intrinsic_is_overloaded(intr_id)
    with llvm_context() as ctx:
        types = []
        if is_overloaded:
            types = [mlir_type_to_llvm_type(a.type, ctx.context) for a in args]
        intr_decl_fn_ty = intrinsic_get_type(ctx.context, intr_id, types)

    ret_type_str = print_type_to_string(intr_decl_fn_ty).split(" (")[0].strip()
    mlir_ret_type = None
    if ret_type_str:
        mlir_ret_type = llvm_type_str_to_mlir_type(ret_type_str)

    return _call_intrinsic(mlir_ret_type, intr_name, args, [], [])


call_intrinsic_ = call_intrinsic
