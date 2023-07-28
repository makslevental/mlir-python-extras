import ctypes
import logging
import os
import sys
import tempfile
import warnings
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
from mlir import _mlir_libs
from mlir.execution_engine import ExecutionEngine
from mlir.ir import StringAttr, UnitAttr, Module
from mlir.passmanager import PassManager
from mlir.runtime import (
    UnrankedMemRefDescriptor,
    get_unranked_memref_descriptor,
    get_ranked_memref_descriptor,
    unranked_memref_to_numpy,
)

from mlir_utils.runtime.passes import Pipeline
from mlir_utils.types import memref_type_to_np_dtype, mlir_type_to_ctype
from mlir_utils.util import disable_multithreading, shlib_ext, find_ops

logger = logging.getLogger(__name__)


# adapted from https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py


def assert_arg_type_is_supported(ty):
    SUPPORTED = [
        np.float16,
        np.float32,
        np.float64,
        np.uint8,
        np.int8,
        np.int32,
        np.int64,
        np.bool_,
    ]
    assert (
        ty in SUPPORTED
    ), f"Only numpy arrays with dtypes in {SUPPORTED} are supported, but got {ty}"


class MlirCompilerError(Exception):
    pass


def get_module_name_for_debug_dump(module):
    if "debug_module_name" not in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: str,
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Lower module in place to make it ready for compiler backends.
        with ExitStack() as stack:
            stack.enter_context(module.context)
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10,
                enable_debug_info=True,
            )
            pm = PassManager.parse(pipeline)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                stack.enter_context(disable_multithreading())
                pm.enable_ir_printing()

            pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:

            {'*' * 80}
            {sys.stderr.getvalue().strip()}
            {'*' * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise MlirCompilerError(trimmed_message)
    finally:
        sys.stderr = original_stderr

    return module


CONSUME_RETURN_CALLBACK_ATTR = "refbackend_consume_return_callback"
refback_cb_attr = CONSUME_RETURN_CALLBACK_ATTR


def get_return_func(module):
    with module.context:
        for func in module.body:
            if CONSUME_RETURN_CALLBACK_ATTR in func.attributes:
                return func.operation


def get_ctype_func(ret_types):
    ctypes_arg = [None]
    legal_ret_types = []
    for type in ret_types:
        if ctype := mlir_type_to_ctype(type):
            ctypes_arg.append(ctype)
            legal_ret_types.append(type)
        elif memref_type_to_np_dtype(type):
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
            legal_ret_types.append(type)
        else:
            warnings.warn(f"Not supported type for callback return: {type=}")

    return ctypes.CFUNCTYPE(*ctypes_arg), legal_ret_types


def convert_returns(args, mlir_types):
    return tuple(
        arg
        if mlir_type_to_ctype(type)
        else unranked_memref_to_numpy(arg, memref_type_to_np_dtype(type))
        for arg, type in zip(args, mlir_types)
    )


# https://stackoverflow.com/a/68198336/9045206
CData = ctypes._SimpleCData.__mro__[-2]


class LLVMJITBackendInvoker:
    return_func: bool

    def __init__(
        self,
        module,
        opt_level=2,
        shared_lib_paths=None,
        return_func_types=None,
        return_func_name=None,
        consume_return_callback=None,
    ):
        if shared_lib_paths is None:
            shared_lib_paths = []
        self.ee = ExecutionEngine(
            module, opt_level=opt_level, shared_libs=shared_lib_paths
        )
        self.results = None
        if return_func_types is not None:
            assert (
                return_func_name is not None
            ), f"must provide return func name when providing return func types"
            ctype_wrapper, ret_types = get_ctype_func(return_func_types)
            self.ret_types = ret_types
            self.return_func = True
            if consume_return_callback is None:

                def consume_return_callback(*args):
                    self.results = convert_returns(args, self.ret_types)

            self.ee.register_runtime(
                return_func_name,
                ctype_wrapper(consume_return_callback),
            )

    def __getattr__(self, function_name: str):
        _get = super().__getattribute__

        def invoke(*args):
            ffi_args = []
            for arg in args:
                if isinstance(arg, CData):
                    ffi_args.append(arg)
                else:
                    assert_arg_type_is_supported(arg.dtype)
                    ffi_args.append(
                        ctypes.pointer(
                            # TODO(max): this is a hack to handle refbackend
                            # in principle this has nothing to do with anything
                            # refbackend related
                            ctypes.pointer(get_unranked_memref_descriptor(arg))
                            if _get("return_func")
                            else ctypes.pointer(get_ranked_memref_descriptor(arg))
                        )
                    )

            self.ee.invoke(function_name, *ffi_args)
            return self.results

        return invoke


if ASYNC_RUNTIME_LIB_PATH := os.getenv("ASYNC_RUNTIME_LIB_PATH"):
    ASYNC_RUNTIME_LIB_PATH = Path(ASYNC_RUNTIME_LIB_PATH)
else:
    ASYNC_RUNTIME_LIB_PATH = (
        Path(_mlir_libs.__file__).parent / f"libmlir_async_runtime.{shlib_ext()}"
    )
if not ASYNC_RUNTIME_LIB_PATH.exists():
    warnings.warn(f"{ASYNC_RUNTIME_LIB_PATH=} doesn't exist")

if C_RUNNER_UTILS_LIB_PATH := os.getenv("C_RUNNER_UTILS_LIB_PATH"):
    C_RUNNER_UTILS_LIB_PATH = Path(C_RUNNER_UTILS_LIB_PATH)
else:
    C_RUNNER_UTILS_LIB_PATH = (
        Path(_mlir_libs.__file__).parent / f"libmlir_c_runner_utils.{shlib_ext()}"
    )

if not C_RUNNER_UTILS_LIB_PATH.exists():
    warnings.warn(f"{C_RUNNER_UTILS_LIB_PATH=} doesn't exist")

if RUNNER_UTILS_LIB_PATH := os.getenv("RUNNER_UTILS_LIB_PATH"):
    RUNNER_UTILS_LIB_PATH = Path(RUNNER_UTILS_LIB_PATH)
else:
    RUNNER_UTILS_LIB_PATH = (
        Path(_mlir_libs.__file__).parent / f"libmlir_runner_utils.{shlib_ext()}"
    )
if not RUNNER_UTILS_LIB_PATH.exists():
    warnings.warn(f"{RUNNER_UTILS_LIB_PATH=} doesn't exist")


class LLVMJITBackend:
    def __init__(
        self,
        shared_lib_paths=None,
    ):
        if shared_lib_paths is None:
            shared_lib_paths = [
                ASYNC_RUNTIME_LIB_PATH,
                C_RUNNER_UTILS_LIB_PATH,
                RUNNER_UTILS_LIB_PATH,
            ]
        if shared_lib_paths is None:
            shared_lib_paths = []
        self.shared_lib_paths = shared_lib_paths
        self.return_func_types = None
        self.return_func_name = None

    def compile(
        self,
        module: Module,
        pipeline: str | Pipeline,
        kernel_name="main",
        enable_ir_printing=False,
    ):
        pipeline = str(pipeline)

        def cb(op):
            try:
                return kernel_name == op.sym_name.value
            except:
                return False

        needs_cface = "to-llvm" in pipeline

        if needs_cface:
            kernel_func = find_ops(module.operation, cb)
            assert len(kernel_func) == 1, f"kernel func {kernel_name} not found"
            kernel_func[0].attributes["llvm.emit_c_interface"] = UnitAttr.get()

        return_func = get_return_func(module)
        if return_func:
            self.return_func_name = return_func.attributes["sym_name"].value
            # this is confusing but it's because the callback takes as operands the return values it's going to consume
            self.return_func_types = [
                i for i in return_func.attributes["function_type"].value.inputs
            ]

        return run_pipeline(
            module,
            pipeline=pipeline,
            description="Lowering IR",
            enable_ir_printing=enable_ir_printing,
        )

    def load(
        self, module, consume_return_callback=None, opt_level=2
    ) -> LLVMJITBackendInvoker:
        return LLVMJITBackendInvoker(
            module,
            opt_level=opt_level,
            shared_lib_paths=[str(p.absolute()) for p in self.shared_lib_paths],
            return_func_types=self.return_func_types,
            return_func_name=self.return_func_name,
            consume_return_callback=consume_return_callback,
        )
