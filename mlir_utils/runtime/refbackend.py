# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from __future__ import annotations

import ctypes
import logging
import os
import sys
import tempfile
import warnings
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from mlir import _mlir_libs
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.ir import StringAttr, UnitAttr, Module
from mlir.runtime import (
    UnrankedMemRefDescriptor,
    get_unranked_memref_descriptor,
    get_ranked_memref_descriptor,
)

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


memref_type_to_np_dtype = {
    "mrf16": np.float16,
    "mrf32": np.float32,
    "mrf64": np.float64,
    "mri1": np.bool_,
    "mri8": np.int8,
    "mri32": np.int32,
    "mri64": np.int64,
}
elemental_type_to_ctype = {
    "i1": ctypes.c_bool,
    "i8": ctypes.c_byte,
    "i64": ctypes.c_int,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}

CONSUME_RETURN_FUNC_PREFIX = "refbackend_consume_func_return_"


class MlirCompilerError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


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
        # Put something descriptive here even if description is empty.
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
        raise MlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr

    return module


def get_return_funcs(module):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    return_funcs = []
    with module.context:
        for func in module.body:
            # Returns strings of the form `"refbackend.."` so `"` is deleted.
            func_name = str(func.attributes["sym_name"]).replace('"', "")
            if func_name[:return_prefix_len] == CONSUME_RETURN_FUNC_PREFIX:
                return_funcs.append(func_name)

    return return_funcs


def get_ctype_func(func_name):
    return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    ret_types = func_name[return_prefix_len:].split("_")
    ctypes_arg = [None]
    for type in ret_types:
        if type in elemental_type_to_ctype:
            ctypes_arg.append(elemental_type_to_ctype[type])
        elif type in memref_type_to_np_dtype:
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
        else:
            assert False, f"Not supported type: {type}"

    return ctypes.CFUNCTYPE(*ctypes_arg), ret_types


# https://stackoverflow.com/a/68198336/9045206
CData = ctypes._SimpleCData.__mro__[-2]


class LLVMJITBackendInvoker:
    return_func: Optional[Callable] = None

    def __init__(
        self, module, consume_return_func=None, opt_level=2, shared_lib_paths=None
    ):
        if shared_lib_paths is None:
            shared_lib_paths = []
        self.ee = ExecutionEngine(
            module, opt_level=opt_level, shared_libs=shared_lib_paths
        )
        if consume_return_func is not None:
            return_funcs = get_return_funcs(module)
            assert len(return_funcs) == 1, f"multiple return funcs not supported"
            self.return_func = return_funcs[0]
            ctype_wrapper, ret_types = get_ctype_func(self.return_func)
            self.ret_types = ret_types
            self.ee.register_runtime(
                self.return_func, ctype_wrapper(consume_return_func)
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
                            if _get("return_func") is not None
                            else ctypes.pointer(get_ranked_memref_descriptor(arg))
                        )
                    )

            self.ee.invoke(function_name, *ffi_args)

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

    def compile(
        self,
        module: Module,
        pipeline: str,
        kernel_name="main",
        enable_ir_printing=False,
    ):
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

        return run_pipeline(
            module,
            pipeline=pipeline,
            description="Lowering IR",
            enable_ir_printing=enable_ir_printing,
        )

    def load(
        self, module, consume_return_func=None, opt_level=2
    ) -> LLVMJITBackendInvoker:
        return LLVMJITBackendInvoker(
            module,
            opt_level=opt_level,
            shared_lib_paths=[str(p.absolute()) for p in self.shared_lib_paths],
            consume_return_func=consume_return_func,
        )
