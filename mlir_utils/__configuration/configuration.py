import argparse
import os
import sys
from pathlib import Path

from mlir_utils.__configuration._module_alias_map import (
    get_meta_path_insertion_index,
    AliasedModuleFinder,
)

THIS_DIR = Path(__file__).resolve().parent
MLIR_PYTHON_PACKAGE_PREFIX_FILE_PATH = THIS_DIR / "__MLIR_PYTHON_PACKAGE_PREFIX__"
__MLIR_PYTHON_PACKAGE_PREFIX__ = None


def load_upstream_bindings():
    global __MLIR_PYTHON_PACKAGE_PREFIX__

    if MLIR_PYTHON_PACKAGE_PREFIX_FILE_PATH.exists():
        with open(MLIR_PYTHON_PACKAGE_PREFIX_FILE_PATH) as f:
            __MLIR_PYTHON_PACKAGE_PREFIX__ = f.read().strip()

    if os.getenv("MLIR_PYTHON_PACKAGE_PREFIX"):
        __MLIR_PYTHON_PACKAGE_PREFIX__ = os.getenv("MLIR_PYTHON_PACKAGE_PREFIX")
    if __MLIR_PYTHON_PACKAGE_PREFIX__ is not None:
        sys.meta_path.insert(
            get_meta_path_insertion_index(),
            AliasedModuleFinder({"mlir": __MLIR_PYTHON_PACKAGE_PREFIX__}),
        )
    elif not (
        sys.argv[0].endswith("configure-mlir-utils")
        or ("-m" in sys.orig_argv and "mlir_utils.__configuration" in sys.orig_argv)
    ):
        raise Exception(
            "mlir-utils not configured and MLIR_PYTHON_PACKAGE_PREFIX env variable not set"
        )


def configure_host_bindings():
    parser = argparse.ArgumentParser(
        prog="configure-mlir-utils",
        description="Configure mlir-utils",
    )
    parser.add_argument("-y", "--yes", action="store_true", default=False)
    parser.add_argument("mlir_python_package_prefix")
    args = parser.parse_args()
    mlir_python_package_prefix = args.mlir_python_package_prefix
    assert mlir_python_package_prefix, "missing mlir_python_package_prefix"
    mlir_python_package_prefix = (
        mlir_python_package_prefix.replace("'", "").replace('"', "").strip()
    )

    if bool(__MLIR_PYTHON_PACKAGE_PREFIX__):
        print(
            f'mlir_python_package_prefix has already been set to "{__MLIR_PYTHON_PACKAGE_PREFIX__}"'
        )
        if not args.yes:
            answer = input("do you want to reset? [y/n]: ")
            if answer.lower() not in {"1", "true", "yes", "y"}:
                return

    if not args.yes:
        answer = input(f"new {mlir_python_package_prefix=}; continue? [y/n]: ")
        if answer.lower() not in {"1", "true", "yes", "y"}:
            return
    else:
        print(f"new {mlir_python_package_prefix=}")

    # check if valid package/module
    try:
        _host_bindings_mlir = __import__(f"{mlir_python_package_prefix}._mlir_libs")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"couldn't import {mlir_python_package_prefix=} due to: {e}")
        raise e

    with open(MLIR_PYTHON_PACKAGE_PREFIX_FILE_PATH, "w") as f:
        f.write(mlir_python_package_prefix)

    load_upstream_bindings()

    from ..dialects.generate_trampolines import generate_all_upstream_trampolines

    generate_all_upstream_trampolines()
