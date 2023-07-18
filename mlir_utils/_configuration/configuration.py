import argparse
import hashlib
import os
import sys
from base64 import urlsafe_b64encode
from importlib.metadata import distribution, packages_distributions
from importlib.resources import files
from pathlib import Path

from .module_alias_map import get_meta_path_insertion_index, AliasedModuleFinder

__MLIR_PYTHON_PACKAGE_PREFIX__ = "__MLIR_PYTHON_PACKAGE_PREFIX__"
PACKAGE = __package__.split(".")[0]
PACKAGE_ROOT_PATH = files(PACKAGE)
DIST = distribution(packages_distributions()[PACKAGE][0])
MLIR_PYTHON_PACKAGE_PREFIX_TOKEN_PATH = (
    Path(__file__).parent / __MLIR_PYTHON_PACKAGE_PREFIX__
).absolute()


def _add_file_to_sources_txt_file(file_path: Path):
    assert file_path.exists(), f"file being added doesn't exist at {file_path}"
    relative_file_path = Path(PACKAGE) / file_path.relative_to(PACKAGE_ROOT_PATH)
    if DIST._read_files_egginfo() is not None:
        with open(DIST._path / "SOURCES.txt", "a") as sources_file:
            sources_file.write(f"\n{relative_file_path}")
    if DIST._read_files_distinfo():
        with open(file_path, "rb") as file, open(
            DIST._path / "RECORD", "a"
        ) as sources_file:
            # https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file
            m = hashlib.sha256()
            file = file.read()
            m.update(file)
            encoded = urlsafe_b64encode(m.digest())
            sources_file.write(
                f"{relative_file_path},sha256={encoded[:-1].decode()},{len(file)}\n"
            )


def _get_mlir_package_prefix():
    mlir_python_package_prefix = None
    if MLIR_PYTHON_PACKAGE_PREFIX_TOKEN_PATH.exists():
        with open(MLIR_PYTHON_PACKAGE_PREFIX_TOKEN_PATH) as f:
            mlir_python_package_prefix = f.read().strip()

    if os.getenv("MLIR_PYTHON_PACKAGE_PREFIX"):
        mlir_python_package_prefix = os.getenv("MLIR_PYTHON_PACKAGE_PREFIX")

    return mlir_python_package_prefix


def alias_upstream_bindings():
    if mlir_python_package_prefix := _get_mlir_package_prefix():
        sys.meta_path.insert(
            get_meta_path_insertion_index(),
            AliasedModuleFinder({"mlir": mlir_python_package_prefix}),
        )
        return True
    elif not (
        sys.argv[0].endswith("configure-mlir-python-utils")
        or ("-m" in sys.orig_argv and __package__ in sys.orig_argv)
    ):
        raise Exception(
            "mlir-python-utils not configured and MLIR_PYTHON_PACKAGE_PREFIX env variable not set"
        )
    return False


def configure_host_bindings():
    parser = argparse.ArgumentParser(
        prog="configure-mlir-python-utils",
        description="Configure mlir-python-utils",
    )
    parser.add_argument("-y", "--yes", action="store_true", default=False)
    parser.add_argument("mlir_python_package_prefix")
    args = parser.parse_args()
    mlir_python_package_prefix = args.mlir_python_package_prefix
    mlir_python_package_prefix = (
        mlir_python_package_prefix.replace("'", "").replace('"', "").strip()
    )

    if current_mlir_python_package_prefix := _get_mlir_package_prefix():
        print(
            f'mlir_python_package_prefix has already been set to "{current_mlir_python_package_prefix}"'
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

    with open(MLIR_PYTHON_PACKAGE_PREFIX_TOKEN_PATH, "w") as f:
        f.write(mlir_python_package_prefix)

    _add_file_to_sources_txt_file(MLIR_PYTHON_PACKAGE_PREFIX_TOKEN_PATH)

    alias_upstream_bindings()

    from .generate_trampolines import generate_all_upstream_trampolines

    generate_all_upstream_trampolines()
