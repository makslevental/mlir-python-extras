import os
import shutil
from datetime import datetime
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools_scm.git import run_git

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# TODO: find from extras maybe
HOST_MLIR_PYTHON_PACKAGE_PREFIX = os.environ.get(
    "HOST_MLIR_PYTHON_PACKAGE_PREFIX", "mlir"
)
PACKAGE_NAME = f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX.replace('.', '-').replace('_', '-')}-python-extras"


def check_env(build):
    return os.environ.get(build, 0) in {"1", "true", "True", "ON", "YES"}


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


class MyInstallData(build_py):
    def run(self):
        build_py.run(self)
        build_dir = os.path.join(
            *([self.build_lib] + HOST_MLIR_PYTHON_PACKAGE_PREFIX.split("."))
        )
        try:
            from llvm import amdgcn

            shutil.copy(
                amdgcn.__file__,
                Path(build_dir) / "extras" / "dialects" / "ext" / "llvm" / "amdgcn.py",
            )
        except ImportError:
            pass


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        pass


now = datetime.now()
version_s = f"0.0.8.{now.year}{now.month:02}{now.day:02}{now.hour:02}+"


local_version = []
GPU = os.getenv("GPU", None)
if GPU not in {None, "none"}:
    local_version += [GPU]

try:
    short_hash = run_git(
        ["rev-parse", "--short", "HEAD"],
        Path(__file__).parent,
    ).parse_success(
        parse=str,
        error_msg="branch err (abbrev-err)",
    )
except Exception as e:
    short_hash = "no-hash"

if local_version:
    version_s += ".".join(local_version + [short_hash])
else:
    version_s += short_hash

packages = (
    [HOST_MLIR_PYTHON_PACKAGE_PREFIX]
    + [
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras.{p}"
        for p in find_namespace_packages(where="mlir/extras")
    ]
    + [f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras"]
)

description = "The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings."
BINDINGS_VERSION = os.getenv("BINDINGS_VERSION", None)
if BINDINGS_VERSION is not None:
    description += f" Includes {BINDINGS_VERSION}"

cmdclass = {"build_py": MyInstallData}
ext_modules = []
if bool(os.getenv("BUNDLE_MLIR_PYTHON_BINDINGS", False)):
    cmdclass["build_ext"] = CMakeBuild
    ext_modules += [CMakeExtension(HOST_MLIR_PYTHON_PACKAGE_PREFIX, sourcedir=".")]

setup(
    name=PACKAGE_NAME,
    version=version_s,
    description=description,
    license="LICENSE",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "test": ["pytest", "mlir-native-tools", "astpretty"],
        "torch-mlir": ["torch-mlir-core"],
        "jax": ["jax[cpu]"],
        "mlir": ["mlir-python-bindings"],
        "eudsl": ["eudsl-llvmpy"],
    },
    python_requires=">=3.8",
    include_package_data=True,
    packages=packages,
    # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}": "mlir",
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras": "mlir/extras",
    },
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
