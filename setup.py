import os
import shutil
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import setup, find_namespace_packages
from setuptools.command.install import install

# TODO: find from extras maybe
HOST_MLIR_PYTHON_PACKAGE_PREFIX = os.environ.get(
    "HOST_MLIR_PYTHON_PACKAGE_PREFIX", "mlir"
)
PACKAGE_NAME = f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX.replace('.', '-').replace('_', '-')}-python-extras"


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


packages = [
    f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras.{p}"
    for p in find_namespace_packages(where="mlir/extras")
] + [f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras"]


class MyInstallData(install):
    def run(self):
        from llvm import amdgcn

        shutil.copy(
            amdgcn.__file__,
            Path(__file__).parent
            / HOST_MLIR_PYTHON_PACKAGE_PREFIX
            / "extras"
            / "dialects"
            / "ext"
            / "llvm"
            / "amdgcn.py",
        )
        
        install.run(self)


setup(
    name=PACKAGE_NAME,
    version="0.0.7",
    description="The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.",
    license="LICENSE",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "test": ["pytest", "mlir-native-tools", "astpretty"],
        "torch-mlir": ["torch-mlir-core"],
        "jax": ["jax[cpu]"],
        "mlir": ["mlir-python-bindings"],
    },
    python_requires=">=3.8",
    packages=packages,
    # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras": "mlir/extras",
    },
    cmdclass={"install": MyInstallData},
)
