import os

from pip._internal.req import parse_requirements
from setuptools import setup

# TODO: find from extras maybe
HOST_MLIR_PYTHON_PACKAGE_PREFIX = os.environ.get(
    "HOST_MLIR_PYTHON_PACKAGE_PREFIX", "mlir"
)
PACKAGE_NAME = f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX.replace('.', '-').replace('_', '-')}-python-utils"


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


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
    python_requires=">=3.10",
    # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.utils": "mlir/utils",
    },
)
