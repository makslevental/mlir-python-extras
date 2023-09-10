import os

from pip._internal.req import parse_requirements
from setuptools import setup

# TODO: find from extras maybe
MLIR_PYTHON_PACKAGE_PREFIX = os.environ.get("MLIR_PYTHON_PACKAGE_PREFIX", "mlir")
PACKAGE_NAME = (
    f"{MLIR_PYTHON_PACKAGE_PREFIX.replace('.', '-').replace('_', '-')}-python-utils"
)


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
    ###
    package_dir={
        # lhs is package namespace, rhs is path (relative to this setup.py)
        f"{MLIR_PYTHON_PACKAGE_PREFIX}.utils": "mlir/utils",
    },
    entry_points={
        "console_scripts": [
            f"{PACKAGE_NAME}-generate-trampolines = {MLIR_PYTHON_PACKAGE_PREFIX}.utils._configuration:generate_trampolines.generate_trampolines",
            f"{PACKAGE_NAME}-generate-all-upstream-trampolines = {MLIR_PYTHON_PACKAGE_PREFIX}.utils._configuration:generate_trampolines.generate_all_upstream_trampolines",
        ],
    },
)
