# mlir-utils

The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.

## Install

This package is meant to work in concert with the upstream bindings.
Practically speaking that means you need to have *some* package installed that includes mlir python bindings.
In addition, you have to do one of two things to **configure this package** (after installing it):

1. `$ configure-mlir-utils -y <MLIR_PYTHON_PACKAGE_PREFIX>`, where `MLIR_PYTHON_PACKAGE_PREFIX` is (as it says) the
   package prefix for your chosen upstream bindings. So for example, for `torch-mlir`, you would
   execute `configure-mlir-utils torch_mlir`, since `torch-mlir`'s bindings are the root of the `torch-mlir` python
   package. **When in doubt about this prefix**, it is everything up until `ir` (e.g., as
   in `from torch_mlir import ir`).
2. `$ export MLIR_PYTHON_PACKAGE_PREFIX=<MLIR_PYTHON_PACKAGE_PREFIX>`, i.e., you can set this string as an environment
   variable each time you use this package. Note, in this case, if you want to make use of the "op trampolines", you
   still need to run `generate_trampolines.generate_all_upstream_trampolines()` by hand at least once.

There is a convenience `extra_requires` in `pyproject.toml` such that you can do this:

```shell
$ pip install .[mlir] -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
$ configure-mlir-utils mlir
```

# Examples

Check out the [tests](tests).

## Dev

```shell
# you need setuptools >= 64 for build_editable
pip install setuptools -U
pip install -e .[torch-mlir-test] \
   -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest \
   -f https://llvm.github.io/torch-mlir/package-index/
```