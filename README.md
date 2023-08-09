# mlir-python-utils

The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.

Note, this is **not a Python compiler**, but just a (hopefully) nice way to emit MLIR using python.

Note, also, there are no docs (because ain't no one got time for that) but that shouldn't be a problem because the package is designed such that you can use/reuse only the pieces/parts you want/understand.

## Install

### TL;DR

```shell
$ pip install mlir-python-utils -f https://makslevental.github.io/wheels/
$ configure-mlir-python-utils <YOUR BINDINGS>
```

## Examples/Demo

There is a convenience `extra_requires` in `pyproject.toml` such that you can install stock upstream bindings like this:

```shell
$ pip install mlir-python-utils[mlir] -f https://makslevental.github.io/wheels/
```

Then configure

```shell
$ configure-mlir-python-utils mlir
```

and then there's a [examples/mwe.py](examples/mwe.py) but also check [tests](tests).

## Details

This package is meant to work in concert with the upstream bindings.
Practically speaking that means you need to have *some* package installed that includes mlir python bindings.
In addition, you have to do one of two things to **configure this package** (after installing it):

1. `$ configure-mlir-python-utils -y <MLIR_PYTHON_PACKAGE_PREFIX>`, where `MLIR_PYTHON_PACKAGE_PREFIX` is (as it says)
   the
   package prefix for your chosen upstream bindings. So for example, for `torch-mlir`, you would
   execute `configure-mlir-python-utils torch_mlir`, since `torch-mlir`'s bindings are the root of the `torch-mlir`
   python package.
   **When in doubt about this prefix**, it is everything up until `ir` when you import your bindings (e.g., as
   `torch_mlir` in `import torch_mlir.ir`).
2. `$ export MLIR_PYTHON_PACKAGE_PREFIX=<MLIR_PYTHON_PACKAGE_PREFIX>`, i.e., you can set this string as an environment
   variable each time you use this package. Note, in this case, if you want to make use of the "op trampolines", you
   still need to run `generate_trampolines.generate_all_upstream_trampolines()` by hand at least once.
