# mlir-python-extras

[![Test](https://github.com/makslevental/mlir-python-extras/actions/workflows/test.yml/badge.svg)](https://github.com/makslevental/mlir-python-extras/actions/workflows/test.yml)
[![Release](https://github.com/makslevental/mlir-python-extras/actions/workflows/release.yml/badge.svg)](https://github.com/makslevental/mlir-python-extras/actions/workflows/release.yml)

The missing pieces (as far as boilerplate reduction goes) of the MLIR python bindings.

* [TL;DR](#tl-dr)
* [5s Intro](#5s-intro)
* [Install](#install)
* [Examples/Demo](#examples-demo)

## TL;DR

Full example at [examples/mwe.py](examples/mwe.py) (i.e., go there if you want to copy-paste).

Turn this 

```python
K = 10
memref_i64 = T.memref(K, K, T.i64)

@func
@canonicalize(using=scf)
def memfoo(A: memref_i64, B: memref_i64, C: memref_i64):
    one = constant(1)
    two = constant(2)
    if one > two:
        three = constant(3)
    else:
        for i in range(0, K):
            for j in range(0, K):
                C[i, j] = A[i, j] * B[i, j]
```

into this

```mlir
func.func @memfoo(%arg0: memref<10x10xi64>, %arg1: memref<10x10xi64>, %arg2: memref<10x10xi64>) {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.cmpi ugt, %c1_i32, %c2_i32 : i32
  scf.if %0 {
    %c3_i32 = arith.constant 3 : i32
  } else {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c10 step %c1 {
      scf.for %arg4 = %c0 to %c10 step %c1 {
        %1 = memref.load %arg0[%arg3, %arg4] : memref<10x10xi64>
        %2 = memref.load %arg1[%arg3, %arg4] : memref<10x10xi64>
        %3 = arith.muli %1, %2 : i64
        memref.store %3, %arg2[%arg3, %arg4] : memref<10x10xi64>
      }
    }
  }
  return
}
```

then run it like this

```python
module = backend.compile(
    ctx.module,
    kernel_name=memfoo.__name__,
    pipeline=Pipeline().bufferize().lower_to_llvm(),
)

A = np.random.randint(0, 10, (K, K))
B = np.random.randint(0, 10, (K, K))
C = np.zeros((K, K), dtype=int)

backend.load(module).memfoo(A, B, C)
assert np.array_equal(A * B, C)
```

## 5s Intro

This is **not a Python compiler**, but just a (hopefully) nice way to emit MLIR using python.

The few main features/affordances:

1. `region_op`s (like `@func` above)
   \
   &nbsp;
   1. These are decorators around ops (bindings for MLIR operations) that have regions (e.g., [in_parallel](https://github.com/makslevental/mlir-python-extras/blob/40c8c44e767fd9ee8fec35a3c3342b3c8623ef1c/mlir/extras/dialects/ext/scf.py#L205)). 
   They turn decorated functions, by executing them "eagerly", into an instance of such an op, e.g., 
      ```python
      @func
      def foo(x: T.i32):
         return
      ```
      becomes `func.func @foo(%arg0: i32) { }`; if the region carrying op produces a result, the identifier for the python function (`foo`) becomes the corresponding `ir.Value` of the result (if the op doesn't produce a result then the identifier becomes the corresponding `ir.OpView`).
      \
      \
      This has been upstreamed to [mlir/python/mlir/extras/meta.py](https://github.com/llvm/llvm-project/blob/24038650d9ca5d66b07d3075afdebe81012ab1f2/mlir/python/mlir/extras/meta.py#L12)
      \
      &nbsp;
2. `@canonicalize` (like `@canonicalize(using=scf)` above)
   \
   &nbsp;
   1. These are decorators that **rewrite the python AST**. They transform a select few forms (basically only `if`s) into a more "canonical" form, in order to more easily map to MLIR. If that scares you, fear not; they are not essential and all target MLIR can still be mapped to without using them (by using the slightly more verbose `region_op`).
      \
      \
      See [mlir/extras.ast.canonicalize](https://github.com/makslevental/mlir-python-extras/blob/40c8c44e767fd9ee8fec35a3c3342b3c8623ef1c/mlir/extras/ast/canonicalize.py) for details.
      \
      &nbsp;
3. `mlir/extras.types` (like `T.memref(K, K, T.i64)` above)
   \
   &nbsp;
   1. These are just convenient wrappers around upstream type constructors. Note, because MLIR types are uniqued to a `ir.Context`, these are all actually functions that return the type.
      \
      \
      These have been upstreamed to [mlir/python/mlir/extras/types.py](https://github.com/llvm/llvm-project/blob/52b18b4e82d412a7d755e89591c6ebcc41c257a1/mlir/python/mlir/extras/types.py)
      \
      &nbsp;
4. `Pipeline()`
   \
   &nbsp;
   1. This is just a (generated) wrapper around available **upstream** passes; it can be used to build pass pipelines (by `str(Pipeline())`). It is mainly convenient with IDEs/editors that will tab-complete the available methods on the `Pipeline` class (which correspond to passes), Note, if your host bindings don't register some upstream passes, then this will generate "illegal" pass pipelines.
      \
      \
      See [scripts/generate_pass_pipeline.py](https://github.com/makslevental/mlir-python-extras/blob/40c8c44e767fd9ee8fec35a3c3342b3c8623ef1c/scripts/generate_pass_pipeline.py) for details on generation
      [mlir/extras.runtime.passes.py](https://github.com/makslevental/mlir-python-extras/blob/40c8c44e767fd9ee8fec35a3c3342b3c8623ef1c/mlir/extras/runtime/passes.py#L80) for the passes themselves.
      \
      &nbsp;



Note, also, there are no docs (because ain't no one got time for that) but that shouldn't be a problem because the package is designed such that you can use/reuse only the pieces/parts you want/understand.
But, open an issue if something isn't clear.


## Install

If you want to just get started/play around:

```shell
$ pip install mlir-python-extras -f https://makslevental.github.io/wheels
```

Alternatively, this [colab notebook](https://drive.google.com/file/d/1NAtf2Yxj_VVnzwn8u_kxtajfVzgbuWhi/view?usp=sharing) (which is the same as [examples/mlir_python_extras.ipynb](examples/mlir_python_extras.ipynb)) has a MWE if you don't want to install anything even.

In reality, this package is meant to work in concert with "host bindings" (some distribution of the actual MLIR Python bindings).
Practically speaking that means you need to have *some* package installed that includes mlir python bindings.

So that means the second line should be amended to

```shell
$ HOST_MLIR_PYTHON_PACKAGE_PREFIX=<YOUR_HOST_MLIR_PYTHON_PACKAGE_PREFIX> \
      pip install git+https://github.com/makslevental/mlir-python-extras
```

where `YOUR_HOST_MLIR_PYTHON_PACKAGE_PREFIX` is (as it says) the package prefix for your chosen host bindings.
**When in doubt about this prefix**, it is everything up until `ir` when you import your bindings, e.g., in `import torch_mlir.ir`, `torch_mlir` is the `HOST_MLIR_PYTHON_PACKAGE_PREFIX` for the torch-mlir bindings.

## Examples/Demo

Check [examples](examples) and [tests](tests) for a plethora of example code.
