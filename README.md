# mlir-python-utils

The missing pieces (as far as boilerplate reduction goes) of the MLIR python bindings.

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
   1. These are decorators around ops (bindings for MLIR operations) that have regions (e.g., [in_parallel](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/dialects/ext/scf.py#L185)). 
   They turn decorated functions, by executing them "eagerly", into an instance of such an op, e.g., 
      ```python
      @func
      def foo(x: T.i32):
         return
      ```
      becomes `func.func @foo(%arg0: i32) { }`; if the region carrying op produces a result, the identifier for the python function (`foo`) becomes the corresponding `ir.Value` of the result (if the op doesn't produce a result then the identifier becomes the corresponding `ir.OpView`).
      \
      \
      See [mlir_utils.util.op_region_builder](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/util.py#L123) for details.
      \
      &nbsp;
2. `@canonicalize` (like `@canonicalize(using=scf)` above)
   \
   &nbsp;
   1. These are decorators that **rewrite the python AST**. They transform a select few forms (basically only `if`s) into a more "canonical" form, in order to more easily map to MLIR. If that scares you, fear not; they are not essential and all target MLIR can still be mapped to without using them (by using the slightly more verbose `region_op`).
      \
      \
      See [mlir_utils.ast.canonicalize](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/ast/canonicalize.py) for details.
      \
      &nbsp;
3. `mlir_utils.types` (like `T.memref(K, K, T.i64)` above)
   \
   &nbsp;
   1. These are just convenient wrappers around upstream type constructors. Note, because MLIR types are uniqued to a `ir.Context`, these are all actually functions that return the type (yes, even `T.i64`, which uses [`__getattr__` on the module](https://github.com/makslevental/mlir-python-utils/blob/2ca62e9c1540b1624c302bc9efb4666ff5d1c133/mlir_utils/types.py#L98)).
      \
      \
      See [mlir_utils.types](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/types.py) for details.
      \
      &nbsp;
4. `Pipeline()`
   \
   &nbsp;
   1. This is just a (generated) wrapper around available **upstream** passes; it can be used to build pass pipelines (by `str(Pipeline())`). It is mainly convenient with IDEs/editors that will tab-complete the available methods on the `Pipeline` class (which correspond to passes), Note, if your host bindings don't register some upstream passes, then this will generate "illegal" pass pipelines.
      \
      \
      See [mlir_utils._configuration.generate_pass_pipeline.py](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/_configuration/generate_pass_pipeline.py) for details on generation
      [mlir_utils.runtime.passes.py](https://github.com/makslevental/mlir-python-utils/blob/a9885db18096a610d29a26293396d860d40ad213/mlir_utils/runtime/passes.py#L80) for the passes themselves.
      \
      &nbsp;



Note, also, there are no docs (because ain't no one got time for that) but that shouldn't be a problem because the package is designed such that you can use/reuse only the pieces/parts you want/understand.
But, open an issue if something isn't clear.


## Install

This package is meant to work in concert with host bindings.
Practically speaking that means you need to have *some* package installed that includes mlir python bindings.

So

```shell
$ YOUR_HOST_MLIR_PYTHON_PACKAGE_PREFIX=<YOUR_HOST_MLIR_PYTHON_PACKAGE_PREFIX> pip install git+https://github.com/makslevental/mlir-python-utils
```

where `YOUR_HOST_MLIR_PYTHON_PACKAGE_PREFIX` is (as it says) the package prefix for your chosen host bindings.
**When in doubt about this prefix**, it is everything up until `ir` when you import your bindings, e.g., in `import torch_mlir.ir`, `torch_mlir` is the `HOST_MLIR_PYTHON_PACKAGE_PREFIX` for the torch-mlir bindings.

If you don't have any such package, but you want to experiment anyway, you can install the "stock" upstream bindings first:

```shell
$ pip install mlir-python-bindings -f https://makslevental.github.io/wheels/
```

and then

```shell
$ pip install git+https://github.com/makslevental/mlir-python-utils
```

## Examples/Demo

Check [tests](tests) for a plethora of example code.