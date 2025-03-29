import random
import time

import mlir.extras.types as T
import numpy as np
from hip import hip

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith

# noinspection PyUnresolvedReferences
from mlir.extras.dialects.ext.gpu import (
    all_reduce,
    wait,
    thread_attr as thread,
    block_idx,
    thread_idx,
    block_dim,
    GPUModuleMeta,
    func as gpu_func,
    set_container_module,
    launch,
    all_reduce_,
    module,
    get_compile_object_bytes,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from util import hip_check, launch_kernel, hip_synchronize

# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref

ctx = RAIIMLIRContextModule()
set_container_module(ctx.module)

M, K, N = 1024, 1024, 1024


@gpu_func
@canonicalize(using=scf.canonicalizer)
def kernel1_naive(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(K, N, T.f32())
):
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x
    if (arith.index_cast(row, to=T.i32()) < M) & (
        arith.index_cast(col, to=T.i32()) < N
    ):
        acc = arith.constant(0.0)
        for k, acc, _ in scf.range_(K, iter_args=[acc]):
            acc += A[row, k] * B[k, col]
            acc = yield acc
        C[row, col] = acc


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode()


@module("kernels", [f'#rocdl.target<chip = "{arch}">'])
def gpu_module():
    kernel1_naive.emit()


gpu_module.operation.verify()

lowered_module = run_pipeline(
    gpu_module,
    Pipeline()
    .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True))
    .rocdl_attach_target(chip=arch)
    .gpu_to_llvm()
    .lower_to_llvm()
    .gpu_module_to_binary(),
)

hsaco = get_compile_object_bytes(lowered_module)
hip_module = hip_check(hip.hipModuleLoadData(hsaco))

a_h = np.random.randint(0, 10, (M, K)).astype(dtype=np.float32)
b_h = np.random.randint(0, 10, (K, N)).astype(dtype=np.float32)
c_h = -3 * np.ones((M, N), dtype=np.float32)

a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize
c_num_bytes = c_h.size * c_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes))
b_d = hip_check(hip.hipMalloc(b_num_bytes))
c_d = hip_check(hip.hipMalloc(c_num_bytes))

hip_check(hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

gridX = max(M // 32, 1)
gridY = max(M // 8, 1)
gridZ = 1
warp_size = 32
num_warps = 8
stream = 0
shared_memory = 0

times = {
    kernel1_naive.__name__: 0,
}
runs = 10
start, stop = hip.hipEventCreate(), hip.hipEventCreate()
for i in range(runs):
    kernels = [kernel1_naive]
    random.shuffle(kernels)
    for kernel in kernels:
        hip_check(
            hip.hipMemcpy(
                a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
            )
        )
        hip_check(
            hip.hipMemcpy(
                b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
            )
        )
        function = hip_check(
            hip.hipModuleGetFunction(hip_module, kernel.__name__.encode())
        )

        start = time.monotonic()
        launch_kernel(
            function.as_c_void_p(),
            gridX,
            gridY,
            gridZ,
            warp_size,
            num_warps,
            stream,
            shared_memory,
            a_d,
            b_d,
            c_d,
        )
        hip_synchronize()
        if i > 0:
            times[kernel.__name__] += time.monotonic() - start

        hip_check(
            hip.hipMemcpy(
                c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        assert np.allclose(a_h @ b_h, c_h)

for k in times:
    times[k] /= runs

for k, v in times.items():
    print(f"{k}: {v:.3e}ms")
