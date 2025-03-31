import time

import mlir.extras.types as T
import numpy as np
from hip import hip

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, gpu
from mlir.dialects import rocdl

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
    lds_space,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from util import hip_check, launch_kernel, hip_synchronize

# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref

ctx = RAIIMLIRContextModule()
set_container_module(ctx.module)

M, K, N = 16, 16, 16


@gpu_func
@canonicalize(using=scf.canonicalizer)
def kernel1_naive(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
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


BN = BK = TILE_SIZE = 16
dtype = T.f32()


@gpu_func
@canonicalize(using=scf.canonicalizer)
def kernel2_lds(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    As = memref.get_global("A_shared")
    Bs = memref.get_global("B_shared")

    thread_row = thread_idx.y
    thread_col = thread_idx.x

    row = block_idx.y * TILE_SIZE + thread_row
    col = block_idx.x * TILE_SIZE + thread_col

    sum = arith.constant(0.0)

    for t, sum, _ in scf.range_(0, N, BK, iter_args=[sum]):
        As[thread_row, thread_col] = A[row, t + thread_col]
        Bs[thread_row, thread_col] = B[t + thread_row, col]

        rocdl.s_barrier()

        for k, sum, _ in scf.range_(K, iter_args=[sum]):
            sum += As[thread_row, k] * Bs[k, thread_col]
            sum = yield sum

        rocdl.s_barrier()
        sum = yield sum

    C[row, col] = sum


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode()


@module("kernels", [f'#rocdl.target<chip = "{arch}">'])
def gpu_module():
    A_shared = memref.global_(
        sym_name="A_shared",
        type=T.memref(TILE_SIZE, TILE_SIZE, T.f32(), memory_space=lds_space()),
    )
    B_shared = memref.global_(
        sym_name="B_shared",
        type=T.memref(TILE_SIZE, TILE_SIZE, T.f32(), memory_space=lds_space()),
    )
    kernel1_naive.emit()
    kernel2_lds.emit()


gpu_module.operation.verify()
print(gpu_module)

lowered_module = run_pipeline(
    gpu_module,
    Pipeline()
    .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True))
    .rocdl_attach_target(chip=arch)
    .gpu_to_llvm()
    .lower_to_llvm(),
)

print(lowered_module)

lowered_module = run_pipeline(lowered_module, Pipeline().gpu_module_to_binary())

hsaco = get_compile_object_bytes(lowered_module)
hip_module = hip_check(hip.hipModuleLoadData(hsaco))

a_h = np.random.randint(0, 10, (M, K)).astype(dtype=np.float32)
b_h = np.random.randint(0, 10, (K, N)).astype(dtype=np.float32)

a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes))
b_d = hip_check(hip.hipMalloc(b_num_bytes))

hip_check(hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

blocks_per_grid_x = max(M // TILE_SIZE, 1)
blocks_per_grid_y = max(N // TILE_SIZE, 1)
blocks_per_grid_z = 1
threads_per_block_x = TILE_SIZE
threads_per_block_y = TILE_SIZE
stream = 0
shared_memory = 0

times = {
    kernel1_naive.__name__: 0,
    kernel2_lds.__name__: 0,
}
runs = 10
start, stop = hip.hipEventCreate(), hip.hipEventCreate()
kernels = [kernel1_naive, kernel2_lds]
for kernel in kernels:
    for i in range(runs):
        # random.shuffle(kernels)
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

        c_h = -3 * np.ones((M, N), dtype=np.float32)
        c_num_bytes = c_h.size * c_h.itemsize
        c_d = hip_check(hip.hipMalloc(c_num_bytes))
        hip_check(
            hip.hipMemcpy(
                c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
            )
        )

        function = hip_check(
            hip.hipModuleGetFunction(hip_module, kernel.__name__.encode())
        )

        start = time.monotonic()
        launch_kernel(
            function.as_c_void_p(),
            blocks_per_grid_x,
            blocks_per_grid_y,
            blocks_per_grid_z,
            threads_per_block_x,
            threads_per_block_y,
            stream,
            shared_memory,
            a_d,
            b_d,
            c_d,
        )
        hip_synchronize()

        hip_check(
            hip.hipMemcpy(
                c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        correct = a_h @ b_h
        if not np.allclose(correct, c_h):
            # print(correct)
            # print(c_h)
            print(f"{kernel.__name__} failed")
        else:
            if i > 0:
                times[kernel.__name__] += time.monotonic() - start
                print(f"{kernel.__name__} : {times[kernel.__name__]}")

for k in times:
    times[k] /= runs

for k, v in times.items():
    print(f"{k}: {v:.3e}ms")
