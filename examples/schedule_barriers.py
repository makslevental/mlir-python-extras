#!/usr/bin/env python
from pathlib import Path

import mlir.extras.types as T
import numpy as np
from hip import hip
from mlir.ir import InsertionPoint

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, rocdl, gpu, llvm, vector

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
    dynamic_shared_memory,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from util import hip_check, launch_kernel, hip_synchronize


def time_to_gflops(time_ms, N):
    return 1e-6 * (N * N * N * 2 + 3 * N * N) // time_ms


# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref

ctx = RAIIMLIRContextModule()
set_container_module(ctx.module)


# just a default attr - actual target is set blow
@module("kernels", [f'#rocdl.target<abi = "500">'])
def gpu_module():
    pass


ip = InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
ip.__enter__()

set_container_module(ctx.module)

v_len = 16
M, K, N = 512, 512, 512
TILE_SIZE = BK = 16
dtype = T.f16()
np_dtype = np.float16
v16 = T.vector(v_len, dtype)


@gpu_func
@canonicalize(using=scf.canonicalizer)
def kernel(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    base = dynamic_shared_memory()
    As = memref.view(base, (TILE_SIZE, TILE_SIZE), dtype=dtype)
    Bs = memref.view(
        base, (TILE_SIZE, TILE_SIZE), dtype=dtype, shift=TILE_SIZE * TILE_SIZE
    )

    row = block_idx.y * TILE_SIZE + thread_idx.y
    col = block_idx.x * TILE_SIZE + thread_idx.x
    lane = thread_idx.x % v_len
    # gpu.printf("(%ld, %ld)\n", row, col)
    # vector.print_(source=row)

    sum = arith.constant(np.full([v_len], 0.0, np_dtype), v16)

    Bs[thread_idx.y, thread_idx.x] = B[col, thread_idx.y + 0]
    As[thread_idx.y, thread_idx.x] = A[row, thread_idx.x + 0]

    for t, sum, _ in scf.range_(BK, N + BK, BK, iter_args=[sum]):
        gpu.barrier()

        a_frag = As @ vector.load(v16) @ [lane, 0]
        b_frag = Bs @ vector.load(v16) @ [lane, 0]

        sum = rocdl.wmma_f16_16x16x16_f16(a_frag, b_frag, sum)

        if arith.index_cast(t, T.i32()) < N:
            Bs[thread_idx.y, thread_idx.x] = B[col, thread_idx.y + t]
            As[thread_idx.y, thread_idx.x] = A[row, thread_idx.x + t]

        sum = yield sum

    C[row, col] = sum[2 * (row // 2)]


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode().split(":")[0]


@module("naive", [f'#rocdl.target<chip = "{arch}", abi = "500">'])
def gpu_module():
    kernel.emit()


ip.__exit__(None, None, None)

# gpu_module = run_pipeline(gpu_module, Pipeline().cse())
# print(gpu_module)

O = 3
output_format = "binary"

lowered_module = run_pipeline(
    gpu_module,
    Pipeline()
    .Gpu(
        Pipeline().convert_gpu_to_rocdl(
            use_bare_ptr_memref_call_conv=True,
            runtime="HIP",
        )
    )
    .rocdl_attach_target(chip=arch, abi="500", O=O)
    .gpu_to_llvm()
    .lower_to_llvm()
    .ensure_debug_info_scope_on_llvm_func(emission_kind="Full")
    .gpu_module_to_binary(format=output_format),
)

hsaco = get_compile_object_bytes(lowered_module)
if output_format == "assembly":
    with open(Path(__file__).parent / f"hsacoO{O}.txt", "wb") as f:
        f.write(hsaco)
        exit()
hip_module = hip_check(hip.hipModuleLoadData(hsaco))
function = hip_check(hip.hipModuleGetFunction(hip_module, kernel.__name__.encode()))

# a_h = np.random.randint(1, 5, (M, K)).astype(dtype=np_dtype)
# b_h = np.random.randint(1, 5, (K, N)).astype(dtype=np_dtype)

# a_h = np.random.rand(M, K).astype(np_dtype)
# b_h = np.random.rand(K, N).astype(np_dtype)

a_h = 3 * np.ones((M, K)).astype(dtype=np_dtype)
a_h[0 : M // 2, 0 : K // 2] = 0
a_h[M // 2 : M, K // 2 : K] = 1
b_h = 2 * np.ones((K, N)).astype(dtype=np_dtype)
b_h[0 : K // 2, 0 : N // 2] = 2
b_h[K // 2 : K, N // 2 : N] = 3

c_h = 0 * np.ones((M, N), dtype=np.float32)
for k in range(K):
    a = a_h.astype(np.float32)[:, k]
    b = b_h.astype(np.float32)[k, :]
    c_h += np.outer(a, b)
assert np.allclose(a_h.astype(np.float32) @ b_h.astype(np.float32), c_h)

c_h = -3 * np.ones((M, N), dtype=np_dtype)
a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize
c_num_bytes = c_h.size * c_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes))
b_d = hip_check(hip.hipMalloc(b_num_bytes))
c_d = hip_check(hip.hipMalloc(c_num_bytes))

hip_check(hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

(
    (
        blocks_per_grid_x,
        blocks_per_grid_y,
        blocks_per_grid_z,
    ),
    (
        threads_per_block_x,
        threads_per_block_y,
        threads_per_block_z,
    ),
    shared_memory,
) = (
    (N // TILE_SIZE, N // TILE_SIZE, 1),
    (TILE_SIZE, TILE_SIZE, 1),
    2 * TILE_SIZE * TILE_SIZE * dtype.width // 8,
)

stream = 0

launch_kernel(
    function.as_c_void_p(),
    blocks_per_grid_x,
    blocks_per_grid_y,
    blocks_per_grid_z,
    threads_per_block_x,
    threads_per_block_y,
    threads_per_block_z,
    stream,
    shared_memory,
    a_d,
    b_d,
    c_d,
)

correct = a_h @ b_h
assert np.allclose(c_h, -3.0)
assert not np.allclose(correct, c_h)
hip_check(hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

if not np.allclose(c_h, correct):
    with np.printoptions(threshold=np.inf, linewidth=np.inf):
        # print("correct\n", correct)
        # print("c_h\n", c_h)
        print("off by atol", np.max(np.abs(correct - c_h)))
        print("off by rtol", np.max(np.abs(correct - c_h) / correct))
        print("num incorrect", np.sum(np.abs(correct - c_h) != 0))
        print("fraction incorrect", np.sum(np.abs(correct - c_h) != 0) / (M * N))


hip_check(hip.hipFree(a_d))
hip_check(hip.hipFree(b_d))
hip_check(hip.hipFree(c_d))

hip_check(hip.hipModuleUnload(hip_module))
