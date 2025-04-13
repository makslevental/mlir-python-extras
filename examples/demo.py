#!/usr/bin/env python

import mlir.extras.types as T
import numpy as np
from hip import hip
from mlir.ir import InsertionPoint

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, rocdl

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


def time_to_gflops(time_ms, N):
    return 1e-6 * (N * N * N * 2 + 3 * N * N) // time_ms


# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref

ctx = RAIIMLIRContextModule()
set_container_module(ctx.module)

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode()


# just a default attr - actual target is set blow
@module("kernels", [f'#rocdl.target<abi = "500">'])
def gpu_module():
    pass


ip = InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
ip.__enter__()

set_container_module(ctx.module)

v_len = 16
M, K, N = 1024, 1024, 1024
v16f16 = T.vector(v_len, T.f16())


@gpu_func
@canonicalize(using=scf.canonicalizer)
def smol_matmul(
    a: T.memref(M, K, T.f16()),
    b: T.memref(K, N, T.f16()),
    c: T.memref(M, N, T.f16()),
):
    lIdx = thread_idx.x
    # a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    # a_frag will store one column of the 16x16 matrix A tile
    # b_frag will store one row of the 16x16 matrix B tile
    a_frag = arith.constant(np.full([v_len], 0.0, np.float16), v16f16)
    b_frag = arith.constant(np.full([v_len], 0.0, np.float16), v16f16)
    c_frag = arith.constant(np.full([v_len], 0.0, np.float16), v16f16)

    # lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    lane = lIdx % v_len
    for ele in range(v_len):
        b_frag[ele] = b[ele, lane]
        a_frag[ele] = a[lane, ele]
        # a_frag, b_frag = yield a_frag, b_frag

    # call the WMMA intrinsic
    false = arith.constant(False, T.bool())
    c_frag = rocdl.wmma_f16_16x16x16_f16(v16f16, [a_frag, b_frag, c_frag, false])

    for ele in range(v_len // 2):
        r = ele * 2 + (lIdx // v_len)
        # store results from unpacked c_frag output
        c[r, lane] = c_frag[ele * 2]


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode().split(":")[0]


@module("naive", [f'#rocdl.target<chip = "{arch}", abi = "500">'])
def gpu_module():
    smol_matmul.emit()


ip.__exit__(None, None, None)

lowered_module = run_pipeline(
    gpu_module,
    Pipeline()
    .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True))
    .rocdl_attach_target(chip=arch, abi="500", O=0)
    .gpu_to_llvm()
    .lower_to_llvm()
    .ensure_debug_info_scope_on_llvm_func(emission_kind="Full")
    .gpu_module_to_binary(),
)

hsaco = get_compile_object_bytes(lowered_module)
hip_module = hip_check(hip.hipModuleLoadData(hsaco))
function = hip_check(
    hip.hipModuleGetFunction(hip_module, smol_matmul.__name__.encode())
)

a_h = np.random.randint(0, 10, (M, K)).astype(dtype=np.float16)
b_h = np.random.randint(0, 10, (K, N)).astype(dtype=np.float16)
c_h = -3 * np.ones((M, N), dtype=np.float16)

a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize
c_num_bytes = c_h.size * c_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes))
b_d = hip_check(hip.hipMalloc(b_num_bytes))
c_d = hip_check(hip.hipMalloc(c_num_bytes))

hip_check(hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

gridX = 32
gridY = 32
gridZ = 1
warp_size = 32
num_warps = 1
stream = 0
shared_memory = 0

launch_kernel(
    function.as_c_void_p(),
    gridX,
    gridY,
    gridZ,
    warp_size,
    num_warps,
    1,
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

# if not np.allclose(c_h, correct):
#     with np.printoptions(threshold=np.inf, linewidth=200):
#         print(correct)
#         print(c_h)
#         assert False

hip_check(hip.hipFree(a_d))
hip_check(hip.hipFree(b_d))
hip_check(hip.hipFree(c_d))

hip_check(hip.hipModuleUnload(hip_module))
