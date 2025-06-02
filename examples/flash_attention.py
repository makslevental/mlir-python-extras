from pathlib import Path

import mlir.extras.types as T
import numpy as np
from hip import hip
from mlir.ir import InsertionPoint, IntegerAttr, UnitAttr
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, gpu, llvm

# noinspection PyUnresolvedReferences
from mlir.extras.dialects.ext.gpu import (
    block_idx,
    thread_idx,
    grid_dim,
    func as gpu_func,
    set_container_module,
    module,
    get_compile_object_bytes,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline
from mlir.extras.util import find_ops

# noinspection PyUnresolvedReferences
from util import hip_check, launch_kernel, hip_synchronize


def init_copy_host_device():
    q_h = np.random.randint(0, 10, (B * nh * N * d)).astype(dtype=np.float32)
    k_h = np.random.randint(0, 10, (B * nh * N * d)).astype(dtype=np.float32)
    v_h = np.random.randint(0, 10, (B * nh * N * d)).astype(dtype=np.float32)
    l_h = np.zeros((B * nh * N), dtype=np.float32)
    m_h = np.full((B * nh * N), float(np.finfo(np.float32).min), dtype=np.float32)
    O_h = np.zeros_like(q_h, dtype=np.float32)

    host = [q_h, k_h, v_h, l_h, m_h, O_h]
    device = [hip_check(hip.hipMalloc(h.size * h.itemsize)) for h in host]

    for dev, h in zip(device, host):
        hip_check(
            hip.hipMemcpy(
                dev, h, h.size * h.itemsize, hip.hipMemcpyKind.hipMemcpyHostToDevice
            )
        )

    return host, device


def copy_device_host(host, device):
    for d, h in zip(device, host):
        hip_check(
            hip.hipMemcpy(
                h, d, h.size * h.itemsize, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        hip_check(hip.hipFree(d))

    return host


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

Bc = 32
Br = 32

B = 16
nh = 12
N = 128
d = 128

import math

Tc = math.ceil(N / Bc)
Tr = math.ceil(N / Br)
softmax_scale = 1.0 / math.sqrt(d)


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def manual_attn(q, k, v):
    # the kernel below overwrites the global math.........
    import math

    q = q.reshape(B, nh, N, d)
    k = k.reshape(B, nh, N, d)
    v = v.reshape(B, nh, N, d)

    att = q @ k.transpose(0, 1, 3, 2) * (1.0 / math.sqrt(k.shape[-1]))
    att = softmax(att, axis=-1)
    y = att @ v
    return y.flatten()


from mlir.dialects import math


# https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
@gpu_func(emit=True)
@canonicalize(using=[scf.canonicalizer, arith.canonicalizer])
def flash_attention(
    Q: T.memref(B * nh * N * d, T.f32()),
    K: T.memref(B * nh * N * d, T.f32()),
    V: T.memref(B * nh * N * d, T.f32()),
    l: T.memref(B * nh * N, T.f32()),
    m: T.memref(B * nh * N, T.f32()),
    O: T.memref(B * nh * N * d, T.f32()),
):
    tx = thread_idx.x
    # batch idx, head_idx
    bx, by = block_idx.x, block_idx.y
    # gpu.printf("bx %ld, by %ld\n", bx, by)

    # Offset into Q,K,V,O,l,m - different for each batch and head
    qkv_offset = bx * nh * N * d + by * N * d
    lm_offset = bx * nh * N + by * N  # offset for l and m

    # Define SRAM for Q,K,V,S
    sram = gpu.dynamic_shared_memory()
    Qi = memref.view(sram, (Br * d,), dtype=T.f32())
    Kj = memref.view(sram, (Bc * d,), dtype=T.f32(), shift=Qi.n_elements)
    Vj = memref.view(
        sram, (Bc * d,), dtype=T.f32(), shift=Qi.n_elements + Kj.n_elements
    )
    S = memref.view(
        sram,
        (Br * Bc,),
        dtype=T.f32(),
        shift=Qi.n_elements + Kj.n_elements + Vj.n_elements,
    )

    for j in scf.range_(0, Tc):
        # Load Kj, Vj to SRAM
        for x in scf.range_(0, d):
            Kj[tx * d + x] = K[qkv_offset + Bc * d * j + tx * d + x]
            Vj[tx * d + x] = V[qkv_offset + Bc * d * j + tx * d + x]

        for i in scf.range_(0, Tr):
            # Load Qi to SRAM, l and m to registers
            for x in scf.range_(0, d):
                ii = qkv_offset + Bc * d * i + tx * d + x
                Qi[tx * d + x] = Q[ii]

            row_m_prev = m[lm_offset + Br * i + tx]
            row_l_prev = l[lm_offset + Br * i + tx]

            # S = QK^T, row_m = rowmax(S)
            row_m: T.f32() = float(np.finfo(np.float32).min)
            for y, row_m, _ in scf.range_(0, Bc, iter_args=[row_m]):
                sum: T.f32() = 0.0
                for x, sum, _ in scf.range_(0, d, iter_args=[sum]):
                    sum += Qi[tx * d + x] * Kj[y * d + x]
                    sum = yield sum

                sum *= softmax_scale
                S[Bc * tx + y] = sum

                if sum > row_m:
                    row_m_ = yield sum
                else:
                    row_m_ = yield row_m

                row_m = yield row_m_

            # P = exp(S - row_m), row_l = rowsum(P)
            row_l: T.f32() = 0.0
            for y, row_l, _ in scf.range_(0, Bc, iter_args=[row_l]):
                S[Bc * tx + y] = math.exp(S[Bc * tx + y] - row_m)
                row_l += S[Bc * tx + y]
                row_l = yield row_l

            # Compute new m and l
            row_m_new = arith.maximumf(row_m_prev, row_m)
            row_l_new = (
                math.exp(row_m_prev - row_m_new) * row_l_prev
                + math.exp(row_m - row_m_new) * row_l
            )
            div = 1.0 / row_l_new
            c = row_l_prev * math.exp(row_m_prev - row_m_new)

            # Write O, l, m to HBM
            for x in scf.range_(0, d):
                pv: T.f32() = 0.0  # Pij * Vj
                for y, pv, _ in scf.range_(0, Bc, iter_args=[pv]):
                    pv += S[Bc * tx + y] * Vj[y * d + x]
                    pv = yield pv

                ii = qkv_offset + Bc * d * i + tx * d + x
                O[ii] = div * (c * O[ii] + math.exp(row_m - row_m_new) * pv)

            m[lm_offset + Br * i + tx] = row_m_new
            l[lm_offset + Br * i + tx] = row_l_new

            gpu.barrier()


ip.__exit__(None, None, None)

sram_size = 4 * Bc * d * np.float32().itemsize

launch_params = {
    flash_attention.__name__: (
        (B, nh, 1),
        (Bc, 1, 1),
        sram_size,
    )
}

simplified_module = run_pipeline(
    ctx.module,
    Pipeline()
    .canonicalize()
    .cse()
    .loop_invariant_code_motion()
    .loop_invariant_subset_hoisting()
    .rocdl_attach_target(chip=arch, O=3, abi="500"),
)

# print(simplified_module)
# exit()

lowered_module = run_pipeline(
    simplified_module,
    Pipeline()
    .Gpu(
        Pipeline().convert_gpu_to_rocdl(
            use_bare_ptr_memref_call_conv=True,
            runtime="HIP",
        )
    )
    .gpu_to_llvm()
    .lower_to_llvm()
    .ensure_debug_info_scope_on_llvm_func(emission_kind="Full"),
    # .Nested("llvm.func", Pipeline().sroa()),
)

# print(lowered_module)
gep = find_ops(lowered_module.operation, lambda o: isinstance(o.opview, llvm.GEPOp))
for g in gep:
    g.attributes["inbounds"] = UnitAttr.get()

kernel_funcs = find_ops(
    lowered_module.operation, lambda o: isinstance(o.opview, llvm.LLVMFuncOp)
)
for k in kernel_funcs:
    if k.sym_name.value != flash_attention.__name__:
        continue
    _, thread_dims, _ = launch_params[k.sym_name.value]
    k.attributes["rocdl.max_flat_work_group_size"] = IntegerAttr.get(
        T.index(), np.prod(thread_dims)
    )

output_format = "bin"
# output_format = "llvm"
# output_format = "isa"

lowered_module = run_pipeline(
    lowered_module, Pipeline().gpu_module_to_binary(format=output_format)
)
hsaco = get_compile_object_bytes(lowered_module)
if output_format in {"isa", "llvm", "offloading"}:
    with open(Path(__file__).parent / "flashattention.amdgcn", "wb") as f:
        f.write(hsaco)
    exit()

hip_module = hip_check(hip.hipModuleLoadData(hsaco))

stream = 0

times = {
    flash_attention: 0,
}
runs = 32
for kernel in times:
    for i in range(runs):
        function = hip_check(
            hip.hipModuleGetFunction(hip_module, kernel.__name__.encode())
        )
        hip_check(hip.hipDeviceSynchronize())

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
        ) = launch_params[kernel.__name__]

        host, device = init_copy_host_device()
        q_h, k_h, v_h, *_ = host
        correct = manual_attn(q_h, k_h, v_h)

        time_compute = launch_kernel(
            function.as_c_void_p(),
            blocks_per_grid_x,
            blocks_per_grid_y,
            blocks_per_grid_z,
            threads_per_block_x,
            threads_per_block_y,
            threads_per_block_z,
            stream,
            shared_memory,
            *device,
        )

        *_, O_h = copy_device_host(host, device)
        if not np.allclose(correct, O_h):
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                print(
                    "correct - output:\n",
                    correct.round().reshape(B, nh, N, d)
                    - O_h.round().reshape(B, nh, N, d),
                )
            print(f"{kernel.__name__} failed\n")
        else:
            print(f"{kernel.__name__}: {time_compute:.03f}ms")

        times[kernel] += time_compute

for k in times:
    times[k] /= runs

for k, v in times.items():
    print(f"{k.__name__}: {v:.03f}ms")
