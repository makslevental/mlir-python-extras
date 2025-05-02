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

batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

Bc = 32
Br = 32

B = batch_size
nh = n_head
N = seq_len
d = head_embd

import math

Tc = math.ceil(N / Bc)
Tr = math.ceil(N / Br)
softmax_scale = 1.0 / math.sqrt(d)
tile_size = Bc * d  # size of Qi, Kj, Vj


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def manual_attn(q, k, v):
    # the kernel below overwrites the global math.........
    import math

    q = q.reshape(batch_size, n_head, seq_len, head_embd)
    k = k.reshape(batch_size, n_head, seq_len, head_embd)
    v = v.reshape(batch_size, n_head, seq_len, head_embd)

    att = q @ k.transpose(0, 1, -2, -1) * (1.0 / math.sqrt(k.shape[-1]))
    att = softmax(att, axis=-1)
    y = att @ v
    return y.flatten()


from mlir.dialects import math


# https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
@gpu_func(emit=True)
@canonicalize(using=[scf.canonicalizer, arith.canonicalizer])
def flash_attention(
    Q: T.memref(batch_size * n_head * seq_len * head_embd, T.f32()),
    K: T.memref(batch_size * n_head * seq_len * head_embd, T.f32()),
    V: T.memref(batch_size * n_head * seq_len * head_embd, T.f32()),
    l: T.memref(B * nh * N, T.f32()),
    m: T.memref(B * nh * N, T.f32()),
    O: T.memref(batch_size * n_head * seq_len * head_embd, T.f32()),
):
    tx = thread_idx.x
    bx = block_idx.x
    by = block_idx.y  # batch and head index

    # Offset into Q,K,V,O,l,m - different for each batch and head
    qkv_offset = bx * grid_dim.y * N * d + by * N * d  # gridDim.y = nh
    lm_offset = bx * grid_dim.y * N + by * N  # offset for l and m

    # Define SRAM for Q,K,V,S
    sram = gpu.dynamic_shared_memory()
    Qi = memref.view(sram, (tile_size,), dtype=T.f32())
    Kj = memref.view(sram, (tile_size,), dtype=T.f32(), shift=tile_size * 1)
    Vj = memref.view(sram, (tile_size,), dtype=T.f32(), shift=tile_size * 2)
    S = memref.view(sram, (tile_size,), dtype=T.f32(), shift=tile_size * 3)

    for j in scf.range_(0, Tc):
        # Load Kj, Vj to SRAM
        for x in scf.range_(0, d):
            Kj[tx * d + x] = K[qkv_offset + tile_size * j + tx * d + x]
            Vj[tx * d + x] = V[qkv_offset + tile_size * j + tx * d + x]

        gpu.barrier()  # such that the inner loop can use the correct Kj, Vj

        for i in scf.range_(0, Tr):
            # Load Qi to SRAM, l and m to registers
            for x in scf.range_(0, d):
                ii = qkv_offset + tile_size * i + tx * d + x
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

                ii = qkv_offset + tile_size * i + tx * d + x
                O[ii] = div * (c * O[ii] + math.exp(row_m - row_m_new) * pv)

            gpu.barrier()  # otherwise, thread can use the wrong Kj, Vj in inner loop

            m[lm_offset + Br * i + tx] = row_m_new
            l[lm_offset + Br * i + tx] = row_l_new

            # gpu.barrier()  # otherwise, thread can use the wrong Kj, Vj in inner loop
        # gpu.barrier()  # otherwise, thread can use the wrong Kj, Vj in inner loop


ip.__exit__(None, None, None)

sram_size = 4 * tile_size * np.float32().itemsize

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
    .lower_to_llvm(),
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

lowered_module = run_pipeline(lowered_module, Pipeline().gpu_module_to_binary())
hsaco = get_compile_object_bytes(lowered_module)

hip_module = hip_check(hip.hipModuleLoadData(hsaco))

q_h = np.random.randint(0, 10, (batch_size * n_head * seq_len * head_embd)).astype(
    dtype=np.float32
)
k_h = np.random.randint(0, 10, (batch_size * n_head * seq_len * head_embd)).astype(
    dtype=np.float32
)
v_h = np.random.randint(0, 10, (batch_size * n_head * seq_len * head_embd)).astype(
    dtype=np.float32
)
l_h = np.zeros((B * nh * N), dtype=np.float32)
m_h = np.full((B * nh * N), float(np.finfo(np.float32).min), dtype=np.float32)
O_h = np.zeros_like(q_h, dtype=np.float32)

q_num_bytes = q_h.size * q_h.itemsize
k_num_bytes = k_h.size * k_h.itemsize
v_num_bytes = v_h.size * v_h.itemsize
l_num_bytes = l_h.size * l_h.itemsize
m_num_bytes = m_h.size * m_h.itemsize
O_num_bytes = O_h.size * O_h.itemsize

q_d = hip_check(hip.hipMalloc(q_num_bytes))
k_d = hip_check(hip.hipMalloc(k_num_bytes))
v_d = hip_check(hip.hipMalloc(v_num_bytes))
l_d = hip_check(hip.hipMalloc(l_num_bytes))
m_d = hip_check(hip.hipMalloc(m_num_bytes))
O_d = hip_check(hip.hipMalloc(O_num_bytes))

stream = 0

times = {
    flash_attention: 0,
}
# random.shuffle(kernels)
runs = 16
for kernel in times:
    for i in range(runs):
        function = hip_check(
            hip.hipModuleGetFunction(hip_module, kernel.__name__.encode())
        )
        hip_check(hip.hipDeviceSynchronize())

        for d, h, num_bytes in zip(
            [q_d, k_d, v_d, l_d, m_d, O_d],
            [q_h, k_h, v_h, l_h, m_h, O_h],
            [
                q_num_bytes,
                k_num_bytes,
                v_num_bytes,
                l_num_bytes,
                m_num_bytes,
                O_num_bytes,
            ],
        ):
            hip_check(
                hip.hipMemcpy(d, h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
            )

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
            q_d,
            k_d,
            v_d,
            l_d,
            m_d,
            O_d,
        )

        hip_check(
            hip.hipMemcpy(
                l_h, l_d, l_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        hip_check(
            hip.hipMemcpy(
                m_h, m_d, m_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        hip_check(
            hip.hipMemcpy(
                O_h, O_d, O_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        correct = manual_attn(q_h, k_h, v_h)
        if not np.allclose(correct, O_h):
            print("correct", correct)
            print("l_h", l_h)
            print("m_h", m_h)
            print("output", O_h)
            print(f"{kernel.__name__} failed")

        times[kernel] += time_compute

for k in times:
    times[k] /= runs

for k, v in times.items():
    print(f"{k.__name__}: {v:.03f}ms")
