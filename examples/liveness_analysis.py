from mlir import ir
from pathlib import Path

import mlir.extras.types as T
import numpy as np
from mlir.ir import InsertionPoint, IntegerAttr, UnitAttr

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, gpu, llvm
from mlir.dialects import math

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
from mlir.extras.util import find_ops, walk_blocks_in_operation, walk_operations
from mlir.extras.util.liveness import (
    BlockInfoBuilder,
    Liveness,
    LiveInterval,
    linear_scan_register_allocation,
)

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

Bc = 32
Br = 32

B = 16
nh = 12
N = 128
d = 128

softmax_scale = 1.0 / float(np.sqrt(d))


rank_reduce = memref.rank_reduce


# https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
@gpu_func(emit=True)
@canonicalize(using=[scf.canonicalizer, arith.canonicalizer])
def flash_attention(
    Q: T.memref(B, nh, N, d, T.f32()),
    K: T.memref(B, nh, N, d, T.f32()),
    V: T.memref(B, nh, N, d, T.f32()),
    l: T.memref(B, nh, N, T.f32()),
    m: T.memref(B, nh, N, T.f32()),
    O: T.memref(B, nh, N, d, T.f32()),
):
    tx = thread_idx.x
    # batch idx, head_idx
    bx, by = block_idx.x, block_idx.y
    # gpu.printf("bx %ld, by %ld\n", bx, by)

    # Offset into Q,K,V,O,l,m - different for each batch and head
    K = K[bx, by, :, :, rank_reduce]
    V = V[bx, by, :, :, rank_reduce]
    Q = Q[bx, by, :, :, rank_reduce]
    O = O[bx, by, :, :, rank_reduce]
    l = l[bx, by, :, rank_reduce]
    m = m[bx, by, :, rank_reduce]

    # Define SRAM for Q,K,V,S
    sram = gpu.dynamic_shared_memory()
    Qi = memref.view(sram, (Br, d), dtype=T.f32())
    Kj = memref.view(sram, (Bc, d), dtype=T.f32(), shift=Qi.n_elements)
    Vj = memref.view(sram, (Bc, d), dtype=T.f32(), shift=Qi.n_elements + Kj.n_elements)
    S = memref.view(
        sram,
        (Br, Bc),
        dtype=T.f32(),
        shift=Qi.n_elements + Kj.n_elements + Vj.n_elements,
    )

    for bc in scf.range_(0, N, Bc):
        # Load Kj, Vj to SRAM
        K_ = K[bc : bc + 1, :]
        V_ = V[bc : bc + 1, :]
        for x in scf.range_(0, d):
            Kj[tx, x] = K_[tx, x]
            Vj[tx, x] = V_[tx, x]

        for br in scf.range_(0, N, Br):
            # Load Qi to SRAM, l and m to registers
            Q_ = Q[br : br + 1, :]
            for x in scf.range_(0, d):
                Qi[tx, x] = Q_[tx, x]

            l_ = l[br : br + 1]
            m_ = m[br : br + 1]
            row_l_prev = l_[tx]
            row_m_prev = m_[tx]

            # S = QK^T, row_m = rowmax(S)
            row_m: T.f32() = float(np.finfo(np.float32).min)
            for y, row_m, _ in scf.range_(0, Bc, iter_args=[row_m]):
                sum: T.f32() = 0.0
                for x, sum, _ in scf.range_(0, d, iter_args=[sum]):
                    sum += Qi[tx, x] * Kj[y, x]
                    sum = yield sum

                sum *= softmax_scale
                S[tx, y] = sum

                if sum > row_m:
                    row_m_ = yield sum
                else:
                    row_m_ = yield row_m

                row_m = yield row_m_

            # P = exp(S - row_m), row_l = rowsum(P)
            row_l: T.f32() = 0.0
            for y, row_l, _ in scf.range_(0, Bc, iter_args=[row_l]):
                S[tx, y] = math.exp(S[tx, y] - row_m)
                row_l += S[tx, y]
                row_l = yield row_l

            # Compute new m and l
            row_m_new = arith.maximumf(row_m_prev, row_m)
            row_l_new = (
                math.exp(row_m_prev - row_m_new) * row_l_prev
                + math.exp(row_m - row_m_new) * row_l
            )
            div = 1.0 / row_l_new
            f1 = row_l_prev * math.exp(row_m_prev - row_m_new)
            f2 = math.exp(row_m - row_m_new)

            # Write O, l, m to HBM
            O_ = O[br : br + 1, :]
            for x in scf.range_(0, d):
                pv: T.f32() = 0.0  # Pij * Vj
                for y, pv, _ in scf.range_(0, Bc, iter_args=[pv]):
                    pv += S[tx, y] * Vj[y, x]
                    pv = yield pv

                O_[tx, x] = div * (f1 * O_[tx, x] + f2 * pv)

            l_[tx] = row_l_new
            m_[tx] = row_m_new

            gpu.barrier()


ip.__exit__(None, None, None)

assert gpu_module.operation.verify()
# l = Liveness(gpu_module)
# print(l)


# https://langdev.stackexchange.com/questions/4325/how-do-modern-compilers-choose-which-variables-to-put-in-registers
x = LiveInterval(1, 3, "x")
t1 = LiveInterval(1, 2, "t1")
y = LiveInterval(2, 5, "y")
z = LiveInterval(3, 4, "z")
t2 = LiveInterval(4, 5, "t2")
y2 = LiveInterval(5, 6, "y2")

register, location = linear_scan_register_allocation([x, t1, y, z, t2, y2], 2)

for v, r in register.items():
    print(v, r)
for v, l in location.items():
    print(v, l)
