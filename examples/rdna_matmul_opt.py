import numpy as np

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import RAIIMLIRContextModule
from mlir.extras.dialects.ext import memref, scf, arith, gpu, llvm
from mlir.dialects import index as index_dialect
from mlir.ir import InsertionPoint, IntegerAttr, UnitAttr, Attribute
import mlir.extras.types as T

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
from mlir.extras.util import find_ops

# noinspection PyUnresolvedReferences
from util import (
    hip_check,
    launch_kernel,
    hip_synchronize,
    hip_bindings_not_installed,
    get_hip_arch,
)


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

M, K, N = 1024, 1024, 1024


@gpu_func(emit=True)
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
            a = A[row, k]
            b = B[k, col]
            acc = llvm.intr_fmuladd(a, b, acc)
            acc = yield acc

        C[row, col] = acc


launch_params = {
    kernel1_naive.__name__: (
        (N // 16, N // 16, 1),
        (16, 16, 1),
        0,
    )
}

BN = BK = TILE_SIZE = 32

A_shared = memref.global_(
    sym_name="A_shared_BN_BK_0",
    type=T.memref(BN, BK, T.f32(), memory_space=lds_space()),
    alignment=16,
)
B_shared = memref.global_(
    sym_name="B_shared_BK_BN_0",
    type=T.memref(BK, BN, T.f32(), memory_space=lds_space()),
    alignment=16,
)

dtype = T.f32()


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel2_lds_shared_direct_load_globals(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    As = memref.get_global(A_shared)
    Bs = memref.get_global(B_shared)

    row = block_idx.y * TILE_SIZE + thread_idx.y
    col = block_idx.x * TILE_SIZE + thread_idx.x

    sum = arith.constant(0.0)

    for t, sum, _ in scf.range_(0, N, BK, iter_args=[sum]):
        Bs[thread_idx.y, thread_idx.x] = B[thread_idx.y + t, col]
        As[thread_idx.y, thread_idx.x] = A[row, thread_idx.x + t]

        gpu.barrier()

        for k in range(BK):
            a = As[thread_idx.y, k]
            b = Bs[k, thread_idx.x]
            sum = llvm.intr_fmuladd(a, b, sum)

        gpu.barrier()

        sum = yield sum

    C[row, col] = sum


launch_params[kernel2_lds_shared_direct_load_globals.__name__] = (
    (N // TILE_SIZE, N // TILE_SIZE, 1),
    (TILE_SIZE, TILE_SIZE, 1),
    0,
)


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel2_lds_shared_direct_dynamic(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    As = memref.get_global(A_shared)
    Bs = memref.get_global(B_shared)

    row = block_idx.y * TILE_SIZE + thread_idx.y
    col = block_idx.x * TILE_SIZE + thread_idx.x

    sum = arith.constant(0.0)

    for t, sum, _ in scf.range_(0, N, BK, iter_args=[sum]):
        Bs[thread_idx.y, thread_idx.x] = B[thread_idx.y + t, col]
        As[thread_idx.y, thread_idx.x] = A[row, thread_idx.x + t]

        gpu.barrier()

        for k in range(BK):
            a = As[thread_idx.y, k]
            b = Bs[k, thread_idx.x]
            sum = llvm.intr_fmuladd(a, b, sum)

        gpu.barrier()

        sum = yield sum

    C[row, col] = sum


launch_params[kernel2_lds_shared_direct_dynamic.__name__] = (
    (N // TILE_SIZE, N // TILE_SIZE, 1),
    (TILE_SIZE, TILE_SIZE, 1),
    2 * TILE_SIZE * TILE_SIZE * T.f32().width // 8,
)


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel2_lds_shared_subview(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    # allocate buffer for current block in fast shared mem
    # shared mem is shared between all threads in a block
    base = gpu.dynamic_shared_memory()
    As = memref.view(base, (TILE_SIZE, TILE_SIZE), dtype=dtype)
    Bs = memref.view(
        base, (TILE_SIZE, TILE_SIZE), dtype=dtype, shift=TILE_SIZE * TILE_SIZE
    )

    # the inner row & col that we're accessing in this thread
    tid = gpu.thread_id()
    thread_row = tid / TILE_SIZE
    thread_col = tid % TILE_SIZE

    # the output block that we want to compute in this threadblock
    c_row = block_idx.x * TILE_SIZE
    c_col = block_idx.y * TILE_SIZE

    tmp = arith.constant(0, type=dtype)

    for bk_idx, tmp, _ in scf.range_(0, K, TILE_SIZE, iter_args=[tmp]):
        A_ = A[c_row : c_row + TILE_SIZE, bk_idx : bk_idx + TILE_SIZE]
        B_ = B[bk_idx : bk_idx + TILE_SIZE, c_col : c_col + TILE_SIZE]

        # Have each thread load one of the elements in A & B
        # Make the threadCol (=thread_idx.x) the consecutive index
        # to allow global memory access coalescing
        As[thread_row, thread_col] = A_[thread_row, thread_col]
        Bs[thread_row, thread_col] = B_[thread_row, thread_col]

        # block threads in this block until cache is fully populated
        gpu.barrier()

        # execute the dotproduct on the currently cached block
        for dot_idx in range(TILE_SIZE):
            a, b = As[thread_row, dot_idx], Bs[dot_idx, thread_col]
            tmp = llvm.intr_fmuladd(a, b, tmp)

        # need to sync again at the end, to avoid faster threads
        # fetching the next block into the cache before slower threads are done
        gpu.barrier()

        tmp = yield tmp

    C_ = C[c_row : c_row + TILE_SIZE, c_col : c_col + TILE_SIZE]
    C_[thread_row, thread_col] = tmp


launch_params[kernel2_lds_shared_subview.__name__] = (
    (N // TILE_SIZE, N // TILE_SIZE, 1),
    (TILE_SIZE, TILE_SIZE, 1),
    2 * TILE_SIZE * TILE_SIZE * T.f32().width // 8,
)

BLOCK_SIZE = 256
# Block Tile size
BN = 128
BM = 128
# Number of Row or column we read per batch
BK = 8

A_shared = memref.global_(
    sym_name="A_shared_BK_BM_1",
    type=T.memref(BK, BM, T.f32(), memory_space=lds_space()),
    alignment=16,
)
B_shared = memref.global_(
    sym_name="B_shared_BK_BN_1",
    type=T.memref(BK, BN, T.f32(), memory_space=lds_space()),
    alignment=16,
)


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel3_registers(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    # Block Tile size
    BN = 128
    BM = 128
    # Number of Row or column we read per batch
    BK = 8

    # Thread Tile size
    TN = 4
    TM = 4

    nbWaves = BLOCK_SIZE // 32
    # Wave Tile size
    WN = 64
    WM = BN * BM // nbWaves // WN

    # Number of wave on X & Y axis in the Block tile
    nbWaveX = BN // WN
    nbWaveY = BM // WM

    waveIndex = thread_idx.x // 32
    waveIdx = waveIndex % nbWaveX
    waveIdy = waveIndex // nbWaveX
    indexInWave = thread_idx.x % 32

    # A wave is a block of 8x4 of the output matrix
    nbThreadXPerWave = 8
    nbThreadYPerWave = 4

    # Thread coordinates in Wave
    idxInWave = indexInWave % nbThreadXPerWave
    idyInWave = indexInWave // nbThreadXPerWave

    nbIterWaveN = WN // (nbThreadXPerWave * TN)
    nbIterWaveM = WM // (nbThreadYPerWave * TM)

    # Wave Sub-tile size
    SUBWN = WN // nbIterWaveN
    SUBWM = WM // nbIterWaveM

    # Thread mapping to read BKxBN block from A
    rAIdx = thread_idx.x % BK
    rAIdy = thread_idx.x // BK
    # Thread mapping to read BNxBK block from B
    rBIdx = thread_idx.x % BN
    rBIdy = thread_idx.x // BN

    strideReadB = BLOCK_SIZE // BN
    strideReadA = BLOCK_SIZE // BK
    nbReadsB = BN * BK // BLOCK_SIZE
    nbReadsA = BM * BK // BLOCK_SIZE

    A_col = memref.alloca([nbIterWaveM * TM], T.f32())
    B_row = memref.alloca([nbIterWaveN * TN], T.f32())

    As = memref.get_global(A_shared)
    Bs = memref.get_global(B_shared)

    l = TM * nbIterWaveM * TN * nbIterWaveN
    c_regs = memref.alloca([l], T.f32())

    c_regs_idx = memref.extract_aligned_pointer_as_index(c_regs)
    c_regs_i64 = arith.index_cast(c_regs_idx, T.i64())
    c_regs_ptr = llvm.inttoptr(llvm.llvm_ptr_t(), c_regs_i64)

    l_4 = llvm.mlir_constant(l * 4)
    c_0 = llvm.mlir_constant(0, T.i8())
    llvm.intr_memset(c_regs_ptr, c_0, l_4, False)

    for kId in scf.range_(0, N, BK):

        for i in range(nbReadsB):
            index_x = BN * block_idx.x + rBIdx
            index_y = rBIdy + i * strideReadB + kId
            Bs[index_y % BK, index_x % BN] = B[index_y, index_x]

        for i in range(nbReadsA):
            index_x = rAIdx + kId
            index_y = BM * block_idx.y + rAIdy + i * strideReadA
            As[index_x % BK, index_y % BM] = A[index_y, index_x]

        gpu.barrier()

        for k in range(BK):
            for iterWave in range(nbIterWaveN):
                for i in range(TN):
                    index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
                    B_row[iterWave * TN + i] = Bs[k, index]

            for iterWave in range(nbIterWaveM):
                for i in range(TM):
                    index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
                    A_col[iterWave * TM + i] = As[k, index]

            for iterWaveM in range(nbIterWaveM):
                for iterWaveN in range(nbIterWaveN):
                    for yt in range(TM):
                        for xt in range(TN):
                            x = iterWaveN * TN + xt
                            y = iterWaveM * TM + yt
                            a = A_col[y]
                            b = B_row[x]
                            c = c_regs[y * TN * nbIterWaveN + x]
                            c = llvm.intr_fmuladd(a, b, c)
                            # c = llvm.intr_fma(a, b, c)
                            c_regs[y * TN * nbIterWaveN + x] = c
                            # c_regs[y * TN * nbIterWaveN + x] += a * b

        gpu.barrier()

    for iterWaveM in range(nbIterWaveM):
        for iterWaveN in range(nbIterWaveN):
            xOut = block_idx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
            yOut = block_idx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
            for yt in range(TM):
                for xt in range(TN):
                    C[yOut + yt, xOut + xt] = c_regs[
                        TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)
                    ]


launch_params[kernel3_registers.__name__] = (
    (N // 128, N // 128, 1),
    (BLOCK_SIZE, 1, 1),
    0,
)


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel4_gmem_db(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    # Thread Tile size
    TN = 4
    TM = 4

    nbWaves = BLOCK_SIZE // 32
    # Wave Tile size
    WN = 64
    WM = BN * BM // nbWaves // WN

    # Number of wave on X & Y axis in the Block tile
    nbWaveX = BN // WN

    # A wave is a block of 8x4 of the output matrix
    nbThreadXPerWave = 8
    nbThreadYPerWave = 4

    nbIterWaveN = WN // (nbThreadXPerWave * TN)
    nbIterWaveM = WM // (nbThreadYPerWave * TM)

    # Wave Sub-tile size
    SUBWN = WN // nbIterWaveN
    SUBWM = WM // nbIterWaveM

    strideReadB = BLOCK_SIZE // BN
    strideReadA = BLOCK_SIZE // BK
    nbReadsB = BN * BK // BLOCK_SIZE
    nbReadsA = BM * BK // BLOCK_SIZE

    waveIndex = thread_idx.x // 32
    waveIdx = waveIndex % nbWaveX
    waveIdy = waveIndex // nbWaveX
    indexInWave = thread_idx.x % 32

    # Thread coordinates in Wave
    idxInWave = indexInWave % nbThreadXPerWave
    idyInWave = indexInWave / nbThreadXPerWave

    # Thread mapping to read BKxBN block from A
    rAIdx = thread_idx.x % BK
    rAIdy = thread_idx.x // BK
    # Thread mapping to read BNxBK block from B
    rBIdx = thread_idx.x % BN
    rBIdy = thread_idx.x // BN

    A_col = memref.alloca([nbIterWaveM * TM], T.f32())
    B_row = memref.alloca([nbIterWaveN * TN], T.f32())

    As = memref.get_global(A_shared)
    Bs = memref.get_global(B_shared)

    l = TM * nbIterWaveM * TN * nbIterWaveN
    c_regs = memref.alloca([l], T.f32())

    c_regs_idx = memref.extract_aligned_pointer_as_index(c_regs)
    c_regs_i64 = arith.index_cast(c_regs_idx, T.i64())
    c_regs_ptr = llvm.inttoptr(llvm.llvm_ptr_t(), c_regs_i64)

    l_4 = llvm.mlir_constant(l * 4)
    c_0 = llvm.mlir_constant(0, T.i8())
    llvm.intr_memset(c_regs_ptr, c_0, l_4, False)

    for i in range(nbReadsB):
        index_x = BN * block_idx.x + rBIdx
        index_y = rBIdy + i * strideReadB
        Bs[index_y % BK, index_x % BN] = B[index_y, index_x]

    for i in range(nbReadsA):
        index_x = rAIdx
        index_y = BM * block_idx.y + rAIdy + i * strideReadA
        As[index_x % BK, index_y % BM] = A[index_y, index_x]

    gpu.barrier()

    regA = memref.alloca([nbReadsA], T.f32())
    regB = memref.alloca([nbReadsB], T.f32())

    N_minus_BK = arith.constant(N - BK, index=True)

    for kId in scf.range_(0, N, BK):

        pred = index_dialect.cmp(index_dialect.IndexCmpPredicate.SLT, kId, N_minus_BK)
        if pred:

            for i in range(nbReadsB):
                index_x = BN * block_idx.x + rBIdx
                index_y = rBIdy + i * strideReadB + kId + BK
                regB[i] = B[index_y, index_x]

            for i in range(nbReadsA):
                index_x = rAIdx + kId + BK
                index_y = BM * block_idx.y + rAIdy + i * strideReadA
                regA[i] = A[index_y, index_x]

        for k in range(BK):

            for iterWave in range(nbIterWaveN):
                for i in range(TN):
                    index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
                    B_row[iterWave * TN + i] = Bs[k, index]

            for iterWave in range(nbIterWaveM):
                for i in range(TM):
                    index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
                    A_col[iterWave * TM + i] = As[k, index]

            for iterWaveM in range(nbIterWaveM):
                for iterWaveN in range(nbIterWaveN):
                    for yt in range(TM):
                        for xt in range(TN):
                            x = iterWaveN * TN + xt
                            y = iterWaveM * TM + yt
                            a = A_col[y]
                            b = B_row[x]
                            c = c_regs[y * TN * nbIterWaveN + x]
                            c = llvm.intr_fmuladd(a, b, c)
                            c_regs[y * TN * nbIterWaveN + x] = c

        gpu.barrier()

        if pred:

            for i in range(nbReadsB):
                index_x = BN * block_idx.x + rBIdx
                index_y = rBIdy + i * strideReadB + kId + BK
                Bs[index_y % BK, index_x % BN] = regB[i]

            for i in range(nbReadsA):
                index_x = rAIdx + kId + BK
                index_y = BM * block_idx.y + rAIdy + i * strideReadA
                As[index_x % BK, index_y % BM] = regA[i]

            gpu.barrier()

    for iterWaveM in range(nbIterWaveM):
        for iterWaveN in range(nbIterWaveN):
            xOut = block_idx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
            yOut = block_idx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
            for yt in range(TM):
                for xt in range(TN):
                    C[yOut + yt, xOut + xt] = c_regs[
                        TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)
                    ]


launch_params[kernel4_gmem_db.__name__] = (
    (N // 128, N // 128, 1),
    (BLOCK_SIZE, 1, 1),
    0,
)

A_shared = memref.global_(
    sym_name="A_shared_BK_BM_times_4",
    type=T.memref(BK, BM * 4, T.f32(), memory_space=lds_space()),
    alignment=16,
)


@gpu_func(emit=True)
@canonicalize(using=scf.canonicalizer)
def kernel5_lds_optim(
    A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())
):
    # Thread Tile size
    TN = 4
    TM = 4

    nbWaves = BLOCK_SIZE // 32
    # Wave Tile size
    WN = 64
    WM = BN * BM // nbWaves // WN

    # Number of wave on X & Y axis in the Block tile
    nbWaveX = BN // WN

    # A wave is a block of 8x4 of the output matrix
    nbThreadXPerWave = 8
    nbThreadYPerWave = 4

    nbIterWaveN = WN // (nbThreadXPerWave * TN)
    nbIterWaveM = WM // (nbThreadYPerWave * TM)

    # Wave Sub-tile size
    SUBWN = WN // nbIterWaveN
    SUBWM = WM // nbIterWaveM

    strideReadB = BLOCK_SIZE // BN
    strideReadA = BLOCK_SIZE // BK
    nbReadsB = BN * BK // BLOCK_SIZE
    nbReadsA = BM * BK // BLOCK_SIZE

    waveIndex = thread_idx.x // 32
    waveIdx = waveIndex % nbWaveX
    waveIdy = waveIndex // nbWaveX
    indexInWave = thread_idx.x % 32

    # Thread coordinates in Wave
    idxInWave = indexInWave % nbThreadXPerWave
    idyInWave = indexInWave / nbThreadXPerWave

    # Thread mapping to read BKxBN block from A
    rAIdx = thread_idx.x % BK
    rAIdy = thread_idx.x // BK
    # Thread mapping to read BNxBK block from B
    rBIdx = thread_idx.x % BN
    rBIdy = thread_idx.x // BN

    A_col = memref.alloca([nbIterWaveM * TM], T.f32())
    B_row = memref.alloca([nbIterWaveN * TN], T.f32())

    As = memref.get_global(A_shared)
    Bs = memref.get_global(B_shared)

    l = TM * nbIterWaveM * TN * nbIterWaveN
    c_regs = memref.alloca([l], T.f32())

    c_regs_idx = memref.extract_aligned_pointer_as_index(c_regs)
    c_regs_i64 = arith.index_cast(c_regs_idx, T.i64())
    c_regs_ptr = llvm.inttoptr(llvm.llvm_ptr_t(), c_regs_i64)

    l_4 = llvm.mlir_constant(l * 4)
    c_0 = llvm.mlir_constant(0, T.i8())
    llvm.intr_memset(c_regs_ptr, c_0, l_4, False)

    for i in range(nbReadsB):
        index_x = BN * block_idx.x + rBIdx
        index_y = rBIdy + i * strideReadB
        Bs[index_y % BK, index_x % BN] = B[index_y, index_x]

    for i in range(nbReadsA):
        index_x = rAIdx
        index_y = BM * block_idx.y + rAIdy + i * strideReadA
        As[index_x % BK, index_y % BM] = A[index_y, index_x]

    gpu.barrier()

    regA = memref.alloca([nbReadsA], T.f32())
    regB = memref.alloca([nbReadsB], T.f32())

    for kId in scf.range_(0, N, BK):

        kId_i32 = arith.index_cast(kId, to=T.i32())
        if kId_i32 < (N - BK):

            for i in range(nbReadsB):
                index_x = BN * block_idx.x + rBIdx
                index_y = rBIdy + i * strideReadB + kId + BK
                regB[i] = B[index_y, index_x]

            for i in range(nbReadsA):
                index_x = rAIdx + kId + BK
                index_y = BM * block_idx.y + rAIdy + i * strideReadA
                regA[i] = A[index_y, index_x]

        for k in range(BK):

            for iterWave in range(nbIterWaveN):
                for i in range(TN):
                    index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
                    B_row[iterWave * TN + i] = Bs[k, index]

            for iterWave in range(nbIterWaveM):
                for i in range(TM):
                    index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
                    A_col[iterWave * TM + i] = As[k, index]

            for iterWaveM in range(nbIterWaveM):
                for iterWaveN in range(nbIterWaveN):
                    for yt in range(TM):
                        for xt in range(TN):
                            x = iterWaveN * TN + xt
                            y = iterWaveM * TM + yt
                            a = A_col[y]
                            b = B_row[x]
                            c = c_regs[y * TN * nbIterWaveN + x]
                            c = llvm.intr_fmuladd(a, b, c)
                            c_regs[y * TN * nbIterWaveN + x] = c

        gpu.barrier()

        if kId_i32 < (N - BK):

            for i in range(nbReadsB):
                index_x = BN * block_idx.x + rBIdx
                index_y = rBIdy + i * strideReadB + kId + BK
                Bs[index_y % BK, index_x % BN] = regB[i]

            for i in range(nbReadsA):
                index_x = rAIdx + kId + BK
                index_y = BM * block_idx.y + rAIdy + i * strideReadA
                As[index_x % BK, index_y % BM] = regA[i]

            gpu.barrier()

    for iterWaveM in range(nbIterWaveM):
        for iterWaveN in range(nbIterWaveN):
            xOut = block_idx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
            yOut = block_idx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
            for yt in range(TM):
                for xt in range(TN):
                    C[yOut + yt, xOut + xt] = c_regs[
                        TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)
                    ]


launch_params[kernel5_lds_optim.__name__] = (
    (N // 128, N // 128, 1),
    (BLOCK_SIZE, 1, 1),
    0,
)


ip.__exit__(None, None, None)

assert gpu_module.operation.verify()

simplified_module = run_pipeline(
    ctx.module,
    Pipeline()
    .canonicalize()
    .cse()
    .loop_invariant_code_motion()
    .loop_invariant_subset_hoisting()
    .rocdl_attach_target(chip=get_hip_arch(), O=3, abi="500"),
)

assert simplified_module.operation.verify()
# print(simplified_module)

lowered_module = run_pipeline(
    simplified_module,
    Pipeline()
    .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True))
    .gpu_to_llvm()
    .lower_to_llvm(),
    # .Nested("llvm.func", Pipeline().sroa()),
)

assert lowered_module.operation.verify()
# print(lowered_module)

gep = find_ops(lowered_module.operation, lambda o: isinstance(o.opview, llvm.GEPOp))
for g in gep:
    g.attributes["inbounds"] = UnitAttr.get()


kernel_funcs = find_ops(
    lowered_module.operation, lambda o: isinstance(o.opview, llvm.LLVMFuncOp)
)
target_flags = "+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32".split(
    ","
)
flags = ", ".join([f'"{t}"' for t in target_flags])
for k in kernel_funcs:
    _, thread_dims, _ = launch_params[k.sym_name.value]
    k.attributes["rocdl.max_flat_work_group_size"] = IntegerAttr.get(
        T.index(), np.prod(thread_dims)
    )
    k.attributes["target_features"] = Attribute.parse(
        f"#llvm.target_features<[{flags}]>"
    )


if hip_bindings_not_installed():
    exit()
from hip import hip


lowered_module = run_pipeline(lowered_module, Pipeline().gpu_module_to_binary())
hsaco = get_compile_object_bytes(lowered_module)
# with open("/home/mlevental/dev_projects/fp32_sgemm_amd/pythonkernels.hsaco", "wb") as f:
#     f.write(hsaco)
hip_module = hip_check(hip.hipModuleLoadData(hsaco))

a_h = np.random.randint(0, 10, (M, K)).astype(dtype=np.float32)
b_h = np.random.randint(0, 10, (K, N)).astype(dtype=np.float32)
# a_h = np.ones((M, K)).astype(dtype=np.float32)
# b_h = np.ones((M, K)).astype(dtype=np.float32)

a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes))
b_d = hip_check(hip.hipMalloc(b_num_bytes))

stream = 0

times = {
    kernel1_naive: 0,
    kernel2_lds_shared_subview: 0,
    kernel2_lds_shared_direct_dynamic: 0,
    kernel2_lds_shared_direct_load_globals: 0,
    kernel3_registers: 0,
    kernel4_gmem_db: 0,
    kernel5_lds_optim: 0,
}
# random.shuffle(kernels)
runs = 16
for kernel in times:
    for i in range(runs):
        function = hip_check(
            hip.hipModuleGetFunction(hip_module, kernel.__name__.encode())
        )
        hip_check(hip.hipDeviceSynchronize())

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
            a_d,
            b_d,
            c_d,
        )

        hip_check(
            hip.hipMemcpy(
                c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
            )
        )
        correct = a_h @ b_h
        if not np.allclose(correct, c_h):
            # with np.printoptions(threshold=np.inf, linewidth=np.inf):
            # print(correct)
            # print(c_h)
            print(f"{kernel.__name__} failed")

        times[kernel] += time_compute

        # print(f"{kernel.__name__} : {time_compute}")

for k in times:
    times[k] /= runs

for k, v in times.items():
    print(f"{k.__name__}: {v:.03f}ms GLOPs {time_to_gflops(v, N)}")
