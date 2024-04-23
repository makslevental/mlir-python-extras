from .gpu import smem_space
from . import arith
from ....dialects.nvgpu import *
from ....ir import Attribute, Type
from ... import types as T


def nvgpu_type(mnemonic, attr_value):
    return Type.parse(f"!nvgpu.{mnemonic}<{attr_value}>")


def barrier_group_t(num_barriers=1, address_space=None):
    if address_space is None:
        address_space = smem_space()
    return nvgpu_type(
        "mbarrier.group", f"memorySpace={address_space}, num_barriers = {num_barriers}"
    )


def warpgroup_accumulator_t(M, N, dtype):
    return nvgpu_type("warpgroup.accumulator", f"fragmented=vector<{M}x{N}x{dtype}>")


def warpgroup_descriptor(M, N, dtype):
    return nvgpu_type(
        "warpgroup.descriptor",
        f"tensor=memref<{M}x{N}x{dtype}, {smem_space()}>",
    )


_mbarrier_init = mbarrier_init


_mbarrier_create = mbarrier_create


def mbarrier_create(num_barriers=1, address_space=None, *, loc=None, ip=None):
    return _mbarrier_create(
        barriers=barrier_group_t(num_barriers, address_space), loc=loc, ip=ip
    )


def mbarrier_init(barriers, count, mbar_id, *, predicate=None, loc=None, ip=None):
    if isinstance(count, int):
        count = arith.constant(count, index=True)
    if isinstance(mbar_id, int):
        mbar_id = arith.constant(mbar_id, index=True)
    return _mbarrier_init(
        barriers=barriers,
        count=count,
        mbar_id=mbar_id,
        predicate=predicate,
        loc=loc,
        ip=ip,
    )


_mbarrier_arrive_expect_tx = mbarrier_arrive_expect_tx


def mbarrier_arrive_expect_tx(
    barriers, txcount, mbar_id, *, predicate=None, loc=None, ip=None
):
    if isinstance(txcount, int):
        txcount = arith.constant(txcount, index=True)
    if isinstance(mbar_id, int):
        mbar_id = arith.constant(mbar_id, index=True)
    return _mbarrier_arrive_expect_tx(
        barriers=barriers,
        txcount=txcount,
        mbar_id=mbar_id,
        predicate=predicate,
        loc=loc,
        ip=ip,
    )


_tma_async_load = tma_async_load


def tma_async_load(
    dst,
    barriers,
    tensor_map_descriptor,
    coordinates,
    mbar_id,
    *,
    multicast_mask=None,
    predicate=None,
    loc=None,
    ip=None,
):
    for i, c in enumerate(coordinates):
        if isinstance(c, int):
            coordinates[i] = arith.constant(c, index=True)

    if isinstance(mbar_id, int):
        mbar_id = arith.constant(mbar_id, index=True)

    return _tma_async_load(
        dst=dst,
        barriers=barriers,
        tensor_map_descriptor=tensor_map_descriptor,
        coordinates=coordinates,
        mbar_id=mbar_id,
        multicast_mask=multicast_mask,
        predicate=predicate,
        loc=loc,
        ip=ip,
    )


_mbarrier_try_wait_parity = mbarrier_try_wait_parity


def mbarrier_try_wait_parity(
    barriers, mbar_id, phase_parity=False, ticks=10000000, *, loc=None, ip=None
):
    if isinstance(ticks, int):
        ticks = arith.constant(ticks, index=True)
    if isinstance(mbar_id, int):
        mbar_id = arith.constant(mbar_id, index=True)
    if isinstance(phase_parity, bool):
        phase_parity = arith.constant(phase_parity, type=T.bool())
    return _mbarrier_try_wait_parity(
        barriers=barriers,
        phase_parity=phase_parity,
        ticks=ticks,
        mbar_id=mbar_id,
        loc=loc,
        ip=ip,
    )


_warpgroup_mma = warpgroup_mma


def warpgroup_mma(
    matrix_c,
    descriptor_a,
    descriptor_b,
    *,
    wait_group=None,
    transpose_a=None,
    transpose_b=None,
    loc=None,
    ip=None,
):
    matrix_d = matrix_c.type
    return _warpgroup_mma(
        matrix_d=matrix_d,
        descriptor_a=descriptor_a,
        descriptor_b=descriptor_b,
        matrix_c=matrix_c,
        wait_group=wait_group,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        loc=loc,
        ip=ip,
    )
