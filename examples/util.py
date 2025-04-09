import ctypes
import sys


def jax_not_installed():
    try:
        from jaxlib import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def mlir_bindings_not_installed():
    try:
        import mlir.extras

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def llvm_bindings_not_installed():
    try:
        import llvm

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def hip_check(call_result):
    from hip import hip

    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def hip_synchronize():
    from hip import hip

    hip.hipDeviceSynchronize()


def hip_bindings_not_installed():
    try:
        from hip import hip

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))

        # don't skip
        return False

    except Exception as e:
        print(e, file=sys.stderr)
        # skip
        return True


def chip_check(status):
    import chip

    if status != 0:
        raise RuntimeError(
            f"HIP Error {status}, {ctypes.string_at(chip.hipGetErrorString(status)).decode()}"
        )


def launch_kernel(
    function,
    blocks_per_grid_x,
    blocks_per_grid_y,
    blocks_per_grid_z,
    threads_per_block_x,
    threads_per_block_y,
    threads_per_block_z,
    stream,
    shared_memory,
    *args,
):
    import chip

    import hip
    from hip._util.types import DeviceArray

    params = [None] * len(args)
    addresses = [None] * len(args)
    for i, p in enumerate(args):
        if isinstance(p, DeviceArray):
            addresses[i] = params[i] = p.createRef().as_c_void_p()
        elif isinstance(p, int):
            params[i] = ctypes.c_int32(p)
            addresses[i] = ctypes.addressof(params[i])
        else:
            raise NotImplementedError(f"{p=} not supported with {p=}")

    c_args = (ctypes.c_void_p * len(addresses))(*addresses)
    function = ctypes.cast(function, chip.hipFunction_t)
    stream = ctypes.cast(stream, chip.hipStream_t)

    tstart = hip_check(hip.hip.hipEventCreate())
    tstop = hip_check(hip.hip.hipEventCreate())
    hip_check(hip.hip.hipEventRecord(tstart, None))

    r = chip.hipModuleLaunchKernel(
        function,
        blocks_per_grid_x,
        blocks_per_grid_y,
        blocks_per_grid_z,
        threads_per_block_x,
        threads_per_block_y,
        threads_per_block_z,
        shared_memory,
        stream,
        c_args,
        None,
    )

    hip_check(hip.hip.hipEventRecord(tstop, None))
    hip_check(hip.hip.hipEventSynchronize(tstop))
    time_compute = hip_check(hip.hip.hipEventElapsedTime(tstart, tstop))

    chip_check(r)

    return time_compute
