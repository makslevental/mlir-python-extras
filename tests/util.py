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


def llvm_amdgcn_bindings_not_installed():
    try:
        from mlir.extras.dialects.ext.llvm import amdgcn

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
    gridX,
    gridY,
    gridZ,
    warp_size,
    num_warps,
    stream,
    shared_memory,
    *args,
):
    import chip

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
    chip_check(
        chip.hipModuleLaunchKernel(
            function,
            gridX,
            gridY,
            gridZ,
            warp_size,
            num_warps,
            1,
            shared_memory,
            stream,
            c_args,
            None,
        )
    )


def get_hip_arch():
    if hip_bindings_not_installed():
        return "gfx1100"

    from hip import hip

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    return props.gcnArchName.decode()
