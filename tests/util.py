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
