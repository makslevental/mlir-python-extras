from ._configuration.configuration import alias_upstream_bindings
import atexit

if alias_upstream_bindings():
    from mlir import ir

    DefaultContext = ir.Context()
    # Push a default context onto the context stack at import time.
    DefaultContext.__enter__()
    DefaultContext.allow_unregistered_dialects = False

    DefaultLocation = ir.Location.unknown()
    DefaultLocation.__enter__()

    @atexit.register
    def __exit_ctxt():
        DefaultContext.__exit__(None, None, None)

    @atexit.register
    def __exit_loc():
        DefaultLocation.__exit__(None, None, None)
