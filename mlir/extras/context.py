import contextlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from .. import ir


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None,
    context: ir.Context = None,
    location: ir.Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with ExitStack() as stack:
        stack.enter_context(context)
        if location is None:
            location = ir.Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = ir.Module.parse(src)
        else:
            module = ir.Module.create()
        ip = ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield MLIRContext(context, module)
    context._clear_live_operations()


@contextlib.contextmanager
def enable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)
