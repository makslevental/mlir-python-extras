from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

import mlir.ir

from mlir_utils import DefaultContext


@dataclass
class MLIRContext:
    context: mlir.ir.Context
    module: mlir.ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None,
    context: mlir.ir.Context = DefaultContext,
    location: mlir.ir.Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = mlir.ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with ExitStack() as stack:
        stack.enter_context(context)
        if location is None:
            location = mlir.ir.Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = mlir.ir.Module.parse(src)
        else:
            module = mlir.ir.Module.create()
        ip = mlir.ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield MLIRContext(context, module)
