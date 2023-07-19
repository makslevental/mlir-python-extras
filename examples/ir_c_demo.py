import ctypes

from mlir_utils._configuration.configuration import _get_mlir_package_prefix
from mlir_utils.ctypesgen.mlir_capi import (
    mlirDialectRegistryCreate,
    mlirRegisterAllDialects,
    mlirContextAppendDialectRegistry,
    mlirDialectRegistryDestroy,
    mlirContextCreate,
    mlirContextGetOrLoadDialect,
    mlirStringRefCreateFromCString,
    mlirLocationUnknownGet,
    mlirModuleCreateEmpty,
    mlirModuleGetBody,
    mlirModuleGetOperation,
    mlirOperationDump,
)
from mlir.ir import Operation


def registerAllUpstreamDialects(ctx):
    registry = mlirDialectRegistryCreate()

    mlirRegisterAllDialects(registry)
    mlirContextAppendDialectRegistry(ctx, registry)
    mlirDialectRegistryDestroy(registry)


ctx = mlirContextCreate()
registerAllUpstreamDialects(ctx)
mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("func"))
mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("memref"))
mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("shape"))
mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("scf"))

location = mlirLocationUnknownGet(ctx)


def makeAndDumpAdd(ctx, location):
    moduleOp = mlirModuleCreateEmpty(location)
    # moduleBody = mlirModuleGetBody(moduleOp)
    return moduleOp


moduleOp = makeAndDumpAdd(ctx, location)
module = mlirModuleGetOperation(moduleOp)
mlirOperationDump(module)


def MAKE_MLIR_PYTHON_QUALNAME(x):
    return f"{_get_mlir_package_prefix()}.{x}"


PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
MLIR_PYTHON_CAPSULE_OPERATION = MAKE_MLIR_PYTHON_QUALNAME("ir.Operation._CAPIPtr")
capsule = PyCapsule_New(
    module.ptr,
    ctypes.c_char_p(MLIR_PYTHON_CAPSULE_OPERATION.encode("utf-8")),
    PyCapsule_Destructor(0),
)

op = Operation._CAPICreate(capsule)
print(op.name)
