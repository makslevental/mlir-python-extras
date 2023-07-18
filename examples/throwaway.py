import inspect
import sys
from functools import wraps
from pathlib import Path

from mlir_utils.context import mlir_mod_ctx
from mlir_utils._configuration.generate_trampolines import (
    generate_all_upstream_trampolines,
)
from mlir_utils._configuration.configuration import _add_file_to_sources_txt_file

# _add_file_to_sources_txt_file(Path("_configuration/__MLIR_PYTHON_PACKAGE_PREFIX__"))
generate_all_upstream_trampolines()
from mlir_utils.dialects.memref import alloca_scope, return_
from mlir_utils.dialects.transform import foreach, yield_
from mlir_utils.dialects import gpu
from mlir_utils.dialects.ext import func


from mlir_utils.dialects.util import constant

# # generate_all_upstream_trampolines()
# from mlir.dialects.scf import WhileOp
# from mlir.ir import InsertionPoint
#
#
# # from mlir_utils.dialects.scf import execute_region, yield_
#
#
# # def doublewrap(f):
# #     """
# #     a decorator decorator, allowing the decorator to be used as:
# #     @decorator(with, arguments, and=kwargs)
# #     or
# #     @decorator
# #     """
# #
# #     @wraps(f)
# #     def new_dec(*args, **kwargs):
# #         if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
# #             # actual decorated function
# #             return f(args[0])
# #         else:
# #             # decorator arguments
# #             return lambda realf: f(realf, *args, **kwargs)
# #
# #     return new_dec
# #
#
#
# with mlir_mod_ctx() as ctx:
#     one = constant(1)
#
#     @func.func
#     def demo_fun1():
#         one = constant(1)
#         return
#
#     demo_fun1()
#     ctx.module.operation.verify()
#
#
# print(ctx.module)
# ctx.module.operation.verify()
# print(ctx.module)
# from importlib.resources import files
