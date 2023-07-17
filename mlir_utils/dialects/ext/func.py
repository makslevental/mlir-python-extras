import inspect
from functools import wraps

from mlir.dialects.func import FuncOp, ReturnOp
from mlir.ir import InsertionPoint, FunctionType, StringAttr, TypeAttr

from mlir_utils.dialects.util import (
    get_result_or_results,
    make_maybe_no_args_decorator,
)


@make_maybe_no_args_decorator
def func(sym_visibility=None, arg_attrs=None, res_attrs=None, loc=None, ip=None):
    ip = ip or InsertionPoint.current

    def builder_wrapper(body_builder):
        @wraps(body_builder)
        def wrapper(*args):
            sig = inspect.signature(body_builder)
            implicit_return = sig.return_annotation is inspect._empty
            function_type = TypeAttr.get(
                FunctionType.get(
                    inputs=[a.type for a in args],
                    results=[] if implicit_return else sig.return_annotation,
                )
            )
            # FuncOp is extended but we do really want the base
            op = FuncOp.__base__(
                body_builder.__name__,
                function_type,
                sym_visibility=StringAttr.get(str(sym_visibility))
                if sym_visibility is not None
                else None,
                arg_attrs=arg_attrs,
                res_attrs=res_attrs,
                loc=loc,
                ip=ip,
            )
            op.regions[0].blocks.append(*[a.type for a in args])
            with InsertionPoint(op.regions[0].blocks[0]):
                r = get_result_or_results(
                    body_builder(*op.regions[0].blocks[0].arguments)
                )
                if r is not None:
                    if isinstance(r, (tuple, list)):
                        ReturnOp(list(r))
                    else:
                        ReturnOp([r])
                else:
                    ReturnOp([])
                return r

        # wrapper.op = op
        return wrapper

    return builder_wrapper
