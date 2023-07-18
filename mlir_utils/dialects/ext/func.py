import inspect
from functools import wraps

from mlir.dialects.func import FuncOp, ReturnOp, CallOp
from mlir.ir import (
    InsertionPoint,
    FunctionType,
    StringAttr,
    TypeAttr,
    FlatSymbolRefAttr,
)

from mlir_utils.dialects.util import (
    get_result_or_results,
    make_maybe_no_args_decorator,
)


@make_maybe_no_args_decorator
def func(sym_visibility=None, arg_attrs=None, res_attrs=None, loc=None, ip=None):
    ip = ip or InsertionPoint.current

    def builder_wrapper(body_builder):
        @wraps(body_builder)
        def wrapper(*call_args):
            sig = inspect.signature(body_builder)
            implicit_return = sig.return_annotation is inspect._empty
            input_types = [a.type for a in call_args]
            function_type = TypeAttr.get(
                FunctionType.get(
                    inputs=input_types,
                    results=[] if implicit_return else sig.return_annotation,
                )
            )
            # FuncOp is extended but we do really want the base
            func_name = body_builder.__name__
            func_op = FuncOp.__base__(
                func_name,
                function_type,
                sym_visibility=StringAttr.get(str(sym_visibility))
                if sym_visibility is not None
                else None,
                arg_attrs=arg_attrs,
                res_attrs=res_attrs,
                loc=loc,
                ip=ip,
            )
            func_op.regions[0].blocks.append(*[a.type for a in call_args])
            with InsertionPoint(func_op.regions[0].blocks[0]):
                results = get_result_or_results(
                    body_builder(*func_op.regions[0].blocks[0].arguments)
                )
                if results is not None:
                    if isinstance(results, (tuple, list)):
                        results = list(results)
                    else:
                        results = [results]
                else:
                    results = []
                ReturnOp(results)
            # Recompute the function type.
            return_types = [v.type for v in results]
            function_type = FunctionType.get(inputs=input_types, results=return_types)
            func_op.attributes["function_type"] = TypeAttr.get(function_type)

            call_op = CallOp(
                [r.type for r in results], FlatSymbolRefAttr.get(func_name), call_args
            )
            if results is None:
                return func_op
            return get_result_or_results(call_op)

        # wrapper.op = op
        return wrapper

    return builder_wrapper
