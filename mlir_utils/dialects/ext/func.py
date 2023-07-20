import inspect
from functools import wraps, partial

from mlir.dialects.func import FuncOp, ReturnOp, CallOp
from mlir.ir import (
    InsertionPoint,
    FunctionType,
    StringAttr,
    TypeAttr,
    FlatSymbolRefAttr,
    Type,
)

from mlir_utils.dialects.util import (
    get_result_or_results,
    make_maybe_no_args_decorator,
    maybe_cast,
)


def func_base(
    FuncOp,
    ReturnOp,
    CallOp,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    ip = ip or InsertionPoint.current

    # if this is set to true then wrapper below won't emit a call op
    # it is set below by a def emit fn that is attached to the body_builder
    # wrapper; thus you can call wrapped_fn.emit() (i.e., without an operands)
    # and the func will be emitted.
    _emit = False

    def builder_wrapper(body_builder):
        @wraps(body_builder)
        def wrapper(*call_args):
            sig = inspect.signature(body_builder)
            implicit_return = sig.return_annotation is inspect._empty
            input_types = [p.annotation for p in sig.parameters.values()]
            if not (
                len(input_types) == len(sig.parameters)
                and all(isinstance(t, Type) for t in input_types)
            ):
                input_types = [a.type for a in call_args]
            function_type = TypeAttr.get(
                FunctionType.get(
                    inputs=input_types,
                    results=[] if implicit_return else sig.return_annotation,
                )
            )
            # FuncOp is extended but we do really want the base
            func_name = body_builder.__name__
            func_op = FuncOp(
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
            func_op.regions[0].blocks.append(*input_types)
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

            if _emit:
                return maybe_cast(get_result_or_results(func_op))
            else:
                call_op = CallOp(
                    [r.type for r in results],
                    FlatSymbolRefAttr.get(func_name),
                    call_args,
                )
                return maybe_cast(get_result_or_results(call_op))

        def emit():
            nonlocal _emit
            _emit = True
            wrapper()

        wrapper.emit = emit
        return wrapper

    return builder_wrapper


func = make_maybe_no_args_decorator(
    partial(func_base, FuncOp=FuncOp.__base__, ReturnOp=ReturnOp, CallOp=CallOp)
)
