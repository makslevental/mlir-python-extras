import inspect

from mlir.dialects.func import FuncOp, ReturnOp, CallOp
from mlir.ir import (
    InsertionPoint,
    FunctionType,
    StringAttr,
    TypeAttr,
    FlatSymbolRefAttr,
    Type,
    Location,
)

from mlir_utils.util import (
    get_result_or_results,
    maybe_cast,
    make_maybe_no_args_decorator,
    get_user_code_loc,
)


class FuncBase:
    def __init__(
        self,
        body_builder,
        func_op_ctor,
        return_op_ctor,
        call_op_ctor,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        loc=None,
        ip=None,
    ):
        assert inspect.isfunction(body_builder), body_builder
        assert inspect.isclass(func_op_ctor), func_op_ctor
        assert inspect.isclass(return_op_ctor), return_op_ctor
        assert inspect.isclass(call_op_ctor), call_op_ctor

        self.body_builder = body_builder
        self.func_name = self.body_builder.__name__

        self.func_op_ctor = func_op_ctor
        self.return_op_ctor = return_op_ctor
        self.call_op_ctor = call_op_ctor
        self.sym_visibility = (
            StringAttr.get(str(sym_visibility)) if sym_visibility is not None else None
        )
        self.arg_attrs = arg_attrs
        self.res_attrs = res_attrs
        self.loc = loc
        self.ip = ip or InsertionPoint.current
        self.emitted = False

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def body_builder_wrapper(self, *call_args):
        sig = inspect.signature(self.body_builder)
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
        func_op = self.func_op_ctor(
            self.func_name,
            function_type,
            sym_visibility=self.sym_visibility,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            loc=self.loc,
            ip=self.ip,
        )
        arg_locs = [get_user_code_loc()] * len(sig.parameters)
        func_op.regions[0].blocks.append(*input_types, arg_locs=arg_locs)
        with InsertionPoint(func_op.regions[0].blocks[0]):
            results = get_result_or_results(
                self.body_builder(
                    *[maybe_cast(a) for a in func_op.regions[0].blocks[0].arguments]
                )
            )
            if results is not None:
                if isinstance(results, (tuple, list)):
                    results = list(results)
                else:
                    results = [results]
            else:
                results = []
            self.return_op_ctor(results)

        return results, input_types, func_op

    def emit(self):
        self.results, input_types, func_op = self.body_builder_wrapper()
        return_types = [v.type for v in self.results]
        function_type = FunctionType.get(inputs=input_types, results=return_types)
        func_op.attributes["function_type"] = TypeAttr.get(function_type)
        self.emitted = True
        # this is the func op itself (funcs never have a resulting ssa value)
        return maybe_cast(get_result_or_results(func_op))

    def __call__(self, *call_args, loc: Location = None):
        if loc is None:
            loc = get_user_code_loc()
        if not self.emitted:
            self.emit()
        call_op = self.call_op_ctor(
            [r.type for r in self.results],
            FlatSymbolRefAttr.get(self.func_name),
            call_args,
            loc=loc,
        )
        return maybe_cast(get_result_or_results(call_op))


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp.__base__,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        loc=loc,
        ip=ip,
    )
