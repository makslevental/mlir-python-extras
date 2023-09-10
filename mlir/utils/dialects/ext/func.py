import inspect
from typing import Union, Optional

from ....dialects.func import FuncOp, ReturnOp, CallOp
from ....ir import (
    InsertionPoint,
    FunctionType,
    StringAttr,
    TypeAttr,
    FlatSymbolRefAttr,
    Type,
    Value,
)

from ...util import get_result_or_results, get_user_code_loc, is_311
from ...meta import make_maybe_no_args_decorator, maybe_cast


def call(
    callee_or_results: Union[FuncOp, list[Type]],
    arguments_or_callee: Union[list[Value], FlatSymbolRefAttr, str],
    arguments: Optional[list] = None,
    *,
    call_op_ctor=CallOp.__base__,
    loc=None,
    ip=None,
):
    """Creates an call operation.

    The constructor accepts three different forms:

      1. A function op to be called followed by a list of arguments.
      2. A list of result types, followed by the name of the function to be
         called as string, following by a list of arguments.
      3. A list of result types, followed by the name of the function to be
         called as symbol reference attribute, followed by a list of arguments.

    For example

        f = func.FuncOp("foo", ...)
        func.CallOp(f, [args])
        func.CallOp([result_types], "foo", [args])

    In all cases, the location and insertion point may be specified as keyword
    arguments if not provided by the surrounding context managers.
    """
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(callee_or_results, FuncOp.__base__):
        if not isinstance(arguments_or_callee, (list, tuple)):
            raise ValueError(
                "when constructing a call to a function, expected "
                + "the second argument to be a list of call arguments, "
                + f"got {type(arguments_or_callee)}"
            )
        if arguments is not None:
            raise ValueError(
                "unexpected third argument when constructing a call" + "to a function"
            )
        return call_op_ctor(
            callee_or_results.function_type.value.results,
            FlatSymbolRefAttr.get(callee_or_results.sym_name.value),
            arguments_or_callee,
            loc=loc,
            ip=ip,
        )

    if isinstance(arguments_or_callee, list):
        raise ValueError(
            "when constructing a call to a function by name, "
            + "expected the second argument to be a string or a "
            + f"FlatSymbolRefAttr, got {type(arguments_or_callee)}"
        )

    if isinstance(arguments_or_callee, FlatSymbolRefAttr):
        return call_op_ctor(
            callee_or_results, arguments_or_callee, arguments, loc=loc, ip=ip
        )
    elif isinstance(arguments_or_callee, str):
        return call_op_ctor(
            callee_or_results,
            FlatSymbolRefAttr.get(arguments_or_callee),
            arguments,
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError(f"unexpected type {callee_or_results=}")


class FuncBase:
    def __init__(
        self,
        body_builder,
        func_op_ctor,
        return_op_ctor,
        call_op_ctor,
        return_types=None,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        func_attrs=None,
        loc=None,
        ip=None,
        qualname=None,
    ):
        assert inspect.isfunction(body_builder), body_builder
        assert inspect.isclass(func_op_ctor), func_op_ctor
        assert inspect.isclass(return_op_ctor), return_op_ctor
        assert inspect.isclass(call_op_ctor), call_op_ctor

        self.body_builder = body_builder
        self.func_name = self.body_builder.__name__

        if return_types is None:
            return_types = []
        sig = inspect.signature(self.body_builder)
        self.input_types, self.return_types, self.arg_locs = self.prep_func_types(
            sig, return_types
        )

        self.func_op_ctor = func_op_ctor
        self.return_op_ctor = return_op_ctor
        self.call_op_ctor = call_op_ctor
        self.sym_visibility = (
            StringAttr.get(str(sym_visibility)) if sym_visibility is not None else None
        )
        self.arg_attrs = arg_attrs
        self.res_attrs = res_attrs
        if func_attrs is None:
            func_attrs = {}
        self.func_attrs = func_attrs
        self.loc = loc
        self.ip = ip or InsertionPoint.current
        self._func_op = None
        # in case this function lives inside a class
        self.qualname = qualname

        if self._is_decl():
            assert len(self.input_types) == len(
                sig.parameters
            ), f"func decl needs all input types annotated"
            self.sym_visibility = StringAttr.get("private")
            self.emit()

    def _is_decl(self):
        # magic constant found from looking at the code for an empty fn
        if is_311():
            return self.body_builder.__code__.co_code == b"\x97\x00d\x00S\x00"
        else:
            return self.body_builder.__code__.co_code == b"d\x00S\x00"

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def prep_func_types(self, sig, return_types):
        assert not (
            not sig.return_annotation is inspect.Signature.empty
            and len(return_types) > 0
        ), f"func can use return annotation or explicit return_types but not both"
        return_types = (
            sig.return_annotation
            if not sig.return_annotation is inspect.Signature.empty
            else return_types
        )
        if not isinstance(return_types, (tuple, list)):
            return_types = [return_types]
        return_types = list(return_types)
        assert all(
            isinstance(r, Type) for r in return_types
        ), f"all return types must be mlir types {return_types=}"

        input_types = [
            p.annotation
            for p in sig.parameters.values()
            if not p.annotation is inspect.Signature.empty
        ]
        assert all(
            isinstance(r, Type) for r in input_types
        ), f"all input types must be mlir types {input_types=}"
        return input_types, return_types, [get_user_code_loc()] * len(sig.parameters)

    def body_builder_wrapper(self, *call_args):
        if len(call_args) == 0:
            input_types = self.input_types
        else:
            input_types = [a.type for a in call_args]
        function_type = TypeAttr.get(
            FunctionType.get(
                inputs=input_types,
                results=self.return_types,
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
        if self._is_decl():
            return self.return_types, input_types, func_op

        func_op.regions[0].blocks.append(*input_types, arg_locs=self.arg_locs)
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
        return_types = [r.type for r in results]
        return return_types, input_types, func_op

    def emit(self) -> FuncOp:
        if self._func_op is None:
            return_types, input_types, func_op = self.body_builder_wrapper()
            function_type = FunctionType.get(inputs=input_types, results=return_types)
            func_op.attributes["function_type"] = TypeAttr.get(function_type)
            for k, v in self.func_attrs.items():
                func_op.attributes[k] = v
            self._func_op = func_op
        return self._func_op

    def __call__(self, *call_args):
        return call(self.emit(), call_args)


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    emit=False,
    loc=None,
    ip=None,
) -> FuncBase:
    if loc is None:
        loc = get_user_code_loc()
    func = FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp.__base__,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        loc=loc,
        ip=ip,
    )
    func.__name__ = f.__name__
    if emit:
        func.emit()
    return func
