import inspect
from functools import partial
from typing import Optional, Any

from ....dialects._ods_common import get_default_loc_context, _cext
from ....dialects.gpu import *
from ....dialects._gpu_ops_gen import _Dialect
from ....ir import (
    Type,
    Attribute,
    AttrBuilder,
    UnitAttr,
    register_attribute_builder,
    Context,
    ArrayAttr,
    InsertionPoint,
    Value,
)

from ... import types as T
from .arith import constant
from .func import FuncBase
from ...meta import (
    region_op,
)
from ....dialects._ods_common import get_op_result_or_op_results
from ...util import get_user_code_loc, make_maybe_no_args_decorator, ModuleMeta


def block_id_x():
    return block_id("x")


def block_id_y():
    return block_id("y")


def gpu_async_token():
    return Type.parse("!gpu.async.token")


def set_container_module(module):
    module.operation.attributes["gpu.container_module"] = UnitAttr.get()
    return module


@register_attribute_builder("DeviceMappingArrayAttr")
def get_device_mapping_array_attr(
    mapping: list[Attribute], context: Optional[Context] = None
) -> ArrayAttr:
    if context is None:
        context = Context.current
    if isinstance(mapping, ArrayAttr):
        return mapping

    return ArrayAttr.get(mapping, context=context)


def device_mapping_attr(mnemonic, mapping_id_enum: MappingId):
    return Attribute.parse(f"#gpu.{mnemonic}<{mapping_id_enum}>")


def thread_attr(thread):
    return device_mapping_attr("thread", thread)


def block_attr(block):
    return device_mapping_attr("block", block)


def warp_attr(warp):
    return device_mapping_attr("warp", warp)


def warpgroup_attr(warpgroup):
    return device_mapping_attr("warpgroup", warpgroup)


def memory_space_attr(address_space: AddressSpace):
    return device_mapping_attr("memory_space", address_space)


@_cext.register_operation(_Dialect, replace=True)
class GPUModuleOp(GPUModuleOp):
    def __init__(
        self, sym_name, targets: Optional[list[Attribute]] = None, *, loc=None, ip=None
    ):
        if loc is None:
            loc = get_user_code_loc()
        if targets is None:
            targets = []
        _ods_context = get_default_loc_context(loc)
        super().__init__(targets=ArrayAttr.get(targets), loc=loc, ip=ip)
        self.regions[0].blocks.append()
        self.operation.attributes["sym_name"] = (
            sym_name
            if (
                issubclass(type(sym_name), Attribute)
                or not AttrBuilder.contains("SymbolNameAttr")
            )
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )

    @property
    def body(self):
        return self.regions[0].blocks[0]


class GPUModuleMeta(ModuleMeta):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = get_user_code_loc()
        targets = kwargs.pop("targets", None)
        if targets is not None:
            for i, t in enumerate(targets):
                if isinstance(t, str):
                    targets[i] = Attribute.parse(t)
        gpu_module_op = GPUModuleOp(
            sym_name=name,
            targets=targets,
            ip=kwargs.pop("ip", None),
            loc=loc,
        )
        ip = InsertionPoint(gpu_module_op.body)
        ip.__enter__()
        return {
            "ip": ip,
            "loc": loc,
            "gpu_module_op": gpu_module_op,
            "module_terminator": module_end,
        }


class GPUFuncOp(GPUFuncOp):
    def __init__(
        self,
        sym_name,
        function_type,
        *,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        workgroup_attrib_attrs=None,
        private_attrib_attrs=None,
        loc=None,
        ip=None,
    ):
        super().__init__(
            function_type=function_type,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
            workgroup_attrib_attrs=workgroup_attrib_attrs,
            private_attrib_attrs=private_attrib_attrs,
            loc=loc,
            ip=ip,
        )
        self.operation.attributes["gpu.kernel"] = UnitAttr.get()
        _ods_context = get_default_loc_context(loc)
        self.operation.attributes["sym_name"] = (
            sym_name
            if (
                issubclass(type(sym_name), Attribute)
                or not AttrBuilder.contains("SymbolNameAttr")
            )
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        if sym_visibility is not None:
            self.operation.attributes["sym_visibility"] = (
                sym_visibility
                if (
                    issubclass(type(sym_visibility), Attribute)
                    or not AttrBuilder.contains("StrAttr")
                )
                else AttrBuilder.get("StrAttr")(sym_visibility, context=_ods_context)
            )


class LaunchOp(LaunchOp):
    def __init__(
        self,
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        _ods_context = get_default_loc_context(loc)
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size

        super().__init__(
            async_token,
            async_dependencies,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*[T.index() for _ in range(12)])


def launch_(
    grid_size: tuple[Any, Any, Any],
    block_size: tuple[Any, Any, Any],
    async_dependencies=None,
    dynamic_shared_memory_size: Optional[Value] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
        for size in [grid_size, block_size]:
            for i, s in enumerate(size):
                if isinstance(s, int):
                    size[i] = constant(s, index=True)
    launch_op = LaunchOp(
        grid_size,
        block_size,
        async_dependencies,
        dynamic_shared_memory_size,
        loc=loc,
        ip=ip,
    )
    return launch_op


launch = region_op(launch_, terminator=lambda *args: TerminatorOp())


class LaunchFuncOp(LaunchFuncOp):
    def __init__(
        self,
        kernel: list[str],
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        kernel_operands: list[Value] = None,
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        async_object=None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        _ods_context = get_default_loc_context(loc)
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size

        super().__init__(
            async_token,
            async_dependencies,
            kernel,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            kernel_operands,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            asyncObject=async_object,
            loc=loc,
            ip=ip,
        )


class GPUFunc(FuncBase):
    def __call__(
        self,
        *kernel_operands: list[Value],
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        stream=None,
    ):
        for size in [grid_size, block_size]:
            for i, s in enumerate(size):
                if isinstance(s, int):
                    size[i] = constant(s, index=True)

        loc = get_user_code_loc()
        return get_op_result_or_op_results(
            LaunchFuncOp(
                [self.qualname, self.func_name]
                if self.qualname is not None
                else [self.func_name],
                grid_size,
                block_size,
                kernel_operands,
                async_dependencies,
                dynamic_shared_memory_size,
                async_object=stream,
                loc=loc,
            )
        )


class Grid:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, item):
        previous_frame = inspect.currentframe().f_back
        var_names = [
            [
                var_name
                for var_name, var_val in previous_frame.f_locals.items()
                if var_val is arg
            ]
            for arg in item
        ]
        kwargs = {}
        for i, it in enumerate(item):
            assert len(var_names[i]) == 1, "expected unique kwarg"
            k = var_names[i][0]
            kwargs[k] = it

        return partial(self.func, **kwargs)

    @property
    def qualname(self):
        return self.func.qualname

    @qualname.setter
    def qualname(self, v):
        self.func.qualname = v

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@make_maybe_no_args_decorator
def gpu_func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    emit=True,
    loc=None,
    ip=None,
) -> Grid:
    if loc is None:
        loc = get_user_code_loc()
    func = GPUFunc(
        body_builder=f,
        func_op_ctor=GPUFuncOp,
        return_op_ctor=ReturnOp,
        call_op_ctor=LaunchFuncOp,
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
    return Grid(func)


def all_reduce__(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return AllReduceOp(value, op=op, uniform=uniform, loc=loc, ip=ip)


def all_reduce_(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    return get_op_result_or_op_results(
        all_reduce__(value, op=op, uniform=uniform, loc=loc, ip=ip)
    )


all_reduce = region_op(all_reduce__, terminator=YieldOp)


def wait(async_dependencies: Optional[list[Value]] = None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = gpu_async_token()
    return get_op_result_or_op_results(
        WaitOp(async_token, async_dependencies, loc=loc, ip=ip)
    )
