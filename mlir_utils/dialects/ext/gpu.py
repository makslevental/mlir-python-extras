from functools import partial
from typing import Optional, Any

from mlir.dialects._ods_common import get_default_loc_context
from mlir.dialects.gpu import (
    AddressSpace,
    MappingId,
    GPUModuleOp,
    GPUFuncOp,
    LaunchFuncOp,
    ReturnOp,
)
from mlir.ir import (
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

from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import FuncBase
from mlir_utils.dialects.gpu import block_id, module_end
from mlir_utils.meta import ModuleMeta, make_maybe_no_args_decorator, maybe_cast
from mlir_utils.util import get_user_code_loc, get_result_or_results


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


class GPUModuleOp(GPUModuleOp):
    def __init__(
        self, sym_name, targets: Optional[list[Attribute]] = None, *, loc=None, ip=None
    ):
        if loc is None:
            loc = get_user_code_loc()
        if targets is None:
            targets = []
        _ods_context = get_default_loc_context(loc)
        attributes = {
            "targets": ArrayAttr.get(targets),
            "sym_name": (
                sym_name
                if (
                    issubclass(type(sym_name), Attribute)
                    or not AttrBuilder.contains("SymbolNameAttr")
                )
                else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
            ),
        }
        super().__init__(
            self.build_generic(
                results=[], operands=[], attributes=attributes, loc=loc, ip=ip
            )
        )
        self.regions[0].blocks.append()

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
        operands = []
        results = []
        attributes = {}
        regions = None
        _ods_context = get_default_loc_context(loc)
        attributes["sym_name"] = (
            sym_name
            if (
                issubclass(type(sym_name), Attribute)
                or not AttrBuilder.contains("SymbolNameAttr")
            )
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        attributes["function_type"] = (
            function_type
            if (
                issubclass(type(function_type), Attribute)
                or not AttrBuilder.contains("anonymous_407")
            )
            else AttrBuilder.get("anonymous_407")(function_type, context=_ods_context)
        )
        if sym_visibility is not None:
            attributes["sym_visibility"] = (
                sym_visibility
                if (
                    issubclass(type(sym_visibility), Attribute)
                    or not AttrBuilder.contains("StrAttr")
                )
                else AttrBuilder.get("StrAttr")(sym_visibility, context=_ods_context)
            )
        if arg_attrs is not None:
            attributes["arg_attrs"] = (
                arg_attrs
                if (
                    issubclass(type(arg_attrs), Attribute)
                    or not AttrBuilder.contains("DictArrayAttr")
                )
                else AttrBuilder.get("DictArrayAttr")(arg_attrs, context=_ods_context)
            )
        if res_attrs is not None:
            attributes["res_attrs"] = (
                res_attrs
                if (
                    issubclass(type(res_attrs), Attribute)
                    or not AttrBuilder.contains("DictArrayAttr")
                )
                else AttrBuilder.get("DictArrayAttr")(res_attrs, context=_ods_context)
            )
        if workgroup_attrib_attrs is not None:
            attributes["workgroup_attrib_attrs"] = (
                workgroup_attrib_attrs
                if (
                    issubclass(type(workgroup_attrib_attrs), Attribute)
                    or not AttrBuilder.contains("DictArrayAttr")
                )
                else AttrBuilder.get("DictArrayAttr")(
                    workgroup_attrib_attrs, context=_ods_context
                )
            )
        if private_attrib_attrs is not None:
            attributes["private_attrib_attrs"] = (
                private_attrib_attrs
                if (
                    issubclass(type(private_attrib_attrs), Attribute)
                    or not AttrBuilder.contains("DictArrayAttr")
                )
                else AttrBuilder.get("DictArrayAttr")(
                    private_attrib_attrs, context=_ods_context
                )
            )

        attributes["gpu.kernel"] = UnitAttr.get()

        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )


class LaunchFuncOp(LaunchFuncOp):
    def __init__(
        self,
        kernel: list[str],
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        kernel_operands: list[Value] = None,
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        _ods_context = get_default_loc_context(loc)
        attributes = {
            "kernel": (
                kernel
                if (
                    issubclass(type(kernel), Attribute)
                    or not AttrBuilder.contains("SymbolRefAttr")
                )
                else AttrBuilder.get("SymbolRefAttr")(kernel, context=_ods_context)
            )
        }

        if async_dependencies is None:
            async_dependencies = []
        results = [gpu_async_token()] * len(async_dependencies)
        async_object = None
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size

        regions = None
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=[
                    async_dependencies,
                    grid_size_x,
                    grid_size_y,
                    grid_size_z,
                    block_size_x,
                    block_size_y,
                    block_size_z,
                    dynamic_shared_memory_size,
                    kernel_operands,
                    async_object,
                ],
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )


class GPUFunc(FuncBase):
    def __call__(
        self,
        *kernel_operands: list[Value],
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
    ):
        for size in [grid_size, block_size]:
            for i, s in enumerate(size):
                if isinstance(s, int):
                    size[i] = constant(s, index=True)

        loc = get_user_code_loc()
        return maybe_cast(
            get_result_or_results(
                LaunchFuncOp(
                    [self.qualname, self.func_name]
                    if self.qualname is not None
                    else [self.func_name],
                    grid_size,
                    block_size,
                    kernel_operands,
                    async_dependencies,
                    dynamic_shared_memory_size,
                    loc=loc,
                )
            )
        )


class Grid:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, item):
        kwargs = {}
        for it in item:
            k, v = it.start, it.stop
            kwargs[k] = v

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
