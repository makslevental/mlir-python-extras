import inspect
from functools import partial
from typing import Any, List, Optional, Tuple, Union


from .arith import constant
from .func import FuncBase
from ... import types as T
from ...meta import (
    region_op,
)
from ...util import (
    ModuleMeta,
    _get_previous_frame_idents,
    get_user_code_loc,
    make_maybe_no_args_decorator,
    find_ops,
)
from ....dialects._gpu_ops_gen import _Dialect
from ....dialects._ods_common import (
    _cext,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from ....dialects.gpu import *
from ....ir import (
    ArrayAttr,
    AttrBuilder,
    Attribute,
    Context,
    InsertionPoint,
    ShapedType,
    Type,
    UnitAttr,
    Value,
    register_attribute_builder,
)

_block_id = block_id
_thread_id = thread_id
_block_dim = block_dim


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class block_idx:
    @classproperty
    def x(cls):
        return _block_id("x")

    @classproperty
    def y(cls):
        return _block_id("y")

    @classproperty
    def z(cls):
        return _block_id("z")


class block_dim:
    @classproperty
    def x(cls):
        return _block_dim("x")

    @classproperty
    def y(cls):
        return _block_dim("y")

    @classproperty
    def z(cls):
        return _block_dim("z")


class thread_idx:
    @classproperty
    def x(cls):
        return _thread_id("x")

    @classproperty
    def y(cls):
        return _thread_id("y")

    @classproperty
    def z(cls):
        return _thread_id("z")


def thread_id():
    return (
        block_dim.x * block_dim.y * thread_idx.z
        + block_dim.x * thread_idx.y
        + thread_idx.x
    )


# TODO(max): replace all the parsing here with upstream bindings work
def gpu_async_token():
    return Type.parse("!gpu.async.token")


def set_container_module(module):
    module.operation.attributes["gpu.container_module"] = UnitAttr.get()
    return module


@register_attribute_builder("DeviceMappingArrayAttr")
def get_device_mapping_array_attr(
    mapping: List[Attribute], context: Optional[Context] = None
) -> ArrayAttr:
    if context is None:
        context = Context.current
    if isinstance(mapping, ArrayAttr):
        return mapping

    return ArrayAttr.get(mapping, context=context)


def gpu_attr(mnemonic, attr_value):
    return Attribute.parse(f"#gpu.{mnemonic}<{attr_value}>")


def thread_attr(thread):
    return gpu_attr("thread", thread)


def block_attr(block):
    return gpu_attr("block", block)


def warp_attr(warp):
    return gpu_attr("warp", warp)


def warpgroup_attr(warpgroup):
    return gpu_attr("warpgroup", warpgroup)


def address_space_attr(address_space: AddressSpace):
    return gpu_attr("address_space", address_space)


_int = int


def smem_space(int=False):
    a = AddressSpace.Workgroup
    if int:
        return _int(a)

    return address_space_attr(a)


@_cext.register_operation(_Dialect, replace=True)
class GPUModuleOp(GPUModuleOp):
    def __init__(
        self, sym_name, targets: Optional[List[Attribute]] = None, *, loc=None, ip=None
    ):
        if loc is None:
            loc = get_user_code_loc()
        if targets is None:
            targets = []
        for i, t in enumerate(targets):
            if isinstance(t, str):
                targets[i] = Attribute.parse(t)
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


module = region_op(GPUModuleOp, terminator=lambda *_args: module_end())


class GPUModuleMeta(ModuleMeta):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = get_user_code_loc()
        targets = kwargs.pop("targets", None)
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
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
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
    grid_size: Tuple[Any, Any, Any],
    block_size: Tuple[Any, Any, Any],
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


launch = region_op(launch_, terminator=lambda *_args: TerminatorOp())


class LaunchFuncOp(LaunchFuncOp):
    def __init__(
        self,
        kernel: List[str],
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        kernel_operands: List[Value] = None,
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
        *kernel_operands: List[Value],
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
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
                (
                    [self.qualname, self.func_name]
                    if self.qualname is not None
                    else [self.func_name]
                ),
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
    def __init__(self, func_):
        self.func = func_

    def __getitem__(self, item):
        previous_frame = inspect.currentframe().f_back
        var_names = [_get_previous_frame_idents(arg, previous_frame) for arg in item]
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
def func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    emit=False,
    generics=None,
    loc=None,
    ip=None,
    emit_grid=False,
) -> Grid:
    if loc is None:
        loc = get_user_code_loc()
    if generics is None and hasattr(f, "__type_params__") and f.__type_params__:
        generics = f.__type_params__
    func_ = GPUFunc(
        body_builder=f,
        func_op_ctor=GPUFuncOp,
        return_op_ctor=ReturnOp,
        call_op_ctor=LaunchFuncOp,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        generics=generics,
        loc=loc,
        ip=ip,
    )
    func_.__name__ = f.__name__
    if emit:
        func_.emit()
    if emit_grid:
        func_ = Grid(func_)
    return func_


def all_reduce__(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return AllReduceOp(value, op=op, uniform=uniform, loc=loc, ip=ip)


def all_reduce_(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    return get_op_result_or_op_results(
        all_reduce__(value, op=op, uniform=uniform, loc=loc, ip=ip)
    )


all_reduce = region_op(all_reduce__, terminator=YieldOp)


def wait(async_dependencies: Optional[List[Value]] = None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = gpu_async_token()
    return get_op_result_or_op_results(
        WaitOp(async_token, async_dependencies, loc=loc, ip=ip)
    )


_alloc = alloc


def alloc(
    sizes: Union[int, Value],
    element_type: Type = None,
    async_dependencies=None,
    dynamic_sizes=None,
    symbol_operands=None,
    host_shared=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if symbol_operands is None:
        symbol_operands = []
    if dynamic_sizes is None:
        dynamic_sizes = []
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()

    memref_shape = []
    for s in sizes:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            dynamic_sizes.append(s)
    memref = T.memref(*memref_shape, element_type)
    return _alloc(
        memref,
        async_token,
        async_dependencies,
        dynamic_sizes,
        symbol_operands,
        host_shared=host_shared,
        loc=loc,
        ip=ip,
    )


_dealloc = dealloc


def dealloc(memref, async_dependencies=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()
    return _dealloc(async_token, async_dependencies, memref, loc=loc, ip=ip)


_memcpy = memcpy


def memcpy(dst, src, async_dependencies=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()
    return _memcpy(
        async_token,
        async_dependencies,
        dst,
        src,
        loc=loc,
        ip=ip,
    )


def get_compile_object_bytes(compiled_module):
    binary = find_ops(compiled_module, lambda o: isinstance(o, BinaryOp), single=True)
    objects = list(map(ObjectAttr, binary.objects))
    return objects[-1].object


_printf = printf


def printf(format, *args):
    loc = get_user_code_loc()
    return _printf(format=format, args=args, loc=loc)


_dynamic_shared_memory = dynamic_shared_memory


def dynamic_shared_memory(*, int=False, loc=None, ip=None):
    return _dynamic_shared_memory(
        T.memref(
            ShapedType.get_dynamic_size(),
            element_type=T.i8(),
            memory_space=smem_space(int),
        ),
        loc=loc,
        ip=ip,
    )


_memset = memset


def memset(dst, value, async_dependencies=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()
    if isinstance(value, (int, float, bool)):
        value = constant(value, type=dst.type.element_type)
    return _memset(async_token, async_dependencies, dst, value, loc=loc, ip=ip)
