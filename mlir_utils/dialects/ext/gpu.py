from typing import Optional

from mlir.ir import (
    Type,
    Attribute,
    UnitAttr,
    register_attribute_builder,
    Context,
    ArrayAttr,
)

from mlir_utils.dialects.ext.gpu_enums import AddressSpace
from mlir_utils.dialects.gpu import block_id


def block_id_x():
    return block_id("x")


def block_id_y():
    return block_id("y")


def warp_group_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.warpgroup<{dim}>")


def warp_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.warp<{dim}>")


def memory_space(address_space: AddressSpace):
    return Attribute.parse(f"#gpu.memory_space<{address_space}>")


def block_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.block<{dim}>")


def thread_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.thread<{dim}>")


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
