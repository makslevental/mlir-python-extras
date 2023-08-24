from typing import Optional

from mlir.dialects.gpu import AddressSpace, MappingId
from mlir.ir import (
    Type,
    Attribute,
    UnitAttr,
    register_attribute_builder,
    Context,
    ArrayAttr,
)

from mlir_utils.dialects.gpu import block_id


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
