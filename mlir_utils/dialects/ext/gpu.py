from typing import Optional

from mlir.ir import (
    Type,
    register_attribute_builder,
    Attribute,
    UnitAttr,
    Context,
    FlatSymbolRefAttr,
    ArrayAttr,
)

from mlir_utils.dialects.gpu import block_id


@register_attribute_builder("GPU_DimensionAttr")
def _dimAttr(dim, context=None):
    return Attribute.parse(f"#gpu<dim {dim}>", context=context)


def block_id_x():
    return block_id("x")


def block_id_y():
    return block_id("y")


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


@register_attribute_builder("GPU_AllReduceOperationAttr")
def gpu_all_reduce_op_attr(
    op: str, context: Optional[Context] = None
) -> FlatSymbolRefAttr:
    if context is None:
        context = Context.current
    ops = {"add", "and", "max", "min", "mul", "or", "xor"}
    if op not in ops:
        raise ValueError(f"{op=} not in {ops=}")
    return Attribute.parse(f"#gpu<all_reduce_op {op}>", context)


@register_attribute_builder("DeviceMappingArrayAttr")
def get_device_mapping_array_attr(
    mapping: list[Attribute], context: Optional[Context] = None
) -> ArrayAttr:
    if context is None:
        context = Context.current
    if isinstance(mapping, ArrayAttr):
        return mapping

    return ArrayAttr.get(mapping, context=context)
