from typing import Union, Sequence

import mlir.dialects.linalg
from mlir.dialects._ods_common import get_op_result_or_value
from mlir.dialects.linalg import LinalgOpConfig, OperandKind
from mlir.dialects.linalg.opdsl.lang.dsl import _prepare_structured_op_outs
from mlir.dialects.linalg.opdsl.lang.emitter import (
    prepare_common_structured_op,
    _get_operand_def_names,
    _BodyBuilder,
)
from mlir.ir import (
    Value,
    OpResultList,
    Context,
    Attribute,
    Operation,
    OpView,
    ShapedType,
    ArrayAttr,
    AffineMap,
    AffineMapAttr,
    InsertionPoint,
)

ValueList = Union[Sequence[Value], OpResultList]


def emit_generic_structured_op(
    op_config,
    *ins: Value,
    outs: ValueList,
    **attrs: Sequence[int],
):
    (
        all_arg_defs,
        in_arg_defs,
        out_arg_defs,
        outs,
        result_types,
        type_mapping,
        indexing_maps_attr,
        iterator_types_attr,
        index_attrs,
        fn_attr_mapping,
        block_arg_types,
    ) = prepare_common_structured_op(op_config, *ins, outs=outs, **attrs)

    # An operation that accesses only scalars and scalar/rank zero tensors is
    # rank polymorhpic. We implement rank polymorphism by generating different
    # indexing maps and iterators that match the rank of the first output tensor.
    # An operation is rank polymorphic if the iteration domain has rank zero.
    if not iterator_types_attr:
        rank = ShapedType(outs[0].type).rank
        iterator_types_attr = ArrayAttr.get(
            [Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        )
        scalar_map = AffineMap.get(rank, 0, [])
        tensor_map = AffineMap.get_identity(rank)
        indexing_maps = []
        for arg_def in all_arg_defs:
            if arg_def.operand_def.kind == OperandKind.SCALAR:
                indexing_maps.append(scalar_map)
            if arg_def.operand_def.is_tensor():
                idx = arg_def.operand_def.registered_index
                if idx < len(ins) and ShapedType(ins[idx].type).rank == 0:
                    indexing_maps.append(scalar_map)
                else:
                    indexing_maps.append(tensor_map)
        indexing_maps_attr = ArrayAttr.get(
            [AffineMapAttr.get(am) for am in indexing_maps]
        )

    generic_op = mlir.dialects.linalg.GenericOp(
        result_tensors=result_types,
        inputs=ins,
        outputs=outs,
        indexing_maps=indexing_maps_attr,
        iterator_types=iterator_types_attr,
        doc=None,  # TODO: Make optional.
        library_call=None,
    )  # TODO: Make optional.

    # Construct the body.
    block_arg_names = _get_operand_def_names(*in_arg_defs, *out_arg_defs)
    block = generic_op.regions[0].blocks.append(*block_arg_types)
    block_arg_mapping = dict(zip(block_arg_names, block.arguments))
    with InsertionPoint(block):
        body_builder = _BodyBuilder(type_mapping, block_arg_mapping, fn_attr_mapping)
        for assignment in op_config.assignments:
            body_builder.assign(assignment)
        body_builder.yield_outputs(*_get_operand_def_names(*out_arg_defs))

    if len(result_types) == 1:
        return generic_op.result
    else:
        if len(generic_op.results):
            return generic_op.results
        else:
            return generic_op


def emit_named_structured_op(
    op_config,
    op_name: str,
    op_class_name: str,
    *ins: Value,
    outs: ValueList,
    **attrs: Sequence[int],
):
    (
        all_arg_defs,
        in_arg_defs,
        out_arg_defs,
        outs,
        result_types,
        type_mapping,
        indexing_maps_attr,
        iterator_types_attr,
        index_attrs,
        fn_attr_mapping,
        block_arg_types,
    ) = prepare_common_structured_op(op_config, *ins, outs=outs, **attrs)

    # If we get here, there must exist a builtin class `op_class_name`.
    ctx = Context.current
    fully_qualified_name = "linalg." + op_name
    if (
        not ctx.is_registered_operation(fully_qualified_name)
        or not op_class_name in mlir.dialects.linalg.__dict__.keys()
    ):
        raise NotImplementedError(
            f"Unknown named op_name / op_class_name: {op_name} / {op_class_name}"
        )

    # Set the index attributes used to compute the indexing maps.
    named_op = getattr(mlir.dialects.linalg, op_class_name)(ins, outs, result_types)
    for name, value in index_attrs.items():
        named_op.operation.attributes[name] = value

    # Compute the function attributes by combining operand kind and function name.
    for name, (fn_name, kind) in fn_attr_mapping.items():
        assert kind.name.lower().endswith("_attr")
        enum_name = kind.name.lower()[:-5]
        named_op.operation.attributes[name] = Attribute.parse(
            f"#linalg.{enum_name}<{fn_name}>"
        )

    mlir.dialects.linalg.fill_builtin_region(named_op.operation)

    if len(result_types) == 1:
        return named_op.result
    else:
        if len(named_op.results):
            return named_op.results
        else:
            return named_op


mlir.dialects.linalg.opdsl.lang.emitter.emit_named_structured_op = (
    emit_named_structured_op
)


def __call__(
    self,
    *ins: Union[Operation, OpView, Value],
    outs,
    **kwargs,
):
    """Emits the corresponding op definition as IR.

    Most arguments are passed through to the underlying emitter. The following
    keyword argument is interpreted here:
      emit_generic: Emits a generic form as appropriate (default True). If
        False, a named form is emitted (which must have been built in to the
        compiler).
    """
    emit_generic = kwargs.pop("emit_generic", False)
    if not isinstance(emit_generic, bool):
        raise ValueError(
            f"The named argument 'emit_generic' needs to be "
            f" of type bool but got {type(emit_generic)}"
        )

    op_configs = LinalgOpConfig.from_linalg_op_def(self.op_def, context=Context.current)

    if len(op_configs) != 1:
        # TODO: Support composite ops.
        raise NotImplementedError(
            f"Emission of composite linalg ops not supported: {op_configs}"
        )

    ctx = Context.current
    linalgDialect = ctx.get_dialect_descriptor("linalg")
    fully_qualified_name = "linalg." + self.op_name
    emit_generic = emit_generic or not ctx.is_registered_operation(fully_qualified_name)

    op_config = op_configs[0]
    out_values = _prepare_structured_op_outs(outs)
    in_values = [get_op_result_or_value(i) for i in ins]
    if op_config.structured_op:
        if emit_generic:
            return emit_generic_structured_op(
                op_config.structured_op, *in_values, outs=out_values, **kwargs
            )
        else:
            return emit_named_structured_op(
                op_config.structured_op,
                self.op_name,
                self.op_def.metadata.cpp_class_name,
                *in_values,
                outs=out_values,
                **kwargs,
            )

    raise NotImplementedError(f"Emission of linalg op type not supported: {op_config}")


mlir.dialects.linalg.opdsl.lang.dsl.DefinedOpCallable.__call__ = __call__
