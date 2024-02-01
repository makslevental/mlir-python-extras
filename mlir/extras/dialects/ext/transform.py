from typing import Optional, Union, Sequence, List

from ...meta import region_op
from ...util import get_user_code_loc
from ....dialects import pdl
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects.transform import *
from ....dialects.transform.extras import OpHandle
from ....dialects.transform.loop import *
from ....dialects.transform.structured import *
from ....dialects._ods_common import (
    _dispatch_mixed_values,
    _dispatch_dynamic_index_list,
)
from ....ir import Type, Operation, StringAttr, Attribute, Value
from ....dialects._structured_transform_ops_gen import (
    TileUsingForallOp,
    MatchOp,
)

pdl_operation_t = lambda: pdl.OperationType.get()


def _unwrap_op_handle(op_handle):
    assert isinstance(op_handle, OpHandle)
    return Value(op_handle)


def transform_any_op_t():
    return AnyOpType.get()


def transform_any_value_t():
    return AnyValueType.get()


def transform_op_t(name):
    return OperationType.get(name)


def sequence_(
    target: Optional[Union[Operation, Value, Type, str]] = None,
    target_tag=None,
    failure_propagation_mode: FailurePropagationMode = None,
    results_: List[Type] = None,
    extra_bindings: List[Value] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if results_ is None:
        results_ = []
    if target is None:
        target = pdl_operation_t()
    # this is a misnomer - it's not about targeting a particular op
    # but about picking which transform sequence runs using
    # transform_dialect_interpreter(debug_transform_root_tag="")
    if target_tag is None:
        target_tag = str(loc).split("/")[-1]
    if extra_bindings is None:
        extra_bindings = []
    if failure_propagation_mode is None:
        failure_propagation_mode = FailurePropagationMode.Propagate

    if isinstance(target, str):
        target = transform_op_t(target)

    seq_op = SequenceOp(
        failure_propagation_mode,
        results_,
        target,
        extra_bindings,  # loc=loc, ip=ip
    )
    seq_op.operation.attributes["transform.target_tag"] = StringAttr.get(target_tag)

    return seq_op


sequence = region_op(sequence_, terminator=YieldOp)

StrOrAttrList = Sequence[Union[StringAttr, str]]


def get_parent(
    target: Value,
    *,
    isolated_from_above: bool = False,
    op_name: Optional[str] = None,
    deduplicate: bool = False,
    nth_parent: int = 1,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    return get_op_result_or_op_results(
        GetParentOp(
            pdl_operation_t(),
            target,
            isolated_from_above=isolated_from_above,
            op_name=op_name,
            deduplicate=deduplicate,
            nth_parent=nth_parent,
            loc=loc,
            ip=ip,
        )
    )


def unroll(target: Value, factor=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return get_op_result_or_op_results(
        LoopUnrollOp(target, factor=factor, loc=loc, ip=ip)
    )


def match(
    target: Value,
    ops=None,
    *,
    interface=None,
    op_attrs=None,
    filter_result_type=None,
    matched_op=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if matched_op is None:
        matched_op = transform_any_op_t()
    return get_op_result_or_op_results(
        MatchOp(
            matched_op,
            target,
            ops=ops,
            interface=interface,
            op_attrs=op_attrs,
            filter_result_type=filter_result_type,
            loc=loc,
            ip=ip,
        )
    )


def tile_to_scf_for(
    target: Value,
    *,
    sizes: List[int],
    interchange=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    t = tuple(
        (
            get_op_result_or_op_results(
                TileUsingForOp(
                    target,
                    sizes=sizes,
                    interchange=interchange,
                    loc=loc,
                    ip=ip,
                )
            )
        )
    )

    return t[0], t[1:]


def tile_to_scf_forall(
    target,
    tile_sizes,
    num_threads=None,
    *,
    mapping=None,
    loc=None,
    ip=None,
):
    if num_threads is None:
        num_threads = []
    if loc is None:
        loc = get_user_code_loc()
    (
        dynamic_num_threads,
        packed_num_threads,
        static_num_threads,
    ) = _dispatch_mixed_values(num_threads)
    (
        dynamic_tile_sizes,
        packed_tile_sizes,
        static_tile_sizes,
    ) = _dispatch_mixed_values(tile_sizes)

    tiled_op = forall_op = target.type

    t = tuple(
        (
            get_op_result_or_op_results(
                TileUsingForallOp(
                    forall_op,
                    tiled_op,
                    target,
                    num_threads=dynamic_num_threads,
                    tile_sizes=dynamic_num_threads,
                    packed_num_threads=packed_num_threads,
                    packed_tile_sizes=packed_tile_sizes,
                    static_num_threads=static_num_threads,
                    static_tile_sizes=static_tile_sizes,
                    mapping=mapping,
                    loc=loc,
                    ip=ip,
                )
            )
        )
    )

    return t[0], t[1:]


def apply_patterns_(
    target,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return ApplyPatternsOp(
        target,
        loc=loc,
        ip=ip,
    )


apply_patterns = region_op(apply_patterns_)


_structured_fuse_into_containing_op = structured_fuse_into_containing_op


def structured_fuse_into_containing_op(
    producer_op,
    containing_op,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return _structured_fuse_into_containing_op(
        transform_any_op_t(),
        transform_any_op_t(),
        producer_op,
        containing_op,
        loc=loc,
        ip=ip,
    )


_split_handle = split_handle


def split_handle(
    handle,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if not isinstance(handle.owner.opview, MatchOp):
        raise ValueError(f"{handle=} must be an instance of MatchOp")
    return _split_handle(
        [transform_any_op_t()] * len(list(handle.owner.opview.ops)),
        handle,
        loc=loc,
        ip=ip,
    )


_structured_pack = structured_pack


def structured_pack(target, packed_sizes, *, packed_op=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()

    (
        dynamic_packed_sizes,
        # packed here means %1:2 packing (results packing)
        _packed_packed_sizes,
        static_packed_sizes,
    ) = _dispatch_mixed_values(packed_sizes)

    if packed_op is None:
        packed_op = transform_any_op_t()

    return _structured_pack(
        packed_op=packed_op,
        target=target,
        packed_sizes=dynamic_packed_sizes,
        static_packed_sizes=static_packed_sizes,
        loc=loc,
        ip=ip,
    )


_get_producer_of_operand = get_producer_of_operand


def get_producer_of_operand(target, operand_number, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _get_producer_of_operand(
        producer=transform_any_op_t(),
        target=target,
        operand_number=operand_number,
        loc=loc,
        ip=ip,
    )


_structured_pack_transpose = structured_pack_transpose


def structured_pack_transpose(
    target_pack_or_un_pack_op,
    target_linalg_op,
    *,
    outer_perm=None,
    inner_perm=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return _structured_pack_transpose(
        packed_op=transform_any_op_t(),
        pack_op=transform_any_op_t(),
        un_pack_op=transform_any_op_t(),
        target_pack_or_un_pack_op=target_pack_or_un_pack_op,
        target_linalg_op=target_linalg_op,
        outer_perm=outer_perm,
        inner_perm=inner_perm,
        loc=loc,
        ip=ip,
    )


_include = include


def include(target, operands, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _include(
        results_=[],
        target=target,
        failure_propagation_mode=FailurePropagationMode.Propagate,
        operands_=operands,
        loc=loc,
        ip=ip,
    )


_structured_bufferize_to_allocation = structured_bufferize_to_allocation


def structured_bufferize_to_allocation(
    target,
    *,
    memory_space=None,
    memcpy_op=None,
    alloc_op=None,
    bufferize_destination_only=None,
    emit_dealloc=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(memory_space, int):
        memory_space = str(memory_space)
    if isinstance(memory_space, str):
        try:
            memory_space = Attribute.parse(memory_space)
        except:
            memory_space = StringAttr.get(memory_space)

    return _structured_bufferize_to_allocation(
        allocated_buffer=transform_any_value_t(),
        new_ops=transform_any_op_t(),
        target=target,
        memory_space=memory_space,
        memcpy_op=memcpy_op,
        alloc_op=alloc_op,
        bufferize_destination_only=bufferize_destination_only,
        emit_dealloc=emit_dealloc,
        loc=loc,
        ip=ip,
    )


_get_consumers_of_result = get_consumers_of_result


def get_consumers_of_result(target, result_number, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _unwrap_op_handle(
        _get_consumers_of_result(
            consumers=transform_any_op_t(),
            target=target,
            result_number=result_number,
            loc=loc,
            ip=ip,
        )
    )


_structured_lower_pack = structured_lower_pack


def structured_lower_pack(target, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _structured_lower_pack(
        pad_op=transform_op_t("tensor.pad"),
        expand_shape_op=transform_op_t("tensor.expand_shape"),
        transpose_op=transform_op_t("linalg.transpose"),
        target=target,
        loc=loc,
        ip=ip,
    )


_structured_vectorize = structured_vectorize


def structured_vectorize(
    target,
    vector_sizes,
    *,
    vectorize_nd_extract=None,
    loc=None,
    ip=None,
):
    (
        dynamic_vector_sizes,
        static_vector_sizes,
        scalable_sizes,
    ) = _dispatch_dynamic_index_list(vector_sizes)

    return _structured_vectorize(
        target=target,
        vector_sizes=dynamic_vector_sizes,
        vectorize_nd_extract=vectorize_nd_extract,
        scalable_sizes=scalable_sizes,
        static_vector_sizes=static_vector_sizes,
        loc=loc,
        ip=ip,
    )
