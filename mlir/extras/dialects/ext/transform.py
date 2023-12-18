from typing import Optional, Union, Sequence

from ... import types as T
from ...meta import region_op
from ...util import get_user_code_loc
from ....dialects.transform.structured import _dispatch_mixed_values, TileUsingForOp
from ....dialects._structured_transform_ops_gen import (
    TileUsingForallOp,
    MatchOp,
)
from ....dialects.transform import ApplyPatternsOp
from ....dialects.transform import (
    SequenceOp,
    FailurePropagationMode,
    YieldOp,
    AnyOpType,
    OperationType,
)
from ....dialects.transform.loop import LoopUnrollOp
from ....dialects.transform import GetParentOp
from ....ir import (
    Type,
    Value,
    Operation,
    StringAttr,
)
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects import pdl


pdl_operation_t = lambda: pdl.OperationType.get()


def transform_any_op_t():
    return AnyOpType.get()


def transform_op_t(name):
    return OperationType.get(name)


def sequence_(
    target: Optional[Union[Operation, Value, Type, str]] = None,
    target_tag=None,
    failure_propagation_mode: FailurePropagationMode = None,
    results_: list[Type] = None,
    extra_bindings: list[Value] = None,
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
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return get_op_result_or_op_results(
        MatchOp(
            transform_any_op_t(),
            target,
            ops=ops,
            interface=interface,
            op_attrs=op_attrs,
            filter_result_type=filter_result_type,
            loc=loc,
            ip=ip,
        )
    )


def tile(
    target: Value,
    *,
    sizes: list[int],
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
