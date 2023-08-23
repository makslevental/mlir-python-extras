from typing import Optional, Union, Sequence

from mlir.dialects._structured_transform_ops_ext import _get_value_list
from mlir.dialects.transform import (
    SequenceOp,
    FailurePropagationMode,
    YieldOp,
)
from mlir.dialects.transform.loop import GetParentForOp, LoopUnrollOp
from mlir.dialects.transform.structured import MatchOp, TileToScfForOp
from mlir.ir import (
    Type,
    Value,
    Operation,
    StringAttr,
    ArrayAttr,
    register_attribute_builder,
    Context,
)

import mlir_utils.types as T
from mlir_utils.util import (
    get_user_code_loc,
    maybe_cast,
    get_result_or_results,
)
from mlir_utils.util import (
    region_op,
)


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
        target = T.pdl_operation
    # this is a misnomer - it's not about targeting a particular op
    # but about picking which transform sequence runs using
    # transform_dialect_interpreter(debug_transform_root_tag="")
    if target_tag is None:
        target_tag = str(loc).split("/")[-1]
    if extra_bindings is None:
        extra_bindings = []
    if failure_propagation_mode is None:
        failure_propagation_mode = FailurePropagationMode.PROPAGATE

    if isinstance(target, str):
        target = T.transform_op(target)

    seq_op = SequenceOp(
        failure_propagation_mode, results_, target, extra_bindings  # , loc=loc, ip=ip
    )
    seq_op.operation.attributes["transform.target_tag"] = StringAttr.get(target_tag)

    return seq_op


sequence = region_op(sequence_, terminator=YieldOp)

StrOrAttrList = Sequence[Union[StringAttr, str]]


@register_attribute_builder("StrArrayAttr")
def _get_str_array_attr(
    values: Optional[Union[ArrayAttr, StrOrAttrList]], context: Context
) -> ArrayAttr:
    if values is None:
        return ArrayAttr.get([], context=context)

    values = _get_value_list(values)
    return ArrayAttr.get(
        [StringAttr.get(v, context=context) for v in values], context=context
    )


def get_parent_for(target: Value, *, num_loops=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()

    return maybe_cast(
        get_result_or_results(
            GetParentForOp(T.pdl_operation, target, num_loops=num_loops, loc=loc, ip=ip)
        )
    )


def unroll(target: Value, factor=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(LoopUnrollOp(target, factor=factor, loc=loc, ip=ip))
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
    return maybe_cast(
        get_result_or_results(
            MatchOp(
                T.transform_any_op(),
                target,
                ops=ops,
                interface=interface,
                op_attrs=op_attrs,
                filter_result_type=filter_result_type,
                loc=loc,
                ip=ip,
            )
        )
    )


def tile_to_scf_for(
    target: Value,
    tile_sizes: list[int],
    *,
    interchange=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    tiled_linalg_op: Type = target.type
    loops: list[Type] = [target.type] * len(tile_sizes)
    dynamic_sizes: list[Value] = []
    static_sizes = tile_sizes

    t = tuple(
        maybe_cast(
            get_result_or_results(
                TileToScfForOp(
                    tiled_linalg_op,
                    loops,
                    target,
                    dynamic_sizes,
                    static_sizes=static_sizes,
                    interchange=interchange,
                    loc=loc,
                    ip=ip,
                )
            )
        )
    )

    return t[0], t[1:]
