from typing import List

from ._shaped_value import ShapedValue
from .arith import ArithValue, FastMathFlags, constant
from ...util import get_user_code_loc
from ...._mlir_libs._mlir import register_value_caster
from ....dialects._ods_common import _dispatch_mixed_values

# noinspection PyUnresolvedReferences
from ....dialects.vector import *
from ....extras import types as T
from ....ir import AffineMap, VectorType


@register_value_caster(VectorType.static_typeid)
class Vector(ShapedValue, ArithValue):
    pass


_transfer_write = transfer_write


def transfer_write(
    vector: Vector,
    source,
    indices,
    *,
    permutation_map=None,
    mask: List[int] = None,
    in_bounds: List[bool] = None,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    if permutation_map is None:
        permutation_map = AffineMap.get_minor_identity(
            source.type.rank, vector.type.rank
        )
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    return _transfer_write(
        result=None,
        vector=vector,
        source=source,
        indices=indices,
        permutation_map=permutation_map,
        mask=mask,
        in_bounds=in_bounds,
        loc=loc,
        ip=ip,
    )


_transfer_read = transfer_read


def transfer_read(
    vector_t,
    source,
    indices,
    *,
    permutation_map=None,
    padding=None,
    mask=None,
    in_bounds=None,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    if permutation_map is None:
        permutation_map = AffineMap.get_minor_identity(source.type.rank, vector_t.rank)
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    if padding is None:
        padding = 0
    if isinstance(padding, int):
        padding = constant(padding, type=source.type.element_type)

    return _transfer_read(
        vector=vector_t,
        source=source,
        indices=indices,
        permutation_map=permutation_map,
        padding=padding,
        mask=mask,
        in_bounds=in_bounds,
        loc=loc,
        ip=ip,
    )


_extract = extract


def extract(vector, position, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    dynamic_position, _packed_position, static_position = _dispatch_mixed_values(
        position
    )
    return _extract(
        vector=vector,
        dynamic_position=dynamic_position,
        static_position=static_position,
        loc=loc,
        ip=ip,
    )


_reduction = reduction


def reduction(
    kind: CombiningKind,
    vector,
    *,
    acc=None,
    fastmath: FastMathFlags = None,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    dest = vector.type.element_type
    return _reduction(
        dest=dest,
        kind=kind,
        vector=vector,
        acc=acc,
        fastmath=fastmath,
        loc=loc,
        ip=ip,
    )


_broadcast = broadcast


def broadcast(vector, source, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(source, (float, int, bool)):
        source = constant(source)
    return _broadcast(vector=vector, source=source, loc=loc, ip=ip)


_extract_strided_slice = extract_strided_slice


def extract_strided_slice(vector, offsets, sizes, strides, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_shape = [int(s) for s in sizes] + vector.type.shape[len(sizes) :]
    result = T.vector(*result_shape, vector.type.element_type)
    return _extract_strided_slice(
        result=result,
        vector=vector,
        offsets=offsets,
        sizes=sizes,
        strides=strides,
        loc=loc,
        ip=ip,
    )
