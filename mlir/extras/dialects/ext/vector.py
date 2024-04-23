import inspect
from typing import List

from ._shaped_value import ShapedValue
from .arith import ArithValue, FastMathFlags, constant, Scalar
from ...util import get_user_code_loc, _update_caller_vars
from ...._mlir_libs._mlir import register_value_caster
from ....dialects._ods_common import _dispatch_mixed_values

# noinspection PyUnresolvedReferences
from ....dialects.vector import *
from ....extras import types as T
from ....ir import AffineMap, VectorType, Value


@register_value_caster(VectorType.static_typeid)
class Vector(ShapedValue, ArithValue):
    def __getitem__(self, idx: tuple) -> "Vector":
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if idx is None:
            raise RuntimeError("None idx not supported")

        idx = list((idx,) if isinstance(idx, (int, Scalar, slice)) else idx)
        return extract(self, tuple(idx), loc=loc)

    def __setitem__(self, idx, val):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, (Scalar, int, Value)) else idx)
        res = insert(self, val, idx, loc=loc)
        previous_frame = inspect.currentframe().f_back
        _update_caller_vars(previous_frame, [self], [res])


_transfer_write = transfer_write


def transfer_write(
    vector: Vector,
    dest,
    indices,
    *,
    permutation_map=None,
    mask: List[int] = None,
    in_bounds: List[bool] = None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if permutation_map is None:
        permutation_map = AffineMap.get_minor_identity(dest.type.rank, vector.type.rank)
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    return _transfer_write(
        result=None,
        vector=vector,
        # no clue why they chose this name...
        source=dest,
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
    ip=None,
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


_insert = insert


def insert(vector, val, position, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    dynamic_position, _packed_position, static_position = _dispatch_mixed_values(
        position
    )
    return _insert(
        val,
        dest=vector,
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
    ip=None,
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
