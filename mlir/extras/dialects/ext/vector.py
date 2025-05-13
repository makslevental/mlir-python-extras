import inspect
from typing import List

from ._shaped_value import ShapedValue, _indices_to_indexer
from .arith import ArithValue, FastMathFlags, constant, Scalar
from ...util import get_user_code_loc, _update_caller_vars, Infix
from ...._mlir_libs._mlir import register_value_caster
from ....dialects._ods_common import _dispatch_mixed_values

# noinspection PyUnresolvedReferences
from ....dialects.vector import *
from ....extras import types as T
from ....ir import AffineMap, VectorType, Value


@register_value_caster(VectorType.static_typeid)
@ShapedValue
class Vector(ArithValue):
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
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)
        if all(isinstance(d, (int, Scalar)) for d in idx):
            return extract(self, tuple(idx), loc=loc)
        else:
            indexer = _indices_to_indexer(idx, self.shape)
            return extract_strided_slice(
                self,
                offsets=indexer.static_offsets(),
                sizes=indexer.static_sizes(),
                strides=indexer.static_strides(),
                loc=loc,
            )

    def __setitem__(self, idx, val):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked vector slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, (Scalar, int, Value, slice)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)
        if all(isinstance(d, Scalar) for d in idx):
            res = insert(self, val, idx, loc=loc)
        else:
            indexer = _indices_to_indexer(tuple(idx), self.shape)
            if indexer.is_constant():
                res = insert_strided_slice(
                    val,
                    self,
                    offsets=indexer.static_offsets(),
                    strides=indexer.static_strides(),
                    loc=loc,
                    ip=None,
                )
            else:
                raise ValueError(f"non-constant indices not supported {indexer}")

        previous_frame = inspect.currentframe().f_back
        _update_caller_vars(previous_frame, [self], [res])


_transfer_write = transfer_write


def transfer_write(
    val: Vector,
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
        permutation_map = AffineMap.get_minor_identity(dest.type.rank, val.type.rank)
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    return _transfer_write(
        result=None,
        value_to_store=val,
        # no clue why they chose this name...
        base=dest,
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
    base,
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
        permutation_map = AffineMap.get_minor_identity(base.type.rank, vector_t.rank)
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    if padding is None:
        padding = 0
    if isinstance(padding, int):
        padding = constant(padding, type=base.type.element_type)
    if in_bounds is None:
        raise ValueError("in_bounds cannot be None")

    return _transfer_read(
        vector=vector_t,
        base=base,
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


def insert(vector, val, positions, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if len(positions) == 0:
        raise ValueError("positions cannot be empty")
    dynamic_position, _packed_position, static_position = _dispatch_mixed_values(
        positions
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


_outerproduct = outerproduct


def outerproduct(lhs, rhs, acc=None, *, kind=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if kind is None:
        kind = CombiningKind.ADD
    result_shape = [lhs.shape[0], rhs.shape[0]]
    result = VectorType.get(result_shape, lhs.type.element_type)
    return OuterProductOp(
        result=result, lhs=lhs, rhs=rhs, acc=acc, kind=kind, loc=loc, ip=ip
    ).result


@Infix
def outer(lhs, rhs, acc=None, *, kind=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return outerproduct(lhs, rhs, acc, kind=kind, loc=loc, ip=ip)


_shuffle = shuffle


@Infix
def shuffle(v1, v2, mask, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return ShuffleOp(v1=v1, v2=v2, mask=mask, loc=loc, ip=ip).result


_load = load


def load_(base, indices, result, *, nontemporal=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    for j, i in enumerate(indices):
        if isinstance(i, int):
            indices[j] = constant(i, index=True)
    return LoadOp(
        result=result,
        base=base,
        indices=indices,
        nontemporal=nontemporal,
        loc=loc,
        ip=ip,
    ).result


load = Infix(load_)
