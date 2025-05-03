import inspect
import operator
from itertools import accumulate, zip_longest
from typing import Sequence, Union, Optional

import numpy as np

from ._shaped_value import ShapedValue, _indices_to_indexer
from .arith import Scalar, constant, index_cast
from .tensor import compute_result_shape_reassoc_list
from .vector import Vector
from ... import types as T
from ...meta import region_op
from ...util import (
    _get_sym_name,
    get_user_code_loc,
    infer_mlir_type,
)
from ...._mlir_libs._mlir import register_value_caster
from ....dialects import memref, arith, vector, builtin
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from ....dialects._ods_common import (
    get_op_result_or_op_results,
    MixedValues,
    _dispatch_mixed_values,
)
from ....dialects.memref import (
    _is_static_int_like,
    _infer_memref_subview_result_type,
    _generated_subview,
)
from ....dialects.memref import *
from ....ir import (
    DenseElementsAttr,
    MemRefType,
    UnrankedMemRefType,
    ShapedType,
    IndexType,
    Type,
    Value,
    SymbolTable,
    InsertionPoint,
    StridedLayoutAttr,
)

S = ShapedType.get_dynamic_size()


def __alloc(
    op_ctor,
    sizes: Sequence[Union[int, Value]],
    element_type: Type,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    dynamic_sizes = []
    memref_shape = []
    for s in sizes:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            dynamic_sizes.append(s)
    result_type = T.memref(
        *memref_shape, element_type=element_type, memory_space=memory_space
    )

    symbol_operands = []
    return get_op_result_or_op_results(
        op_ctor(
            result_type,
            dynamic_sizes,
            symbol_operands,
            alignment=alignment,
            loc=loc,
            ip=ip,
        )
    )


_alloc = alloc


def alloc(
    sizes: Union[int, Value],
    element_type: Type = None,
    *,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __alloc(
        AllocOp,
        sizes,
        element_type,
        memory_space=memory_space,
        alignment=alignment,
        loc=loc,
        ip=ip,
    )


_alloca = alloca


def alloca(
    sizes: Union[int, Value],
    element_type: Type = None,
    memory_space=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return __alloc(
        AllocaOp,
        sizes,
        element_type,
        memory_space=memory_space,
        alignment=alignment,
        loc=loc,
        ip=ip,
    )


def load(memref: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
        elif isinstance(i, Value):
            if not _is_index_type(i.type):
                i = index_cast(i, to=IndexType.get(), loc=loc, ip=ip)
                indices[idx] = i
        else:
            raise TypeError(f"expected {i=} to be either int or Value")
    return get_op_result_or_op_results(LoadOp(memref, indices, loc=loc, ip=ip))


def store(
    value: Value,
    memref: Value,
    indices: Sequence[Union[Value, int]],
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
        elif isinstance(i, Value):
            if not _is_index_type(i.type):
                i = index_cast(i, to=IndexType.get(), loc=loc, ip=ip)
                indices[idx] = i
        else:
            raise TypeError(f"expected {i=} to be either int or Value")
    return get_op_result_or_op_results(StoreOp(value, memref, indices, loc=loc, ip=ip))


@register_value_caster(MemRefType.static_typeid)
@ShapedValue
class MemRef(Value):
    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

    def __repr__(self):
        return str(self)

    rank_reduce = object()

    def __getitem__(self, idx: tuple) -> "MemRef":
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if idx is None:
            return expand_shape(self, (0,), loc=loc)

        idx = list((idx,) if isinstance(idx, (int, Scalar, slice)) else idx)
        rank_reduce = MemRef.rank_reduce in idx
        if rank_reduce:
            idx.remove(MemRef.rank_reduce)

        for i, d in enumerate(idx):
            # TODO(max): rethink this since subview and etc probably take constant attributes?
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            return load(self, idx, loc=loc)
        else:
            return _subview(self, tuple(idx), rank_reduce=rank_reduce, loc=loc)

    def __setitem__(self, idx, val):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, (Scalar, int, Value)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            if isinstance(val, (int, float)):
                # TODO: this is an unchecked conversion
                val = Scalar(val, dtype=self.dtype)
            assert isinstance(
                val, (Scalar, Vector)
            ), f"coordinate insert on ranked memref {self.type} requires scalar element but got {val=}"
            if isinstance(val, Scalar):
                store(val, self, idx, loc=loc)
            elif isinstance(val, Vector):
                return vector.StoreOp(
                    valueToStore=val,
                    base=self,
                    indices=idx,
                    loc=loc,
                )
        else:
            _copy_to_subview(self, val, tuple(idx), loc=loc)


def expand_shape(
    inp,
    newaxis_dims,
    *,
    loc=None,
    ip=None,
) -> MemRef:
    """Expand the shape of a memref.

    Insert a new axis that will appear at the `axis` position in the expanded
    memref shape.

    Args:
      inp: Input memref-like.
      axis: Position in the expanded axes where the new axis (or axes) is placed.

    Returns:
       View of `a` with the number of dimensions increased.

    """
    if loc is None:
        loc = get_user_code_loc()

    if len(newaxis_dims) == 0:
        return inp

    result_shape, reassoc_list = compute_result_shape_reassoc_list(
        inp.shape, newaxis_dims
    )

    return MemRef(
        memref.expand_shape(
            T.memref(*result_shape, inp.dtype),
            inp,
            reassoc_list,
            output_shape=[],
            static_output_shape=result_shape,
            loc=loc,
            ip=ip,
        )
    )


def _maybe_compute_size(start, stop, step):
    # TODO(max): figure out how to use actual canonicalizers
    if (
        isinstance(start, Value)
        and isinstance(stop, Value)
        and stop.owner.operands[0]._eq(start)
        and stop.owner.operands[1].is_constant()
    ):
        return stop.owner.operands[1].literal_value
    elif (
        isinstance(start, Value)
        and isinstance(start.owner.opview, arith.MulIOp)
        and isinstance(stop, Value)
        and isinstance(stop.owner.opview, arith.MulIOp)
        and isinstance(stop.owner.operands[0].owner.opview, arith.AddIOp)
        and start.owner.operands[0] == stop.owner.operands[0].owner.operands[0]
        and stop.owner.operands[1].is_constant()
        and isinstance(step, int)
        or (isinstance(step, Scalar) and step.is_constant())
    ):
        # looks like this
        # l = lambda l: l * D
        # r = lambda r: (r + 1) * D
        # a, b, c = (
        #     A[l(i) : r(i), l(j) : r(j)],
        #     B[l(i) : r(i), l(j) : r(j)],
        #     C[l(i) : r(i), l(j) : r(j)],
        # )
        return stop.owner.operands[1]
    else:
        return stop - start


def subview(
    source: Value,
    offsets: MixedValues,
    sizes: MixedValues,
    strides: MixedValues,
    *,
    rank_reduce=False,
    result_type: Optional[MemRefType] = None,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if sizes is None:
        sizes = []
    if strides is None:
        strides = []
    source_strides, source_offset = source.type.get_strides_and_offset()
    if result_type is None and all(
        all(_is_static_int_like(i) for i in s) for s in [sizes, strides, source_strides]
    ):
        # If any are arith.constant results then this will canonicalize to python int
        # (which can then be used to fully specify the subview).
        (
            offsets,
            sizes,
            strides,
            result_type,
        ) = _infer_memref_subview_result_type(source.type, offsets, sizes, strides)
    elif result_type is None:
        raise ValueError(
            "mixed static/dynamic offset/sizes/strides requires explicit result type."
        )

    offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(offsets)
    sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
    strides, _packed_strides, static_strides = _dispatch_mixed_values(strides)

    if rank_reduce:
        result_shape = list(result_type.shape)
        layout_strides = None
        if result_type.layout:
            layout_strides = result_type.layout.strides
        for i, (s, ss) in reversed(
            list(enumerate(list(zip_longest(sizes, static_sizes))))
        ):
            if (
                s is not None and _is_static_int_like(s) and s.literal_value == 1
            ) or ss == 1:
                del result_shape[i]
                if layout_strides is not None:
                    del layout_strides[i]
        reduced_layout = None
        if layout_strides is not None:
            reduced_layout = StridedLayoutAttr.get(
                result_type.layout.offset, layout_strides
            )
        result_type = MemRefType.get(
            result_shape,
            result_type.element_type,
            reduced_layout,
            result_type.memory_space,
        )

    return _generated_subview(
        result_type,
        source,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )


def _subview(
    mem: MemRef,
    idx,
    *,
    rank_reduce=False,
    loc=None,
    ip=None,
) -> MemRef:
    if loc is None:
        loc = get_user_code_loc()

    indexer = _indices_to_indexer(idx, mem.shape)
    out = mem

    if indexer.is_constant():
        offsets = indexer.static_offsets()
        sizes = indexer.static_sizes()
        strides = indexer.static_strides()
    else:
        # special tile case
        offsets = [None] * len(indexer.in_shape)
        sizes = [None] * len(indexer.in_shape)
        strides = [None] * len(indexer.in_shape)
        for i, ind in enumerate(indexer.indices):
            if isinstance(ind, slice):
                maybe_size = _maybe_compute_size(ind.start, ind.stop, ind.step)
                if maybe_size is None:
                    raise RuntimeError(
                        f"failed to canonicalize start, stop, step: {ind=}"
                    )
                offsets[i] = ind.start
                sizes[i] = maybe_size
                strides[i] = (
                    ind.step.literal_value if isinstance(ind.step, Scalar) else ind.step
                )
            elif isinstance(ind, Value):
                offsets[i] = ind
                sizes[i] = 1
                strides[i] = 1
            else:
                raise RuntimeError(f"indexing of {mem=} not supported by {ind=}")
        assert all(
            map(lambda x: x is not None, offsets + sizes + strides)
        ), f"not each slice is statically known: {indexer.indices}"

    out = subview(
        out,
        offsets=offsets,
        sizes=sizes,
        strides=strides,
        rank_reduce=rank_reduce,
        loc=loc,
        ip=ip,
    )

    # This adds newaxis/None dimensions.
    return expand_shape(out, indexer.newaxis_dims, loc=loc, ip=ip)


def _copy_to_subview(
    dest: MemRef,
    source: MemRef,
    idx,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(source, Scalar):
        source = expand_shape(source, (0,), loc=loc, ip=ip)

    dest_subview = _subview(dest, idx, loc=loc, ip=ip)
    assert (
        dest_subview.shape == source.shape
    ), f"Expected matching shape for dest subview {dest_subview.shape} and source {source.shape=}"

    return memref.copy(source, dest_subview, loc=loc, ip=ip)


alloca_scope = region_op(AllocaScopeOp)

_dim = dim


def dim(source, index, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(index, int):
        index = constant(index, index=True)
    return _dim(source=source, index=index, loc=loc, ip=ip)


def global_(
    initial_value=None,
    sym_name=None,
    type=None,
    sym_visibility="private",
    constant=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if sym_name is None:
        previous_frame = inspect.currentframe().f_back
        sym_name = _get_sym_name(
            previous_frame, check_func_call="memref\\.global_|global_"
        )
        assert (
            sym_name is not None
        ), "couldn't automatically find sym_name in previous frame"
    if loc is None:
        loc = get_user_code_loc()
    if initial_value is None:
        assert type is not None
    else:
        assert isinstance(initial_value, np.ndarray)
        if type is None:
            type = infer_mlir_type(initial_value, memref=True)
        initial_value = DenseElementsAttr.get(
            initial_value,
            type=type.element_type,
            context=None,
        )
        constant = True

    return memref.global_(
        sym_name,
        type,
        sym_visibility=sym_visibility,
        initial_value=initial_value,
        constant=constant,
        alignment=alignment,
        loc=loc,
        ip=ip,
    ).opview


def view(source, shape, dtype=None, shift=0, memory_space=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if dtype is None:
        dtype = source.type.element_type
    byte_width_dtype = dtype.width // 8
    byte_shift = shift * byte_width_dtype
    if isinstance(byte_shift, int):
        byte_shift = constant(byte_shift, index=True)
    elif isinstance(byte_shift, Value):
        if not _is_index_type(byte_shift.type):
            byte_shift = index_cast(byte_shift, to=IndexType.get(), loc=loc, ip=ip)
    else:
        raise TypeError(f"expected {byte_shift=} to be either int or Value")
    assert _is_index_type(byte_shift.type), "expected index type for byte-shift"
    if memory_space is None and source:
        memory_space = source.type.memory_space

    dynamic_sizes = []
    memref_shape = []
    for s in shape:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            if not _is_index_type(s.type):
                s = index_cast(s, to=IndexType.get(), loc=loc, ip=ip)
            dynamic_sizes.append(s)

    return memref.view(
        T.memref(*memref_shape, element_type=dtype, memory_space=memory_space),
        source,
        byte_shift,
        dynamic_sizes,
        loc=loc,
        ip=ip,
    )


_get_global = get_global


def get_global(
    name_or_global, *, name=None, global_=None, result=None, loc=None, ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(name_or_global, GlobalOp):
        global_ = name_or_global
    elif isinstance(name_or_global, str):
        name = name_or_global
    elif name_or_global is not None:
        raise ValueError(
            f"only string or GlobalOp can be provided; got {name_or_global}"
        )

    if global_ is None:
        assert name is not None, "name must be provided"

        if result is None:

            def callback(symbol_table_op, _uses_visible):
                nonlocal global_
                sym_table = SymbolTable(symbol_table_op)
                if name in sym_table:
                    global_ = sym_table[name]

            current_owner = InsertionPoint.current.block.owner
            while not isinstance(current_owner.opview, builtin.ModuleOp):
                current_owner = current_owner.parent
            SymbolTable.walk_symbol_tables(current_owner, True, callback)
            if global_ is None:
                raise RuntimeError(f"couldn't find symbol for {name}")

    if not isinstance(global_, GlobalOp):
        raise RuntimeError(f"expected memref.global, got {global_}")
    result = global_.type_.value
    name = global_.sym_name.value
    return GetGlobalOp(result=result, name=name, loc=loc, ip=ip).result


def reinterpret_cast(
    source: Value,
    offsets: MixedValues = None,
    sizes: MixedValues = None,
    strides: MixedValues = None,
    *,
    loc=None,
    ip=None,
) -> Value:

    if offsets is None:
        offsets = []
    if sizes is None:
        sizes = []

    offsets_, _packed_offsets, static_offsets = _dispatch_mixed_values(offsets)
    sizes_, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
    strides_, _packed_strides, static_strides = _dispatch_mixed_values(strides)

    if offsets_ or sizes_ or strides_:
        raise NotImplementedError("only static offsets and sizes and strides supported")

    default_strides = None
    if not static_strides and all(_is_static_int_like(s) for s in static_sizes):
        default_strides = list(accumulate(list(static_sizes)[1:][::-1], operator.mul))[
            ::-1
        ] + [1]
        static_strides = default_strides

    target_offset = 0
    for offset, target_stride in zip(static_offsets, static_strides):
        target_offset += offset * target_stride

    if static_strides == default_strides and target_offset == 0:
        layout = None
    else:
        layout = StridedLayoutAttr.get(target_offset, static_strides)

    result = MemRefType.get(
        static_sizes, source.type.element_type, layout, source.type.memory_space
    )
    return ReinterpretCastOp(
        result=result,
        source=source,
        offsets=offsets_,
        sizes=sizes_,
        strides=strides_,
        static_offsets=static_offsets,
        static_sizes=static_sizes,
        static_strides=static_strides,
        loc=loc,
        ip=ip,
    ).result
