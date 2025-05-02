import inspect
from typing import List, Optional, Tuple, Union, Sequence

# noinspection PyUnresolvedReferences
import numpy as np

from ._shaped_value import (
    ShapedValue,
    _indices_to_indexer,
    _is_scalar,
    _is_int_arraylike,
)
from .arith import ArithValue, Scalar, constant
from ... import types as T
from ...util import (
    _unpack_sizes_element_type,
    _update_caller_vars,
    get_user_code_loc,
    mlir_type_to_np_dtype,
)
from ...._mlir_libs._mlir import register_value_caster
from ....dialects import tensor
from ....dialects._ods_common import _dispatch_mixed_values, get_op_result_or_op_results
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from ....dialects.tensor import *
from ....dialects.transform.structured import _get_int_array_array_attr
from ....ir import RankedTensorType, ShapedType, Type, Value

S = ShapedType.get_dynamic_size()


def empty(*sizes: Union[int, Value], element_type: Type = None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if element_type is None:
        sizes, element_type = _unpack_sizes_element_type(sizes)
    return get_op_result_or_op_results(
        tensor.EmptyOp(sizes, element_type, loc=loc, ip=ip)
    )


def extract_slice(
    source: "Tensor",
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if offsets is None:
        offsets = []
    if strides is None:
        strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    assert offsets or static_offsets and bool(offsets) != bool(static_offsets)
    assert strides or static_strides and bool(strides) != bool(static_strides)
    sizes = []
    result = T.tensor(*static_sizes, source.dtype)
    return tensor.extract_slice(
        result,
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


def insert_slice(
    source: Value,
    dest: Value,
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if offsets is None:
        offsets = []
    if strides is None:
        strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    assert offsets or static_offsets and bool(offsets) != bool(static_offsets)
    assert strides or static_strides and bool(strides) != bool(static_strides)
    sizes = []
    return tensor.insert_slice(
        source,
        dest,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )


def _is_index_tensor(x):
    """Returns True if x is a Tensor with index dtype, False otherwise."""
    return isinstance(x, Value) and isinstance(x, Tensor) and _is_index_type(x.dtype)


# TODO(max): unify vector/memref/tensor
@register_value_caster(RankedTensorType.static_typeid)
@ShapedValue
class Tensor(ArithValue):
    def __getitem__(self, idx: tuple) -> "Tensor":
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked tensor slicing/indexing supported")

        if idx is None:
            return expand_dims(self, (0,), loc=loc)
        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) or i is None for i in idx):
            nones = [i for i, n in enumerate(idx) if n is None]
            return expand_dims(self, nones, loc=loc)

        idx = list((idx,) if isinstance(idx, int) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            return tensor.extract(self, idx, loc=loc)
        else:
            if any(_is_index_tensor(i) or _is_int_arraylike(i) for i in idx):
                raise ValueError("indexing by tensor is not currently supported")

            indexer = _indices_to_indexer(tuple(idx), self.shape)
            out = self

            if indexer.is_full():
                out = out
            elif indexer.is_constant():
                out = extract_slice(
                    out,
                    static_offsets=indexer.static_offsets(),
                    static_sizes=indexer.static_sizes(),
                    static_strides=indexer.static_strides(),
                    loc=loc,
                    ip=None,
                )
            else:
                raise ValueError(f"non-constant indices not supported {indexer}")

            # This adds newaxis/None dimensions.
            return expand_dims(out, indexer.newaxis_dims, loc=loc, ip=None)

    def __setitem__(self, idx, source):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked tensor slicing/indexing supported")
        if not source.has_rank():
            raise ValueError("only ranked tensor slicing/indexing supported")

        if (
            idx == Ellipsis
            or idx == slice(None)
            or (isinstance(idx, tuple) and all(i == slice(None) for i in idx))
        ):
            assert (
                self.shape == source.shape
            ), f"Expected matching shape for dest slice {self.shape=} and source {source.shape=}"
            return self

        idx = list((idx,) if isinstance(idx, int) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) and d.fold() for d in idx) and len(idx) == len(
            self.shape
        ):
            assert isinstance(
                source, Scalar
            ), "coordinate insert requires scalar element"
            res = tensor.insert(source, self, idx, loc=loc)
        else:
            if any(_is_index_tensor(i) or _is_int_arraylike(i) for i in idx):
                raise ValueError("indexing by tensor is not currently supported")
            indexer = _indices_to_indexer(tuple(idx), self.shape)
            if indexer.is_constant():
                assert (
                    indexer.static_sizes() == source.shape
                ), f"Expected matching shape for dest slice {indexer.static_sizes()=} and source {source.shape=}"
                res = insert_slice(
                    source,
                    self,
                    static_offsets=indexer.static_offsets(),
                    static_sizes=indexer.static_sizes(),
                    static_strides=indexer.static_strides(),
                    loc=loc,
                    ip=None,
                )
            else:
                raise ValueError(f"non-constant indices not supported {indexer}")

        previous_frame = inspect.currentframe().f_back
        _update_caller_vars(previous_frame, [self], [res])

    def coerce(
        self,
        other,
        *,
        loc=None,
        ip=None,
    ) -> Tuple["Tensor", "Tensor"]:
        if loc is None:
            loc = get_user_code_loc()
        if isinstance(other, np.ndarray):
            other = Tensor(other)
            return other
        elif _is_scalar(other):
            if not self.has_static_shape():
                raise ValueError(
                    f"can't coerce {other=} because {self=} doesn't have static shape"
                )
            if isinstance(other, (int, float)):
                np_dtype = mlir_type_to_np_dtype(self.dtype)
                other = Tensor(
                    np.full(self.shape, other, dtype=np_dtype),
                    dtype=self.dtype,
                    loc=loc,
                    ip=ip,
                )
                return other
            elif isinstance(other, Scalar):
                other = tensor.splat(
                    RankedTensorType.get(self.shape, other.dtype),
                    other,
                    [],
                    loc=loc,
                    ip=ip,
                )
                return other

        raise ValueError(f"can't coerce unknown {other=}")


def compute_result_shape_reassoc_list(inp_shape, newaxis_dims):
    newaxis_dims = sorted(newaxis_dims)
    if len(set(newaxis_dims)) != len(newaxis_dims):
        raise ValueError(f"repeated axis in expand_dims: {newaxis_dims}")

    ndim_out = len(inp_shape) + len(newaxis_dims)
    if not all(0 <= d < ndim_out for d in newaxis_dims):
        raise ValueError("no negative dims allowed")
    result_shape = list(inp_shape)
    for i in reversed(newaxis_dims):
        result_shape.insert(i, 1)
    reassoc_list = [[i] for i in range(len(inp_shape))]
    for i, d in enumerate(newaxis_dims):
        reassoc_list.append([len(inp_shape) + i])
        if d == 0:
            d = 1
        reassoc_list[max(d - 1, 0)].extend(reassoc_list.pop(d))

    reassoc_list = _get_int_array_array_attr(reassoc_list)
    return result_shape, reassoc_list


def expand_dims(
    inp,
    newaxis_dims,
    *,
    loc=None,
    ip=None,
) -> Tensor:
    """Expand the shape of a tensor.

    Insert a new axis that will appear at the `axis` position in the expanded
    tensor shape.

    Args:
      inp: Input tensor-like.
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
    if inp.fold():
        return Tensor(inp.literal_value.reshape(result_shape))

    return Tensor(
        tensor.expand_shape(
            RankedTensorType.get(result_shape, inp.dtype),
            inp,
            reassoc_list,
            output_shape=[],
            static_output_shape=result_shape,
            loc=loc,
            ip=ip,
        )
    )


def parallel_insert_slice(
    source,
    dest,
    offsets=None,
    sizes=None,
    strides=None,
    static_offsets=None,
    static_sizes=None,
    static_strides=None,
):
    if static_offsets is None:
        assert offsets is not None
        static_offsets = [S, S]
    if static_sizes is None:
        assert sizes is not None
        static_sizes = [S, S]
    if static_strides is None:
        assert strides is not None
        static_strides = [S, S]
    if offsets is None:
        assert static_offsets
        offsets = []
    if sizes is None:
        assert static_sizes
        sizes = []
    if strides is None:
        assert static_strides
        strides = []

    return tensor.parallel_insert_slice(
        source,
        dest,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
    )


def pad_(
    source: Value,
    low: List[int],
    high: List[int],
    *,
    nofold=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    assert all(
        isinstance(l, int) for l in low
    ), f"only literal pad values supported: {low=}"
    assert all(
        isinstance(l, int) for l in high
    ), f"only literal pad values supported: {high=}"

    dim_sizes = []
    source_type = source.type
    for dim in range(source_type.rank):
        dim_sizes.append(source_type.get_dim_size(dim) + low[dim] + high[dim])
    result_type = RankedTensorType.get(dim_sizes, source_type.element_type)

    return tensor.PadOp(
        result_type,
        source,
        [],
        [],
        low,
        high,
        nofold=nofold,
        loc=loc,
        ip=ip,
    )


pad = region_op(pad_, terminator=lambda args: tensor.YieldOp(args[0]))

generate = region_op(
    lambda result, dynamic_extents: tensor.GenerateOp(result, dynamic_extents)
)
