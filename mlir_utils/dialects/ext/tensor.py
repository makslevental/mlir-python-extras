import inspect
from dataclasses import dataclass
from functools import cached_property
from typing import Union, Tuple, Sequence, Optional, Any

import numpy as np
from mlir.dialects._structured_transform_ops_ext import _get_int_int_array_attr
from mlir.dialects.linalg.opdsl.lang.emitter import _is_index_type
from mlir.ir import (
    Type,
    Value,
    RankedTensorType,
    DenseElementsAttr,
    ShapedType,
)

import mlir_utils.types as T
from mlir_utils.dialects import tensor
from mlir_utils.dialects.ext.arith import ArithValue, Scalar, constant
from mlir_utils.util import (
    get_result_or_results,
    maybe_cast,
    register_value_caster,
    _update_caller_vars,
    get_user_code_loc,
)

S = ShapedType.get_dynamic_size()


def empty(sizes: Sequence[Union[int, Value]], element_type: Type, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(tensor.EmptyOp(sizes, element_type, loc=loc, ip=ip))
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


@register_value_caster(RankedTensorType.static_typeid)
class Tensor(ArithValue):
    @staticmethod
    def isinstance(other: Value):
        return isinstance(other, Value) and RankedTensorType.isinstance(other.type)

    @cached_property
    def literal_value(self) -> np.ndarray:
        if not self.is_constant:
            raise ValueError("Can't build literal from non-constant Tensor")
        return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

    @cached_property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    def has_static_shape(self) -> bool:
        return self._shaped_type.has_static_shape

    def has_rank(self) -> bool:
        return self._shaped_type.has_rank

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shaped_type.shape)

    @cached_property
    def dtype(self) -> Type:
        return self._shaped_type.element_type

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
            return _extract_slice(self, tuple(idx), loc=loc)

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
            res = _insert_slice(self, source, tuple(idx), loc=loc)

        previous_frame = inspect.currentframe().f_back
        _update_caller_vars(previous_frame, [self], [res])

    def coerce(
        self,
        other,
        *,
        loc=None,
        ip=None,
    ) -> tuple["Tensor", "Tensor"]:
        if loc is None:
            loc = get_user_code_loc()
        if isinstance(other, np.ndarray):
            other = Tensor(other)
            return self, other
        elif _is_scalar(other):
            if not self.has_static_shape():
                raise ValueError(
                    f"can't coerce {other=} because {self=} doesn't have static shape"
                )
            if isinstance(other, (int, float)):
                other = Tensor(
                    np.full(self.shape, other), dtype=self.dtype, loc=loc, ip=ip
                )
                return self, other
            elif isinstance(other, Scalar):
                other = tensor.splat(
                    RankedTensorType.get(self.shape, other.dtype), other, loc=loc, ip=ip
                )
                return self, other

        raise ValueError(f"can't coerce unknown {other=}")


@dataclass(frozen=True)
class _Indexer:
    indices: tuple[Union[int, Scalar, slice, "Ellipsis", None]]
    newaxis_dims: tuple[int, ...]
    in_shape: tuple[Value | int]

    def is_constant(self):
        return all(_is_constant_index(i) for i in self.indices)

    def is_full(self):
        return all(
            isinstance(idx, slice)
            and len(range(*idx.indices(self.in_shape[i]))) == self.in_shape[i]
            for i, idx in enumerate(self.indices)
        )

    # waiting on hashable slices in 3.12 https://stackoverflow.com/a/76562346
    # @lru_cache(maxsize=1)
    def static_offsets(self):
        offsets = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                offsets.append(int(i))
            elif isinstance(i, slice):
                offsets.append(int(i.start))
            else:
                raise ValueError(f"idx {i} not supported with static offsets")
        return tuple(offsets)

    # @lru_cache(maxsize=1)
    def static_sizes(self):
        sizes = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                sizes.append(1)
            elif isinstance(i, slice):
                start, stop, step = map(int, (i.start, i.stop, i.step))
                if all(isinstance(j, int) for j in (start, stop, step)):
                    s = ((stop - start) // step) + 1
                    if (stop - start) % step == 0:
                        s -= 1
                    sizes.append(s)
                else:
                    raise ValueError(f"idx {i} not supported with static sizes")

            else:
                raise ValueError(f"idx {i} not supported with static sizes")
        return tuple(sizes)

    # @lru_cache(maxsize=1)
    def static_strides(self):
        strides = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                strides.append(1)
            elif isinstance(i, slice):
                strides.append(int(i.step))
            else:
                raise ValueError(f"idx {i} not supported with static strides")
        return tuple(strides)


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

    reassoc_list = _get_int_int_array_attr(reassoc_list)
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
            loc=loc,
            ip=ip,
        )
    )


def _has_index_type(e: Any) -> bool:
    """Checks whether e has MLIR index type or a Python value that can be used
    to construct an index type.

    Args:
      e: Anything
    """
    return (
        isinstance(e, int)
        or isinstance(e, np.ndarray)
        and e.dtype in {np.intp}
        or isinstance(e, (Tensor, Scalar))
        and _is_index_type(e.dtype)
    )


def _is_scalar(e: Any) -> bool:
    """Checks whether e is a Scalar or can be used to construct a Scalar.

    Args:
      e: Anything
    """
    return isinstance(e, Scalar) or isinstance(e, (int, float, bool))


def _is_constant_scalar(e: Any) -> bool:
    return (
        isinstance(e, Scalar)
        and e.is_constant()
        or (isinstance(e, (int, float, bool)) and e != ShapedType.get_dynamic_size())
        or e is None
    )


def _is_constant_index(e: Any) -> bool:
    return (
        isinstance(e, Scalar)
        and e.is_constant()
        or isinstance(e, (int, float, bool))
        or isinstance(e, slice)
        and _is_constant_scalar(e.start)
        and _is_constant_scalar(e.stop)
        and _is_constant_scalar(e.step)
    )


def _is_index_tensor(x):
    """Returns True if x is a Tensor with index dtype, False otherwise."""
    return isinstance(x, Value) and Tensor.isinstance(x) and _is_index_type(x.dtype)


def _is_int_arraylike(x):
    """Returns True if x is array-like with integer dtype, False otherwise.

    Positive (i.e., return True) examples are e.g., [[0], [1]], [[0, 1]],
    [[[0, 1]], [[0, 1]]].
    """
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        or isinstance(x, (list, tuple))
        and all(_is_int_arraylike(e) for e in x)
    )


def _canonicalize_tuple_index(idx: Tuple[Any], rank: int):
    """Helper to
    1. remove Ellipsis and replace with implicit trailing slice(None)s.
    2. cast Python lists of lists or numpy arrays to index Tensors

    Args:
      rank: Rank of tensor.
      idx: Index object (Scalar, Tensor, slice, Ellipse, or None).

    Returns:
      Tuple of index objects with no ellipses.
    """

    len_without_none = 0
    for e in idx:
        if e is None or e is Ellipsis:
            continue
        else:
            len_without_none += 1

    if len_without_none > rank:
        raise IndexError(
            f"Too many indices for shaped type with rank: {len_without_none} "
            f"non-None/Ellipsis indices for dim {rank}."
        )
    ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
    ellipsis_index = next(ellipses, None)
    if ellipsis_index is not None:
        if next(ellipses, None) is not None:
            raise IndexError(
                f"Multiple ellipses (...) not supported: {list(map(type, idx))}."
            )
        colons = (slice(None),) * (rank - len_without_none)
        idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1 :]
    elif len_without_none < rank:
        colons = (slice(None),) * (rank - len_without_none)
        idx = tuple(idx) + colons
    return idx


def _indices_to_indexer(
    idx: tuple[Union[Scalar, slice, "Ellipsis", None]], in_shape: tuple[int]
) -> _Indexer:
    """Processes sequence of index objects and constructs _Indexer with
    corresponding indexing tensor and collapse dims (i.e., scatter/gather dims).

    Args:
      idx: Sequence (list or tuple) of slices, ellipses, Scalar, or Tensors.
      in_shape: The shape of the tensor being indexed into.

    Returns:
      _Indexer object.

    """
    idx = _canonicalize_tuple_index(idx, len(in_shape))

    in_axis = 0  # Current axis in input.
    out_axis = 0  # Current axis in output.
    indices: list[Union[Scalar, slice, Ellipsis, None]] = [slice(None)] * len(in_shape)
    newaxis_dims: list[int] = []

    if any(_is_index_tensor(i) or _is_int_arraylike(i) for i in idx):
        raise ValueError("indexing by tensor is not currently supported")

    # nb: idx_e <-> idx_element
    for idx_i, idx_e in enumerate(idx):
        if _is_scalar(idx_e) and _has_index_type(idx_e):
            # Handle basic Scalar indexes.
            indices[in_axis] = idx_e
            in_axis += 1
        # Handle newaxis (None)
        elif idx_e is None:
            newaxis_dims.append(out_axis)
            out_axis += 1
        elif isinstance(idx_e, slice):
            # Normalize the slice to use None when possible
            start, stop, step = idx_e.start, idx_e.stop, idx_e.step
            if step is None or isinstance(step, int) and step == 1:
                step = None
            if step is None:
                if start is None or isinstance(start, int) and start == 0:
                    start = None
                if stop is None or isinstance(stop, int) and stop >= in_shape[in_axis]:
                    stop = None
            # Handle slice(None) and slice(None, None, -1)
            if (
                start is None
                and stop is None
                and (step is None or isinstance(step, int) and step == -1)
            ):
                if step == -1:
                    raise IndexError(
                        f"Negative step indexing mode not yet supported:\n{idx}"
                    )
                indices[in_axis] = slice(None)
                out_axis += 1
                in_axis += 1

            # Handle slice index (only static shape supported)
            else:
                if (
                    not isinstance(in_shape[in_axis], int)
                    or in_shape[in_axis] == ShapedType.get_dynamic_size()
                ):
                    msg = (
                        "Cannot use NumPy slice indexing on an array dimension whose "
                        f"size is not statically known ({in_shape[in_axis]}). "
                    )
                    raise IndexError(msg)

                if step is None:
                    step = 1
                indices[in_axis] = slice(start, stop, step)

                out_axis += 1
                in_axis += 1
        else:
            raise IndexError(f"Indexing mode not yet supported:\n{idx}")

    for i, idx in enumerate(indices):
        if _is_constant_index(idx) and _is_constant_scalar(in_shape[i]):
            if isinstance(idx, slice):
                indices[i] = slice(*idx.indices(int(in_shape[i])))
            elif isinstance(idx, Scalar):
                indices[i] = int(idx)

    return _Indexer(
        newaxis_dims=tuple(newaxis_dims), indices=tuple(indices), in_shape=in_shape
    )


def _extract_slice(
    ten: Tensor,
    idx,
    *,
    loc=None,
    ip=None,
) -> Tensor:
    if loc is None:
        loc = get_user_code_loc()

    indexer = _indices_to_indexer(idx, ten.shape)
    out = ten

    if indexer.is_full():
        out = out
    elif indexer.is_constant():
        out = extract_slice(
            out,
            static_offsets=indexer.static_offsets(),
            static_sizes=indexer.static_sizes(),
            static_strides=indexer.static_strides(),
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError(f"non-constant indices not supported {indexer}")

    # This adds newaxis/None dimensions.
    return expand_dims(out, indexer.newaxis_dims, loc=loc, ip=ip)


def _insert_slice(
    dest: Tensor,
    source: Tensor,
    idx,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if isinstance(source, Scalar):
        source = expand_dims(source, (0,), loc=loc, ip=ip)

    indexer = _indices_to_indexer(idx, dest.shape)

    if indexer.is_constant():
        assert (
            indexer.static_sizes() == source.shape
        ), f"Expected matching shape for dest slice {indexer.static_sizes()=} and source {source.shape=}"
        out = insert_slice(
            source,
            dest,
            static_offsets=indexer.static_offsets(),
            static_sizes=indexer.static_sizes(),
            static_strides=indexer.static_strides(),
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError(f"non-constant indices not supported {indexer}")

    return out
