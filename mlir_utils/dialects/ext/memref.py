import inspect
from functools import cached_property
from typing import Tuple, Sequence

from mlir.ir import (
    Type,
    Value,
    MemRefType,
    ShapedType,
)

from mlir_utils.dialects.ext.arith import ArithValue, Scalar, constant
from mlir_utils.dialects.memref import LoadOp, StoreOp
from mlir_utils.util import (
    register_value_caster,
    _update_caller_vars,
    get_user_code_loc,
    maybe_cast,
    get_result_or_results,
)

S = ShapedType.get_dynamic_size()


def load(memref: Value, indices: Sequence[Value | int], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return maybe_cast(
        get_result_or_results(LoadOp.__base__(memref, indices, loc=loc, ip=ip))
    )


def store(
    value: Value, memref: Value, indices: Sequence[Value | int], *, loc=None, ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return maybe_cast(
        get_result_or_results(StoreOp(value, memref, indices, loc=loc, ip=ip))
    )


@register_value_caster(MemRefType.static_typeid)
class MemRef(ArithValue):
    @staticmethod
    def isinstance(other: Value):
        return isinstance(other, Value) and MemRefType.isinstance(other.type)

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

    def __getitem__(self, idx: tuple) -> "MemRef":
        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if idx is None:
            raise ValueError(f"None indexing not supported yet")

        idx = list((idx,) if isinstance(idx, int) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            return load(self, idx)
        else:
            raise ValueError(f"unsupported {idx=} for memref getitem")

    def __setitem__(self, idx, val):
        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, int) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            assert isinstance(val, Scalar), "coordinate insert requires scalar element"
            res = store(val, self, idx)
        else:
            raise ValueError(f"unsupported {idx=} for memref setitem")

        if len(res.results):
            assert len(res.results) == 1
            previous_frame = inspect.currentframe().f_back
            _update_caller_vars(previous_frame, [self], [res])
