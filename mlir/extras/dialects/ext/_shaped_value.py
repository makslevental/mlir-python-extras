from functools import cached_property, reduce
from typing import Tuple

import numpy as np

from ....ir import DenseElementsAttr, ShapedType, Type

S = ShapedType.get_dynamic_size()


# mixin that requires `is_constant`
def ShapedValue(cls):
    @cached_property
    def literal_value(self) -> np.ndarray:
        if not self.is_constant:
            raise ValueError("Can't build literal from non-constant value")
        return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

    @cached_property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    def has_static_shape(self) -> bool:
        return self._shaped_type.has_static_shape

    def has_rank(self) -> bool:
        return self._shaped_type.has_rank

    @cached_property
    def rank(self) -> int:
        return self._shaped_type.rank

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shaped_type.shape)

    @cached_property
    def n_elements(self) -> int:
        assert self.has_static_shape()
        return reduce(lambda acc, v: acc * v, self._shaped_type.shape, 1)

    @cached_property
    def dtype(self) -> Type:
        return self._shaped_type.element_type

    setattr(cls, "literal_value", literal_value)
    cls.literal_value.__set_name__(None, "literal_value")
    setattr(cls, "_shaped_type", _shaped_type)
    cls._shaped_type.__set_name__(None, "_shaped_type")

    setattr(cls, "has_static_shape", has_static_shape)
    setattr(cls, "has_rank", has_rank)

    setattr(cls, "rank", rank)
    cls.rank.__set_name__(None, "rank")
    setattr(cls, "shape", shape)
    cls.shape.__set_name__(None, "shape")
    setattr(cls, "n_elements", n_elements)
    cls.n_elements.__set_name__(None, "n_elements")
    setattr(cls, "dtype", dtype)
    cls.dtype.__set_name__(None, "dtype")

    return cls
