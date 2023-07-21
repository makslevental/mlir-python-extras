from functools import cached_property
from typing import Union, Tuple, Sequence

import numpy as np
from mlir.dialects.tensor import EmptyOp, GenerateOp
from mlir.ir import Type, Value, RankedTensorType, DenseElementsAttr, ShapedType

from mlir_utils.dialects.ext.arith import ArithValue
from mlir_utils.dialects.util import register_value_caster

try:
    from mlir_utils.dialects.tensor import *
except ModuleNotFoundError:
    pass

S = ShapedType.get_dynamic_size()


def empty(sizes: Sequence[Union[int, Value]], element_type: Type, *, loc=None, ip=None):
    from mlir_utils.dialects.util import maybe_cast
    from mlir_utils.dialects.util import get_result_or_results

    return maybe_cast(
        get_result_or_results(EmptyOp(sizes, element_type, loc=loc, ip=ip))
    )


class Tensor(ArithValue):
    """Value subclass TensorValue that adds convenience methods
    for getting dtype, shape and (possibly) the stored literal value.

    Note, order matters in the superclasses above; ArithValue is first so that
    e.g. __init__, and __str__ from ArithValue are used instead of
    from TensorValue.
    """

    @staticmethod
    def isinstance(other: Value):
        return isinstance(other, Value) and RankedTensorType.isinstance(other.type)

    @cached_property
    def literal_value(self) -> np.ndarray:
        if not self.is_constant():
            raise ValueError("Can't build literal from non-constant Tensor")
        return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

    @cached_property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shaped_type.shape)

    @cached_property
    def dtype(self) -> Type:
        return self._shaped_type.element_type

    @classmethod
    def empty(
        cls,
        shape: Union[list[Union[int, Value]], tuple[Union[int, Value], ...]],
        el_type: Type,
    ) -> "Tensor":
        return cls(EmptyOp(shape, el_type).result)


register_value_caster(RankedTensorType.static_typeid, Tensor)
