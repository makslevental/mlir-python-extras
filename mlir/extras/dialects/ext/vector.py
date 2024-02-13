from ._shaped_value import ShapedValue
from .arith import ArithValue
from ...._mlir_libs._mlir import register_value_caster

# noinspection PyUnresolvedReferences
from ....dialects.vector import *
from ....ir import VectorType


@register_value_caster(VectorType.static_typeid)
class Vector(ShapedValue, ArithValue):
    pass
