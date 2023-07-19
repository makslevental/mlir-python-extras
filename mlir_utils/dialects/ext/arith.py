from copy import deepcopy
from functools import lru_cache, partialmethod, cached_property
from typing import Union, Optional

import numpy as np
from mlir.dialects import arith as arith_dialect
from mlir.dialects._arith_ops_ext import _is_integer_like_type
from mlir.dialects._ods_common import get_op_result_or_value
from mlir.dialects.linalg.opdsl.lang.emitter import (
    _is_floating_point_type,
    _is_integer_type,
    _is_complex_type,
    _is_index_type,
)
from mlir.ir import (
    OpView,
    Operation,
    Type,
    Value,
    IndexType,
    RankedTensorType,
    DenseElementsAttr,
)

from mlir_utils.dialects.util import get_result_or_results, maybe_cast

try:
    from mlir_utils.dialects.arith import *
except ModuleNotFoundError:
    pass

from mlir_utils.types import infer_mlir_type, MLIR_TYPE_TO_NP_DTYPE


def constant(
    value: Union[int, float, bool, np.ndarray],
    type: Optional[Type] = None,
    index: Optional[bool] = None,
) -> arith_dialect.ConstantOp:
    """Instantiate arith.constant with value `value`.

    Args:
      value: Python value that determines the value attribute of the
        arith.constant op.
      type: Optional MLIR type that type of the value attribute of the
        arith.constant op; if omitted the type of the value attribute
        will be inferred from the value.
      index: Whether the MLIR type should be an index type; if passed the
        type argument will be ignored.

    Returns:
      ir.OpView instance that corresponds to instantiated arith.constant op.
    """
    if index is not None and index:
        type = IndexType.get()
    if type is None:
        type = infer_mlir_type(value)
    elif RankedTensorType.isinstance(type) and isinstance(value, (int, float, bool)):
        ranked_tensor_type = RankedTensorType(type)
        value = np.ones(
            ranked_tensor_type.shape,
            dtype=MLIR_TYPE_TO_NP_DTYPE()[ranked_tensor_type.element_type],
        )
    assert type is not None

    if isinstance(value, np.ndarray):
        value = DenseElementsAttr.get(
            value,
            type=type,
        )

    return maybe_cast(get_result_or_results(arith_dialect.ConstantOp(type, value)))


class ArithValueMeta(type(Value)):
    """Metaclass that orchestrates the Python object protocol
    (i.e., calling __new__ and __init__) for Indexing dialect extension values
    (created using `mlir_value_subclass`).

    The purpose/benefit of handling the `__new__` and `__init__` calls
    explicitly/manually is we can then wrap arbitrary Python objects; e.g.
    all three of the following wrappers are equivalent:

    ```
    s1 = Scalar(arith.ConstantOp(f64, 0.0).result)
    s2 = Scalar(arith.ConstantOp(f64, 0.0))
    s3 = Scalar(0.0)
    ```

    In general the Python object protocol for an object instance is determined
    by `__call__` of the object class's metaclass, thus here we overload
    `__call__` and branch on what we're wrapping there.

    Why not just overload __new__ and be done with it? Because then we can't
    choose what get's passed to __init__: by default (i.e., without overloading
    __call__ here) the same arguments are passed to both __new__ and __init__.

    Note, this class inherits from `type(Value)` (i.e., the metaclass of
    `ir.Value`) rather than `type` or `abc.ABCMeta` or something like this because
    the metaclass of a derived class must be a (non-strict) subclass of the
    metaclasses of all its bases and so all the extension classes
    (`ScalarValue`, `TensorValue`), which are derived classes of `ir.Value` must
    have metaclasses that inherit from the metaclass of `ir.Value`. Without this
    hierarchy Python will throw `TypeError: metaclass conflict`.
    """

    def __call__(cls, *args, **kwargs):
        """Orchestrate the Python object protocol for Indexing dialect extension
        values in order to handle wrapper arbitrary Python objects.

        Args:
          *args: Position arguments to the class constructor. Note, currently,
            only one positional arg is supported (so constructing something like a
            tuple type from element objects isn't supported).
          **kwargs: Keyword arguments to the class constructor. Note, currently,
            we only look for `dtype` (an `ir.Type`).

        Returns:
          A fully constructed and initialized instance of the class.
        """
        if len(args) != 1:
            raise ValueError("Only one non-kw arg supported.")
        arg = args[0]
        arg_copy = None
        if isinstance(arg, (OpView, Operation, Value)):
            # wrap an already created Value (or op the produces a Value)
            if isinstance(arg, (Operation, OpView)):
                assert len(arg.results) == 1
            val = get_op_result_or_value(arg)
        elif isinstance(arg, (int, float, bool, np.ndarray)):
            # wrap a Python value, effectively a scalar or tensor literal
            dtype = kwargs.get("dtype")
            if dtype is not None and not isinstance(dtype, Type):
                raise ValueError(f"{dtype=} is expected to be an ir.Type.")
            # If we're wrapping a numpy array (effectively a tensor literal),
            # then we want to make sure no one else has access to that memory.
            # Otherwise, the array will get funneled down to DenseElementsAttr.get,
            # which by default (through the Python buffer protocol) does not copy;
            # see mlir/lib/Bindings/Python/IRAttributes.cpp#L556
            arg_copy = deepcopy(arg)
            val = constant(arg, dtype).result
        else:
            raise NotImplementedError(f"{cls.__name__} doesn't support wrapping {arg}.")

        # The mlir_value_subclass mechanism works through __new__
        # (see mlir/Bindings/Python/PybindAdaptors.h#L502)
        # So we have to pass the wrapped Value to the __new__ of the subclass
        cls_obj = cls.__new__(cls, val)
        # We also have to pass it to __init__ because that is required by
        # the Python object protocol; first an object is new'ed and then
        # it is init'ed. Note we pass arg_copy here in case a subclass wants to
        # inspect the literal.
        cls.__init__(cls_obj, val, arg_copy)
        return cls_obj


class ArithValue(Value, metaclass=ArithValueMeta):
    """Class for functionality shared by Value subclasses that support
    arithmetic operations.

    Note, since we bind the ArithValueMeta here, it is here that the __new__ and
    __init__ must be defined. To be precise, the callchain, starting from
    ArithValueMeta is:

    ArithValueMeta.__call__ -> mlir_value_subclass.__new__ ->
                          (mlir_value_subclass.__init__ == ArithValue.__init__) ->
                          Value.__init__
    """

    def __init__(
        self,
        val,
        arg: Optional[Union[int, float, bool, np.ndarray]] = None,
    ):
        self.__arg = arg
        super().__init__(val)

    @lru_cache(maxsize=1)
    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

    @lru_cache(maxsize=1)
    def __repr__(self):
        return str(self)

    def __binary_op(self, other: "ArithValue", op: str) -> "ArithValue":
        """Generic for handling infix binary operator dispatch.

        Args:
          self: E.g. Scalar or Tensor below.
          other: Scalar or Tensor with type matching self.
          op: Binary operator, currently only add, sub, mul
            supported.

        Returns:
          Result of binary operation. This will be a handle to an arith(add|sub|mul) op.
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other, dtype=self.type)

        assert op in {"add", "sub", "mul"}
        if self.type != other.type:
            raise ValueError(f"{self=} {other=} must have the same type.")

        op = op.capitalize()
        lhs, rhs = self, other
        if _is_floating_point_type(self.dtype):
            op = getattr(arith_dialect, f"{op}FOp")
        elif _is_integer_like_type(self.dtype):
            op = getattr(arith_dialect, f"{op}IOp")
        else:
            raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")
        return self.__class__(op(lhs, rhs), dtype=self.dtype)

    # partialmethod differs from partial in that it also binds the object instance
    # to the first arg (i.e., self)
    __add__ = partialmethod(__binary_op, op="add")
    __sub__ = partialmethod(__binary_op, op="sub")
    __mul__ = partialmethod(__binary_op, op="mul")
    __radd__ = partialmethod(__binary_op, op="add")
    __rsub__ = partialmethod(__binary_op, op="sub")
    __rmul__ = partialmethod(__binary_op, op="mul")


class Scalar(ArithValue):
    """Value subclass ScalarValue that adds convenience methods
    for getting dtype and (possibly) the stored literal value.

    Note, order matters in the superclasses above; ArithValue is first so that
    e.g. __init__, and __str__ from ArithValue are used instead of
    from ScalarValue.
    """

    @cached_property
    def dtype(self) -> Type:
        return self.type

    @staticmethod
    def isinstance(other: Value):
        return (
            isinstance(other, Value)
            and _is_integer_type(other.type)
            or _is_floating_point_type(other.type)
            or _is_index_type(other.type)
            or _is_complex_type(other.type)
        )
