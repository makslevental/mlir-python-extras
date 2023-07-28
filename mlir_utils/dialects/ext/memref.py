from typing import Sequence

from mlir.dialects.memref import LoadOp, StoreOp
from mlir.ir import Value

from mlir_utils.dialects.ext.arith import constant
from mlir_utils.util import get_user_code_loc, maybe_cast, get_result_or_results


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
