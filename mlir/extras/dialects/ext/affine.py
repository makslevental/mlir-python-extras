from ....dialects._ods_common import get_op_result_or_op_results, get_op_result_or_value
from ....dialects.affine import *
from ....ir import Value, IndexType
from ...util import get_user_code_loc


def delinearize_index(
    linear_index, static_basis=None, *, dynamic_basis=None, loc=None, ip=None
) -> Value:
    if loc is None:
        loc = get_user_code_loc()

    if dynamic_basis is None:
        dynamic_basis = []
    if static_basis and dynamic_basis:
        raise ValueError("Cannot specify both static and dynamic basis")
    multi_index = [IndexType.get() for _ in static_basis or dynamic_basis]
    return get_op_result_or_op_results(
        AffineDelinearizeIndexOp(
            multi_index=multi_index,
            linear_index=linear_index,
            dynamic_basis=dynamic_basis,
            static_basis=static_basis,
            loc=loc,
            ip=ip,
        )
    )
