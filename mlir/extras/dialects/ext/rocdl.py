from . import arith
from ...util import get_user_code_loc

from ....dialects._ods_common import (
    _dispatch_mixed_values,
    _cext,
    get_op_results_or_values,
    get_default_loc_context,
    get_op_result_or_op_results,
    get_default_loc_context,
    segmented_accessor,
)

# noinspection PyUnresolvedReferences
from ....dialects.rocdl import *
from ....dialects._rocdl_ops_gen import _Dialect
from .... import ir


@_cext.register_operation(_Dialect, replace=True)
class WMMA_F16_16X16X16_F16(ir.OpView):
    OPERATION_NAME = "rocdl.wmma.f16.16x16x16.f16"

    _ODS_REGIONS = (0, True)

    def __init__(self, res, args, *, loc=None, ip=None):
        if loc is None:
            loc = get_user_code_loc()
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.extend(get_op_results_or_values(args))
        _ods_context = get_default_loc_context(loc)
        results.append(res)
        _ods_successors = None
        super().__init__(
            self.OPERATION_NAME,
            self._ODS_REGIONS,
            self._ODS_OPERAND_SEGMENTS,
            self._ODS_RESULT_SEGMENTS,
            attributes=attributes,
            results=results,
            operands=operands,
            successors=_ods_successors,
            regions=regions,
            loc=loc,
            ip=ip,
        )

    @property
    def args(self):
        _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
        return self.operation.operands[0 : 0 + _ods_variadic_group_length]

    @property
    def res(self):
        return self.operation.results[0]


def wmma_f16_16x16x16_f16(A, B, C, *, OPSEL=False, loc=None, ip=None) -> ir.Value:
    if loc is None:
        loc = get_user_code_loc()

    opsel = arith.constant(OPSEL, ir.IntegerType.get_signless(1))
    args = [A, B, C, opsel]
    v16 = ir.VectorType.get((16,), ir.F16Type.get())
    return WMMA_F16_16X16X16_F16(res=v16, args=args, loc=loc, ip=ip).result
