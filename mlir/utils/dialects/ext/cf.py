from ...meta import maybe_cast
from ...util import get_user_code_loc, get_result_or_results, Successor
from ....dialects import cf
from ....dialects._ods_common import (
    get_op_results_or_values,
    get_default_loc_context,
    segmented_accessor,
    get_op_result_or_value,
)
from ....ir import Value, InsertionPoint, Block, OpView


class BranchOp(cf.BranchOp.__base__):
    OPERATION_NAME = "cf.br"

    _ODS_REGIONS = (0, True)

    def __init__(self, destOperands, dest=None, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.extend(get_op_results_or_values(destOperands))
        _ods_context = get_default_loc_context(loc)
        _ods_successors = []
        if dest is not None:
            _ods_successors.append(dest)
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )


class CondBranchOp(OpView):
    OPERATION_NAME = "cf.cond_br"

    _ODS_OPERAND_SEGMENTS = [1, -1, -1]

    _ODS_REGIONS = (0, True)

    def __init__(
        self,
        condition,
        trueDestOperands=None,
        falseDestOperands=None,
        trueDest=None,
        falseDest=None,
        *,
        loc=None,
        ip=None
    ):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(condition))
        if trueDestOperands is None:
            trueDestOperands = []
        if falseDestOperands is None:
            falseDestOperands = []
        operands.append(get_op_results_or_values(trueDestOperands))
        operands.append(get_op_results_or_values(falseDestOperands))
        _ods_context = get_default_loc_context(loc)
        _ods_successors = []
        if trueDest is not None:
            _ods_successors.append(trueDest)
        if falseDest is not None:
            _ods_successors.append(falseDest)
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )

    @property
    def condition(self):
        operand_range = segmented_accessor(
            self.operation.operands, self.operation.attributes["operandSegmentSizes"], 0
        )
        return operand_range[0]

    @property
    def trueDestOperands(self):
        operand_range = segmented_accessor(
            self.operation.operands, self.operation.attributes["operandSegmentSizes"], 1
        )
        return operand_range

    @property
    def falseDestOperands(self):
        operand_range = segmented_accessor(
            self.operation.operands, self.operation.attributes["operandSegmentSizes"], 2
        )
        return operand_range

    @property
    def true(self):
        return Successor(self, self.trueDestOperands, self.successors[0], 0)

    @property
    def false(self):
        return Successor(self, self.falseDestOperands, self.successors[1], 1)


def br(dest: Value | Block = None, *dest_operands: list[Value], loc=None, ip=None):
    if isinstance(dest, Value):
        dest_operands = [dest] + list(dest_operands)
        dest = None
    if dest is None:
        dest = InsertionPoint.current.block
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(BranchOp(dest_operands, dest, loc=loc, ip=ip))
    )


def cond_br(
    condition: Value,
    true_dest: Value | Block = None,
    false_dest: Value | Block = None,
    true_dest_operands: list[Value] = None,
    false_dest_operands: list[Value] = None,
    *,
    loc=None,
    ip=None
):
    if true_dest is None:
        true_dest = InsertionPoint.current.block
    if false_dest is None:
        false_dest = InsertionPoint.current.block
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            CondBranchOp(
                condition,
                true_dest_operands,
                false_dest_operands,
                true_dest,
                false_dest,
                loc=loc,
                ip=ip,
            )
        )
    )
