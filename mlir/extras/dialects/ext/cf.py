from typing import List, Union

from ...util import Successor, get_user_code_loc
from ....dialects._cf_ops_gen import _Dialect
from ....dialects._ods_common import (
    _cext,
)
from ....dialects.cf import *
from ....ir import Block, InsertionPoint, Value


@_cext.register_operation(_Dialect, replace=True)
class CondBranchOp(CondBranchOp):
    @property
    def true(self):
        return Successor(self, self.trueDestOperands, self.successors[0], 0)

    @property
    def false(self):
        return Successor(self, self.falseDestOperands, self.successors[1], 1)


def br(
    dest: Union[Value, Block] = None, *dest_operands: List[Value], loc=None, ip=None
):
    if isinstance(dest, Value):
        dest_operands = [dest] + list(dest_operands)
        dest = None
    if dest is None:
        dest = InsertionPoint.current.block
    if loc is None:
        loc = get_user_code_loc()
    return BranchOp(dest_operands, dest, loc=loc, ip=ip)


def cond_br(
    condition: Value,
    true_dest: Union[Value, Block] = None,
    false_dest: Union[Value, Block] = None,
    true_dest_operands: List[Value] = None,
    false_dest_operands: List[Value] = None,
    *,
    loc=None,
    ip=None,
):
    if true_dest is None:
        true_dest = InsertionPoint.current.block
    if false_dest is None:
        false_dest = InsertionPoint.current.block
    if true_dest_operands is None:
        true_dest_operands = []
    if false_dest_operands is None:
        false_dest_operands = []
    if loc is None:
        loc = get_user_code_loc()
    return CondBranchOp(
        condition,
        true_dest_operands,
        false_dest_operands,
        true_dest,
        false_dest,
        loc=loc,
        ip=ip,
    )
