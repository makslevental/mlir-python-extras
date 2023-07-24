from textwrap import dedent

import pytest

from mlir_utils.ast.canonicalize import FuncIdentTypeTable, get_module_cst
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.scf import (
    ReplaceSCFYield,
    ReplaceSCFCond,
    InsertEndIfs,
    InsertSCFYield,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import f64_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def transform_func(f, *transformer_ctors):
    module_cst = get_module_cst(f)
    func_sym_table = FuncIdentTypeTable(f)
    for transformer_ctor in transformer_ctors:
        func_node = module_cst.body[0]
        replace = transformer_ctor(context=None, func_sym_table=func_sym_table)
        new_func = func_node._visit_and_replace_children(replace)
        module_cst = module_cst.deep_replace(func_node, new_func)

    return module_cst.code


def test_if_replace_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    code = transform_func(iffoo, ReplaceSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield_()
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        return

    code = transform_func(iffoo, InsertSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0); yield_()
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield three
        return

    code = transform_func(iffoo, ReplaceSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield_(three)
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield three, three
        return

    code = transform_func(iffoo, ReplaceSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield_(three, three)
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield three, three, three
        return

    code = transform_func(iffoo, ReplaceSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield_(three, three, three)
        return
    """
    )
    filecheck(correct, code)


def test_if_replace_cond(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    code = transform_func(iffoo, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two):
            three = constant(3.0)
            yield
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            yield three
        return

    code = transform_func(iffoo, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := stack_if(one < two, (f64_t,), True):
            three = constant(3.0)
            yield three
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            yield three, three
        return

    code = transform_func(iffoo, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := stack_if(one < two, (f64_t, f64_t), True):
            three = constant(3.0)
            yield three, three
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            yield three, three, three
        return

    code = transform_func(iffoo, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t, f64_t)
        if res := stack_if(one < two, (f64_t, f64_t, f64_t), True):
            three = constant(3.0)
            yield three, three, three
        return
    """
    )
    filecheck(correct, code)


def test_insert_end_ifs(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    code = transform_func(iffoo, InsertEndIfs)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield; stack_endif_branch(); stack_endif()
        return
    """
    )
    filecheck(correct, code)

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            yield three
        else:
            four = constant(4.0)
            yield four
        return

    code = transform_func(iffoo, InsertEndIfs)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            yield three; stack_endif_branch()
        else:
            stack_else(); four = constant(4.0)
            yield four; stack_endif_branch(); stack_endif()
        return
    """
    )
    filecheck(correct, code)


def test_if_nested_no_else_no_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
        return

    iffoo()

    code = transform_func(iffoo, InsertSCFYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0); yield_()
            yield_()
        return
    """
    )
    filecheck(correct, code)


def test_if_nested_with_else_no_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)
        return

    iffoo()

    code = transform_func(iffoo, InsertSCFYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0); yield_()
            else:
                five = constant(5.0); yield_()
            yield_()
        return
    """
    )
    filecheck(correct, code)


def test_insert_end_ifs_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        else:
            four = constant(4.0)
            yield
        return

    code = transform_func(iffoo, InsertEndIfs)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield; stack_endif_branch()
        else:
            stack_else(); four = constant(4.0)
            yield; stack_endif_branch(); stack_endif()
        return
    """
    )
    filecheck(correct, code)


def test_if_else_with_nested_no_yields_yield_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            yield three
        else:
            five = constant(5.0)
            yield five
        return

    code = transform_func(
        iffoo, InsertSCFYield, ReplaceSCFYield, ReplaceSCFCond, InsertEndIfs
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: f64_t
        if res := stack_if(one < two, (f64_t,), True):
            three = constant(3.0)
            if stack_if(two < three):
                four = constant(4.0); yield_(); stack_endif_branch(); stack_endif()
            yield_(three); stack_endif_branch()
        else:
            stack_else(); five = constant(5.0)
            yield_(five); stack_endif_branch(); stack_endif()
        return
    """
    )
    filecheck(correct, code)


def test_if_else_with_nested_no_yields_yield_multiple_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            yield three, three
        else:
            five = constant(5.0)
            yield five, five
        return

    code = transform_func(
        iffoo, InsertSCFYield, ReplaceSCFYield, ReplaceSCFCond, InsertEndIfs
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        res: (f64_t, f64_t)
        if res := stack_if(one < two, (f64_t, f64_t), True):
            three = constant(3.0)
            if stack_if(two < three):
                four = constant(4.0); yield_(); stack_endif_branch(); stack_endif()
            yield_(three, three); stack_endif_branch()
        else:
            stack_else(); five = constant(5.0)
            yield_(five, five); stack_endif_branch(); stack_endif()
        return
    """
    )
    filecheck(correct, code)
