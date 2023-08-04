from textwrap import dedent

import pytest

from mlir_utils.ast.canonicalize import get_module_cst
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.scf import (
    ReplaceYieldWithSCFYield,
    ReplaceSCFCond,
    InsertEndIfs,
    InsertEmptyYield,
    CanonicalizeElIfs,
    range_,
)

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def transform_func(f, *transformer_ctors):
    module_cst = get_module_cst(f)
    for transformer_ctor in transformer_ctors:
        func_node = module_cst.body[0]
        replace = transformer_ctor(context=None)
        new_func = func_node._visit_and_replace_children(replace)
        module_cst = module_cst.deep_replace(func_node, new_func)
        # print(module_cst.code)

    return module_cst.code


def test_if_replace_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield)

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
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        return

    code = transform_func(iffoo, InsertEmptyYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0); yield
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield_(three)
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield_(three, three)
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2, res3 = yield three, three, three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2, res3 = yield_(three, three, three)
        return
    """
    )
    assert correct == code


def test_if_replace_cond(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (), has_else = False):
            three = constant(3.0)
            yield_()
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (_placeholder_opaque_t(),), has_else = False):
            three = constant(3.0)
            res = yield_(three)
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (_placeholder_opaque_t(), _placeholder_opaque_t()), has_else = False):
            three = constant(3.0)
            res1, res2 = yield_(three, three)
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2, res3 = yield three, three, three
        return

    code = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceSCFCond)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (_placeholder_opaque_t(), _placeholder_opaque_t(), _placeholder_opaque_t()), has_else = False):
            three = constant(3.0)
            res1, res2, res3 = yield_(three, three, three)
        return
    """
    )
    assert correct == code


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
            yield; end_if()
        return
    """
    )
    assert correct == code

    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
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
        if res := one < two:
            three = constant(3.0)
            yield three; end_branch()
        else:
            else_(); four = constant(4.0)
            yield four; end_if()
        return
    """
    )
    assert correct == code


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

    code = transform_func(iffoo, InsertEmptyYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0); yield
            yield
        return
    """
    )
    assert correct == code


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

    code = transform_func(iffoo, InsertEmptyYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0); yield
            else:
                five = constant(5.0); yield
            yield
        return
    """
    )
    assert correct == code


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
            yield; end_branch()
        else:
            else_(); four = constant(4.0)
            yield; end_if()
        return
    """
    )
    assert correct == code


def test_if_else_with_nested_no_yields_yield_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            res = yield three
        else:
            five = constant(5.0)
            res = yield five
        return

    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (_placeholder_opaque_t(),), has_else = True):
            three = constant(3.0)
            if stack_if(two < three, (), has_else = False):
                four = constant(4.0); yield_(); end_if()
            res = yield_(three); end_branch()
        else:
            else_(); five = constant(5.0)
            res = yield_(five); end_if()
        return
    """
    )
    assert correct == code


def test_if_else_with_nested_no_yields_yield_multiple_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if two < three:
                four = constant(4.0)
            res = yield three, three
        else:
            five = constant(5.0)
            res = yield five, five
        return

    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if stack_if(one < two, (_placeholder_opaque_t(), _placeholder_opaque_t()), has_else = True):
            three = constant(3.0)
            if stack_if(two < three, (), has_else = False):
                four = constant(4.0); yield_(); end_if()
            res = yield_(three, three); end_branch()
        else:
            else_(); five = constant(5.0)
            res = yield_(five, five); end_if()
        return
    """
    )
    assert correct == code


def test_if_nested_with_else_no_yield_insert_order(ctx: MLIRContext):
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

    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if stack_if(one < two, (), has_else = False):
            three = constant(3.0)
            if stack_if(one < two, (), has_else = True):
                four = constant(4.0); yield_(); end_branch()
            else:
                else_(); five = constant(5.0); yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_else_with_nested_no_yields_insert_order(ctx: MLIRContext):
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
    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if stack_if(one < two, (), has_else = True):
            three = constant(3.0)
            if stack_if(one < two, (), has_else = False):
                four = constant(4.0); yield_(); end_if()
            yield_(); end_branch()
        else:
            else_(); five = constant(5.0); yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_nested_with_else_no_yields_insert_order(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)
        else:
            six = constant(6.0)

        return

    iffoo()
    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if stack_if(one < two, (), has_else = True):
            three = constant(3.0)
            if stack_if(one < two, (), has_else = True):
                four = constant(4.0); yield_(); end_branch()
            else:
                else_(); five = constant(5.0); yield_(); end_if()
            yield_(); end_branch()
        else:
            else_(); six = constant(6.0); yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_else_else_with_yields(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if one < two:
            three = constant(3.0)
            yield
        else:
            if one < two:
                four = constant(4.0)
                yield
            else:
                five = constant(5.0)
                yield

        return

    code = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if stack_if(one < two, (), has_else = True):
            three = constant(3.0)
            yield_(); end_branch()
        else:
            else_()
            if stack_if(one < two, (), has_else = True):
                four = constant(4.0)
                yield_(); end_branch()
            else:
                else_(); five = constant(5.0)
                yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_canonicalize_elif(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield
        else:
            six = constant(6.0)
            yield

        return

    code = transform_func(iffoo, CanonicalizeElIfs)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        else:
            if two < three:
                five = constant(5.0)
                yield
            else:
                six = constant(6.0)
                yield

        return
    """
    )
    assert correct == code


def test_if_canonicalize_elif_elif(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield
        elif two < three:
            six = constant(6.0)
            yield
        else:
            seven = constant(7.0)
            yield

        return

    code = transform_func(iffoo, CanonicalizeElIfs)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        else:
            if two < three:
                five = constant(5.0)
                yield
            else:
                if two < three:
                    six = constant(6.0)
                    yield
                else:
                    seven = constant(7.0)
                    yield

        return
    """
    )
    assert correct == code


def test_if_with_elif_with_yields(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield
        else:
            six = constant(6.0)
            yield

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if stack_if(one < two, (), has_else = True):
            four = constant(4.0)
            yield_(); end_branch()
        else:
            else_()
            if stack_if(two < three, (), has_else = True):
                five = constant(5.0)
                yield_(); end_branch()
            else:
                else_(); six = constant(6.0)
                yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_elif_with_yields(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield
        elif three < four:
            six = constant(6.0)
            yield
        else:
            six = constant(7.0)
            yield

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if stack_if(one < two, (), has_else = True):
            five = constant(5.0)
            yield_(); end_branch()
        else:
            else_()
            if stack_if(two < three, (), has_else = True):
                five = constant(5.0)
                yield_(); end_branch()
            else:
                else_()
                if stack_if(three < four, (), has_else = True):
                    six = constant(6.0)
                    yield_(); end_branch()
                else:
                    else_(); six = constant(7.0)
                    yield_(); end_if()
                yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_no_else(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            yield
        elif two < three:
            five = constant(5.0)
            yield

        return

    try:
        code = transform_func(
            iffoo,
            InsertEmptyYield,
            ReplaceYieldWithSCFYield,
            ReplaceSCFCond,
            InsertEndIfs,
        )
    except Exception as e:
        assert str(e) == "conditional must have else branch"


def test_if_with_else_nested_elif(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
        else:
            if two < three:
                six = constant(6.0)
            elif three < four:
                seven = constant(7.0)
            else:
                eight = constant(8.0)

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if stack_if(one < two, (), has_else = True):
            five = constant(5.0); yield_(); end_branch()
        else:
            else_()
            if stack_if(two < three, (), has_else = True):
                six = constant(6.0); yield_(); end_branch()
            else:
                else_()
                if stack_if(three < four, (), has_else = True):
                    seven = constant(7.0); yield_(); end_branch()
                else:
                    else_(); eight = constant(8.0); yield_(); end_if()
                yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_no_yields(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
        elif two < three:
            five = constant(5.0)
        else:
            six = constant(6.0)

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if stack_if(one < two, (), has_else = True):
            four = constant(4.0); yield_(); end_branch()
        else:
            else_()
            if stack_if(two < three, (), has_else = True):
                five = constant(5.0); yield_(); end_branch()
            else:
                else_(); six = constant(6.0); yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_elif_no_yields(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
        elif two < three:
            six = constant(6.0)
        elif three < four:
            seven = constant(7.0)
        else:
            eight = constant(8.0)

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if stack_if(one < two, (), has_else = True):
            five = constant(5.0); yield_(); end_branch()
        else:
            else_()
            if stack_if(two < three, (), has_else = True):
                six = constant(6.0); yield_(); end_branch()
            else:
                else_()
                if stack_if(three < four, (), has_else = True):
                    seven = constant(7.0); yield_(); end_branch()
                else:
                    else_(); eight = constant(8.0); yield_(); end_if()
                yield_(); end_if()
            yield_(); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_yields_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
            res = yield four
        elif two < three:
            five = constant(5.0)
            res1 = yield five
        else:
            six = constant(6.0)
            res2 = yield six

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        # InsertEmptyYield,
        # ReplaceYieldWithSCFYield,
        # ReplaceSCFCond,
        # InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        
        if one < two:
            four = constant(4.0)
            res = yield four
        else:
            if two < three:
                five = constant(5.0)
                res1 = yield five
            else:
                six = constant(6.0)
                res2 = yield six
            res1 = yield res1
            
        return
    """
    )
    assert correct == code

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if stack_if(one < two, (_placeholder_opaque_t(),), has_else = True):
            four = constant(4.0)
            res = yield_(four); end_branch()
        else:
            else_()
            if stack_if(two < three, (_placeholder_opaque_t(),), has_else = True):
                five = constant(5.0)
                res1 = yield_(five); end_branch()
            else:
                else_(); six = constant(6.0)
                res2 = yield_(six); end_if()
            res1 = yield_(res1); end_if()

        return
    """
    )
    assert correct == code


def test_if_with_elif_elif_yields_results(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            res1, res2 = yield five, five
        elif two < three:
            six = constant(6.0)
            res1, res2 = yield six, six
        elif three < four:
            seven = constant(7.0)
            res1, res2 = yield seven, seven
        else:
            eight = constant(8.0)
            res1, res2 = yield eight, eight

        return

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        # InsertEmptyYield,
        # ReplaceYieldWithSCFYield,
        # ReplaceSCFCond,
        # InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if one < two:
            five = constant(5.0)
            res1, res2 = yield five, five
        else:
            if two < three:
                six = constant(6.0)
                res1, res2 = yield six, six
            else:
                if three < four:
                    seven = constant(7.0)
                    res1, res2 = yield seven, seven
                else:
                    eight = constant(8.0)
                    res1, res2 = yield eight, eight
                res1, res2 = yield res1, res2
            res1, res2 = yield res1, res2
        
        return
    """
    )
    assert correct == code

    code = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceSCFCond,
        InsertEndIfs,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)

        if stack_if(one < two, (_placeholder_opaque_t(), _placeholder_opaque_t()), has_else = True):
            five = constant(5.0)
            res1, res2 = yield_(five, five); end_branch()
        else:
            else_()
            if stack_if(two < three, (_placeholder_opaque_t(), _placeholder_opaque_t()), has_else = True):
                six = constant(6.0)
                res1, res2 = yield_(six, six); end_branch()
            else:
                else_()
                if stack_if(three < four, (_placeholder_opaque_t(), _placeholder_opaque_t()), has_else = True):
                    seven = constant(7.0)
                    res1, res2 = yield_(seven, seven); end_branch()
                else:
                    else_(); eight = constant(8.0)
                    res1, res2 = yield_(eight, eight); end_if()
                res1, res2 = yield_(res1, res2); end_if()
            res1, res2 = yield_(res1, res2); end_if()

        return
    """
    )
    assert correct == code


def test_for(ctx: MLIRContext):
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, i1 in range_(0, 10, iter_args=[one]):
            three = constant(3.0)
            res = yield three

    foo()

    code = transform_func(
        foo,
        ReplaceYieldWithSCFYield,
    )

    correct = dedent(
        """\
    def foo():
        one = constant(1.0)
        two = constant(1.0)

        _i = 0
        for i, i1 in range_(0, 10, iter_args=[one]):
            three = constant(3.0)
            res = yield_(three)
    """
    )
    assert correct == code
