import ast
import sys
from textwrap import dedent
from typing import Tuple

import astpretty
import pytest

from mlir.extras.ast.canonicalize import transform_func
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.scf import (
    CanonicalizeElIfs,
    ReplaceIfWithWith,
    ReplaceYieldWithSCFYield,
    InsertEmptyYield,
    CanonicalizeWhile,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def _fields(n: ast.AST, show_offsets: bool = True) -> Tuple[str, ...]:
    strip = {"type_ignores", "decorator_list", "type_comment", "ctx", "kind"}
    fields = tuple(f for f in n._fields if f not in strip)
    attributes = ("lineno",) if "lineno" in n._attributes else ()
    return attributes + fields


# astpretty._leaf = _leaf
astpretty._fields = _fields


if sys.version_info.minor != 12:
    pytest.skip("only check latest", allow_module_level=True)


def test_if_handle_yield_1():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield)

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
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=6,
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )
    assert correct.strip() == dump


def test_if_handle_yield_2():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        return

    mod = transform_func(iffoo, InsertEmptyYield)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return\
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=5,
                                value=Yield(lineno=5, value=None),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=6, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )
    assert correct.strip() == dump


def test_if_handle_yield_3():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield)

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
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=6,
                                targets=[Name(lineno=6, id='res')],
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[Name(lineno=6, id='three')],
                                    keywords=[],
                                ),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_handle_yield_4():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield)

    if sys.version_info.minor >= 11:
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
    elif sys.version_info.minor == 10:
        correct = dedent(
            """\
        def iffoo():
            one = constant(1.0)
            two = constant(2.0)
            if one < two:
                three = constant(3.0)
                (res1, res2) = yield_(three, three)
            return
        """
        )
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=6,
                                targets=[
                                    Tuple(
                                        lineno=6,
                                        elts=[
                                            Name(lineno=6, id='res1'),
                                            Name(lineno=6, id='res2'),
                                        ],
                                    ),
                                ],
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[
                                        Name(lineno=6, id='three'),
                                        Name(lineno=6, id='three'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_nested_no_else_no_yield():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
        return

    mod = transform_func(iffoo, InsertEmptyYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
                yield
            yield
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            If(
                                lineno=6,
                                test=Compare(
                                    lineno=6,
                                    left=Name(lineno=6, id='one'),
                                    ops=[Lt()],
                                    comparators=[Name(lineno=6, id='two')],
                                ),
                                body=[
                                    Assign(
                                        lineno=7,
                                        targets=[Name(lineno=7, id='four')],
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='constant'),
                                            args=[Constant(lineno=7, value=4.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=7,
                                        value=Yield(lineno=7, value=None),
                                    ),
                                ],
                                orelse=[],
                            ),
                            Expr(
                                lineno=7,
                                value=Yield(lineno=7, value=None),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=8, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_replace_cond_1():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            yield
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceIfWithWith)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, ()) as __if_op__4:
            three = constant(3.0)
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(lineno=4, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=6,
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_replace_cond_2():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res = yield three
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceIfWithWith)

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, (placeholder_opaque_t(),)) as __if_op__4:
            three = constant(3.0)
            res = yield_(three)
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(
                                            lineno=4,
                                            elts=[
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=6,
                                targets=[Name(lineno=6, id='res')],
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[Name(lineno=6, id='three')],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_replace_cond_3():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            res1, res2 = yield three, three
        return

    mod = transform_func(iffoo, ReplaceYieldWithSCFYield, ReplaceIfWithWith)

    if sys.version_info.minor >= 11:
        correct = dedent(
            """\
        def iffoo():
            one = constant(1.0)
            two = constant(2.0)
            with if_ctx_manager(one < two, (placeholder_opaque_t(), placeholder_opaque_t())) as __if_op__4:
                three = constant(3.0)
                res1, res2 = yield_(three, three)
            return
        """
        )
    elif sys.version_info.minor == 10:
        correct = dedent(
            """\
        def iffoo():
            one = constant(1.0)
            two = constant(2.0)
            with if_ctx_manager(one < two, (placeholder_opaque_t(), placeholder_opaque_t())) as __if_op__4:
                three = constant(3.0)
                (res1, res2) = yield_(three, three)
            return
        """
        )
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(
                                            lineno=4,
                                            elts=[
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=6,
                                targets=[
                                    Tuple(
                                        lineno=6,
                                        elts=[
                                            Name(lineno=6, id='res1'),
                                            Name(lineno=6, id='res2'),
                                        ],
                                    ),
                                ],
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[
                                        Name(lineno=6, id='three'),
                                        Name(lineno=6, id='three'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=7, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )
    assert correct.strip() == dump


def test_if_nested_with_else_no_yield():
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

    mod = transform_func(iffoo, CanonicalizeElIfs, InsertEmptyYield)
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
            if one < two:
                four = constant(4.0)
                yield
            else:
                five = constant(5.0)
                yield
            yield
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)
    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    If(
                        lineno=4,
                        test=Compare(
                            lineno=4,
                            left=Name(lineno=4, id='one'),
                            ops=[Lt()],
                            comparators=[Name(lineno=4, id='two')],
                        ),
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            If(
                                lineno=6,
                                test=Compare(
                                    lineno=6,
                                    left=Name(lineno=6, id='one'),
                                    ops=[Lt()],
                                    comparators=[Name(lineno=6, id='two')],
                                ),
                                body=[
                                    Assign(
                                        lineno=7,
                                        targets=[Name(lineno=7, id='four')],
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='constant'),
                                            args=[Constant(lineno=7, value=4.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=7,
                                        value=Yield(lineno=7, value=None),
                                    ),
                                ],
                                orelse=[
                                    Assign(
                                        lineno=9,
                                        targets=[Name(lineno=9, id='five')],
                                        value=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='constant'),
                                            args=[Constant(lineno=9, value=5.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=9,
                                        value=Yield(lineno=9, value=None),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=9,
                                value=Yield(lineno=9, value=None),
                            ),
                        ],
                        orelse=[],
                    ),
                    Return(lineno=10, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_insert_end_ifs_yield():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        if one < two:
            three = constant(3.0)
        else:
            four = constant(4.0)
        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, ()) as __if_op__4:
            three = constant(3.0)
            yield_()
        with else_ctx_manager(__if_op__4):
            four = constant(4.0)
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)
    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(lineno=4, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=5,
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=6,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='else_ctx_manager'),
                                    args=[Name(lineno=6, id='__if_op__4')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=7,
                                targets=[Name(lineno=7, id='four')],
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='constant'),
                                    args=[Constant(lineno=7, value=4.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=7,
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=8, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_else_with_nested_no_yields_yield_results():
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

    mod = transform_func(
        iffoo,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, (placeholder_opaque_t(),)) as __if_op__4:
            three = constant(3.0)
            with if_ctx_manager(two < three, ()) as __if_op__6:
                four = constant(4.0)
                yield_()
            res = yield_(three)
        with else_ctx_manager(__if_op__4):
            five = constant(5.0)
            res = yield_(five)
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(
                                            lineno=4,
                                            elts=[
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            With(
                                lineno=6,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=6,
                                            func=Name(lineno=6, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=6,
                                                    left=Name(lineno=6, id='two'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=6, id='three')],
                                                ),
                                                Tuple(lineno=6, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=6, id='__if_op__6'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=7,
                                        targets=[Name(lineno=7, id='four')],
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='constant'),
                                            args=[Constant(lineno=7, value=4.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=7,
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Assign(
                                lineno=8,
                                targets=[Name(lineno=8, id='res')],
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='yield_'),
                                    args=[Name(lineno=8, id='three')],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=9,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=9,
                                    func=Name(lineno=9, id='else_ctx_manager'),
                                    args=[Name(lineno=9, id='__if_op__4')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=10,
                                targets=[Name(lineno=10, id='five')],
                                value=Call(
                                    lineno=10,
                                    func=Name(lineno=10, id='constant'),
                                    args=[Constant(lineno=10, value=5.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=11,
                                targets=[Name(lineno=11, id='res')],
                                value=Call(
                                    lineno=11,
                                    func=Name(lineno=11, id='yield_'),
                                    args=[Name(lineno=11, id='five')],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=12, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_else_with_nested_no_yields_yield_multiple_results():
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

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, (placeholder_opaque_t(), placeholder_opaque_t())) as __if_op__4:
            three = constant(3.0)
            with if_ctx_manager(two < three, ()) as __if_op__6:
                four = constant(4.0)
                yield_()
            res = yield_(three, three)
        with else_ctx_manager(__if_op__4):
            five = constant(5.0)
            res = yield_(five, five)
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)
    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=4,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=4,
                                    func=Name(lineno=4, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=4,
                                            left=Name(lineno=4, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=4, id='two')],
                                        ),
                                        Tuple(
                                            lineno=4,
                                            elts=[
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                                Call(
                                                    lineno=4,
                                                    func=Name(lineno=4, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=4, id='__if_op__4'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=5,
                                targets=[Name(lineno=5, id='three')],
                                value=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='constant'),
                                    args=[Constant(lineno=5, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            With(
                                lineno=6,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=6,
                                            func=Name(lineno=6, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=6,
                                                    left=Name(lineno=6, id='two'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=6, id='three')],
                                                ),
                                                Tuple(lineno=6, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=6, id='__if_op__6'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=7,
                                        targets=[Name(lineno=7, id='four')],
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='constant'),
                                            args=[Constant(lineno=7, value=4.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=7,
                                        value=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Assign(
                                lineno=8,
                                targets=[Name(lineno=8, id='res')],
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='yield_'),
                                    args=[
                                        Name(lineno=8, id='three'),
                                        Name(lineno=8, id='three'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=9,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=9,
                                    func=Name(lineno=9, id='else_ctx_manager'),
                                    args=[Name(lineno=9, id='__if_op__4')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=10,
                                targets=[Name(lineno=10, id='five')],
                                value=Call(
                                    lineno=10,
                                    func=Name(lineno=10, id='constant'),
                                    args=[Constant(lineno=10, value=5.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=11,
                                targets=[Name(lineno=11, id='res')],
                                value=Call(
                                    lineno=11,
                                    func=Name(lineno=11, id='yield_'),
                                    args=[
                                        Name(lineno=11, id='five'),
                                        Name(lineno=11, id='five'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=12, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_if_with_else_else_with_yields():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)

        if one < two:
            three = constant(3.0)
        else:
            if one < two:
                four = constant(4.0)
            else:
                five = constant(5.0)

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        with if_ctx_manager(one < two, ()) as __if_op__5:
            three = constant(3.0)
            yield_()
        with else_ctx_manager(__if_op__5):
            with if_ctx_manager(one < two, ()) as __if_op__8:
                four = constant(4.0)
                yield_()
            with else_ctx_manager(__if_op__8):
                five = constant(5.0)
                yield_()
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=5,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=5,
                                    func=Name(lineno=5, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=5,
                                            left=Name(lineno=5, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=5, id='two')],
                                        ),
                                        Tuple(lineno=5, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=5, id='__if_op__5'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=6,
                                targets=[Name(lineno=6, id='three')],
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='constant'),
                                    args=[Constant(lineno=6, value=3.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=6,
                                value=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=7,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='else_ctx_manager'),
                                    args=[Name(lineno=7, id='__if_op__5')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=8,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=8,
                                            func=Name(lineno=8, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=8,
                                                    left=Name(lineno=8, id='one'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=8, id='two')],
                                                ),
                                                Tuple(lineno=8, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=8, id='__if_op__8'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=9,
                                        targets=[Name(lineno=9, id='four')],
                                        value=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='constant'),
                                            args=[Constant(lineno=9, value=4.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=9,
                                        value=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=10,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='else_ctx_manager'),
                                            args=[Name(lineno=10, id='__if_op__8')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=11,
                                        targets=[Name(lineno=11, id='five')],
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='constant'),
                                            args=[Constant(lineno=11, value=5.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=11,
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=11,
                                value=Call(
                                    lineno=11,
                                    func=Name(lineno=11, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=13, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )
    assert correct.strip() == dump


def test_if_canonicalize_elif_elif():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)

        if one < two:
            four = constant(4.0)
        else:
            if two < three:
                five = constant(5.0)
            else:
                if two < three:
                    six = constant(6.0)
                else:
                    seven = constant(7.0)

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )
    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        with if_ctx_manager(one < two, ()) as __if_op__6:
            four = constant(4.0)
            yield_()
        with else_ctx_manager(__if_op__6):
            with if_ctx_manager(two < three, ()) as __if_op__9:
                five = constant(5.0)
                yield_()
            with else_ctx_manager(__if_op__9):
                with if_ctx_manager(two < three, ()) as __if_op__12:
                    six = constant(6.0)
                    yield_()
                with else_ctx_manager(__if_op__12):
                    seven = constant(7.0)
                    yield_()
                yield_()
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=6,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=6,
                                            left=Name(lineno=6, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=6, id='two')],
                                        ),
                                        Tuple(lineno=6, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=6, id='__if_op__6'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=7,
                                targets=[Name(lineno=7, id='four')],
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='constant'),
                                    args=[Constant(lineno=7, value=4.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=7,
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=8,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='else_ctx_manager'),
                                    args=[Name(lineno=8, id='__if_op__6')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=9,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=9,
                                                    left=Name(lineno=9, id='two'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=9, id='three')],
                                                ),
                                                Tuple(lineno=9, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=9, id='__if_op__9'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=10,
                                        targets=[Name(lineno=10, id='five')],
                                        value=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='constant'),
                                            args=[Constant(lineno=10, value=5.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=10,
                                        value=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=11,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='else_ctx_manager'),
                                            args=[Name(lineno=11, id='__if_op__9')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    With(
                                        lineno=12,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=12,
                                                    func=Name(lineno=12, id='if_ctx_manager'),
                                                    args=[
                                                        Compare(
                                                            lineno=12,
                                                            left=Name(lineno=12, id='two'),
                                                            ops=[Lt()],
                                                            comparators=[Name(lineno=12, id='three')],
                                                        ),
                                                        Tuple(lineno=12, elts=[]),
                                                    ],
                                                    keywords=[],
                                                ),
                                                optional_vars=Name(lineno=12, id='__if_op__12'),
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=13,
                                                targets=[Name(lineno=13, id='six')],
                                                value=Call(
                                                    lineno=13,
                                                    func=Name(lineno=13, id='constant'),
                                                    args=[Constant(lineno=13, value=6.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=13,
                                                value=Call(
                                                    lineno=13,
                                                    func=Name(lineno=13, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    With(
                                        lineno=14,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=14,
                                                    func=Name(lineno=14, id='else_ctx_manager'),
                                                    args=[Name(lineno=14, id='__if_op__12')],
                                                    keywords=[],
                                                ),
                                                optional_vars=None,
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=15,
                                                targets=[Name(lineno=15, id='seven')],
                                                value=Call(
                                                    lineno=15,
                                                    func=Name(lineno=15, id='constant'),
                                                    args=[Constant(lineno=15, value=7.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=15,
                                                value=Call(
                                                    lineno=15,
                                                    func=Name(lineno=15, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    Expr(
                                        lineno=15,
                                        value=Call(
                                            lineno=15,
                                            func=Name(lineno=15, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=15,
                                value=Call(
                                    lineno=15,
                                    func=Name(lineno=15, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=17, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_elif_1():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        if one < two:
            five = constant(5.0)
        elif three < four:
            six = constant(6.0)
        else:
            seven = constant(7.0)

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        with if_ctx_manager(one < two, ()) as __if_op__6:
            five = constant(5.0)
            yield_()
        with else_ctx_manager(__if_op__6):
            with if_ctx_manager(three < four, ()) as __if_op__8:
                six = constant(6.0)
                yield_()
            with else_ctx_manager(__if_op__8):
                seven = constant(7.0)
                yield_()
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=5,
                        targets=[Name(lineno=5, id='four')],
                        value=Call(
                            lineno=5,
                            func=Name(lineno=5, id='constant'),
                            args=[Constant(lineno=5, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=6,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=6,
                                            left=Name(lineno=6, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=6, id='two')],
                                        ),
                                        Tuple(lineno=6, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=6, id='__if_op__6'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=7,
                                targets=[Name(lineno=7, id='five')],
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='constant'),
                                    args=[Constant(lineno=7, value=5.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=7,
                                value=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=8,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='else_ctx_manager'),
                                    args=[Name(lineno=8, id='__if_op__6')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=8,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=8,
                                            func=Name(lineno=8, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=8,
                                                    left=Name(lineno=8, id='three'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=8, id='four')],
                                                ),
                                                Tuple(lineno=8, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=8, id='__if_op__8'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=9,
                                        targets=[Name(lineno=9, id='six')],
                                        value=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='constant'),
                                            args=[Constant(lineno=9, value=6.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=9,
                                        value=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=10,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='else_ctx_manager'),
                                            args=[Name(lineno=10, id='__if_op__8')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=11,
                                        targets=[Name(lineno=11, id='seven')],
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='constant'),
                                            args=[Constant(lineno=11, value=7.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=11,
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=11,
                                value=Call(
                                    lineno=11,
                                    func=Name(lineno=11, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=13, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_elif_2():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)  # line 8
        elif three < four:
            seven = constant(7.0)  # line 10
        elif four < five:
            seven = constant(8.0)  # line 12
        else:
            seven = constant(9.0)  # line 14

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        five = constant(5.0)
        with if_ctx_manager(one < two, ()) as __if_op__7:
            six = constant(6.0)
            yield_()
        with else_ctx_manager(__if_op__7):
            with if_ctx_manager(three < four, ()) as __if_op__9:
                seven = constant(7.0)
                yield_()
            with else_ctx_manager(__if_op__9):
                with if_ctx_manager(four < five, ()) as __if_op__11:
                    seven = constant(8.0)
                    yield_()
                with else_ctx_manager(__if_op__11):
                    seven = constant(9.0)
                    yield_()
                yield_()
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=5,
                        targets=[Name(lineno=5, id='four')],
                        value=Call(
                            lineno=5,
                            func=Name(lineno=5, id='constant'),
                            args=[Constant(lineno=5, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=6,
                        targets=[Name(lineno=6, id='five')],
                        value=Call(
                            lineno=6,
                            func=Name(lineno=6, id='constant'),
                            args=[Constant(lineno=6, value=5.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=7,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=7,
                                            left=Name(lineno=7, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=7, id='two')],
                                        ),
                                        Tuple(lineno=7, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=7, id='__if_op__7'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=8,
                                targets=[Name(lineno=8, id='six')],
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='constant'),
                                    args=[Constant(lineno=8, value=6.0)],
                                    keywords=[],
                                ),
                            ),
                            Expr(
                                lineno=8,
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=9,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=9,
                                    func=Name(lineno=9, id='else_ctx_manager'),
                                    args=[Name(lineno=9, id='__if_op__7')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=9,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=9,
                                                    left=Name(lineno=9, id='three'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=9, id='four')],
                                                ),
                                                Tuple(lineno=9, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=9, id='__if_op__9'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=10,
                                        targets=[Name(lineno=10, id='seven')],
                                        value=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='constant'),
                                            args=[Constant(lineno=10, value=7.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=10,
                                        value=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=11,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='else_ctx_manager'),
                                            args=[Name(lineno=11, id='__if_op__9')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    With(
                                        lineno=11,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=11,
                                                    func=Name(lineno=11, id='if_ctx_manager'),
                                                    args=[
                                                        Compare(
                                                            lineno=11,
                                                            left=Name(lineno=11, id='four'),
                                                            ops=[Lt()],
                                                            comparators=[Name(lineno=11, id='five')],
                                                        ),
                                                        Tuple(lineno=11, elts=[]),
                                                    ],
                                                    keywords=[],
                                                ),
                                                optional_vars=Name(lineno=11, id='__if_op__11'),
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=12,
                                                targets=[Name(lineno=12, id='seven')],
                                                value=Call(
                                                    lineno=12,
                                                    func=Name(lineno=12, id='constant'),
                                                    args=[Constant(lineno=12, value=8.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=12,
                                                value=Call(
                                                    lineno=12,
                                                    func=Name(lineno=12, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    With(
                                        lineno=13,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=13,
                                                    func=Name(lineno=13, id='else_ctx_manager'),
                                                    args=[Name(lineno=13, id='__if_op__11')],
                                                    keywords=[],
                                                ),
                                                optional_vars=None,
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=14,
                                                targets=[Name(lineno=14, id='seven')],
                                                value=Call(
                                                    lineno=14,
                                                    func=Name(lineno=14, id='constant'),
                                                    args=[Constant(lineno=14, value=9.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=14,
                                                value=Call(
                                                    lineno=14,
                                                    func=Name(lineno=14, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    Expr(
                                        lineno=14,
                                        value=Call(
                                            lineno=14,
                                            func=Name(lineno=14, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=14,
                                value=Call(
                                    lineno=14,
                                    func=Name(lineno=14, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=16, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_elif_3():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        if one < two:  # line 6
            if one < two:
                five = constant(5.0)  # line 8
            elif three < four:
                six = constant(6.0)  # line 10
            else:
                seven = constant(7.0)  # line 12
        elif three < four:
            six = constant(6.0)  # line 14
        else:
            seven = constant(7.0)  # line 16

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    correct = dedent(
        """\
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(3.0)
        with if_ctx_manager(one < two, ()) as __if_op__6:
            with if_ctx_manager(one < two, ()) as __if_op__7:
                five = constant(5.0)
                yield_()
            with else_ctx_manager(__if_op__7):
                with if_ctx_manager(three < four, ()) as __if_op__9:
                    six = constant(6.0)
                    yield_()
                with else_ctx_manager(__if_op__9):
                    seven = constant(7.0)
                    yield_()
                yield_()
            yield_()
        with else_ctx_manager(__if_op__6):
            with if_ctx_manager(three < four, ()) as __if_op__13:
                six = constant(6.0)
                yield_()
            with else_ctx_manager(__if_op__13):
                seven = constant(7.0)
                yield_()
            yield_()
        return
    """
    )
    assert correct.strip() == ast.unparse(mod)

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=5,
                        targets=[Name(lineno=5, id='four')],
                        value=Call(
                            lineno=5,
                            func=Name(lineno=5, id='constant'),
                            args=[Constant(lineno=5, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=6,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=6,
                                    func=Name(lineno=6, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=6,
                                            left=Name(lineno=6, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=6, id='two')],
                                        ),
                                        Tuple(lineno=6, elts=[]),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=6, id='__if_op__6'),
                            ),
                        ],
                        body=[
                            With(
                                lineno=7,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=7,
                                            func=Name(lineno=7, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=7,
                                                    left=Name(lineno=7, id='one'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=7, id='two')],
                                                ),
                                                Tuple(lineno=7, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=7, id='__if_op__7'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=8,
                                        targets=[Name(lineno=8, id='five')],
                                        value=Call(
                                            lineno=8,
                                            func=Name(lineno=8, id='constant'),
                                            args=[Constant(lineno=8, value=5.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=8,
                                        value=Call(
                                            lineno=8,
                                            func=Name(lineno=8, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=9,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=9,
                                            func=Name(lineno=9, id='else_ctx_manager'),
                                            args=[Name(lineno=9, id='__if_op__7')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    With(
                                        lineno=9,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=9,
                                                    func=Name(lineno=9, id='if_ctx_manager'),
                                                    args=[
                                                        Compare(
                                                            lineno=9,
                                                            left=Name(lineno=9, id='three'),
                                                            ops=[Lt()],
                                                            comparators=[Name(lineno=9, id='four')],
                                                        ),
                                                        Tuple(lineno=9, elts=[]),
                                                    ],
                                                    keywords=[],
                                                ),
                                                optional_vars=Name(lineno=9, id='__if_op__9'),
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=10,
                                                targets=[Name(lineno=10, id='six')],
                                                value=Call(
                                                    lineno=10,
                                                    func=Name(lineno=10, id='constant'),
                                                    args=[Constant(lineno=10, value=6.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=10,
                                                value=Call(
                                                    lineno=10,
                                                    func=Name(lineno=10, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    With(
                                        lineno=11,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=11,
                                                    func=Name(lineno=11, id='else_ctx_manager'),
                                                    args=[Name(lineno=11, id='__if_op__9')],
                                                    keywords=[],
                                                ),
                                                optional_vars=None,
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=12,
                                                targets=[Name(lineno=12, id='seven')],
                                                value=Call(
                                                    lineno=12,
                                                    func=Name(lineno=12, id='constant'),
                                                    args=[Constant(lineno=12, value=7.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Expr(
                                                lineno=12,
                                                value=Call(
                                                    lineno=12,
                                                    func=Name(lineno=12, id='yield_'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    Expr(
                                        lineno=12,
                                        value=Call(
                                            lineno=12,
                                            func=Name(lineno=12, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=12,
                                value=Call(
                                    lineno=12,
                                    func=Name(lineno=12, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=13,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=13,
                                    func=Name(lineno=13, id='else_ctx_manager'),
                                    args=[Name(lineno=13, id='__if_op__6')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=13,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=13,
                                            func=Name(lineno=13, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=13,
                                                    left=Name(lineno=13, id='three'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=13, id='four')],
                                                ),
                                                Tuple(lineno=13, elts=[]),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=13, id='__if_op__13'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=14,
                                        targets=[Name(lineno=14, id='six')],
                                        value=Call(
                                            lineno=14,
                                            func=Name(lineno=14, id='constant'),
                                            args=[Constant(lineno=14, value=6.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=14,
                                        value=Call(
                                            lineno=14,
                                            func=Name(lineno=14, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=15,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=15,
                                            func=Name(lineno=15, id='else_ctx_manager'),
                                            args=[Name(lineno=15, id='__if_op__13')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=16,
                                        targets=[Name(lineno=16, id='seven')],
                                        value=Call(
                                            lineno=16,
                                            func=Name(lineno=16, id='constant'),
                                            args=[Constant(lineno=16, value=7.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Expr(
                                        lineno=16,
                                        value=Call(
                                            lineno=16,
                                            func=Name(lineno=16, id='yield_'),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Expr(
                                lineno=16,
                                value=Call(
                                    lineno=16,
                                    func=Name(lineno=16, id='yield_'),
                                    args=[],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=18, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_elif_nested_else_branch():
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)  # line 8
            res1 = yield six  # line 9
        elif two < three:
            ten = constant(10.0)  # line 11
            res5 = yield ten
        else:  # line 13
            if three < four:
                seven = constant(7.0)  # line 15
                res2 = yield seven
            elif four < five:  # line 17
                eight = constant(8.0)
                res3 = yield eight  # line 19
            else:
                nine = constant(9.0)  # line 21
                res4 = yield nine

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=5,
                        targets=[Name(lineno=5, id='four')],
                        value=Call(
                            lineno=5,
                            func=Name(lineno=5, id='constant'),
                            args=[Constant(lineno=5, value=4.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=6,
                        targets=[Name(lineno=6, id='five')],
                        value=Call(
                            lineno=6,
                            func=Name(lineno=6, id='constant'),
                            args=[Constant(lineno=6, value=5.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=7,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=7,
                                            left=Name(lineno=7, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=7, id='two')],
                                        ),
                                        Tuple(
                                            lineno=7,
                                            elts=[
                                                Call(
                                                    lineno=7,
                                                    func=Name(lineno=7, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=7, id='__if_op__7'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=8,
                                targets=[Name(lineno=8, id='six')],
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='constant'),
                                    args=[Constant(lineno=8, value=6.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=9,
                                targets=[Name(lineno=9, id='res1')],
                                value=Call(
                                    lineno=9,
                                    func=Name(lineno=9, id='yield_'),
                                    args=[Name(lineno=9, id='six')],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=10,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=10,
                                    func=Name(lineno=10, id='else_ctx_manager'),
                                    args=[Name(lineno=10, id='__if_op__7')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=10,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=10,
                                                    left=Name(lineno=10, id='two'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=10, id='three')],
                                                ),
                                                Tuple(
                                                    lineno=10,
                                                    elts=[
                                                        Call(
                                                            lineno=10,
                                                            func=Name(lineno=10, id='placeholder_opaque_t'),
                                                            args=[],
                                                            keywords=[],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=10, id='__if_op__10'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=11,
                                        targets=[Name(lineno=11, id='ten')],
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='constant'),
                                            args=[Constant(lineno=11, value=10.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Assign(
                                        lineno=12,
                                        targets=[Name(lineno=12, id='res5')],
                                        value=Call(
                                            lineno=12,
                                            func=Name(lineno=12, id='yield_'),
                                            args=[Name(lineno=12, id='ten')],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=13,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=13,
                                            func=Name(lineno=13, id='else_ctx_manager'),
                                            args=[Name(lineno=13, id='__if_op__10')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    With(
                                        lineno=14,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=14,
                                                    func=Name(lineno=14, id='if_ctx_manager'),
                                                    args=[
                                                        Compare(
                                                            lineno=14,
                                                            left=Name(lineno=14, id='three'),
                                                            ops=[Lt()],
                                                            comparators=[Name(lineno=14, id='four')],
                                                        ),
                                                        Tuple(
                                                            lineno=14,
                                                            elts=[
                                                                Call(
                                                                    lineno=14,
                                                                    func=Name(lineno=14, id='placeholder_opaque_t'),
                                                                    args=[],
                                                                    keywords=[],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    keywords=[],
                                                ),
                                                optional_vars=Name(lineno=14, id='__if_op__14'),
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=15,
                                                targets=[Name(lineno=15, id='seven')],
                                                value=Call(
                                                    lineno=15,
                                                    func=Name(lineno=15, id='constant'),
                                                    args=[Constant(lineno=15, value=7.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Assign(
                                                lineno=16,
                                                targets=[Name(lineno=16, id='res2')],
                                                value=Call(
                                                    lineno=16,
                                                    func=Name(lineno=16, id='yield_'),
                                                    args=[Name(lineno=16, id='seven')],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    With(
                                        lineno=17,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=17,
                                                    func=Name(lineno=17, id='else_ctx_manager'),
                                                    args=[Name(lineno=17, id='__if_op__14')],
                                                    keywords=[],
                                                ),
                                                optional_vars=None,
                                            ),
                                        ],
                                        body=[
                                            With(
                                                lineno=17,
                                                items=[
                                                    withitem(
                                                        context_expr=Call(
                                                            lineno=17,
                                                            func=Name(lineno=17, id='if_ctx_manager'),
                                                            args=[
                                                                Compare(
                                                                    lineno=17,
                                                                    left=Name(lineno=17, id='four'),
                                                                    ops=[Lt()],
                                                                    comparators=[Name(lineno=17, id='five')],
                                                                ),
                                                                Tuple(
                                                                    lineno=17,
                                                                    elts=[
                                                                        Call(
                                                                            lineno=17,
                                                                            func=Name(lineno=17, id='placeholder_opaque_t'),
                                                                            args=[],
                                                                            keywords=[],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                            keywords=[],
                                                        ),
                                                        optional_vars=Name(lineno=17, id='__if_op__17'),
                                                    ),
                                                ],
                                                body=[
                                                    Assign(
                                                        lineno=18,
                                                        targets=[Name(lineno=18, id='eight')],
                                                        value=Call(
                                                            lineno=18,
                                                            func=Name(lineno=18, id='constant'),
                                                            args=[Constant(lineno=18, value=8.0)],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                    Assign(
                                                        lineno=19,
                                                        targets=[Name(lineno=19, id='res3')],
                                                        value=Call(
                                                            lineno=19,
                                                            func=Name(lineno=19, id='yield_'),
                                                            args=[Name(lineno=19, id='eight')],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            With(
                                                lineno=20,
                                                items=[
                                                    withitem(
                                                        context_expr=Call(
                                                            lineno=20,
                                                            func=Name(lineno=20, id='else_ctx_manager'),
                                                            args=[Name(lineno=20, id='__if_op__17')],
                                                            keywords=[],
                                                        ),
                                                        optional_vars=None,
                                                    ),
                                                ],
                                                body=[
                                                    Assign(
                                                        lineno=21,
                                                        targets=[Name(lineno=21, id='nine')],
                                                        value=Call(
                                                            lineno=21,
                                                            func=Name(lineno=21, id='constant'),
                                                            args=[Constant(lineno=21, value=9.0)],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                    Assign(
                                                        lineno=22,
                                                        targets=[Name(lineno=22, id='res4')],
                                                        value=Call(
                                                            lineno=22,
                                                            func=Name(lineno=22, id='yield_'),
                                                            args=[Name(lineno=22, id='nine')],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            Assign(
                                                lineno=22,
                                                targets=[Name(lineno=22, id='res3')],
                                                value=Call(
                                                    lineno=22,
                                                    func=Name(lineno=22, id='yield_'),
                                                    args=[Name(lineno=22, id='res3')],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    Assign(
                                        lineno=22,
                                        targets=[Name(lineno=22, id='res2')],
                                        value=Call(
                                            lineno=22,
                                            func=Name(lineno=22, id='yield_'),
                                            args=[Name(lineno=22, id='res2')],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Assign(
                                lineno=22,
                                targets=[Name(lineno=22, id='res5')],
                                value=Call(
                                    lineno=22,
                                    func=Name(lineno=22, id='yield_'),
                                    args=[Name(lineno=22, id='res5')],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=24, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_elif_nested_else_branch_multiple_yield(ctx: MLIRContext):
    def iffoo():
        one = constant(1.0)
        two = constant(2.0)
        three = constant(3.0)
        four = constant(4.0)
        five = constant(5.0)
        if one < two:
            six = constant(6.0)
            res1, res2 = yield six, six
        elif two < three:
            ten = constant(10.0)
            res3, res4 = yield ten, ten
        else:
            if three < four:
                seven = constant(7.0)
                res5, res6 = yield seven, seven
            elif four < five:
                eight = constant(8.0)
                res7, res8 = yield eight, eight
            else:
                nine = constant(9.0)
                res9, res10 = yield nine, nine

        return

    mod = transform_func(
        iffoo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
    )

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='iffoo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='one')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='constant'),
                            args=[Constant(lineno=2, value=1.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=3,
                        targets=[Name(lineno=3, id='two')],
                        value=Call(
                            lineno=3,
                            func=Name(lineno=3, id='constant'),
                            args=[Constant(lineno=3, value=2.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=4,
                        targets=[Name(lineno=4, id='three')],
                        value=Call(
                            lineno=4,
                            func=Name(lineno=4, id='constant'),
                            args=[Constant(lineno=4, value=3.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=5,
                        targets=[Name(lineno=5, id='four')],
                        value=Call(
                            lineno=5,
                            func=Name(lineno=5, id='constant'),
                            args=[Constant(lineno=5, value=4.0)],
                            keywords=[],
                        ),
                    ),
                    Assign(
                        lineno=6,
                        targets=[Name(lineno=6, id='five')],
                        value=Call(
                            lineno=6,
                            func=Name(lineno=6, id='constant'),
                            args=[Constant(lineno=6, value=5.0)],
                            keywords=[],
                        ),
                    ),
                    With(
                        lineno=7,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=7,
                                    func=Name(lineno=7, id='if_ctx_manager'),
                                    args=[
                                        Compare(
                                            lineno=7,
                                            left=Name(lineno=7, id='one'),
                                            ops=[Lt()],
                                            comparators=[Name(lineno=7, id='two')],
                                        ),
                                        Tuple(
                                            lineno=7,
                                            elts=[
                                                Call(
                                                    lineno=7,
                                                    func=Name(lineno=7, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                                Call(
                                                    lineno=7,
                                                    func=Name(lineno=7, id='placeholder_opaque_t'),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                            ],
                                        ),
                                    ],
                                    keywords=[],
                                ),
                                optional_vars=Name(lineno=7, id='__if_op__7'),
                            ),
                        ],
                        body=[
                            Assign(
                                lineno=8,
                                targets=[Name(lineno=8, id='six')],
                                value=Call(
                                    lineno=8,
                                    func=Name(lineno=8, id='constant'),
                                    args=[Constant(lineno=8, value=6.0)],
                                    keywords=[],
                                ),
                            ),
                            Assign(
                                lineno=9,
                                targets=[
                                    Tuple(
                                        lineno=9,
                                        elts=[
                                            Name(lineno=9, id='res1'),
                                            Name(lineno=9, id='res2'),
                                        ],
                                    ),
                                ],
                                value=Call(
                                    lineno=9,
                                    func=Name(lineno=9, id='yield_'),
                                    args=[
                                        Name(lineno=9, id='six'),
                                        Name(lineno=9, id='six'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    With(
                        lineno=10,
                        items=[
                            withitem(
                                context_expr=Call(
                                    lineno=10,
                                    func=Name(lineno=10, id='else_ctx_manager'),
                                    args=[Name(lineno=10, id='__if_op__7')],
                                    keywords=[],
                                ),
                                optional_vars=None,
                            ),
                        ],
                        body=[
                            With(
                                lineno=10,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=10,
                                            func=Name(lineno=10, id='if_ctx_manager'),
                                            args=[
                                                Compare(
                                                    lineno=10,
                                                    left=Name(lineno=10, id='two'),
                                                    ops=[Lt()],
                                                    comparators=[Name(lineno=10, id='three')],
                                                ),
                                                Tuple(
                                                    lineno=10,
                                                    elts=[
                                                        Call(
                                                            lineno=10,
                                                            func=Name(lineno=10, id='placeholder_opaque_t'),
                                                            args=[],
                                                            keywords=[],
                                                        ),
                                                        Call(
                                                            lineno=10,
                                                            func=Name(lineno=10, id='placeholder_opaque_t'),
                                                            args=[],
                                                            keywords=[],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            keywords=[],
                                        ),
                                        optional_vars=Name(lineno=10, id='__if_op__10'),
                                    ),
                                ],
                                body=[
                                    Assign(
                                        lineno=11,
                                        targets=[Name(lineno=11, id='ten')],
                                        value=Call(
                                            lineno=11,
                                            func=Name(lineno=11, id='constant'),
                                            args=[Constant(lineno=11, value=10.0)],
                                            keywords=[],
                                        ),
                                    ),
                                    Assign(
                                        lineno=12,
                                        targets=[
                                            Tuple(
                                                lineno=12,
                                                elts=[
                                                    Name(lineno=12, id='res3'),
                                                    Name(lineno=12, id='res4'),
                                                ],
                                            ),
                                        ],
                                        value=Call(
                                            lineno=12,
                                            func=Name(lineno=12, id='yield_'),
                                            args=[
                                                Name(lineno=12, id='ten'),
                                                Name(lineno=12, id='ten'),
                                            ],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            With(
                                lineno=13,
                                items=[
                                    withitem(
                                        context_expr=Call(
                                            lineno=13,
                                            func=Name(lineno=13, id='else_ctx_manager'),
                                            args=[Name(lineno=13, id='__if_op__10')],
                                            keywords=[],
                                        ),
                                        optional_vars=None,
                                    ),
                                ],
                                body=[
                                    With(
                                        lineno=14,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=14,
                                                    func=Name(lineno=14, id='if_ctx_manager'),
                                                    args=[
                                                        Compare(
                                                            lineno=14,
                                                            left=Name(lineno=14, id='three'),
                                                            ops=[Lt()],
                                                            comparators=[Name(lineno=14, id='four')],
                                                        ),
                                                        Tuple(
                                                            lineno=14,
                                                            elts=[
                                                                Call(
                                                                    lineno=14,
                                                                    func=Name(lineno=14, id='placeholder_opaque_t'),
                                                                    args=[],
                                                                    keywords=[],
                                                                ),
                                                                Call(
                                                                    lineno=14,
                                                                    func=Name(lineno=14, id='placeholder_opaque_t'),
                                                                    args=[],
                                                                    keywords=[],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    keywords=[],
                                                ),
                                                optional_vars=Name(lineno=14, id='__if_op__14'),
                                            ),
                                        ],
                                        body=[
                                            Assign(
                                                lineno=15,
                                                targets=[Name(lineno=15, id='seven')],
                                                value=Call(
                                                    lineno=15,
                                                    func=Name(lineno=15, id='constant'),
                                                    args=[Constant(lineno=15, value=7.0)],
                                                    keywords=[],
                                                ),
                                            ),
                                            Assign(
                                                lineno=16,
                                                targets=[
                                                    Tuple(
                                                        lineno=16,
                                                        elts=[
                                                            Name(lineno=16, id='res5'),
                                                            Name(lineno=16, id='res6'),
                                                        ],
                                                    ),
                                                ],
                                                value=Call(
                                                    lineno=16,
                                                    func=Name(lineno=16, id='yield_'),
                                                    args=[
                                                        Name(lineno=16, id='seven'),
                                                        Name(lineno=16, id='seven'),
                                                    ],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    With(
                                        lineno=17,
                                        items=[
                                            withitem(
                                                context_expr=Call(
                                                    lineno=17,
                                                    func=Name(lineno=17, id='else_ctx_manager'),
                                                    args=[Name(lineno=17, id='__if_op__14')],
                                                    keywords=[],
                                                ),
                                                optional_vars=None,
                                            ),
                                        ],
                                        body=[
                                            With(
                                                lineno=17,
                                                items=[
                                                    withitem(
                                                        context_expr=Call(
                                                            lineno=17,
                                                            func=Name(lineno=17, id='if_ctx_manager'),
                                                            args=[
                                                                Compare(
                                                                    lineno=17,
                                                                    left=Name(lineno=17, id='four'),
                                                                    ops=[Lt()],
                                                                    comparators=[Name(lineno=17, id='five')],
                                                                ),
                                                                Tuple(
                                                                    lineno=17,
                                                                    elts=[
                                                                        Call(
                                                                            lineno=17,
                                                                            func=Name(lineno=17, id='placeholder_opaque_t'),
                                                                            args=[],
                                                                            keywords=[],
                                                                        ),
                                                                        Call(
                                                                            lineno=17,
                                                                            func=Name(lineno=17, id='placeholder_opaque_t'),
                                                                            args=[],
                                                                            keywords=[],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                            keywords=[],
                                                        ),
                                                        optional_vars=Name(lineno=17, id='__if_op__17'),
                                                    ),
                                                ],
                                                body=[
                                                    Assign(
                                                        lineno=18,
                                                        targets=[Name(lineno=18, id='eight')],
                                                        value=Call(
                                                            lineno=18,
                                                            func=Name(lineno=18, id='constant'),
                                                            args=[Constant(lineno=18, value=8.0)],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                    Assign(
                                                        lineno=19,
                                                        targets=[
                                                            Tuple(
                                                                lineno=19,
                                                                elts=[
                                                                    Name(lineno=19, id='res7'),
                                                                    Name(lineno=19, id='res8'),
                                                                ],
                                                            ),
                                                        ],
                                                        value=Call(
                                                            lineno=19,
                                                            func=Name(lineno=19, id='yield_'),
                                                            args=[
                                                                Name(lineno=19, id='eight'),
                                                                Name(lineno=19, id='eight'),
                                                            ],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            With(
                                                lineno=20,
                                                items=[
                                                    withitem(
                                                        context_expr=Call(
                                                            lineno=20,
                                                            func=Name(lineno=20, id='else_ctx_manager'),
                                                            args=[Name(lineno=20, id='__if_op__17')],
                                                            keywords=[],
                                                        ),
                                                        optional_vars=None,
                                                    ),
                                                ],
                                                body=[
                                                    Assign(
                                                        lineno=21,
                                                        targets=[Name(lineno=21, id='nine')],
                                                        value=Call(
                                                            lineno=21,
                                                            func=Name(lineno=21, id='constant'),
                                                            args=[Constant(lineno=21, value=9.0)],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                    Assign(
                                                        lineno=22,
                                                        targets=[
                                                            Tuple(
                                                                lineno=22,
                                                                elts=[
                                                                    Name(lineno=22, id='res9'),
                                                                    Name(lineno=22, id='res10'),
                                                                ],
                                                            ),
                                                        ],
                                                        value=Call(
                                                            lineno=22,
                                                            func=Name(lineno=22, id='yield_'),
                                                            args=[
                                                                Name(lineno=22, id='nine'),
                                                                Name(lineno=22, id='nine'),
                                                            ],
                                                            keywords=[],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            Assign(
                                                lineno=22,
                                                targets=[
                                                    Tuple(
                                                        lineno=22,
                                                        elts=[
                                                            Name(lineno=22, id='res7'),
                                                            Name(lineno=22, id='res8'),
                                                        ],
                                                    ),
                                                ],
                                                value=Call(
                                                    lineno=22,
                                                    func=Name(lineno=22, id='yield_'),
                                                    args=[
                                                        Name(lineno=22, id='res7'),
                                                        Name(lineno=22, id='res8'),
                                                    ],
                                                    keywords=[],
                                                ),
                                            ),
                                        ],
                                    ),
                                    Assign(
                                        lineno=22,
                                        targets=[
                                            Tuple(
                                                lineno=22,
                                                elts=[
                                                    Name(lineno=22, id='res5'),
                                                    Name(lineno=22, id='res6'),
                                                ],
                                            ),
                                        ],
                                        value=Call(
                                            lineno=22,
                                            func=Name(lineno=22, id='yield_'),
                                            args=[
                                                Name(lineno=22, id='res5'),
                                                Name(lineno=22, id='res6'),
                                            ],
                                            keywords=[],
                                        ),
                                    ),
                                ],
                            ),
                            Assign(
                                lineno=22,
                                targets=[
                                    Tuple(
                                        lineno=22,
                                        elts=[
                                            Name(lineno=22, id='res3'),
                                            Name(lineno=22, id='res4'),
                                        ],
                                    ),
                                ],
                                value=Call(
                                    lineno=22,
                                    func=Name(lineno=22, id='yield_'),
                                    args=[
                                        Name(lineno=22, id='res3'),
                                        Name(lineno=22, id='res4'),
                                    ],
                                    keywords=[],
                                ),
                            ),
                        ],
                    ),
                    Return(lineno=24, value=None),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump


def test_while_canonicalize(ctx: MLIRContext):
    one = constant(1)
    two = constant(2)

    def foo():
        while inits := one < two:
            r = yield inits

    foo()

    mod = transform_func(
        foo,
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
        CanonicalizeWhile,
    )

    dump = astpretty.pformat(mod, show_offsets=True)

    correct = dedent(
        """\
    Module(
        body=[
            FunctionDef(
                lineno=1,
                name='foo',
                args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[
                    Assign(
                        lineno=2,
                        targets=[Name(lineno=2, id='w_2')],
                        value=Call(
                            lineno=2,
                            func=Name(lineno=2, id='while__'),
                            args=[
                                Compare(
                                    lineno=2,
                                    left=Name(lineno=2, id='one'),
                                    ops=[Lt()],
                                    comparators=[Name(lineno=2, id='two')],
                                ),
                            ],
                            keywords=[],
                        ),
                    ),
                    While(
                        lineno=2,
                        test=NamedExpr(
                            lineno=2,
                            target=Name(lineno=2, id='inits'),
                            value=Call(
                                lineno=2,
                                func=Name(lineno=2, id='next'),
                                args=[
                                    Name(lineno=2, id='w_2'),
                                    Constant(lineno=2, value=False),
                                ],
                                keywords=[],
                            ),
                        ),
                        body=[
                            Assign(
                                lineno=3,
                                targets=[Name(lineno=3, id='r')],
                                value=Call(
                                    lineno=3,
                                    func=Name(lineno=3, id='yield_'),
                                    args=[Name(lineno=3, id='inits')],
                                    keywords=[],
                                ),
                            ),
                        ],
                        orelse=[],
                    ),
                ],
                returns=None,
                type_params=[],
            ),
        ],
    )
    """
    )

    assert correct.strip() == dump
