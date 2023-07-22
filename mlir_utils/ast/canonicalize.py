import ast
import functools
import inspect
import types
from abc import ABC
from textwrap import dedent
from types import CodeType


class StrictTransformer(ast.NodeTransformer):
    def __init__(self, context=None):
        self.context = context

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node


def bind(func, instance, as_name=None):
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def copy_func(f, new_code):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(
        new_code,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g.__kwdefaults__ = f.__kwdefaults__
    g.__dict__.update(f.__dict__)
    g = functools.update_wrapper(g, f)

    if inspect.ismethod(f):
        g = bind(g, f.__self__)
    return g


def rewrite_ast(f, rewriters: list[StrictTransformer.__class__] = None):
    if rewriters is None:
        rewriters = []
    tree = ast.parse(dedent(inspect.getsource(f)))
    assert isinstance(
        tree.body[0], ast.FunctionDef
    ), f"unexpected ast node {tree.body[0]}"
    func_node = tree.body[0]
    context = types.SimpleNamespace()
    for rewriter in rewriters:
        for i, b in enumerate(func_node.body):
            func_node.body[i] = rewriter(context).visit(b)

    tree = ast.Module([func_node], type_ignores=[])

    tree = ast.fix_missing_locations(tree)
    tree = ast.increment_lineno(tree, f.__code__.co_firstlineno - 1)
    module_code_o = compile(tree, f.__code__.co_filename, "exec")
    new_f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == f.__name__
    )

    return copy_func(f, new_f_code_o)


class Canonicalizer(ABC):
    @property
    def ast_rewriters(self) -> list[StrictTransformer]:
        pass

    @property
    def bytecode_rewriters(self) -> list[StrictTransformer]:
        pass


def canonicalize(*, with_: Canonicalizer):
    def wrapper(f):
        f = rewrite_ast(f, with_.ast_rewriters)
        return f

    return wrapper
