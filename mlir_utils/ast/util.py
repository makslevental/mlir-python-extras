import functools
import inspect
import types
from textwrap import dedent

import libcst as cst


def ast_call(name, args=None, keywords=None):
    if keywords is None:
        keywords = []
    if args is None:
        args = []
    call = cst.Call(
        func=cst.Name(value=name),
        args=args + keywords,
    )
    return call


def get_module_cst(f):
    f_src = dedent(inspect.getsource(f))
    tree = cst.parse_module(f_src)
    assert isinstance(
        tree.body[0], cst.FunctionDef
    ), f"unexpected ast node {tree.body[0]}"
    return tree


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
