import ast
import functools
import inspect
import types
from textwrap import dedent


def set_lineno(node, n=1):
    for child in ast.walk(node):
        child.lineno = n
        child.end_lineno = n
    return node


def ast_call(name, args=None, keywords=None):
    if keywords is None:
        keywords = []
    if args is None:
        args = []
    call = ast.Call(
        func=ast.Name(name, ctx=ast.Load()),
        args=args,
        keywords=keywords,
    )
    return call


def get_module_cst(f):
    f_src = dedent(inspect.getsource(f))
    # tree = cst.parse_module(f_src)
    tree = ast.parse(f_src)
    assert isinstance(
        tree.body[0], ast.FunctionDef
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
        code=new_code,
        globals={
            **f.__globals__,
            **{
                fr: f.__closure__[i].cell_contents
                for i, fr in enumerate(f.__code__.co_freevars)
            },
        },
        name=f.__name__,
        argdefs=f.__defaults__,
        # TODO(max): ValueError: foo requires closure of length 0, not 1
        # closure=f.__closure__,
    )
    g.__kwdefaults__ = f.__kwdefaults__
    g.__dict__.update(f.__dict__)
    g = functools.update_wrapper(g, f)

    if inspect.ismethod(f):
        g = bind(g, f.__self__)
    return g


def append_hidden_node(node_body, new_node):
    last_statement = node_body[-1]
    new_node = ast.fix_missing_locations(
        set_lineno(new_node, last_statement.end_lineno)
    )
    node_body.append(new_node)
    return node_body
