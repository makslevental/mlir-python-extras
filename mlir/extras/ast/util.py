import ast
import inspect
from itertools import dropwhile
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
    lines, _lnum = inspect.getsourcelines(f)
    f_src = dedent("".join(list(dropwhile(lambda l: l.startswith("@"), lines))))
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


def append_hidden_node(node_body, new_node):
    last_statement = node_body[-1]
    new_node = ast.fix_missing_locations(
        set_lineno(new_node, last_statement.end_lineno)
    )
    node_body.append(new_node)
    return node_body
