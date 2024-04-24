import ast
import functools
import inspect
import types
from opcode import opmap
from textwrap import dedent
from typing import Dict, TypeVar

from bytecode import ConcreteBytecode
from cloudpickle import cloudpickle
from einspect import ptr
from einspect.structs import PyVarObject, PyObject


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


# based on https://github.com/python/cpython/blob/6078f2033ea15a16cf52fe8d644a95a3be72d2e3/Tools/build/deepfreeze.py#L48
def get_localsplus_name_to_idx(code: types.CodeType):
    localsplus = code.co_varnames + code.co_cellvars + code.co_freevars
    return localsplus, {v: i for i, v in enumerate(localsplus)}


class _empty_cell_value:
    """Sentinel for empty closures."""

    @classmethod
    def __reduce__(cls):
        return cls.__name__


_empty_cell_value = _empty_cell_value()


# based on https://github.com/cloudpipe/cloudpickle/blob/f111f7ab6d302e9b1e2a568d0e4c574895db6a6e/cloudpickle/cloudpickle.py#L513
def make_empty_cell():
    if False:
        # trick the compiler into creating an empty cell in our lambda
        cell = None
        raise AssertionError("this route should not be executed")

    return (lambda: cell).__closure__[0]


def make_cell(value=_empty_cell_value):
    cell = make_empty_cell()
    if value is not _empty_cell_value:
        cell.cell_contents = value
    return cell


# based on https://github.com/python/cpython/blob/a4b44d39cd6941cc03590fee7538776728bdfd0a/Lib/test/test_code.py#L197
def replace_closure(code, new_closure: Dict):
    COPY_FREE_VARS = opmap["COPY_FREE_VARS"]
    LOAD_DEREF = opmap["LOAD_DEREF"]

    # get the orig localplus that will be loaded from by the orig bytecode LOAD_DEREF arg_i
    localsplus, localsplus_name_to_idx = get_localsplus_name_to_idx(code)

    # closure vars go into co_freevars
    new_code = code.replace(co_freevars=tuple(new_closure.keys()))
    # closure is a tuple of cells
    closure = tuple(
        make_cell(v) if not isinstance(v, types.CellType) else v
        for v in new_closure.values()
    )

    new_code = ConcreteBytecode.from_code(new_code)
    # update how many closure vars are loaded from frame
    # see https://github.com/python/cpython/blob/6078f2033ea15a16cf52fe8d644a95a3be72d2e3/Python/bytecodes.c#L1571
    assert new_code[0].opcode == COPY_FREE_VARS, f"{new_code[0].opcode=}"
    new_code[0].arg = len(closure)

    # map orig localsplus arg_i to new localplus position/arg_i
    new_localsplus = new_code.varnames + new_code.cellvars + new_code.freevars
    new_localsplus_name_to_idx = {v: i for i, v in enumerate(new_localsplus)}
    for c in new_code:
        if c.opcode == LOAD_DEREF and c.arg < len(localsplus):
            c.arg = new_localsplus_name_to_idx[localsplus[c.arg]]
    new_code = new_code.to_code()

    return new_code, closure


# Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard);
# potentially more complete approach https://stackoverflow.com/a/56901529/9045206
def copy_func(f, new_closure: Dict = None):
    if new_closure is not None:
        code, closure = replace_closure(f.__code__, new_closure)
    else:
        # see https://github.com/cloudpipe/cloudpickle/blob/f111f7ab6d302e9b1e2a568d0e4c574895db6a6e/cloudpickle/cloudpickle.py#L813
        # for how this trick is accomplished (dill and pickle both fail to pickle eg generic typevars)
        closure = cloudpickle.loads(cloudpickle.dumps(f.__closure__))
        code = f.__code__

    g = types.FunctionType(
        code=code,
        globals=f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=closure,
    )
    g.__kwdefaults__ = f.__kwdefaults__
    g.__dict__.update(f.__dict__)
    g = functools.update_wrapper(g, f)

    if inspect.ismethod(f):
        g = bind(g, f.__self__)
    return g


def append_hidden_node(node_body, new_node):
    last_statement = node_body[-1]
    assert (
        last_statement.end_lineno is not None
    ), f"last_statement {ast.unparse(last_statement)} must have end_lineno"
    new_node = ast.fix_missing_locations(
        set_lineno(new_node, last_statement.end_lineno)
    )
    node_body.append(new_node)
    return node_body


def find_func_in_code_object(co, func_name):
    for c in co.co_consts:
        if type(c) is types.CodeType:
            if c.co_name == func_name:
                return c
            else:
                f = find_func_in_code_object(c, func_name)
                if f is not None:
                    return f


_T = TypeVar("_T")


# https://github.com/python/cpython/blob/809aa9a682fc865f7502e7421da0a74d204aab6d/Objects/typevarobject.c#L29
class PyTypeVarObject(PyVarObject[_T, None, None]):
    name: ptr[PyObject]
    # not sure why but this is the only thing that works but that's fine because it's the only thing we need
    bound: ptr[PyObject]
