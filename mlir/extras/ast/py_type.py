from __future__ import annotations

from ctypes import (
    c_char,
    c_char_p,
    c_uint,
    c_ulong,
    pointer,
)
from typing import TypeVar

from einspect.structs.include.descrobject_h import PyGetSetDef, PyMemberDef
from einspect.structs.include.methodobject_h import PyMethodDef
from einspect.structs.include.object_h import (
    destructor,
    getattrfunc,
    setattrfunc,
    PyAsyncMethods,
    reprfunc,
    PyNumberMethods,
    PySequenceMethods,
    PyMappingMethods,
    hashfunc,
    ternaryfunc,
    getattrofunc,
    setattrofunc,
    PyBufferProcs,
    traverseproc,
    inquiry,
    richcmpfunc,
    getiterfunc,
    iternextfunc,
    descrgetfunc,
    descrsetfunc,
    initproc,
    allocfunc,
    newfunc,
    freefunc,
    vectorcallfunc,
)
from einspect.structs.py_object import PyObject, PyVarObject
from einspect.types import ptr
from typing_extensions import Annotated, Self

__all__ = ("PyTypeObject",)

_T = TypeVar("_T")

DEFAULT = object()


# noinspection PyPep8Naming
class PyTypeObject(PyVarObject[_T, None, None]):
    """
    Defines a PyTypeObject Structure.

    https://github.com/python/cpython/blob/3.11/Doc/includes/typestruct.h

    ..
        source: Include/cpython/object.h (struct _typeobject)
    """

    tp_name: Annotated[bytes, c_char_p]
    # For allocation
    tp_basicsize: int
    tp_itemsize: int
    # Methods to implement standard operations
    tp_dealloc: destructor
    tp_vectorcall_offset: int
    tp_getattr: getattrfunc
    tp_setattr: setattrfunc
    # formerly known as tp_compare (Python 2) or tp_reserved (Python 3)
    tp_as_async: ptr[PyAsyncMethods]

    tp_repr: reprfunc

    # Method suites for standard classes
    tp_as_number: ptr[PyNumberMethods]
    tp_as_sequence: ptr[PySequenceMethods]
    tp_as_mapping: ptr[PyMappingMethods]

    # More standard operations (here for binary compatibility)
    tp_hash: hashfunc
    tp_call: ternaryfunc
    tp_str: reprfunc
    tp_getattro: getattrofunc
    tp_setattro: setattrofunc

    # Functions to access object as input/output buffer
    tp_as_buffer: ptr[PyBufferProcs]

    # Flags to define presence of optional/expanded features
    tp_flags: Annotated[int, c_ulong]

    tp_doc: Annotated[bytes, c_char_p]  # Documentation string

    # Assigned meaning in release 2.0
    # call function for all accessible objects
    tp_traverse: traverseproc

    tp_clear: inquiry  # delete references to contained objects

    # Assigned meaning in release 2.1
    # rich comparisons
    tp_richcompare: richcmpfunc

    tp_weaklistoffset: int  # weak reference enabler

    # Iterators
    tp_iter: getiterfunc
    tp_iternext: iternextfunc

    # Attribute descriptor and subclassing stuff
    tp_methods: ptr[PyMethodDef]
    tp_members: ptr[PyMemberDef]
    tp_getset: ptr[PyGetSetDef]

    # Strong reference on a heap type, borrowed reference on a static type
    tp_base: pointer[Self]
    tp_dict: ptr[PyObject]
    tp_descr_get: descrgetfunc
    tp_descr_set: descrsetfunc
    tp_dictoffset: int
    tp_init: initproc
    tp_alloc: allocfunc
    tp_new: newfunc
    tp_free: freefunc  # Low-level free-memory routine
    tp_is_gc: inquiry  # For PyObject_IS_GC
    tp_bases: ptr[PyObject]
    tp_mro: ptr[PyObject]  # method resolution order
    tp_cache: ptr[PyObject]
    tp_subclasses: ptr[PyObject]  # for static builtin types this is an index
    tp_weaklist: ptr[PyObject]
    tp_del: destructor

    # Type attribute cache version tag. Added in version 2.6
    tp_version_tag: Annotated[int, c_uint]

    tp_finalize: destructor
    tp_vectorcall: vectorcallfunc

    # bitset of which type-watchers care about this type
    tp_watched: c_char


# https://github.com/python/cpython/blob/809aa9a682fc865f7502e7421da0a74d204aab6d/Objects/typevarobject.c#L29
class PyTypeVarObject(PyVarObject[_T, None, None]):
    name: ptr[PyObject]
    # not sure why but this is the only thing that works but that's fine because it's the only thing we need
    bound: ptr[PyObject]


if __name__ == "__main__":
    print(PyTypeObject.from_object(type(3.14)))
