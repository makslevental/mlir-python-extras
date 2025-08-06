import sys
from ctypes import (
    POINTER,
    PYFUNCTYPE,
    Structure,
    _Pointer,
    c_char,
    c_char_p,
    c_int,
    c_ssize_t,
    c_uint,
    c_uint16,
    c_uint32,
    c_uint8,
    c_ulong,
    c_void_p,
    cast,
    pointer,
    py_object,
)
from typing import get_origin, _SpecialForm, _GenericAlias


_SelfPtr = object()


@_SpecialForm
def Self(self, parameters):
    raise TypeError(f"{self} is not subscriptable")


class _Ptr(_Pointer):
    def __new__(cls, *args, **kwargs):
        return pointer(*args, **kwargs)

    def __class_getitem__(cls, item):
        """Return a `ctypes.POINTER` of the given type."""
        # For ptr[Self], return a special object
        if item is Self:
            return _SelfPtr

        # Get base of generic alias
        # noinspection PyUnresolvedReferences, PyProtectedMember
        if isinstance(item, _GenericAlias):
            item = get_origin(item)

        try:
            return POINTER(item)
        except TypeError as e:
            raise TypeError(f"{e} (During POINTER({item}))") from e


def address(obj) -> int:
    source = py_object(obj)
    addr = c_void_p.from_buffer(source).value
    if addr is None:
        raise ValueError("address: NULL object")  # pragma: no cover
    return addr


# https://github.com/python/cpython/blob/3.11/Include/object.h#L196-L227
unaryfunc = PYFUNCTYPE(py_object, py_object)
binaryfunc = PYFUNCTYPE(py_object, py_object, py_object)
ternaryfunc = PYFUNCTYPE(py_object, py_object, py_object, py_object)
inquiry = PYFUNCTYPE(c_int, py_object)
lenfunc = PYFUNCTYPE(c_ssize_t, py_object)
ssizeargfunc = PYFUNCTYPE(py_object, py_object, c_ssize_t)
ssizessizeargfunc = PYFUNCTYPE(py_object, py_object, c_ssize_t, c_ssize_t)
ssizeobjargproc = PYFUNCTYPE(c_int, py_object, c_ssize_t, py_object)
ssizessizeobjargproc = PYFUNCTYPE(c_int, py_object, c_ssize_t, c_ssize_t, py_object)
objobjargproc = PYFUNCTYPE(c_int, py_object, py_object, py_object)

objobjproc = PYFUNCTYPE(c_int, py_object, py_object)
visitproc = PYFUNCTYPE(c_int, py_object, c_void_p)
traverseproc = PYFUNCTYPE(c_int, py_object, visitproc, c_void_p)

freefunc = PYFUNCTYPE(None, c_void_p)
destructor = PYFUNCTYPE(None, py_object)
getattrfunc = PYFUNCTYPE(py_object, py_object, c_char_p)
getattrofunc = PYFUNCTYPE(py_object, py_object, py_object)
setattrfunc = PYFUNCTYPE(c_int, py_object, c_char_p, py_object)
setattrofunc = PYFUNCTYPE(c_int, py_object, py_object, py_object)
reprfunc = PYFUNCTYPE(py_object, py_object)
hashfunc = PYFUNCTYPE(c_ssize_t, py_object)
richcmpfunc = PYFUNCTYPE(py_object, py_object, py_object, c_int)
getiterfunc = PYFUNCTYPE(py_object, py_object)
iternextfunc = PYFUNCTYPE(py_object, py_object)

descrgetfunc = PYFUNCTYPE(py_object, py_object, py_object, py_object)
descrsetfunc = PYFUNCTYPE(c_int, py_object, py_object, py_object)
initproc = PYFUNCTYPE(c_int, py_object, py_object, py_object)
newfunc = PYFUNCTYPE(py_object, py_object, py_object, py_object)
allocfunc = PYFUNCTYPE(py_object, py_object, c_ssize_t)

# PyObject *(*vectorcallfunc)(PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames)
vectorcallfunc = PYFUNCTYPE(py_object, py_object, _Ptr[py_object], c_ssize_t, py_object)
# PySendResult (*sendfunc)(PyObject *iter, PyObject *value, PyObject **result)
sendfunc = PYFUNCTYPE(c_int, py_object, py_object, _Ptr[py_object])


def _is_gil_enabled():
    try:
        return sys._is_gil_enabled()
    except:
        return True


if _is_gil_enabled():
    _py_object_fields = [
        ("ob_refcnt", c_ssize_t),
        ("ob_type", _Ptr[py_object]),
    ]
else:
    # https://github.com/python/cpython/blob/main/Include/object.h#L168
    _py_object_fields = [
        ("ob_tid", c_ssize_t),
        ("ob_flags", c_uint16),
        # https://github.com/python/cpython/blob/main/Include/cpython/pylock.h#L29
        ("ob_mutex", c_uint8),
        ("ob_gc_bits", c_uint8),
        ("ob_ref_local", c_uint32),
        ("ob_ref_shared", c_ssize_t),
        ("ob_type", _Ptr[py_object]),
    ]


class PyObject(Structure):
    _fields_ = _py_object_fields

    def as_ref(self) -> _Ptr[Self]:
        """Return a pointer to the Structure."""
        return pointer(self)  # type: ignore

    def into_object(self):
        """Cast the PyObject into a Python object."""
        py_obj = cast(self.as_ref(), py_object)
        return py_obj.value


# https://github.com/python/cpython/blob/3.11/Doc/includes/typestruct.h
class PyTypeObject(Structure):
    _fields_ = _py_object_fields + [
        ("ob_size", c_ssize_t),
        ("tp_name", c_char_p),
        ("tp_basicsize", c_ssize_t),
        ("tp_itemsize", c_ssize_t),
        ("tp_dealloc", destructor),
        ("tp_vectorcall_offset", c_ssize_t),
        ("tp_getattr", getattrfunc),
        ("tp_setattr", setattrfunc),
        # ("tp_as_async", _Ptr[PyAsyncMethods]),
        ("tp_as_async", _Ptr[PyObject]),
        ("tp_repr", reprfunc),
        # ("tp_as_number", _Ptr[PyNumberMethods]),
        ("tp_as_number", _Ptr[PyObject]),
        # ("tp_as_sequence", _Ptr[PySequenceMethods]),
        ("tp_as_sequence", _Ptr[PyObject]),
        # ("tp_as_mapping", _Ptr[PyMappingMethods]),
        ("tp_as_mapping", _Ptr[PyObject]),
        ("tp_hash", hashfunc),
        ("tp_call", ternaryfunc),
        ("tp_str", reprfunc),
        ("tp_getattro", getattrofunc),
        ("tp_setattro", setattrofunc),
        # ("tp_as_buffer", _Ptr[PyBufferProcs]),
        ("tp_as_buffer", _Ptr[PyObject]),
        ("tp_flags", c_ulong),
        ("tp_doc", c_char_p),
        ("tp_traverse", traverseproc),
        ("tp_clear", inquiry),
        ("tp_richcompare", richcmpfunc),
        ("tp_weaklistoffset", c_ssize_t),
        ("tp_iter", getiterfunc),
        ("tp_iternext", iternextfunc),
        ("tp_methods", _Ptr[PyObject]),
        ("tp_members", _Ptr[PyObject]),
        ("tp_getset", _Ptr[PyObject]),
        # ("tp_base", _Ptr[PyTypeObject]),
        ("tp_base", _Ptr[PyObject]),
        ("tp_dict", _Ptr[PyObject]),
        ("tp_descr_get", descrgetfunc),
        ("tp_descr_set", descrsetfunc),
        ("tp_dictoffset", c_ssize_t),
        ("tp_init", initproc),
        ("tp_alloc", allocfunc),
        ("tp_new", newfunc),
        ("tp_free", freefunc),
        ("tp_is_gc", inquiry),
        ("tp_bases", _Ptr[PyObject]),
        ("tp_mro", _Ptr[PyObject]),
        ("tp_cache", _Ptr[PyObject]),
        ("tp_subclasses", _Ptr[PyObject]),
        ("tp_weaklist", _Ptr[PyObject]),
        ("tp_del", destructor),
        ("tp_version_tag", c_uint),
        ("tp_finalize", descrsetfunc),
        ("tp_vectorcall", vectorcallfunc),
        ("tp_watched", c_char),
    ]

    @classmethod
    def from_object(cls, obj) -> Self:
        """Create a PyObject from an object."""
        return cls.from_address(address(obj))


class PyTypeVarObject(Structure):
    _fields_ = _py_object_fields + [
        ("ob_size", c_ssize_t),
        ("name", _Ptr[PyObject]),
        # not sure why but this is the only thing that works but that's fine because it's the only thing we need
        ("bound", _Ptr[PyObject]),
    ]

    @classmethod
    def from_object(cls, obj) -> Self:
        """Create a PyObject from an object."""
        return cls.from_address(address(obj))


if __name__ == "__main__":
    # for k, v in PyObject._fields_:
    #     print(f"('{k}', {v.__name__}),")
    # for k, v in PyVarObject._fields_:
    #     print(f"('{k}', {v.__name__}),")
    # for k, v in PyTypeObject._fields_:
    #     print(f"('{k}', {v.__name__}),")
    print(PyTypeObject.from_object(type(3.14)).tp_name)
    print(PyTypeObject.from_object(type(3.14)).tp_base)
    print(PyTypeObject.from_object(type(3.14)).ob_type)
    for k, v in PyTypeVarObject._fields_:
        print(f"('{k}', {v.__name__}),")
