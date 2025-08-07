# https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from .. import ir
from .ast.py_type import PyTypeObject

nb_meta_cls = type(ir.Value)

_Py_TPFLAGS_BASETYPE = 1 << 10
PyTypeObject.from_object(nb_meta_cls).tp_flags |= _Py_TPFLAGS_BASETYPE
