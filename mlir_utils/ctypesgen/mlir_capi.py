r"""Wrapper for ctypesgen

Generated with:
/Users/mlevental/dev_projects/mlir-utils/mlir_utils/ctypesgen/generate.py

Do not modify this file.
"""

__docformat__ = "restructuredtext"

# Begin preamble for Python

import ctypes
import sys
from ctypes import *  # noqa: F401, F403

_int_types = (ctypes.c_int16, ctypes.c_int32)
if hasattr(ctypes, "c_int64"):
    # Some builds of ctypes apparently do not have ctypes.c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (ctypes.c_int64,)
for t in _int_types:
    if ctypes.sizeof(t) == ctypes.sizeof(ctypes.c_size_t):
        c_ptrdiff_t = t
del t
del _int_types



class UserString:
    def __init__(self, seq):
        if isinstance(seq, bytes):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq).encode()

    def __bytes__(self):
        return self.data

    def __str__(self):
        return self.data.decode()

    def __repr__(self):
        return repr(self.data)

    def __int__(self):
        return int(self.data.decode())

    def __long__(self):
        return int(self.data.decode())

    def __float__(self):
        return float(self.data.decode())

    def __complex__(self):
        return complex(self.data.decode())

    def __hash__(self):
        return hash(self.data)

    def __le__(self, string):
        if isinstance(string, UserString):
            return self.data <= string.data
        else:
            return self.data <= string

    def __lt__(self, string):
        if isinstance(string, UserString):
            return self.data < string.data
        else:
            return self.data < string

    def __ge__(self, string):
        if isinstance(string, UserString):
            return self.data >= string.data
        else:
            return self.data >= string

    def __gt__(self, string):
        if isinstance(string, UserString):
            return self.data > string.data
        else:
            return self.data > string

    def __eq__(self, string):
        if isinstance(string, UserString):
            return self.data == string.data
        else:
            return self.data == string

    def __ne__(self, string):
        if isinstance(string, UserString):
            return self.data != string.data
        else:
            return self.data != string

    def __contains__(self, char):
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, bytes):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other).encode())

    def __radd__(self, other):
        if isinstance(other, bytes):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other).encode() + self.data)

    def __mul__(self, n):
        return self.__class__(self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.__class__(self.data.capitalize())

    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))

    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(sub, start, end)

    def decode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())

    def encode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))

    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))

    def lower(self):
        return self.__class__(self.data.lower())

    def lstrip(self, chars=None):
        return self.__class__(self.data.lstrip(chars))

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.__class__(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=0):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.__class__(self.data.strip(chars))

    def swapcase(self):
        return self.__class__(self.data.swapcase())

    def title(self):
        return self.__class__(self.data.title())

    def translate(self, *args):
        return self.__class__(self.data.translate(*args))

    def upper(self):
        return self.__class__(self.data.upper())

    def zfill(self, width):
        return self.__class__(self.data.zfill(width))


class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""

    def __init__(self, string=""):
        self.data = string

    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")

    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + sub + self.data[index + 1 :]

    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + self.data[index + 1 :]

    def __setslice__(self, start, end, sub):
        start = max(start, 0)
        end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start] + sub.data + self.data[end:]
        elif isinstance(sub, bytes):
            self.data = self.data[:start] + sub + self.data[end:]
        else:
            self.data = self.data[:start] + str(sub).encode() + self.data[end:]

    def __delslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]

    def immutable(self):
        return UserString(self.data)

    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, bytes):
            self.data += other
        else:
            self.data += str(other).encode()
        return self

    def __imul__(self, n):
        self.data *= n
        return self


class String(MutableString, ctypes.Union):

    _fields_ = [("raw", ctypes.POINTER(ctypes.c_char)), ("data", ctypes.c_char_p)]

    def __init__(self, obj=b""):
        if isinstance(obj, (bytes, UserString)):
            self.data = bytes(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(ctypes.POINTER(ctypes.c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from bytes
        elif isinstance(obj, bytes):
            return cls(obj)

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj.encode())

        # Convert from c_char_p
        elif isinstance(obj, ctypes.c_char_p):
            return obj

        # Convert from POINTER(ctypes.c_char)
        elif isinstance(obj, ctypes.POINTER(ctypes.c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(ctypes.cast(obj, ctypes.POINTER(ctypes.c_char)))

        # Convert from ctypes.c_char array
        elif isinstance(obj, ctypes.c_char * len(obj)):
            return obj

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)

    from_param = classmethod(from_param)


def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)


# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to ctypes.c_void_p.
def UNCHECKED(type):
    if hasattr(type, "_type_") and isinstance(type._type_, str) and type._type_ != "P":
        return type
    else:
        return ctypes.c_void_p


# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self, func, restype, argtypes, errcheck):
        self.func = func
        self.func.restype = restype
        self.argtypes = argtypes
        if errcheck:
            self.func.errcheck = errcheck

    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func

    def __call__(self, *args):
        fixed_args = []
        i = 0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i += 1
        return self.func(*fixed_args + list(args[i:]))


def ord_if_char(value):
    """
    Simple helper used for casts to simple builtin types:  if the argument is a
    string type, it will be converted to it's ordinal value.

    This function will raise an exception if the argument is string with more
    than one characters.
    """
    return ord(value) if (isinstance(value, bytes) or isinstance(value, str)) else value

# End preamble

_libs = {}
_libdirs = ['/Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs']

# Begin loader

"""
Load libraries - appropriately for all our supported platforms
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import ctypes
import ctypes.util
import glob
import os.path
import platform
import re
import sys


def _environ_path(name):
    """Split an environment variable into a path-like list elements"""
    if name in os.environ:
        return os.environ[name].split(":")
    return []


class LibraryLoader:
    """
    A base class For loading of libraries ;-)
    Subclasses load libraries for specific platforms.
    """

    # library names formatted specifically for platforms
    name_formats = ["%s"]

    class Lookup:
        """Looking up calling conventions for a platform"""

        mode = ctypes.DEFAULT_MODE

        def __init__(self, path):
            super(LibraryLoader.Lookup, self).__init__()
            self.access = dict(cdecl=ctypes.CDLL(path, self.mode))

        def get(self, name, calling_convention="cdecl"):
            """Return the given name according to the selected calling convention"""
            if calling_convention not in self.access:
                raise LookupError(
                    "Unknown calling convention '{}' for function '{}'".format(
                        calling_convention, name
                    )
                )
            return getattr(self.access[calling_convention], name)

        def has(self, name, calling_convention="cdecl"):
            """Return True if this given calling convention finds the given 'name'"""
            if calling_convention not in self.access:
                return False
            return hasattr(self.access[calling_convention], name)

        def __getattr__(self, name):
            return getattr(self.access["cdecl"], name)

    def __init__(self):
        self.other_dirs = []

    def __call__(self, libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            # noinspection PyBroadException
            try:
                return self.Lookup(path)
            except Exception:  # pylint: disable=broad-except
                pass

        raise ImportError("Could not load %s." % libname)

    def getpaths(self, libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # search through a prioritized series of locations for the library

            # we first search any specific directories identified by user
            for dir_i in self.other_dirs:
                for fmt in self.name_formats:
                    # dir_i should be absolute already
                    yield os.path.join(dir_i, fmt % libname)

            # check if this code is even stored in a physical file
            try:
                this_file = __file__
            except NameError:
                this_file = None

            # then we search the directory where the generated python interface is stored
            if this_file is not None:
                for fmt in self.name_formats:
                    yield os.path.abspath(os.path.join(os.path.dirname(__file__), fmt % libname))

            # now, use the ctypes tools to try to find the library
            for fmt in self.name_formats:
                path = ctypes.util.find_library(fmt % libname)
                if path:
                    yield path

            # then we search all paths identified as platform-specific lib paths
            for path in self.getplatformpaths(libname):
                yield path

            # Finally, we'll try the users current working directory
            for fmt in self.name_formats:
                yield os.path.abspath(os.path.join(os.path.curdir, fmt % libname))

    def getplatformpaths(self, _libname):  # pylint: disable=no-self-use
        """Return all the library paths available in this platform"""
        return []


# Darwin (Mac OS X)


class DarwinLibraryLoader(LibraryLoader):
    """Library loader for MacOS"""

    name_formats = [
        "lib%s.dylib",
        "lib%s.so",
        "lib%s.bundle",
        "%s.dylib",
        "%s.so",
        "%s.bundle",
        "%s",
    ]

    class Lookup(LibraryLoader.Lookup):
        """
        Looking up library files for this platform (Darwin aka MacOS)
        """

        # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
        # of the default RTLD_LOCAL.  Without this, you end up with
        # libraries not being loadable, resulting in "Symbol not found"
        # errors
        mode = ctypes.RTLD_GLOBAL

    def getplatformpaths(self, libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [fmt % libname for fmt in self.name_formats]

        for directory in self.getdirs(libname):
            for name in names:
                yield os.path.join(directory, name)

    @staticmethod
    def getdirs(libname):
        """Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        """

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [
                os.path.expanduser("~/lib"),
                "/usr/local/lib",
                "/usr/lib",
            ]

        dirs = []

        if "/" in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
            dirs.extend(_environ_path("LD_RUN_PATH"))

        if hasattr(sys, "frozen") and getattr(sys, "frozen") == "macosx_app":
            dirs.append(os.path.join(os.environ["RESOURCEPATH"], "..", "Frameworks"))

        dirs.extend(dyld_fallback_library_path)

        return dirs


# Posix


class PosixLibraryLoader(LibraryLoader):
    """Library loader for POSIX-like systems (including Linux)"""

    _ld_so_cache = None

    _include = re.compile(r"^\s*include\s+(?P<pattern>.*)")

    name_formats = ["lib%s.so", "%s.so", "%s"]

    class _Directories(dict):
        """Deal with directories"""

        def __init__(self):
            dict.__init__(self)
            self.order = 0

        def add(self, directory):
            """Add a directory to our current set of directories"""
            if len(directory) > 1:
                directory = directory.rstrip(os.path.sep)
            # only adds and updates order if exists and not already in set
            if not os.path.exists(directory):
                return
            order = self.setdefault(directory, self.order)
            if order == self.order:
                self.order += 1

        def extend(self, directories):
            """Add a list of directories to our set"""
            for a_dir in directories:
                self.add(a_dir)

        def ordered(self):
            """Sort the list of directories"""
            return (i[0] for i in sorted(self.items(), key=lambda d: d[1]))

    def _get_ld_so_conf_dirs(self, conf, dirs):
        """
        Recursive function to help parse all ld.so.conf files, including proper
        handling of the `include` directive.
        """

        try:
            with open(conf) as fileobj:
                for dirname in fileobj:
                    dirname = dirname.strip()
                    if not dirname:
                        continue

                    match = self._include.match(dirname)
                    if not match:
                        dirs.add(dirname)
                    else:
                        for dir2 in glob.glob(match.group("pattern")):
                            self._get_ld_so_conf_dirs(dir2, dirs)
        except IOError:
            pass

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = self._Directories()
        for name in (
            "LD_LIBRARY_PATH",
            "SHLIB_PATH",  # HP-UX
            "LIBPATH",  # OS/2, AIX
            "LIBRARY_PATH",  # BE/OS
        ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))

        self._get_ld_so_conf_dirs("/etc/ld.so.conf", directories)

        bitage = platform.architecture()[0]

        unix_lib_dirs_list = []
        if bitage.startswith("64"):
            # prefer 64 bit if that is our arch
            unix_lib_dirs_list += ["/lib64", "/usr/lib64"]

        # must include standard libs, since those paths are also used by 64 bit
        # installs
        unix_lib_dirs_list += ["/lib", "/usr/lib"]
        if sys.platform.startswith("linux"):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            if bitage.startswith("32"):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ["/lib/i386-linux-gnu", "/usr/lib/i386-linux-gnu"]
            elif bitage.startswith("64"):
                # Assume Intel/AMD x86 compatible
                unix_lib_dirs_list += [
                    "/lib/x86_64-linux-gnu",
                    "/usr/lib/x86_64-linux-gnu",
                ]
            else:
                # guess...
                unix_lib_dirs_list += glob.glob("/lib/*linux-gnu")
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r"lib(.*)\.s[ol]")
        # ext_re = re.compile(r"\.s[ol]$")
        for our_dir in directories.ordered():
            try:
                for path in glob.glob("%s/*.s[ol]*" % our_dir):
                    file = os.path.basename(path)

                    # Index by filename
                    cache_i = cache.setdefault(file, set())
                    cache_i.add(path)

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        cache_i = cache.setdefault(library, set())
                        cache_i.add(path)
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname, set())
        for i in result:
            # we iterate through all found paths for library, since we may have
            # actually found multiple architectures or other library types that
            # may not load
            yield i


# Windows


class WindowsLibraryLoader(LibraryLoader):
    """Library loader for Microsoft Windows"""

    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll", "%s"]

    class Lookup(LibraryLoader.Lookup):
        """Lookup class for Windows libraries..."""

        def __init__(self, path):
            super(WindowsLibraryLoader.Lookup, self).__init__(path)
            self.access["stdcall"] = ctypes.windll.LoadLibrary(path)


# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin": DarwinLibraryLoader,
    "cygwin": WindowsLibraryLoader,
    "win32": WindowsLibraryLoader,
    "msys": WindowsLibraryLoader,
}

load_library = loaderclass.get(sys.platform, PosixLibraryLoader)()


def add_library_search_dirs(other_dirs):
    """
    Add libraries to search paths.
    If library paths are relative, convert them to absolute with respect to this
    file's directory
    """
    for path in other_dirs:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        load_library.other_dirs.append(path)


del loaderclass

# End loader

add_library_search_dirs(['/Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs'])

# Begin libraries
_libs["MLIRPythonCAPI"] = load_library("MLIRPythonCAPI")

# 1 libraries
# End libraries

# No modules

uint8_t = c_ubyte# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint8_t.h: 31

uint16_t = c_ushort# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint16_t.h: 31

uint32_t = c_uint# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint32_t.h: 31

uint64_t = c_ulonglong# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint64_t.h: 31

__darwin_intptr_t = c_long# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/arm/_types.h: 27

intptr_t = __darwin_intptr_t# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_intptr_t.h: 32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 60
class struct_MlirLlvmThreadPool(Structure):
    pass

struct_MlirLlvmThreadPool.__slots__ = [
    'ptr',
]
struct_MlirLlvmThreadPool._fields_ = [
    ('ptr', POINTER(None)),
]

MlirLlvmThreadPool = struct_MlirLlvmThreadPool# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 60

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 61
class struct_MlirTypeID(Structure):
    pass

struct_MlirTypeID.__slots__ = [
    'ptr',
]
struct_MlirTypeID._fields_ = [
    ('ptr', POINTER(None)),
]

MlirTypeID = struct_MlirTypeID# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 61

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 62
class struct_MlirTypeIDAllocator(Structure):
    pass

struct_MlirTypeIDAllocator.__slots__ = [
    'ptr',
]
struct_MlirTypeIDAllocator._fields_ = [
    ('ptr', POINTER(None)),
]

MlirTypeIDAllocator = struct_MlirTypeIDAllocator# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 62

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 73
class struct_MlirStringRef(Structure):
    pass

struct_MlirStringRef.__slots__ = [
    'data',
    'length',
]
struct_MlirStringRef._fields_ = [
    ('data', String),
    ('length', c_size_t),
]

MlirStringRef = struct_MlirStringRef# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 77

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 84
for _lib in _libs.values():
    try:
        result = (MlirStringRef).in_dll(_lib, "result")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 93
if _libs["MLIRPythonCAPI"].has("mlirStringRefCreateFromCString", "cdecl"):
    mlirStringRefCreateFromCString = _libs["MLIRPythonCAPI"].get("mlirStringRefCreateFromCString", "cdecl")
    mlirStringRefCreateFromCString.argtypes = [String]
    mlirStringRefCreateFromCString.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 96
if _libs["MLIRPythonCAPI"].has("mlirStringRefEqual", "cdecl"):
    mlirStringRefEqual = _libs["MLIRPythonCAPI"].get("mlirStringRefEqual", "cdecl")
    mlirStringRefEqual.argtypes = [MlirStringRef, MlirStringRef]
    mlirStringRefEqual.restype = c_bool

MlirStringCallback = CFUNCTYPE(UNCHECKED(None), MlirStringRef, POINTER(None))# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 105

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 116
class struct_MlirLogicalResult(Structure):
    pass

struct_MlirLogicalResult.__slots__ = [
    'value',
]
struct_MlirLogicalResult._fields_ = [
    ('value', c_int8),
]

MlirLogicalResult = struct_MlirLogicalResult# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 119

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 133
for _lib in _libs.values():
    try:
        res = (MlirLogicalResult).in_dll(_lib, "res")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 139
for _lib in _libs.values():
    try:
        res = (MlirLogicalResult).in_dll(_lib, "res")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 149
if _libs["MLIRPythonCAPI"].has("mlirLlvmThreadPoolCreate", "cdecl"):
    mlirLlvmThreadPoolCreate = _libs["MLIRPythonCAPI"].get("mlirLlvmThreadPoolCreate", "cdecl")
    mlirLlvmThreadPoolCreate.argtypes = []
    mlirLlvmThreadPoolCreate.restype = MlirLlvmThreadPool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 152
if _libs["MLIRPythonCAPI"].has("mlirLlvmThreadPoolDestroy", "cdecl"):
    mlirLlvmThreadPoolDestroy = _libs["MLIRPythonCAPI"].get("mlirLlvmThreadPoolDestroy", "cdecl")
    mlirLlvmThreadPoolDestroy.argtypes = [MlirLlvmThreadPool]
    mlirLlvmThreadPoolDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 160
if _libs["MLIRPythonCAPI"].has("mlirTypeIDCreate", "cdecl"):
    mlirTypeIDCreate = _libs["MLIRPythonCAPI"].get("mlirTypeIDCreate", "cdecl")
    mlirTypeIDCreate.argtypes = [POINTER(None)]
    mlirTypeIDCreate.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 166
if _libs["MLIRPythonCAPI"].has("mlirTypeIDEqual", "cdecl"):
    mlirTypeIDEqual = _libs["MLIRPythonCAPI"].get("mlirTypeIDEqual", "cdecl")
    mlirTypeIDEqual.argtypes = [MlirTypeID, MlirTypeID]
    mlirTypeIDEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 169
if _libs["MLIRPythonCAPI"].has("mlirTypeIDHashValue", "cdecl"):
    mlirTypeIDHashValue = _libs["MLIRPythonCAPI"].get("mlirTypeIDHashValue", "cdecl")
    mlirTypeIDHashValue.argtypes = [MlirTypeID]
    mlirTypeIDHashValue.restype = c_size_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 176
if _libs["MLIRPythonCAPI"].has("mlirTypeIDAllocatorCreate", "cdecl"):
    mlirTypeIDAllocatorCreate = _libs["MLIRPythonCAPI"].get("mlirTypeIDAllocatorCreate", "cdecl")
    mlirTypeIDAllocatorCreate.argtypes = []
    mlirTypeIDAllocatorCreate.restype = MlirTypeIDAllocator

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 180
if _libs["MLIRPythonCAPI"].has("mlirTypeIDAllocatorDestroy", "cdecl"):
    mlirTypeIDAllocatorDestroy = _libs["MLIRPythonCAPI"].get("mlirTypeIDAllocatorDestroy", "cdecl")
    mlirTypeIDAllocatorDestroy.argtypes = [MlirTypeIDAllocator]
    mlirTypeIDAllocatorDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 184
if _libs["MLIRPythonCAPI"].has("mlirTypeIDAllocatorAllocateTypeID", "cdecl"):
    mlirTypeIDAllocatorAllocateTypeID = _libs["MLIRPythonCAPI"].get("mlirTypeIDAllocatorAllocateTypeID", "cdecl")
    mlirTypeIDAllocatorAllocateTypeID.argtypes = [MlirTypeIDAllocator]
    mlirTypeIDAllocatorAllocateTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 51
class struct_MlirBytecodeWriterConfig(Structure):
    pass

struct_MlirBytecodeWriterConfig.__slots__ = [
    'ptr',
]
struct_MlirBytecodeWriterConfig._fields_ = [
    ('ptr', POINTER(None)),
]

MlirBytecodeWriterConfig = struct_MlirBytecodeWriterConfig# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 51

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 52
class struct_MlirContext(Structure):
    pass

struct_MlirContext.__slots__ = [
    'ptr',
]
struct_MlirContext._fields_ = [
    ('ptr', POINTER(None)),
]

MlirContext = struct_MlirContext# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 52

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 53
class struct_MlirDialect(Structure):
    pass

struct_MlirDialect.__slots__ = [
    'ptr',
]
struct_MlirDialect._fields_ = [
    ('ptr', POINTER(None)),
]

MlirDialect = struct_MlirDialect# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 53

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 54
class struct_MlirDialectRegistry(Structure):
    pass

struct_MlirDialectRegistry.__slots__ = [
    'ptr',
]
struct_MlirDialectRegistry._fields_ = [
    ('ptr', POINTER(None)),
]

MlirDialectRegistry = struct_MlirDialectRegistry# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 54

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 55
class struct_MlirOperation(Structure):
    pass

struct_MlirOperation.__slots__ = [
    'ptr',
]
struct_MlirOperation._fields_ = [
    ('ptr', POINTER(None)),
]

MlirOperation = struct_MlirOperation# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 55

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 56
class struct_MlirOpOperand(Structure):
    pass

struct_MlirOpOperand.__slots__ = [
    'ptr',
]
struct_MlirOpOperand._fields_ = [
    ('ptr', POINTER(None)),
]

MlirOpOperand = struct_MlirOpOperand# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 56

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 57
class struct_MlirOpPrintingFlags(Structure):
    pass

struct_MlirOpPrintingFlags.__slots__ = [
    'ptr',
]
struct_MlirOpPrintingFlags._fields_ = [
    ('ptr', POINTER(None)),
]

MlirOpPrintingFlags = struct_MlirOpPrintingFlags# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 57

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 58
class struct_MlirBlock(Structure):
    pass

struct_MlirBlock.__slots__ = [
    'ptr',
]
struct_MlirBlock._fields_ = [
    ('ptr', POINTER(None)),
]

MlirBlock = struct_MlirBlock# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 58

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 59
class struct_MlirRegion(Structure):
    pass

struct_MlirRegion.__slots__ = [
    'ptr',
]
struct_MlirRegion._fields_ = [
    ('ptr', POINTER(None)),
]

MlirRegion = struct_MlirRegion# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 59

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 60
class struct_MlirSymbolTable(Structure):
    pass

struct_MlirSymbolTable.__slots__ = [
    'ptr',
]
struct_MlirSymbolTable._fields_ = [
    ('ptr', POINTER(None)),
]

MlirSymbolTable = struct_MlirSymbolTable# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 60

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 62
class struct_MlirAttribute(Structure):
    pass

struct_MlirAttribute.__slots__ = [
    'ptr',
]
struct_MlirAttribute._fields_ = [
    ('ptr', POINTER(None)),
]

MlirAttribute = struct_MlirAttribute# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 62

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 63
class struct_MlirIdentifier(Structure):
    pass

struct_MlirIdentifier.__slots__ = [
    'ptr',
]
struct_MlirIdentifier._fields_ = [
    ('ptr', POINTER(None)),
]

MlirIdentifier = struct_MlirIdentifier# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 63

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 64
class struct_MlirLocation(Structure):
    pass

struct_MlirLocation.__slots__ = [
    'ptr',
]
struct_MlirLocation._fields_ = [
    ('ptr', POINTER(None)),
]

MlirLocation = struct_MlirLocation# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 65
class struct_MlirModule(Structure):
    pass

struct_MlirModule.__slots__ = [
    'ptr',
]
struct_MlirModule._fields_ = [
    ('ptr', POINTER(None)),
]

MlirModule = struct_MlirModule# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 65

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 66
class struct_MlirType(Structure):
    pass

struct_MlirType.__slots__ = [
    'ptr',
]
struct_MlirType._fields_ = [
    ('ptr', POINTER(None)),
]

MlirType = struct_MlirType# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 66

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 67
class struct_MlirValue(Structure):
    pass

struct_MlirValue.__slots__ = [
    'ptr',
]
struct_MlirValue._fields_ = [
    ('ptr', POINTER(None)),
]

MlirValue = struct_MlirValue# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 67

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 76
class struct_MlirNamedAttribute(Structure):
    pass

struct_MlirNamedAttribute.__slots__ = [
    'name',
    'attribute',
]
struct_MlirNamedAttribute._fields_ = [
    ('name', MlirIdentifier),
    ('attribute', MlirAttribute),
]

MlirNamedAttribute = struct_MlirNamedAttribute# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 80

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 88
if _libs["MLIRPythonCAPI"].has("mlirContextCreate", "cdecl"):
    mlirContextCreate = _libs["MLIRPythonCAPI"].get("mlirContextCreate", "cdecl")
    mlirContextCreate.argtypes = []
    mlirContextCreate.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 93
if _libs["MLIRPythonCAPI"].has("mlirContextCreateWithThreading", "cdecl"):
    mlirContextCreateWithThreading = _libs["MLIRPythonCAPI"].get("mlirContextCreateWithThreading", "cdecl")
    mlirContextCreateWithThreading.argtypes = [c_bool]
    mlirContextCreateWithThreading.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 97
if _libs["MLIRPythonCAPI"].has("mlirContextCreateWithRegistry", "cdecl"):
    mlirContextCreateWithRegistry = _libs["MLIRPythonCAPI"].get("mlirContextCreateWithRegistry", "cdecl")
    mlirContextCreateWithRegistry.argtypes = [MlirDialectRegistry, c_bool]
    mlirContextCreateWithRegistry.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 101
if _libs["MLIRPythonCAPI"].has("mlirContextEqual", "cdecl"):
    mlirContextEqual = _libs["MLIRPythonCAPI"].get("mlirContextEqual", "cdecl")
    mlirContextEqual.argtypes = [MlirContext, MlirContext]
    mlirContextEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 109
if _libs["MLIRPythonCAPI"].has("mlirContextDestroy", "cdecl"):
    mlirContextDestroy = _libs["MLIRPythonCAPI"].get("mlirContextDestroy", "cdecl")
    mlirContextDestroy.argtypes = [MlirContext]
    mlirContextDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 113
if _libs["MLIRPythonCAPI"].has("mlirContextSetAllowUnregisteredDialects", "cdecl"):
    mlirContextSetAllowUnregisteredDialects = _libs["MLIRPythonCAPI"].get("mlirContextSetAllowUnregisteredDialects", "cdecl")
    mlirContextSetAllowUnregisteredDialects.argtypes = [MlirContext, c_bool]
    mlirContextSetAllowUnregisteredDialects.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 117
if _libs["MLIRPythonCAPI"].has("mlirContextGetAllowUnregisteredDialects", "cdecl"):
    mlirContextGetAllowUnregisteredDialects = _libs["MLIRPythonCAPI"].get("mlirContextGetAllowUnregisteredDialects", "cdecl")
    mlirContextGetAllowUnregisteredDialects.argtypes = [MlirContext]
    mlirContextGetAllowUnregisteredDialects.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 122
if _libs["MLIRPythonCAPI"].has("mlirContextGetNumRegisteredDialects", "cdecl"):
    mlirContextGetNumRegisteredDialects = _libs["MLIRPythonCAPI"].get("mlirContextGetNumRegisteredDialects", "cdecl")
    mlirContextGetNumRegisteredDialects.argtypes = [MlirContext]
    mlirContextGetNumRegisteredDialects.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 127
if _libs["MLIRPythonCAPI"].has("mlirContextAppendDialectRegistry", "cdecl"):
    mlirContextAppendDialectRegistry = _libs["MLIRPythonCAPI"].get("mlirContextAppendDialectRegistry", "cdecl")
    mlirContextAppendDialectRegistry.argtypes = [MlirContext, MlirDialectRegistry]
    mlirContextAppendDialectRegistry.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 132
if _libs["MLIRPythonCAPI"].has("mlirContextGetNumLoadedDialects", "cdecl"):
    mlirContextGetNumLoadedDialects = _libs["MLIRPythonCAPI"].get("mlirContextGetNumLoadedDialects", "cdecl")
    mlirContextGetNumLoadedDialects.argtypes = [MlirContext]
    mlirContextGetNumLoadedDialects.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 139
if _libs["MLIRPythonCAPI"].has("mlirContextGetOrLoadDialect", "cdecl"):
    mlirContextGetOrLoadDialect = _libs["MLIRPythonCAPI"].get("mlirContextGetOrLoadDialect", "cdecl")
    mlirContextGetOrLoadDialect.argtypes = [MlirContext, MlirStringRef]
    mlirContextGetOrLoadDialect.restype = MlirDialect

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 143
if _libs["MLIRPythonCAPI"].has("mlirContextEnableMultithreading", "cdecl"):
    mlirContextEnableMultithreading = _libs["MLIRPythonCAPI"].get("mlirContextEnableMultithreading", "cdecl")
    mlirContextEnableMultithreading.argtypes = [MlirContext, c_bool]
    mlirContextEnableMultithreading.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 149
if _libs["MLIRPythonCAPI"].has("mlirContextLoadAllAvailableDialects", "cdecl"):
    mlirContextLoadAllAvailableDialects = _libs["MLIRPythonCAPI"].get("mlirContextLoadAllAvailableDialects", "cdecl")
    mlirContextLoadAllAvailableDialects.argtypes = [MlirContext]
    mlirContextLoadAllAvailableDialects.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 155
if _libs["MLIRPythonCAPI"].has("mlirContextIsRegisteredOperation", "cdecl"):
    mlirContextIsRegisteredOperation = _libs["MLIRPythonCAPI"].get("mlirContextIsRegisteredOperation", "cdecl")
    mlirContextIsRegisteredOperation.argtypes = [MlirContext, MlirStringRef]
    mlirContextIsRegisteredOperation.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 162
if _libs["MLIRPythonCAPI"].has("mlirContextSetThreadPool", "cdecl"):
    mlirContextSetThreadPool = _libs["MLIRPythonCAPI"].get("mlirContextSetThreadPool", "cdecl")
    mlirContextSetThreadPool.argtypes = [MlirContext, MlirLlvmThreadPool]
    mlirContextSetThreadPool.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 170
if _libs["MLIRPythonCAPI"].has("mlirDialectGetContext", "cdecl"):
    mlirDialectGetContext = _libs["MLIRPythonCAPI"].get("mlirDialectGetContext", "cdecl")
    mlirDialectGetContext.argtypes = [MlirDialect]
    mlirDialectGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 179
if _libs["MLIRPythonCAPI"].has("mlirDialectEqual", "cdecl"):
    mlirDialectEqual = _libs["MLIRPythonCAPI"].get("mlirDialectEqual", "cdecl")
    mlirDialectEqual.argtypes = [MlirDialect, MlirDialect]
    mlirDialectEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 183
if _libs["MLIRPythonCAPI"].has("mlirDialectGetNamespace", "cdecl"):
    mlirDialectGetNamespace = _libs["MLIRPythonCAPI"].get("mlirDialectGetNamespace", "cdecl")
    mlirDialectGetNamespace.argtypes = [MlirDialect]
    mlirDialectGetNamespace.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 201
class struct_MlirDialectHandle(Structure):
    pass

struct_MlirDialectHandle.__slots__ = [
    'ptr',
]
struct_MlirDialectHandle._fields_ = [
    ('ptr', POINTER(None)),
]

MlirDialectHandle = struct_MlirDialectHandle# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 204

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 212
if _libs["MLIRPythonCAPI"].has("mlirDialectHandleGetNamespace", "cdecl"):
    mlirDialectHandleGetNamespace = _libs["MLIRPythonCAPI"].get("mlirDialectHandleGetNamespace", "cdecl")
    mlirDialectHandleGetNamespace.argtypes = [MlirDialectHandle]
    mlirDialectHandleGetNamespace.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 216
if _libs["MLIRPythonCAPI"].has("mlirDialectHandleInsertDialect", "cdecl"):
    mlirDialectHandleInsertDialect = _libs["MLIRPythonCAPI"].get("mlirDialectHandleInsertDialect", "cdecl")
    mlirDialectHandleInsertDialect.argtypes = [MlirDialectHandle, MlirDialectRegistry]
    mlirDialectHandleInsertDialect.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 220
if _libs["MLIRPythonCAPI"].has("mlirDialectHandleRegisterDialect", "cdecl"):
    mlirDialectHandleRegisterDialect = _libs["MLIRPythonCAPI"].get("mlirDialectHandleRegisterDialect", "cdecl")
    mlirDialectHandleRegisterDialect.argtypes = [MlirDialectHandle, MlirContext]
    mlirDialectHandleRegisterDialect.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 224
if _libs["MLIRPythonCAPI"].has("mlirDialectHandleLoadDialect", "cdecl"):
    mlirDialectHandleLoadDialect = _libs["MLIRPythonCAPI"].get("mlirDialectHandleLoadDialect", "cdecl")
    mlirDialectHandleLoadDialect.argtypes = [MlirDialectHandle, MlirContext]
    mlirDialectHandleLoadDialect.restype = MlirDialect

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 232
if _libs["MLIRPythonCAPI"].has("mlirDialectRegistryCreate", "cdecl"):
    mlirDialectRegistryCreate = _libs["MLIRPythonCAPI"].get("mlirDialectRegistryCreate", "cdecl")
    mlirDialectRegistryCreate.argtypes = []
    mlirDialectRegistryCreate.restype = MlirDialectRegistry

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 241
if _libs["MLIRPythonCAPI"].has("mlirDialectRegistryDestroy", "cdecl"):
    mlirDialectRegistryDestroy = _libs["MLIRPythonCAPI"].get("mlirDialectRegistryDestroy", "cdecl")
    mlirDialectRegistryDestroy.argtypes = [MlirDialectRegistry]
    mlirDialectRegistryDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 249
if _libs["MLIRPythonCAPI"].has("mlirLocationGetAttribute", "cdecl"):
    mlirLocationGetAttribute = _libs["MLIRPythonCAPI"].get("mlirLocationGetAttribute", "cdecl")
    mlirLocationGetAttribute.argtypes = [MlirLocation]
    mlirLocationGetAttribute.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 253
if _libs["MLIRPythonCAPI"].has("mlirLocationFromAttribute", "cdecl"):
    mlirLocationFromAttribute = _libs["MLIRPythonCAPI"].get("mlirLocationFromAttribute", "cdecl")
    mlirLocationFromAttribute.argtypes = [MlirAttribute]
    mlirLocationFromAttribute.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 256
if _libs["MLIRPythonCAPI"].has("mlirLocationFileLineColGet", "cdecl"):
    mlirLocationFileLineColGet = _libs["MLIRPythonCAPI"].get("mlirLocationFileLineColGet", "cdecl")
    mlirLocationFileLineColGet.argtypes = [MlirContext, MlirStringRef, c_uint, c_uint]
    mlirLocationFileLineColGet.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 260
if _libs["MLIRPythonCAPI"].has("mlirLocationCallSiteGet", "cdecl"):
    mlirLocationCallSiteGet = _libs["MLIRPythonCAPI"].get("mlirLocationCallSiteGet", "cdecl")
    mlirLocationCallSiteGet.argtypes = [MlirLocation, MlirLocation]
    mlirLocationCallSiteGet.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 265
if _libs["MLIRPythonCAPI"].has("mlirLocationFusedGet", "cdecl"):
    mlirLocationFusedGet = _libs["MLIRPythonCAPI"].get("mlirLocationFusedGet", "cdecl")
    mlirLocationFusedGet.argtypes = [MlirContext, intptr_t, POINTER(MlirLocation), MlirAttribute]
    mlirLocationFusedGet.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 271
if _libs["MLIRPythonCAPI"].has("mlirLocationNameGet", "cdecl"):
    mlirLocationNameGet = _libs["MLIRPythonCAPI"].get("mlirLocationNameGet", "cdecl")
    mlirLocationNameGet.argtypes = [MlirContext, MlirStringRef, MlirLocation]
    mlirLocationNameGet.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 276
if _libs["MLIRPythonCAPI"].has("mlirLocationUnknownGet", "cdecl"):
    mlirLocationUnknownGet = _libs["MLIRPythonCAPI"].get("mlirLocationUnknownGet", "cdecl")
    mlirLocationUnknownGet.argtypes = [MlirContext]
    mlirLocationUnknownGet.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 279
if _libs["MLIRPythonCAPI"].has("mlirLocationGetContext", "cdecl"):
    mlirLocationGetContext = _libs["MLIRPythonCAPI"].get("mlirLocationGetContext", "cdecl")
    mlirLocationGetContext.argtypes = [MlirLocation]
    mlirLocationGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 287
if _libs["MLIRPythonCAPI"].has("mlirLocationEqual", "cdecl"):
    mlirLocationEqual = _libs["MLIRPythonCAPI"].get("mlirLocationEqual", "cdecl")
    mlirLocationEqual.argtypes = [MlirLocation, MlirLocation]
    mlirLocationEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 292
if _libs["MLIRPythonCAPI"].has("mlirLocationPrint", "cdecl"):
    mlirLocationPrint = _libs["MLIRPythonCAPI"].get("mlirLocationPrint", "cdecl")
    mlirLocationPrint.argtypes = [MlirLocation, MlirStringCallback, POINTER(None)]
    mlirLocationPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 301
if _libs["MLIRPythonCAPI"].has("mlirModuleCreateEmpty", "cdecl"):
    mlirModuleCreateEmpty = _libs["MLIRPythonCAPI"].get("mlirModuleCreateEmpty", "cdecl")
    mlirModuleCreateEmpty.argtypes = [MlirLocation]
    mlirModuleCreateEmpty.restype = MlirModule

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 304
if _libs["MLIRPythonCAPI"].has("mlirModuleCreateParse", "cdecl"):
    mlirModuleCreateParse = _libs["MLIRPythonCAPI"].get("mlirModuleCreateParse", "cdecl")
    mlirModuleCreateParse.argtypes = [MlirContext, MlirStringRef]
    mlirModuleCreateParse.restype = MlirModule

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 308
if _libs["MLIRPythonCAPI"].has("mlirModuleGetContext", "cdecl"):
    mlirModuleGetContext = _libs["MLIRPythonCAPI"].get("mlirModuleGetContext", "cdecl")
    mlirModuleGetContext.argtypes = [MlirModule]
    mlirModuleGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 311
if _libs["MLIRPythonCAPI"].has("mlirModuleGetBody", "cdecl"):
    mlirModuleGetBody = _libs["MLIRPythonCAPI"].get("mlirModuleGetBody", "cdecl")
    mlirModuleGetBody.argtypes = [MlirModule]
    mlirModuleGetBody.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 317
if _libs["MLIRPythonCAPI"].has("mlirModuleDestroy", "cdecl"):
    mlirModuleDestroy = _libs["MLIRPythonCAPI"].get("mlirModuleDestroy", "cdecl")
    mlirModuleDestroy.argtypes = [MlirModule]
    mlirModuleDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 320
if _libs["MLIRPythonCAPI"].has("mlirModuleGetOperation", "cdecl"):
    mlirModuleGetOperation = _libs["MLIRPythonCAPI"].get("mlirModuleGetOperation", "cdecl")
    mlirModuleGetOperation.argtypes = [MlirModule]
    mlirModuleGetOperation.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 324
if _libs["MLIRPythonCAPI"].has("mlirModuleFromOperation", "cdecl"):
    mlirModuleFromOperation = _libs["MLIRPythonCAPI"].get("mlirModuleFromOperation", "cdecl")
    mlirModuleFromOperation.argtypes = [MlirOperation]
    mlirModuleFromOperation.restype = MlirModule

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 340
class struct_MlirOperationState(Structure):
    pass

struct_MlirOperationState.__slots__ = [
    'name',
    'location',
    'nResults',
    'results',
    'nOperands',
    'operands',
    'nRegions',
    'regions',
    'nSuccessors',
    'successors',
    'nAttributes',
    'attributes',
    'enableResultTypeInference',
]
struct_MlirOperationState._fields_ = [
    ('name', MlirStringRef),
    ('location', MlirLocation),
    ('nResults', intptr_t),
    ('results', POINTER(MlirType)),
    ('nOperands', intptr_t),
    ('operands', POINTER(MlirValue)),
    ('nRegions', intptr_t),
    ('regions', POINTER(MlirRegion)),
    ('nSuccessors', intptr_t),
    ('successors', POINTER(MlirBlock)),
    ('nAttributes', intptr_t),
    ('attributes', POINTER(MlirNamedAttribute)),
    ('enableResultTypeInference', c_bool),
]

MlirOperationState = struct_MlirOperationState# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 355

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 358
if _libs["MLIRPythonCAPI"].has("mlirOperationStateGet", "cdecl"):
    mlirOperationStateGet = _libs["MLIRPythonCAPI"].get("mlirOperationStateGet", "cdecl")
    mlirOperationStateGet.argtypes = [MlirStringRef, MlirLocation]
    mlirOperationStateGet.restype = MlirOperationState

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 362
if _libs["MLIRPythonCAPI"].has("mlirOperationStateAddResults", "cdecl"):
    mlirOperationStateAddResults = _libs["MLIRPythonCAPI"].get("mlirOperationStateAddResults", "cdecl")
    mlirOperationStateAddResults.argtypes = [POINTER(MlirOperationState), intptr_t, POINTER(MlirType)]
    mlirOperationStateAddResults.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 366
if _libs["MLIRPythonCAPI"].has("mlirOperationStateAddOperands", "cdecl"):
    mlirOperationStateAddOperands = _libs["MLIRPythonCAPI"].get("mlirOperationStateAddOperands", "cdecl")
    mlirOperationStateAddOperands.argtypes = [POINTER(MlirOperationState), intptr_t, POINTER(MlirValue)]
    mlirOperationStateAddOperands.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 369
if _libs["MLIRPythonCAPI"].has("mlirOperationStateAddOwnedRegions", "cdecl"):
    mlirOperationStateAddOwnedRegions = _libs["MLIRPythonCAPI"].get("mlirOperationStateAddOwnedRegions", "cdecl")
    mlirOperationStateAddOwnedRegions.argtypes = [POINTER(MlirOperationState), intptr_t, POINTER(MlirRegion)]
    mlirOperationStateAddOwnedRegions.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 372
if _libs["MLIRPythonCAPI"].has("mlirOperationStateAddSuccessors", "cdecl"):
    mlirOperationStateAddSuccessors = _libs["MLIRPythonCAPI"].get("mlirOperationStateAddSuccessors", "cdecl")
    mlirOperationStateAddSuccessors.argtypes = [POINTER(MlirOperationState), intptr_t, POINTER(MlirBlock)]
    mlirOperationStateAddSuccessors.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 375
if _libs["MLIRPythonCAPI"].has("mlirOperationStateAddAttributes", "cdecl"):
    mlirOperationStateAddAttributes = _libs["MLIRPythonCAPI"].get("mlirOperationStateAddAttributes", "cdecl")
    mlirOperationStateAddAttributes.argtypes = [POINTER(MlirOperationState), intptr_t, POINTER(MlirNamedAttribute)]
    mlirOperationStateAddAttributes.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 384
if _libs["MLIRPythonCAPI"].has("mlirOperationStateEnableResultTypeInference", "cdecl"):
    mlirOperationStateEnableResultTypeInference = _libs["MLIRPythonCAPI"].get("mlirOperationStateEnableResultTypeInference", "cdecl")
    mlirOperationStateEnableResultTypeInference.argtypes = [POINTER(MlirOperationState)]
    mlirOperationStateEnableResultTypeInference.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 395
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsCreate", "cdecl"):
    mlirOpPrintingFlagsCreate = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsCreate", "cdecl")
    mlirOpPrintingFlagsCreate.argtypes = []
    mlirOpPrintingFlagsCreate.restype = MlirOpPrintingFlags

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 398
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsDestroy", "cdecl"):
    mlirOpPrintingFlagsDestroy = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsDestroy", "cdecl")
    mlirOpPrintingFlagsDestroy.argtypes = [MlirOpPrintingFlags]
    mlirOpPrintingFlagsDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 405
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsElideLargeElementsAttrs", "cdecl"):
    mlirOpPrintingFlagsElideLargeElementsAttrs = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsElideLargeElementsAttrs", "cdecl")
    mlirOpPrintingFlagsElideLargeElementsAttrs.argtypes = [MlirOpPrintingFlags, intptr_t]
    mlirOpPrintingFlagsElideLargeElementsAttrs.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 412
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsEnableDebugInfo", "cdecl"):
    mlirOpPrintingFlagsEnableDebugInfo = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsEnableDebugInfo", "cdecl")
    mlirOpPrintingFlagsEnableDebugInfo.argtypes = [MlirOpPrintingFlags, c_bool, c_bool]
    mlirOpPrintingFlagsEnableDebugInfo.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 417
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsPrintGenericOpForm", "cdecl"):
    mlirOpPrintingFlagsPrintGenericOpForm = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsPrintGenericOpForm", "cdecl")
    mlirOpPrintingFlagsPrintGenericOpForm.argtypes = [MlirOpPrintingFlags]
    mlirOpPrintingFlagsPrintGenericOpForm.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 424
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsUseLocalScope", "cdecl"):
    mlirOpPrintingFlagsUseLocalScope = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsUseLocalScope", "cdecl")
    mlirOpPrintingFlagsUseLocalScope.argtypes = [MlirOpPrintingFlags]
    mlirOpPrintingFlagsUseLocalScope.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 428
if _libs["MLIRPythonCAPI"].has("mlirOpPrintingFlagsAssumeVerified", "cdecl"):
    mlirOpPrintingFlagsAssumeVerified = _libs["MLIRPythonCAPI"].get("mlirOpPrintingFlagsAssumeVerified", "cdecl")
    mlirOpPrintingFlagsAssumeVerified.argtypes = [MlirOpPrintingFlags]
    mlirOpPrintingFlagsAssumeVerified.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 437
if _libs["MLIRPythonCAPI"].has("mlirBytecodeWriterConfigCreate", "cdecl"):
    mlirBytecodeWriterConfigCreate = _libs["MLIRPythonCAPI"].get("mlirBytecodeWriterConfigCreate", "cdecl")
    mlirBytecodeWriterConfigCreate.argtypes = []
    mlirBytecodeWriterConfigCreate.restype = MlirBytecodeWriterConfig

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 441
if _libs["MLIRPythonCAPI"].has("mlirBytecodeWriterConfigDestroy", "cdecl"):
    mlirBytecodeWriterConfigDestroy = _libs["MLIRPythonCAPI"].get("mlirBytecodeWriterConfigDestroy", "cdecl")
    mlirBytecodeWriterConfigDestroy.argtypes = [MlirBytecodeWriterConfig]
    mlirBytecodeWriterConfigDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 445
if _libs["MLIRPythonCAPI"].has("mlirBytecodeWriterConfigDesiredEmitVersion", "cdecl"):
    mlirBytecodeWriterConfigDesiredEmitVersion = _libs["MLIRPythonCAPI"].get("mlirBytecodeWriterConfigDesiredEmitVersion", "cdecl")
    mlirBytecodeWriterConfigDesiredEmitVersion.argtypes = [MlirBytecodeWriterConfig, c_int64]
    mlirBytecodeWriterConfigDesiredEmitVersion.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 460
if _libs["MLIRPythonCAPI"].has("mlirOperationCreate", "cdecl"):
    mlirOperationCreate = _libs["MLIRPythonCAPI"].get("mlirOperationCreate", "cdecl")
    mlirOperationCreate.argtypes = [POINTER(MlirOperationState)]
    mlirOperationCreate.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 469
if _libs["MLIRPythonCAPI"].has("mlirOperationCreateParse", "cdecl"):
    mlirOperationCreateParse = _libs["MLIRPythonCAPI"].get("mlirOperationCreateParse", "cdecl")
    mlirOperationCreateParse.argtypes = [MlirContext, MlirStringRef, MlirStringRef]
    mlirOperationCreateParse.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 474
if _libs["MLIRPythonCAPI"].has("mlirOperationClone", "cdecl"):
    mlirOperationClone = _libs["MLIRPythonCAPI"].get("mlirOperationClone", "cdecl")
    mlirOperationClone.argtypes = [MlirOperation]
    mlirOperationClone.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 477
if _libs["MLIRPythonCAPI"].has("mlirOperationDestroy", "cdecl"):
    mlirOperationDestroy = _libs["MLIRPythonCAPI"].get("mlirOperationDestroy", "cdecl")
    mlirOperationDestroy.argtypes = [MlirOperation]
    mlirOperationDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 481
if _libs["MLIRPythonCAPI"].has("mlirOperationRemoveFromParent", "cdecl"):
    mlirOperationRemoveFromParent = _libs["MLIRPythonCAPI"].get("mlirOperationRemoveFromParent", "cdecl")
    mlirOperationRemoveFromParent.argtypes = [MlirOperation]
    mlirOperationRemoveFromParent.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 488
if _libs["MLIRPythonCAPI"].has("mlirOperationEqual", "cdecl"):
    mlirOperationEqual = _libs["MLIRPythonCAPI"].get("mlirOperationEqual", "cdecl")
    mlirOperationEqual.argtypes = [MlirOperation, MlirOperation]
    mlirOperationEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 492
if _libs["MLIRPythonCAPI"].has("mlirOperationGetContext", "cdecl"):
    mlirOperationGetContext = _libs["MLIRPythonCAPI"].get("mlirOperationGetContext", "cdecl")
    mlirOperationGetContext.argtypes = [MlirOperation]
    mlirOperationGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 495
if _libs["MLIRPythonCAPI"].has("mlirOperationGetLocation", "cdecl"):
    mlirOperationGetLocation = _libs["MLIRPythonCAPI"].get("mlirOperationGetLocation", "cdecl")
    mlirOperationGetLocation.argtypes = [MlirOperation]
    mlirOperationGetLocation.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 500
if _libs["MLIRPythonCAPI"].has("mlirOperationGetTypeID", "cdecl"):
    mlirOperationGetTypeID = _libs["MLIRPythonCAPI"].get("mlirOperationGetTypeID", "cdecl")
    mlirOperationGetTypeID.argtypes = [MlirOperation]
    mlirOperationGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 503
if _libs["MLIRPythonCAPI"].has("mlirOperationGetName", "cdecl"):
    mlirOperationGetName = _libs["MLIRPythonCAPI"].get("mlirOperationGetName", "cdecl")
    mlirOperationGetName.argtypes = [MlirOperation]
    mlirOperationGetName.restype = MlirIdentifier

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 507
if _libs["MLIRPythonCAPI"].has("mlirOperationGetBlock", "cdecl"):
    mlirOperationGetBlock = _libs["MLIRPythonCAPI"].get("mlirOperationGetBlock", "cdecl")
    mlirOperationGetBlock.argtypes = [MlirOperation]
    mlirOperationGetBlock.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 512
if _libs["MLIRPythonCAPI"].has("mlirOperationGetParentOperation", "cdecl"):
    mlirOperationGetParentOperation = _libs["MLIRPythonCAPI"].get("mlirOperationGetParentOperation", "cdecl")
    mlirOperationGetParentOperation.argtypes = [MlirOperation]
    mlirOperationGetParentOperation.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 515
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNumRegions", "cdecl"):
    mlirOperationGetNumRegions = _libs["MLIRPythonCAPI"].get("mlirOperationGetNumRegions", "cdecl")
    mlirOperationGetNumRegions.argtypes = [MlirOperation]
    mlirOperationGetNumRegions.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 518
if _libs["MLIRPythonCAPI"].has("mlirOperationGetRegion", "cdecl"):
    mlirOperationGetRegion = _libs["MLIRPythonCAPI"].get("mlirOperationGetRegion", "cdecl")
    mlirOperationGetRegion.argtypes = [MlirOperation, intptr_t]
    mlirOperationGetRegion.restype = MlirRegion

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 523
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNextInBlock", "cdecl"):
    mlirOperationGetNextInBlock = _libs["MLIRPythonCAPI"].get("mlirOperationGetNextInBlock", "cdecl")
    mlirOperationGetNextInBlock.argtypes = [MlirOperation]
    mlirOperationGetNextInBlock.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 526
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNumOperands", "cdecl"):
    mlirOperationGetNumOperands = _libs["MLIRPythonCAPI"].get("mlirOperationGetNumOperands", "cdecl")
    mlirOperationGetNumOperands.argtypes = [MlirOperation]
    mlirOperationGetNumOperands.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 529
if _libs["MLIRPythonCAPI"].has("mlirOperationGetOperand", "cdecl"):
    mlirOperationGetOperand = _libs["MLIRPythonCAPI"].get("mlirOperationGetOperand", "cdecl")
    mlirOperationGetOperand.argtypes = [MlirOperation, intptr_t]
    mlirOperationGetOperand.restype = MlirValue

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 533
if _libs["MLIRPythonCAPI"].has("mlirOperationSetOperand", "cdecl"):
    mlirOperationSetOperand = _libs["MLIRPythonCAPI"].get("mlirOperationSetOperand", "cdecl")
    mlirOperationSetOperand.argtypes = [MlirOperation, intptr_t, MlirValue]
    mlirOperationSetOperand.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 537
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNumResults", "cdecl"):
    mlirOperationGetNumResults = _libs["MLIRPythonCAPI"].get("mlirOperationGetNumResults", "cdecl")
    mlirOperationGetNumResults.argtypes = [MlirOperation]
    mlirOperationGetNumResults.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 540
if _libs["MLIRPythonCAPI"].has("mlirOperationGetResult", "cdecl"):
    mlirOperationGetResult = _libs["MLIRPythonCAPI"].get("mlirOperationGetResult", "cdecl")
    mlirOperationGetResult.argtypes = [MlirOperation, intptr_t]
    mlirOperationGetResult.restype = MlirValue

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 544
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNumSuccessors", "cdecl"):
    mlirOperationGetNumSuccessors = _libs["MLIRPythonCAPI"].get("mlirOperationGetNumSuccessors", "cdecl")
    mlirOperationGetNumSuccessors.argtypes = [MlirOperation]
    mlirOperationGetNumSuccessors.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 547
if _libs["MLIRPythonCAPI"].has("mlirOperationGetSuccessor", "cdecl"):
    mlirOperationGetSuccessor = _libs["MLIRPythonCAPI"].get("mlirOperationGetSuccessor", "cdecl")
    mlirOperationGetSuccessor.argtypes = [MlirOperation, intptr_t]
    mlirOperationGetSuccessor.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 551
if _libs["MLIRPythonCAPI"].has("mlirOperationGetNumAttributes", "cdecl"):
    mlirOperationGetNumAttributes = _libs["MLIRPythonCAPI"].get("mlirOperationGetNumAttributes", "cdecl")
    mlirOperationGetNumAttributes.argtypes = [MlirOperation]
    mlirOperationGetNumAttributes.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 555
if _libs["MLIRPythonCAPI"].has("mlirOperationGetAttribute", "cdecl"):
    mlirOperationGetAttribute = _libs["MLIRPythonCAPI"].get("mlirOperationGetAttribute", "cdecl")
    mlirOperationGetAttribute.argtypes = [MlirOperation, intptr_t]
    mlirOperationGetAttribute.restype = MlirNamedAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 559
if _libs["MLIRPythonCAPI"].has("mlirOperationGetAttributeByName", "cdecl"):
    mlirOperationGetAttributeByName = _libs["MLIRPythonCAPI"].get("mlirOperationGetAttributeByName", "cdecl")
    mlirOperationGetAttributeByName.argtypes = [MlirOperation, MlirStringRef]
    mlirOperationGetAttributeByName.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 563
if _libs["MLIRPythonCAPI"].has("mlirOperationSetAttributeByName", "cdecl"):
    mlirOperationSetAttributeByName = _libs["MLIRPythonCAPI"].get("mlirOperationSetAttributeByName", "cdecl")
    mlirOperationSetAttributeByName.argtypes = [MlirOperation, MlirStringRef, MlirAttribute]
    mlirOperationSetAttributeByName.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 569
if _libs["MLIRPythonCAPI"].has("mlirOperationRemoveAttributeByName", "cdecl"):
    mlirOperationRemoveAttributeByName = _libs["MLIRPythonCAPI"].get("mlirOperationRemoveAttributeByName", "cdecl")
    mlirOperationRemoveAttributeByName.argtypes = [MlirOperation, MlirStringRef]
    mlirOperationRemoveAttributeByName.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 575
if _libs["MLIRPythonCAPI"].has("mlirOperationPrint", "cdecl"):
    mlirOperationPrint = _libs["MLIRPythonCAPI"].get("mlirOperationPrint", "cdecl")
    mlirOperationPrint.argtypes = [MlirOperation, MlirStringCallback, POINTER(None)]
    mlirOperationPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 581
if _libs["MLIRPythonCAPI"].has("mlirOperationPrintWithFlags", "cdecl"):
    mlirOperationPrintWithFlags = _libs["MLIRPythonCAPI"].get("mlirOperationPrintWithFlags", "cdecl")
    mlirOperationPrintWithFlags.argtypes = [MlirOperation, MlirOpPrintingFlags, MlirStringCallback, POINTER(None)]
    mlirOperationPrintWithFlags.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 587
if _libs["MLIRPythonCAPI"].has("mlirOperationWriteBytecode", "cdecl"):
    mlirOperationWriteBytecode = _libs["MLIRPythonCAPI"].get("mlirOperationWriteBytecode", "cdecl")
    mlirOperationWriteBytecode.argtypes = [MlirOperation, MlirStringCallback, POINTER(None)]
    mlirOperationWriteBytecode.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 593
if _libs["MLIRPythonCAPI"].has("mlirOperationWriteBytecodeWithConfig", "cdecl"):
    mlirOperationWriteBytecodeWithConfig = _libs["MLIRPythonCAPI"].get("mlirOperationWriteBytecodeWithConfig", "cdecl")
    mlirOperationWriteBytecodeWithConfig.argtypes = [MlirOperation, MlirBytecodeWriterConfig, MlirStringCallback, POINTER(None)]
    mlirOperationWriteBytecodeWithConfig.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 598
if _libs["MLIRPythonCAPI"].has("mlirOperationDump", "cdecl"):
    mlirOperationDump = _libs["MLIRPythonCAPI"].get("mlirOperationDump", "cdecl")
    mlirOperationDump.argtypes = [MlirOperation]
    mlirOperationDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 601
if _libs["MLIRPythonCAPI"].has("mlirOperationVerify", "cdecl"):
    mlirOperationVerify = _libs["MLIRPythonCAPI"].get("mlirOperationVerify", "cdecl")
    mlirOperationVerify.argtypes = [MlirOperation]
    mlirOperationVerify.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 607
if _libs["MLIRPythonCAPI"].has("mlirOperationMoveAfter", "cdecl"):
    mlirOperationMoveAfter = _libs["MLIRPythonCAPI"].get("mlirOperationMoveAfter", "cdecl")
    mlirOperationMoveAfter.argtypes = [MlirOperation, MlirOperation]
    mlirOperationMoveAfter.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 614
if _libs["MLIRPythonCAPI"].has("mlirOperationMoveBefore", "cdecl"):
    mlirOperationMoveBefore = _libs["MLIRPythonCAPI"].get("mlirOperationMoveBefore", "cdecl")
    mlirOperationMoveBefore.argtypes = [MlirOperation, MlirOperation]
    mlirOperationMoveBefore.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 621
if _libs["MLIRPythonCAPI"].has("mlirRegionCreate", "cdecl"):
    mlirRegionCreate = _libs["MLIRPythonCAPI"].get("mlirRegionCreate", "cdecl")
    mlirRegionCreate.argtypes = []
    mlirRegionCreate.restype = MlirRegion

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 624
if _libs["MLIRPythonCAPI"].has("mlirRegionDestroy", "cdecl"):
    mlirRegionDestroy = _libs["MLIRPythonCAPI"].get("mlirRegionDestroy", "cdecl")
    mlirRegionDestroy.argtypes = [MlirRegion]
    mlirRegionDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 631
if _libs["MLIRPythonCAPI"].has("mlirRegionEqual", "cdecl"):
    mlirRegionEqual = _libs["MLIRPythonCAPI"].get("mlirRegionEqual", "cdecl")
    mlirRegionEqual.argtypes = [MlirRegion, MlirRegion]
    mlirRegionEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 634
if _libs["MLIRPythonCAPI"].has("mlirRegionGetFirstBlock", "cdecl"):
    mlirRegionGetFirstBlock = _libs["MLIRPythonCAPI"].get("mlirRegionGetFirstBlock", "cdecl")
    mlirRegionGetFirstBlock.argtypes = [MlirRegion]
    mlirRegionGetFirstBlock.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 637
if _libs["MLIRPythonCAPI"].has("mlirRegionAppendOwnedBlock", "cdecl"):
    mlirRegionAppendOwnedBlock = _libs["MLIRPythonCAPI"].get("mlirRegionAppendOwnedBlock", "cdecl")
    mlirRegionAppendOwnedBlock.argtypes = [MlirRegion, MlirBlock]
    mlirRegionAppendOwnedBlock.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 644
if _libs["MLIRPythonCAPI"].has("mlirRegionInsertOwnedBlock", "cdecl"):
    mlirRegionInsertOwnedBlock = _libs["MLIRPythonCAPI"].get("mlirRegionInsertOwnedBlock", "cdecl")
    mlirRegionInsertOwnedBlock.argtypes = [MlirRegion, intptr_t, MlirBlock]
    mlirRegionInsertOwnedBlock.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 649
if _libs["MLIRPythonCAPI"].has("mlirRegionInsertOwnedBlockAfter", "cdecl"):
    mlirRegionInsertOwnedBlockAfter = _libs["MLIRPythonCAPI"].get("mlirRegionInsertOwnedBlockAfter", "cdecl")
    mlirRegionInsertOwnedBlockAfter.argtypes = [MlirRegion, MlirBlock, MlirBlock]
    mlirRegionInsertOwnedBlockAfter.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 656
if _libs["MLIRPythonCAPI"].has("mlirRegionInsertOwnedBlockBefore", "cdecl"):
    mlirRegionInsertOwnedBlockBefore = _libs["MLIRPythonCAPI"].get("mlirRegionInsertOwnedBlockBefore", "cdecl")
    mlirRegionInsertOwnedBlockBefore.argtypes = [MlirRegion, MlirBlock, MlirBlock]
    mlirRegionInsertOwnedBlockBefore.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 661
if _libs["MLIRPythonCAPI"].has("mlirOperationGetFirstRegion", "cdecl"):
    mlirOperationGetFirstRegion = _libs["MLIRPythonCAPI"].get("mlirOperationGetFirstRegion", "cdecl")
    mlirOperationGetFirstRegion.argtypes = [MlirOperation]
    mlirOperationGetFirstRegion.restype = MlirRegion

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 665
if _libs["MLIRPythonCAPI"].has("mlirRegionGetNextInOperation", "cdecl"):
    mlirRegionGetNextInOperation = _libs["MLIRPythonCAPI"].get("mlirRegionGetNextInOperation", "cdecl")
    mlirRegionGetNextInOperation.argtypes = [MlirRegion]
    mlirRegionGetNextInOperation.restype = MlirRegion

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 673
if _libs["MLIRPythonCAPI"].has("mlirBlockCreate", "cdecl"):
    mlirBlockCreate = _libs["MLIRPythonCAPI"].get("mlirBlockCreate", "cdecl")
    mlirBlockCreate.argtypes = [intptr_t, POINTER(MlirType), POINTER(MlirLocation)]
    mlirBlockCreate.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 678
if _libs["MLIRPythonCAPI"].has("mlirBlockDestroy", "cdecl"):
    mlirBlockDestroy = _libs["MLIRPythonCAPI"].get("mlirBlockDestroy", "cdecl")
    mlirBlockDestroy.argtypes = [MlirBlock]
    mlirBlockDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 681
if _libs["MLIRPythonCAPI"].has("mlirBlockDetach", "cdecl"):
    mlirBlockDetach = _libs["MLIRPythonCAPI"].get("mlirBlockDetach", "cdecl")
    mlirBlockDetach.argtypes = [MlirBlock]
    mlirBlockDetach.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 688
if _libs["MLIRPythonCAPI"].has("mlirBlockEqual", "cdecl"):
    mlirBlockEqual = _libs["MLIRPythonCAPI"].get("mlirBlockEqual", "cdecl")
    mlirBlockEqual.argtypes = [MlirBlock, MlirBlock]
    mlirBlockEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 691
if _libs["MLIRPythonCAPI"].has("mlirBlockGetParentOperation", "cdecl"):
    mlirBlockGetParentOperation = _libs["MLIRPythonCAPI"].get("mlirBlockGetParentOperation", "cdecl")
    mlirBlockGetParentOperation.argtypes = [MlirBlock]
    mlirBlockGetParentOperation.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 694
if _libs["MLIRPythonCAPI"].has("mlirBlockGetParentRegion", "cdecl"):
    mlirBlockGetParentRegion = _libs["MLIRPythonCAPI"].get("mlirBlockGetParentRegion", "cdecl")
    mlirBlockGetParentRegion.argtypes = [MlirBlock]
    mlirBlockGetParentRegion.restype = MlirRegion

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 698
if _libs["MLIRPythonCAPI"].has("mlirBlockGetNextInRegion", "cdecl"):
    mlirBlockGetNextInRegion = _libs["MLIRPythonCAPI"].get("mlirBlockGetNextInRegion", "cdecl")
    mlirBlockGetNextInRegion.argtypes = [MlirBlock]
    mlirBlockGetNextInRegion.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 701
if _libs["MLIRPythonCAPI"].has("mlirBlockGetFirstOperation", "cdecl"):
    mlirBlockGetFirstOperation = _libs["MLIRPythonCAPI"].get("mlirBlockGetFirstOperation", "cdecl")
    mlirBlockGetFirstOperation.argtypes = [MlirBlock]
    mlirBlockGetFirstOperation.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 704
if _libs["MLIRPythonCAPI"].has("mlirBlockGetTerminator", "cdecl"):
    mlirBlockGetTerminator = _libs["MLIRPythonCAPI"].get("mlirBlockGetTerminator", "cdecl")
    mlirBlockGetTerminator.argtypes = [MlirBlock]
    mlirBlockGetTerminator.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 707
if _libs["MLIRPythonCAPI"].has("mlirBlockAppendOwnedOperation", "cdecl"):
    mlirBlockAppendOwnedOperation = _libs["MLIRPythonCAPI"].get("mlirBlockAppendOwnedOperation", "cdecl")
    mlirBlockAppendOwnedOperation.argtypes = [MlirBlock, MlirOperation]
    mlirBlockAppendOwnedOperation.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 713
if _libs["MLIRPythonCAPI"].has("mlirBlockInsertOwnedOperation", "cdecl"):
    mlirBlockInsertOwnedOperation = _libs["MLIRPythonCAPI"].get("mlirBlockInsertOwnedOperation", "cdecl")
    mlirBlockInsertOwnedOperation.argtypes = [MlirBlock, intptr_t, MlirOperation]
    mlirBlockInsertOwnedOperation.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 721
if _libs["MLIRPythonCAPI"].has("mlirBlockInsertOwnedOperationAfter", "cdecl"):
    mlirBlockInsertOwnedOperationAfter = _libs["MLIRPythonCAPI"].get("mlirBlockInsertOwnedOperationAfter", "cdecl")
    mlirBlockInsertOwnedOperationAfter.argtypes = [MlirBlock, MlirOperation, MlirOperation]
    mlirBlockInsertOwnedOperationAfter.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 728
if _libs["MLIRPythonCAPI"].has("mlirBlockInsertOwnedOperationBefore", "cdecl"):
    mlirBlockInsertOwnedOperationBefore = _libs["MLIRPythonCAPI"].get("mlirBlockInsertOwnedOperationBefore", "cdecl")
    mlirBlockInsertOwnedOperationBefore.argtypes = [MlirBlock, MlirOperation, MlirOperation]
    mlirBlockInsertOwnedOperationBefore.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 732
if _libs["MLIRPythonCAPI"].has("mlirBlockGetNumArguments", "cdecl"):
    mlirBlockGetNumArguments = _libs["MLIRPythonCAPI"].get("mlirBlockGetNumArguments", "cdecl")
    mlirBlockGetNumArguments.argtypes = [MlirBlock]
    mlirBlockGetNumArguments.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 736
if _libs["MLIRPythonCAPI"].has("mlirBlockAddArgument", "cdecl"):
    mlirBlockAddArgument = _libs["MLIRPythonCAPI"].get("mlirBlockAddArgument", "cdecl")
    mlirBlockAddArgument.argtypes = [MlirBlock, MlirType, MlirLocation]
    mlirBlockAddArgument.restype = MlirValue

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 741
if _libs["MLIRPythonCAPI"].has("mlirBlockGetArgument", "cdecl"):
    mlirBlockGetArgument = _libs["MLIRPythonCAPI"].get("mlirBlockGetArgument", "cdecl")
    mlirBlockGetArgument.argtypes = [MlirBlock, intptr_t]
    mlirBlockGetArgument.restype = MlirValue

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 748
if _libs["MLIRPythonCAPI"].has("mlirBlockPrint", "cdecl"):
    mlirBlockPrint = _libs["MLIRPythonCAPI"].get("mlirBlockPrint", "cdecl")
    mlirBlockPrint.argtypes = [MlirBlock, MlirStringCallback, POINTER(None)]
    mlirBlockPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 758
if _libs["MLIRPythonCAPI"].has("mlirValueEqual", "cdecl"):
    mlirValueEqual = _libs["MLIRPythonCAPI"].get("mlirValueEqual", "cdecl")
    mlirValueEqual.argtypes = [MlirValue, MlirValue]
    mlirValueEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 761
if _libs["MLIRPythonCAPI"].has("mlirValueIsABlockArgument", "cdecl"):
    mlirValueIsABlockArgument = _libs["MLIRPythonCAPI"].get("mlirValueIsABlockArgument", "cdecl")
    mlirValueIsABlockArgument.argtypes = [MlirValue]
    mlirValueIsABlockArgument.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 764
if _libs["MLIRPythonCAPI"].has("mlirValueIsAOpResult", "cdecl"):
    mlirValueIsAOpResult = _libs["MLIRPythonCAPI"].get("mlirValueIsAOpResult", "cdecl")
    mlirValueIsAOpResult.argtypes = [MlirValue]
    mlirValueIsAOpResult.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 768
if _libs["MLIRPythonCAPI"].has("mlirBlockArgumentGetOwner", "cdecl"):
    mlirBlockArgumentGetOwner = _libs["MLIRPythonCAPI"].get("mlirBlockArgumentGetOwner", "cdecl")
    mlirBlockArgumentGetOwner.argtypes = [MlirValue]
    mlirBlockArgumentGetOwner.restype = MlirBlock

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 771
if _libs["MLIRPythonCAPI"].has("mlirBlockArgumentGetArgNumber", "cdecl"):
    mlirBlockArgumentGetArgNumber = _libs["MLIRPythonCAPI"].get("mlirBlockArgumentGetArgNumber", "cdecl")
    mlirBlockArgumentGetArgNumber.argtypes = [MlirValue]
    mlirBlockArgumentGetArgNumber.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 774
if _libs["MLIRPythonCAPI"].has("mlirBlockArgumentSetType", "cdecl"):
    mlirBlockArgumentSetType = _libs["MLIRPythonCAPI"].get("mlirBlockArgumentSetType", "cdecl")
    mlirBlockArgumentSetType.argtypes = [MlirValue, MlirType]
    mlirBlockArgumentSetType.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 779
if _libs["MLIRPythonCAPI"].has("mlirOpResultGetOwner", "cdecl"):
    mlirOpResultGetOwner = _libs["MLIRPythonCAPI"].get("mlirOpResultGetOwner", "cdecl")
    mlirOpResultGetOwner.argtypes = [MlirValue]
    mlirOpResultGetOwner.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 783
if _libs["MLIRPythonCAPI"].has("mlirOpResultGetResultNumber", "cdecl"):
    mlirOpResultGetResultNumber = _libs["MLIRPythonCAPI"].get("mlirOpResultGetResultNumber", "cdecl")
    mlirOpResultGetResultNumber.argtypes = [MlirValue]
    mlirOpResultGetResultNumber.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 786
if _libs["MLIRPythonCAPI"].has("mlirValueGetType", "cdecl"):
    mlirValueGetType = _libs["MLIRPythonCAPI"].get("mlirValueGetType", "cdecl")
    mlirValueGetType.argtypes = [MlirValue]
    mlirValueGetType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 789
if _libs["MLIRPythonCAPI"].has("mlirValueDump", "cdecl"):
    mlirValueDump = _libs["MLIRPythonCAPI"].get("mlirValueDump", "cdecl")
    mlirValueDump.argtypes = [MlirValue]
    mlirValueDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 795
if _libs["MLIRPythonCAPI"].has("mlirValuePrint", "cdecl"):
    mlirValuePrint = _libs["MLIRPythonCAPI"].get("mlirValuePrint", "cdecl")
    mlirValuePrint.argtypes = [MlirValue, MlirStringCallback, POINTER(None)]
    mlirValuePrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 798
if _libs["MLIRPythonCAPI"].has("mlirValuePrintAsOperand", "cdecl"):
    mlirValuePrintAsOperand = _libs["MLIRPythonCAPI"].get("mlirValuePrintAsOperand", "cdecl")
    mlirValuePrintAsOperand.argtypes = [MlirValue, MlirOpPrintingFlags, MlirStringCallback, POINTER(None)]
    mlirValuePrintAsOperand.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 805
if _libs["MLIRPythonCAPI"].has("mlirValueGetFirstUse", "cdecl"):
    mlirValueGetFirstUse = _libs["MLIRPythonCAPI"].get("mlirValueGetFirstUse", "cdecl")
    mlirValueGetFirstUse.argtypes = [MlirValue]
    mlirValueGetFirstUse.restype = MlirOpOperand

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 810
if _libs["MLIRPythonCAPI"].has("mlirValueReplaceAllUsesOfWith", "cdecl"):
    mlirValueReplaceAllUsesOfWith = _libs["MLIRPythonCAPI"].get("mlirValueReplaceAllUsesOfWith", "cdecl")
    mlirValueReplaceAllUsesOfWith.argtypes = [MlirValue, MlirValue]
    mlirValueReplaceAllUsesOfWith.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 818
if _libs["MLIRPythonCAPI"].has("mlirOpOperandIsNull", "cdecl"):
    mlirOpOperandIsNull = _libs["MLIRPythonCAPI"].get("mlirOpOperandIsNull", "cdecl")
    mlirOpOperandIsNull.argtypes = [MlirOpOperand]
    mlirOpOperandIsNull.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 821
if _libs["MLIRPythonCAPI"].has("mlirOpOperandGetOwner", "cdecl"):
    mlirOpOperandGetOwner = _libs["MLIRPythonCAPI"].get("mlirOpOperandGetOwner", "cdecl")
    mlirOpOperandGetOwner.argtypes = [MlirOpOperand]
    mlirOpOperandGetOwner.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 825
if _libs["MLIRPythonCAPI"].has("mlirOpOperandGetOperandNumber", "cdecl"):
    mlirOpOperandGetOperandNumber = _libs["MLIRPythonCAPI"].get("mlirOpOperandGetOperandNumber", "cdecl")
    mlirOpOperandGetOperandNumber.argtypes = [MlirOpOperand]
    mlirOpOperandGetOperandNumber.restype = c_uint

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 830
if _libs["MLIRPythonCAPI"].has("mlirOpOperandGetNextUse", "cdecl"):
    mlirOpOperandGetNextUse = _libs["MLIRPythonCAPI"].get("mlirOpOperandGetNextUse", "cdecl")
    mlirOpOperandGetNextUse.argtypes = [MlirOpOperand]
    mlirOpOperandGetNextUse.restype = MlirOpOperand

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 837
if _libs["MLIRPythonCAPI"].has("mlirTypeParseGet", "cdecl"):
    mlirTypeParseGet = _libs["MLIRPythonCAPI"].get("mlirTypeParseGet", "cdecl")
    mlirTypeParseGet.argtypes = [MlirContext, MlirStringRef]
    mlirTypeParseGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 841
if _libs["MLIRPythonCAPI"].has("mlirTypeGetContext", "cdecl"):
    mlirTypeGetContext = _libs["MLIRPythonCAPI"].get("mlirTypeGetContext", "cdecl")
    mlirTypeGetContext.argtypes = [MlirType]
    mlirTypeGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 844
if _libs["MLIRPythonCAPI"].has("mlirTypeGetTypeID", "cdecl"):
    mlirTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirTypeGetTypeID", "cdecl")
    mlirTypeGetTypeID.argtypes = [MlirType]
    mlirTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 847
if _libs["MLIRPythonCAPI"].has("mlirTypeGetDialect", "cdecl"):
    mlirTypeGetDialect = _libs["MLIRPythonCAPI"].get("mlirTypeGetDialect", "cdecl")
    mlirTypeGetDialect.argtypes = [MlirType]
    mlirTypeGetDialect.restype = MlirDialect

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 853
if _libs["MLIRPythonCAPI"].has("mlirTypeEqual", "cdecl"):
    mlirTypeEqual = _libs["MLIRPythonCAPI"].get("mlirTypeEqual", "cdecl")
    mlirTypeEqual.argtypes = [MlirType, MlirType]
    mlirTypeEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 859
if _libs["MLIRPythonCAPI"].has("mlirTypePrint", "cdecl"):
    mlirTypePrint = _libs["MLIRPythonCAPI"].get("mlirTypePrint", "cdecl")
    mlirTypePrint.argtypes = [MlirType, MlirStringCallback, POINTER(None)]
    mlirTypePrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 862
if _libs["MLIRPythonCAPI"].has("mlirTypeDump", "cdecl"):
    mlirTypeDump = _libs["MLIRPythonCAPI"].get("mlirTypeDump", "cdecl")
    mlirTypeDump.argtypes = [MlirType]
    mlirTypeDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 869
if _libs["MLIRPythonCAPI"].has("mlirAttributeParseGet", "cdecl"):
    mlirAttributeParseGet = _libs["MLIRPythonCAPI"].get("mlirAttributeParseGet", "cdecl")
    mlirAttributeParseGet.argtypes = [MlirContext, MlirStringRef]
    mlirAttributeParseGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 873
if _libs["MLIRPythonCAPI"].has("mlirAttributeGetContext", "cdecl"):
    mlirAttributeGetContext = _libs["MLIRPythonCAPI"].get("mlirAttributeGetContext", "cdecl")
    mlirAttributeGetContext.argtypes = [MlirAttribute]
    mlirAttributeGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 876
if _libs["MLIRPythonCAPI"].has("mlirAttributeGetType", "cdecl"):
    mlirAttributeGetType = _libs["MLIRPythonCAPI"].get("mlirAttributeGetType", "cdecl")
    mlirAttributeGetType.argtypes = [MlirAttribute]
    mlirAttributeGetType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 879
if _libs["MLIRPythonCAPI"].has("mlirAttributeGetTypeID", "cdecl"):
    mlirAttributeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirAttributeGetTypeID", "cdecl")
    mlirAttributeGetTypeID.argtypes = [MlirAttribute]
    mlirAttributeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 882
if _libs["MLIRPythonCAPI"].has("mlirAttributeGetDialect", "cdecl"):
    mlirAttributeGetDialect = _libs["MLIRPythonCAPI"].get("mlirAttributeGetDialect", "cdecl")
    mlirAttributeGetDialect.argtypes = [MlirAttribute]
    mlirAttributeGetDialect.restype = MlirDialect

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 888
if _libs["MLIRPythonCAPI"].has("mlirAttributeEqual", "cdecl"):
    mlirAttributeEqual = _libs["MLIRPythonCAPI"].get("mlirAttributeEqual", "cdecl")
    mlirAttributeEqual.argtypes = [MlirAttribute, MlirAttribute]
    mlirAttributeEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 893
if _libs["MLIRPythonCAPI"].has("mlirAttributePrint", "cdecl"):
    mlirAttributePrint = _libs["MLIRPythonCAPI"].get("mlirAttributePrint", "cdecl")
    mlirAttributePrint.argtypes = [MlirAttribute, MlirStringCallback, POINTER(None)]
    mlirAttributePrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 898
if _libs["MLIRPythonCAPI"].has("mlirAttributeDump", "cdecl"):
    mlirAttributeDump = _libs["MLIRPythonCAPI"].get("mlirAttributeDump", "cdecl")
    mlirAttributeDump.argtypes = [MlirAttribute]
    mlirAttributeDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 901
if _libs["MLIRPythonCAPI"].has("mlirNamedAttributeGet", "cdecl"):
    mlirNamedAttributeGet = _libs["MLIRPythonCAPI"].get("mlirNamedAttributeGet", "cdecl")
    mlirNamedAttributeGet.argtypes = [MlirIdentifier, MlirAttribute]
    mlirNamedAttributeGet.restype = MlirNamedAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 909
if _libs["MLIRPythonCAPI"].has("mlirIdentifierGet", "cdecl"):
    mlirIdentifierGet = _libs["MLIRPythonCAPI"].get("mlirIdentifierGet", "cdecl")
    mlirIdentifierGet.argtypes = [MlirContext, MlirStringRef]
    mlirIdentifierGet.restype = MlirIdentifier

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 913
if _libs["MLIRPythonCAPI"].has("mlirIdentifierGetContext", "cdecl"):
    mlirIdentifierGetContext = _libs["MLIRPythonCAPI"].get("mlirIdentifierGetContext", "cdecl")
    mlirIdentifierGetContext.argtypes = [MlirIdentifier]
    mlirIdentifierGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 916
if _libs["MLIRPythonCAPI"].has("mlirIdentifierEqual", "cdecl"):
    mlirIdentifierEqual = _libs["MLIRPythonCAPI"].get("mlirIdentifierEqual", "cdecl")
    mlirIdentifierEqual.argtypes = [MlirIdentifier, MlirIdentifier]
    mlirIdentifierEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 920
if _libs["MLIRPythonCAPI"].has("mlirIdentifierStr", "cdecl"):
    mlirIdentifierStr = _libs["MLIRPythonCAPI"].get("mlirIdentifierStr", "cdecl")
    mlirIdentifierStr.argtypes = [MlirIdentifier]
    mlirIdentifierStr.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 928
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableGetSymbolAttributeName", "cdecl"):
    mlirSymbolTableGetSymbolAttributeName = _libs["MLIRPythonCAPI"].get("mlirSymbolTableGetSymbolAttributeName", "cdecl")
    mlirSymbolTableGetSymbolAttributeName.argtypes = []
    mlirSymbolTableGetSymbolAttributeName.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 932
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableGetVisibilityAttributeName", "cdecl"):
    mlirSymbolTableGetVisibilityAttributeName = _libs["MLIRPythonCAPI"].get("mlirSymbolTableGetVisibilityAttributeName", "cdecl")
    mlirSymbolTableGetVisibilityAttributeName.argtypes = []
    mlirSymbolTableGetVisibilityAttributeName.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 937
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableCreate", "cdecl"):
    mlirSymbolTableCreate = _libs["MLIRPythonCAPI"].get("mlirSymbolTableCreate", "cdecl")
    mlirSymbolTableCreate.argtypes = [MlirOperation]
    mlirSymbolTableCreate.restype = MlirSymbolTable

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 946
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableDestroy", "cdecl"):
    mlirSymbolTableDestroy = _libs["MLIRPythonCAPI"].get("mlirSymbolTableDestroy", "cdecl")
    mlirSymbolTableDestroy.argtypes = [MlirSymbolTable]
    mlirSymbolTableDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 952
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableLookup", "cdecl"):
    mlirSymbolTableLookup = _libs["MLIRPythonCAPI"].get("mlirSymbolTableLookup", "cdecl")
    mlirSymbolTableLookup.argtypes = [MlirSymbolTable, MlirStringRef]
    mlirSymbolTableLookup.restype = MlirOperation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 961
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableInsert", "cdecl"):
    mlirSymbolTableInsert = _libs["MLIRPythonCAPI"].get("mlirSymbolTableInsert", "cdecl")
    mlirSymbolTableInsert.argtypes = [MlirSymbolTable, MlirOperation]
    mlirSymbolTableInsert.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 964
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableErase", "cdecl"):
    mlirSymbolTableErase = _libs["MLIRPythonCAPI"].get("mlirSymbolTableErase", "cdecl")
    mlirSymbolTableErase.argtypes = [MlirSymbolTable, MlirOperation]
    mlirSymbolTableErase.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 971
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableReplaceAllSymbolUses", "cdecl"):
    mlirSymbolTableReplaceAllSymbolUses = _libs["MLIRPythonCAPI"].get("mlirSymbolTableReplaceAllSymbolUses", "cdecl")
    mlirSymbolTableReplaceAllSymbolUses.argtypes = [MlirStringRef, MlirStringRef, MlirOperation]
    mlirSymbolTableReplaceAllSymbolUses.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 980
if _libs["MLIRPythonCAPI"].has("mlirSymbolTableWalkSymbolTables", "cdecl"):
    mlirSymbolTableWalkSymbolTables = _libs["MLIRPythonCAPI"].get("mlirSymbolTableWalkSymbolTables", "cdecl")
    mlirSymbolTableWalkSymbolTables.argtypes = [MlirOperation, c_bool, CFUNCTYPE(UNCHECKED(None), MlirOperation, c_bool, POINTER(None)), POINTER(None)]
    mlirSymbolTableWalkSymbolTables.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 28
if _libs["MLIRPythonCAPI"].has("mlirOperationImplementsInterface", "cdecl"):
    mlirOperationImplementsInterface = _libs["MLIRPythonCAPI"].get("mlirOperationImplementsInterface", "cdecl")
    mlirOperationImplementsInterface.argtypes = [MlirOperation, MlirTypeID]
    mlirOperationImplementsInterface.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 36
if _libs["MLIRPythonCAPI"].has("mlirOperationImplementsInterfaceStatic", "cdecl"):
    mlirOperationImplementsInterfaceStatic = _libs["MLIRPythonCAPI"].get("mlirOperationImplementsInterfaceStatic", "cdecl")
    mlirOperationImplementsInterfaceStatic.argtypes = [MlirStringRef, MlirContext, MlirTypeID]
    mlirOperationImplementsInterfaceStatic.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 45
if _libs["MLIRPythonCAPI"].has("mlirInferTypeOpInterfaceTypeID", "cdecl"):
    mlirInferTypeOpInterfaceTypeID = _libs["MLIRPythonCAPI"].get("mlirInferTypeOpInterfaceTypeID", "cdecl")
    mlirInferTypeOpInterfaceTypeID.argtypes = []
    mlirInferTypeOpInterfaceTypeID.restype = MlirTypeID

MlirTypesCallback = CFUNCTYPE(UNCHECKED(None), intptr_t, POINTER(MlirType), POINTER(None))# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 51

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 57
if _libs["MLIRPythonCAPI"].has("mlirInferTypeOpInterfaceInferReturnTypes", "cdecl"):
    mlirInferTypeOpInterfaceInferReturnTypes = _libs["MLIRPythonCAPI"].get("mlirInferTypeOpInterfaceInferReturnTypes", "cdecl")
    mlirInferTypeOpInterfaceInferReturnTypes.argtypes = [MlirStringRef, MlirContext, MlirLocation, intptr_t, POINTER(MlirValue), MlirAttribute, POINTER(None), intptr_t, POINTER(MlirRegion), MlirTypesCallback, POINTER(None)]
    mlirInferTypeOpInterfaceInferReturnTypes.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 68
if _libs["MLIRPythonCAPI"].has("mlirInferShapedTypeOpInterfaceTypeID", "cdecl"):
    mlirInferShapedTypeOpInterfaceTypeID = _libs["MLIRPythonCAPI"].get("mlirInferShapedTypeOpInterfaceTypeID", "cdecl")
    mlirInferShapedTypeOpInterfaceTypeID.argtypes = []
    mlirInferShapedTypeOpInterfaceTypeID.restype = MlirTypeID

MlirShapedTypeComponentsCallback = CFUNCTYPE(UNCHECKED(None), c_bool, intptr_t, POINTER(c_int64), MlirType, MlirAttribute, POINTER(None))# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 77

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Interfaces.h: 84
if _libs["MLIRPythonCAPI"].has("mlirInferShapedTypeOpInterfaceInferReturnTypes", "cdecl"):
    mlirInferShapedTypeOpInterfaceInferReturnTypes = _libs["MLIRPythonCAPI"].get("mlirInferShapedTypeOpInterfaceInferReturnTypes", "cdecl")
    mlirInferShapedTypeOpInterfaceInferReturnTypes.argtypes = [MlirStringRef, MlirContext, MlirLocation, intptr_t, POINTER(MlirValue), MlirAttribute, POINTER(None), intptr_t, POINTER(MlirRegion), MlirShapedTypeComponentsCallback, POINTER(None)]
    mlirInferShapedTypeOpInterfaceInferReturnTypes.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Debug.h: 19
if _libs["MLIRPythonCAPI"].has("mlirEnableGlobalDebug", "cdecl"):
    mlirEnableGlobalDebug = _libs["MLIRPythonCAPI"].get("mlirEnableGlobalDebug", "cdecl")
    mlirEnableGlobalDebug.argtypes = [c_bool]
    mlirEnableGlobalDebug.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Debug.h: 22
if _libs["MLIRPythonCAPI"].has("mlirIsGlobalDebugEnabled", "cdecl"):
    mlirIsGlobalDebugEnabled = _libs["MLIRPythonCAPI"].get("mlirIsGlobalDebugEnabled", "cdecl")
    mlirIsGlobalDebugEnabled.argtypes = []
    mlirIsGlobalDebugEnabled.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 31
class struct_MlirExecutionEngine(Structure):
    pass

struct_MlirExecutionEngine.__slots__ = [
    'ptr',
]
struct_MlirExecutionEngine._fields_ = [
    ('ptr', POINTER(None)),
]

MlirExecutionEngine = struct_MlirExecutionEngine# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 31

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 45
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineCreate", "cdecl"):
    mlirExecutionEngineCreate = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineCreate", "cdecl")
    mlirExecutionEngineCreate.argtypes = [MlirModule, c_int, c_int, POINTER(MlirStringRef), c_bool]
    mlirExecutionEngineCreate.restype = MlirExecutionEngine

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 50
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineDestroy", "cdecl"):
    mlirExecutionEngineDestroy = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineDestroy", "cdecl")
    mlirExecutionEngineDestroy.argtypes = [MlirExecutionEngine]
    mlirExecutionEngineDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 62
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineInvokePacked", "cdecl"):
    mlirExecutionEngineInvokePacked = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineInvokePacked", "cdecl")
    mlirExecutionEngineInvokePacked.argtypes = [MlirExecutionEngine, MlirStringRef, POINTER(POINTER(None))]
    mlirExecutionEngineInvokePacked.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 67
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineLookupPacked", "cdecl"):
    mlirExecutionEngineLookupPacked = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineLookupPacked", "cdecl")
    mlirExecutionEngineLookupPacked.argtypes = [MlirExecutionEngine, MlirStringRef]
    mlirExecutionEngineLookupPacked.restype = POINTER(c_ubyte)
    mlirExecutionEngineLookupPacked.errcheck = lambda v,*a : cast(v, c_void_p)

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 72
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineLookup", "cdecl"):
    mlirExecutionEngineLookup = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineLookup", "cdecl")
    mlirExecutionEngineLookup.argtypes = [MlirExecutionEngine, MlirStringRef]
    mlirExecutionEngineLookup.restype = POINTER(c_ubyte)
    mlirExecutionEngineLookup.errcheck = lambda v,*a : cast(v, c_void_p)

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 78
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineRegisterSymbol", "cdecl"):
    mlirExecutionEngineRegisterSymbol = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineRegisterSymbol", "cdecl")
    mlirExecutionEngineRegisterSymbol.argtypes = [MlirExecutionEngine, MlirStringRef, POINTER(None)]
    mlirExecutionEngineRegisterSymbol.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 83
if _libs["MLIRPythonCAPI"].has("mlirExecutionEngineDumpToObjectFile", "cdecl"):
    mlirExecutionEngineDumpToObjectFile = _libs["MLIRPythonCAPI"].get("mlirExecutionEngineDumpToObjectFile", "cdecl")
    mlirExecutionEngineDumpToObjectFile.argtypes = [MlirExecutionEngine, MlirStringRef]
    mlirExecutionEngineDumpToObjectFile.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 38
class struct_MlirAffineExpr(Structure):
    pass

struct_MlirAffineExpr.__slots__ = [
    'ptr',
]
struct_MlirAffineExpr._fields_ = [
    ('ptr', POINTER(None)),
]

MlirAffineExpr = struct_MlirAffineExpr# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 38

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 39
class struct_MlirAffineMap(Structure):
    pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 46
if _libs["MLIRPythonCAPI"].has("mlirAffineExprGetContext", "cdecl"):
    mlirAffineExprGetContext = _libs["MLIRPythonCAPI"].get("mlirAffineExprGetContext", "cdecl")
    mlirAffineExprGetContext.argtypes = [MlirAffineExpr]
    mlirAffineExprGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 49
if _libs["MLIRPythonCAPI"].has("mlirAffineExprEqual", "cdecl"):
    mlirAffineExprEqual = _libs["MLIRPythonCAPI"].get("mlirAffineExprEqual", "cdecl")
    mlirAffineExprEqual.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineExprEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 61
if _libs["MLIRPythonCAPI"].has("mlirAffineExprPrint", "cdecl"):
    mlirAffineExprPrint = _libs["MLIRPythonCAPI"].get("mlirAffineExprPrint", "cdecl")
    mlirAffineExprPrint.argtypes = [MlirAffineExpr, MlirStringCallback, POINTER(None)]
    mlirAffineExprPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 66
if _libs["MLIRPythonCAPI"].has("mlirAffineExprDump", "cdecl"):
    mlirAffineExprDump = _libs["MLIRPythonCAPI"].get("mlirAffineExprDump", "cdecl")
    mlirAffineExprDump.argtypes = [MlirAffineExpr]
    mlirAffineExprDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 71
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsSymbolicOrConstant", "cdecl"):
    mlirAffineExprIsSymbolicOrConstant = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsSymbolicOrConstant", "cdecl")
    mlirAffineExprIsSymbolicOrConstant.argtypes = [MlirAffineExpr]
    mlirAffineExprIsSymbolicOrConstant.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 75
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsPureAffine", "cdecl"):
    mlirAffineExprIsPureAffine = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsPureAffine", "cdecl")
    mlirAffineExprIsPureAffine.argtypes = [MlirAffineExpr]
    mlirAffineExprIsPureAffine.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 80
if _libs["MLIRPythonCAPI"].has("mlirAffineExprGetLargestKnownDivisor", "cdecl"):
    mlirAffineExprGetLargestKnownDivisor = _libs["MLIRPythonCAPI"].get("mlirAffineExprGetLargestKnownDivisor", "cdecl")
    mlirAffineExprGetLargestKnownDivisor.argtypes = [MlirAffineExpr]
    mlirAffineExprGetLargestKnownDivisor.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 83
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsMultipleOf", "cdecl"):
    mlirAffineExprIsMultipleOf = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsMultipleOf", "cdecl")
    mlirAffineExprIsMultipleOf.argtypes = [MlirAffineExpr, c_int64]
    mlirAffineExprIsMultipleOf.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 88
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsFunctionOfDim", "cdecl"):
    mlirAffineExprIsFunctionOfDim = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsFunctionOfDim", "cdecl")
    mlirAffineExprIsFunctionOfDim.argtypes = [MlirAffineExpr, intptr_t]
    mlirAffineExprIsFunctionOfDim.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 92
if _libs["MLIRPythonCAPI"].has("mlirAffineExprCompose", "cdecl"):
    mlirAffineExprCompose = _libs["MLIRPythonCAPI"].get("mlirAffineExprCompose", "cdecl")
    mlirAffineExprCompose.argtypes = [MlirAffineExpr, struct_MlirAffineMap]
    mlirAffineExprCompose.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 100
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsADim", "cdecl"):
    mlirAffineExprIsADim = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsADim", "cdecl")
    mlirAffineExprIsADim.argtypes = [MlirAffineExpr]
    mlirAffineExprIsADim.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 103
if _libs["MLIRPythonCAPI"].has("mlirAffineDimExprGet", "cdecl"):
    mlirAffineDimExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineDimExprGet", "cdecl")
    mlirAffineDimExprGet.argtypes = [MlirContext, intptr_t]
    mlirAffineDimExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 108
if _libs["MLIRPythonCAPI"].has("mlirAffineDimExprGetPosition", "cdecl"):
    mlirAffineDimExprGetPosition = _libs["MLIRPythonCAPI"].get("mlirAffineDimExprGetPosition", "cdecl")
    mlirAffineDimExprGetPosition.argtypes = [MlirAffineExpr]
    mlirAffineDimExprGetPosition.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 115
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsASymbol", "cdecl"):
    mlirAffineExprIsASymbol = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsASymbol", "cdecl")
    mlirAffineExprIsASymbol.argtypes = [MlirAffineExpr]
    mlirAffineExprIsASymbol.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 118
if _libs["MLIRPythonCAPI"].has("mlirAffineSymbolExprGet", "cdecl"):
    mlirAffineSymbolExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineSymbolExprGet", "cdecl")
    mlirAffineSymbolExprGet.argtypes = [MlirContext, intptr_t]
    mlirAffineSymbolExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 123
if _libs["MLIRPythonCAPI"].has("mlirAffineSymbolExprGetPosition", "cdecl"):
    mlirAffineSymbolExprGetPosition = _libs["MLIRPythonCAPI"].get("mlirAffineSymbolExprGetPosition", "cdecl")
    mlirAffineSymbolExprGetPosition.argtypes = [MlirAffineExpr]
    mlirAffineSymbolExprGetPosition.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 130
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsAConstant", "cdecl"):
    mlirAffineExprIsAConstant = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsAConstant", "cdecl")
    mlirAffineExprIsAConstant.argtypes = [MlirAffineExpr]
    mlirAffineExprIsAConstant.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 133
if _libs["MLIRPythonCAPI"].has("mlirAffineConstantExprGet", "cdecl"):
    mlirAffineConstantExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineConstantExprGet", "cdecl")
    mlirAffineConstantExprGet.argtypes = [MlirContext, c_int64]
    mlirAffineConstantExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 138
if _libs["MLIRPythonCAPI"].has("mlirAffineConstantExprGetValue", "cdecl"):
    mlirAffineConstantExprGetValue = _libs["MLIRPythonCAPI"].get("mlirAffineConstantExprGetValue", "cdecl")
    mlirAffineConstantExprGetValue.argtypes = [MlirAffineExpr]
    mlirAffineConstantExprGetValue.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 145
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsAAdd", "cdecl"):
    mlirAffineExprIsAAdd = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsAAdd", "cdecl")
    mlirAffineExprIsAAdd.argtypes = [MlirAffineExpr]
    mlirAffineExprIsAAdd.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 148
if _libs["MLIRPythonCAPI"].has("mlirAffineAddExprGet", "cdecl"):
    mlirAffineAddExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineAddExprGet", "cdecl")
    mlirAffineAddExprGet.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineAddExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 156
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsAMul", "cdecl"):
    mlirAffineExprIsAMul = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsAMul", "cdecl")
    mlirAffineExprIsAMul.argtypes = [MlirAffineExpr]
    mlirAffineExprIsAMul.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 159
if _libs["MLIRPythonCAPI"].has("mlirAffineMulExprGet", "cdecl"):
    mlirAffineMulExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineMulExprGet", "cdecl")
    mlirAffineMulExprGet.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineMulExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 167
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsAMod", "cdecl"):
    mlirAffineExprIsAMod = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsAMod", "cdecl")
    mlirAffineExprIsAMod.argtypes = [MlirAffineExpr]
    mlirAffineExprIsAMod.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 170
if _libs["MLIRPythonCAPI"].has("mlirAffineModExprGet", "cdecl"):
    mlirAffineModExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineModExprGet", "cdecl")
    mlirAffineModExprGet.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineModExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 178
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsAFloorDiv", "cdecl"):
    mlirAffineExprIsAFloorDiv = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsAFloorDiv", "cdecl")
    mlirAffineExprIsAFloorDiv.argtypes = [MlirAffineExpr]
    mlirAffineExprIsAFloorDiv.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 181
if _libs["MLIRPythonCAPI"].has("mlirAffineFloorDivExprGet", "cdecl"):
    mlirAffineFloorDivExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineFloorDivExprGet", "cdecl")
    mlirAffineFloorDivExprGet.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineFloorDivExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 189
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsACeilDiv", "cdecl"):
    mlirAffineExprIsACeilDiv = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsACeilDiv", "cdecl")
    mlirAffineExprIsACeilDiv.argtypes = [MlirAffineExpr]
    mlirAffineExprIsACeilDiv.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 192
if _libs["MLIRPythonCAPI"].has("mlirAffineCeilDivExprGet", "cdecl"):
    mlirAffineCeilDivExprGet = _libs["MLIRPythonCAPI"].get("mlirAffineCeilDivExprGet", "cdecl")
    mlirAffineCeilDivExprGet.argtypes = [MlirAffineExpr, MlirAffineExpr]
    mlirAffineCeilDivExprGet.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 200
if _libs["MLIRPythonCAPI"].has("mlirAffineExprIsABinary", "cdecl"):
    mlirAffineExprIsABinary = _libs["MLIRPythonCAPI"].get("mlirAffineExprIsABinary", "cdecl")
    mlirAffineExprIsABinary.argtypes = [MlirAffineExpr]
    mlirAffineExprIsABinary.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 205
if _libs["MLIRPythonCAPI"].has("mlirAffineBinaryOpExprGetLHS", "cdecl"):
    mlirAffineBinaryOpExprGetLHS = _libs["MLIRPythonCAPI"].get("mlirAffineBinaryOpExprGetLHS", "cdecl")
    mlirAffineBinaryOpExprGetLHS.argtypes = [MlirAffineExpr]
    mlirAffineBinaryOpExprGetLHS.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 210
if _libs["MLIRPythonCAPI"].has("mlirAffineBinaryOpExprGetRHS", "cdecl"):
    mlirAffineBinaryOpExprGetRHS = _libs["MLIRPythonCAPI"].get("mlirAffineBinaryOpExprGetRHS", "cdecl")
    mlirAffineBinaryOpExprGetRHS.argtypes = [MlirAffineExpr]
    mlirAffineBinaryOpExprGetRHS.restype = MlirAffineExpr

struct_MlirAffineMap.__slots__ = [
    'ptr',
]
struct_MlirAffineMap._fields_ = [
    ('ptr', POINTER(None)),
]

MlirAffineMap = struct_MlirAffineMap# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 39

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 44
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetContext", "cdecl"):
    mlirAffineMapGetContext = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetContext", "cdecl")
    mlirAffineMapGetContext.argtypes = [MlirAffineMap]
    mlirAffineMapGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 52
if _libs["MLIRPythonCAPI"].has("mlirAffineMapEqual", "cdecl"):
    mlirAffineMapEqual = _libs["MLIRPythonCAPI"].get("mlirAffineMapEqual", "cdecl")
    mlirAffineMapEqual.argtypes = [MlirAffineMap, MlirAffineMap]
    mlirAffineMapEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 57
if _libs["MLIRPythonCAPI"].has("mlirAffineMapPrint", "cdecl"):
    mlirAffineMapPrint = _libs["MLIRPythonCAPI"].get("mlirAffineMapPrint", "cdecl")
    mlirAffineMapPrint.argtypes = [MlirAffineMap, MlirStringCallback, POINTER(None)]
    mlirAffineMapPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 62
if _libs["MLIRPythonCAPI"].has("mlirAffineMapDump", "cdecl"):
    mlirAffineMapDump = _libs["MLIRPythonCAPI"].get("mlirAffineMapDump", "cdecl")
    mlirAffineMapDump.argtypes = [MlirAffineMap]
    mlirAffineMapDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 66
if _libs["MLIRPythonCAPI"].has("mlirAffineMapEmptyGet", "cdecl"):
    mlirAffineMapEmptyGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapEmptyGet", "cdecl")
    mlirAffineMapEmptyGet.argtypes = [MlirContext]
    mlirAffineMapEmptyGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 70
if _libs["MLIRPythonCAPI"].has("mlirAffineMapZeroResultGet", "cdecl"):
    mlirAffineMapZeroResultGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapZeroResultGet", "cdecl")
    mlirAffineMapZeroResultGet.argtypes = [MlirContext, intptr_t, intptr_t]
    mlirAffineMapZeroResultGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 77
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGet", "cdecl"):
    mlirAffineMapGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapGet", "cdecl")
    mlirAffineMapGet.argtypes = [MlirContext, intptr_t, intptr_t, intptr_t, POINTER(MlirAffineExpr)]
    mlirAffineMapGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 85
if _libs["MLIRPythonCAPI"].has("mlirAffineMapConstantGet", "cdecl"):
    mlirAffineMapConstantGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapConstantGet", "cdecl")
    mlirAffineMapConstantGet.argtypes = [MlirContext, c_int64]
    mlirAffineMapConstantGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 91
if _libs["MLIRPythonCAPI"].has("mlirAffineMapMultiDimIdentityGet", "cdecl"):
    mlirAffineMapMultiDimIdentityGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapMultiDimIdentityGet", "cdecl")
    mlirAffineMapMultiDimIdentityGet.argtypes = [MlirContext, intptr_t]
    mlirAffineMapMultiDimIdentityGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 97
if _libs["MLIRPythonCAPI"].has("mlirAffineMapMinorIdentityGet", "cdecl"):
    mlirAffineMapMinorIdentityGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapMinorIdentityGet", "cdecl")
    mlirAffineMapMinorIdentityGet.argtypes = [MlirContext, intptr_t, intptr_t]
    mlirAffineMapMinorIdentityGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 104
if _libs["MLIRPythonCAPI"].has("mlirAffineMapPermutationGet", "cdecl"):
    mlirAffineMapPermutationGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapPermutationGet", "cdecl")
    mlirAffineMapPermutationGet.argtypes = [MlirContext, intptr_t, POINTER(c_uint)]
    mlirAffineMapPermutationGet.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 110
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsIdentity", "cdecl"):
    mlirAffineMapIsIdentity = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsIdentity", "cdecl")
    mlirAffineMapIsIdentity.argtypes = [MlirAffineMap]
    mlirAffineMapIsIdentity.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 113
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsMinorIdentity", "cdecl"):
    mlirAffineMapIsMinorIdentity = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsMinorIdentity", "cdecl")
    mlirAffineMapIsMinorIdentity.argtypes = [MlirAffineMap]
    mlirAffineMapIsMinorIdentity.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 116
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsEmpty", "cdecl"):
    mlirAffineMapIsEmpty = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsEmpty", "cdecl")
    mlirAffineMapIsEmpty.argtypes = [MlirAffineMap]
    mlirAffineMapIsEmpty.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 120
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsSingleConstant", "cdecl"):
    mlirAffineMapIsSingleConstant = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsSingleConstant", "cdecl")
    mlirAffineMapIsSingleConstant.argtypes = [MlirAffineMap]
    mlirAffineMapIsSingleConstant.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 125
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetSingleConstantResult", "cdecl"):
    mlirAffineMapGetSingleConstantResult = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetSingleConstantResult", "cdecl")
    mlirAffineMapGetSingleConstantResult.argtypes = [MlirAffineMap]
    mlirAffineMapGetSingleConstantResult.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 128
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetNumDims", "cdecl"):
    mlirAffineMapGetNumDims = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetNumDims", "cdecl")
    mlirAffineMapGetNumDims.argtypes = [MlirAffineMap]
    mlirAffineMapGetNumDims.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 131
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetNumSymbols", "cdecl"):
    mlirAffineMapGetNumSymbols = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetNumSymbols", "cdecl")
    mlirAffineMapGetNumSymbols.argtypes = [MlirAffineMap]
    mlirAffineMapGetNumSymbols.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 134
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetNumResults", "cdecl"):
    mlirAffineMapGetNumResults = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetNumResults", "cdecl")
    mlirAffineMapGetNumResults.argtypes = [MlirAffineMap]
    mlirAffineMapGetNumResults.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 138
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetResult", "cdecl"):
    mlirAffineMapGetResult = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetResult", "cdecl")
    mlirAffineMapGetResult.argtypes = [MlirAffineMap, intptr_t]
    mlirAffineMapGetResult.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 142
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetNumInputs", "cdecl"):
    mlirAffineMapGetNumInputs = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetNumInputs", "cdecl")
    mlirAffineMapGetNumInputs.argtypes = [MlirAffineMap]
    mlirAffineMapGetNumInputs.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 147
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsProjectedPermutation", "cdecl"):
    mlirAffineMapIsProjectedPermutation = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsProjectedPermutation", "cdecl")
    mlirAffineMapIsProjectedPermutation.argtypes = [MlirAffineMap]
    mlirAffineMapIsProjectedPermutation.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 151
if _libs["MLIRPythonCAPI"].has("mlirAffineMapIsPermutation", "cdecl"):
    mlirAffineMapIsPermutation = _libs["MLIRPythonCAPI"].get("mlirAffineMapIsPermutation", "cdecl")
    mlirAffineMapIsPermutation.argtypes = [MlirAffineMap]
    mlirAffineMapIsPermutation.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 154
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetSubMap", "cdecl"):
    mlirAffineMapGetSubMap = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetSubMap", "cdecl")
    mlirAffineMapGetSubMap.argtypes = [MlirAffineMap, intptr_t, POINTER(intptr_t)]
    mlirAffineMapGetSubMap.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 163
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetMajorSubMap", "cdecl"):
    mlirAffineMapGetMajorSubMap = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetMajorSubMap", "cdecl")
    mlirAffineMapGetMajorSubMap.argtypes = [MlirAffineMap, intptr_t]
    mlirAffineMapGetMajorSubMap.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 170
if _libs["MLIRPythonCAPI"].has("mlirAffineMapGetMinorSubMap", "cdecl"):
    mlirAffineMapGetMinorSubMap = _libs["MLIRPythonCAPI"].get("mlirAffineMapGetMinorSubMap", "cdecl")
    mlirAffineMapGetMinorSubMap.argtypes = [MlirAffineMap, intptr_t]
    mlirAffineMapGetMinorSubMap.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 175
if _libs["MLIRPythonCAPI"].has("mlirAffineMapReplace", "cdecl"):
    mlirAffineMapReplace = _libs["MLIRPythonCAPI"].get("mlirAffineMapReplace", "cdecl")
    mlirAffineMapReplace.argtypes = [MlirAffineMap, MlirAffineExpr, MlirAffineExpr, intptr_t, intptr_t]
    mlirAffineMapReplace.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 186
if _libs["MLIRPythonCAPI"].has("mlirAffineMapCompressUnusedSymbols", "cdecl"):
    mlirAffineMapCompressUnusedSymbols = _libs["MLIRPythonCAPI"].get("mlirAffineMapCompressUnusedSymbols", "cdecl")
    mlirAffineMapCompressUnusedSymbols.argtypes = [POINTER(MlirAffineMap), intptr_t, POINTER(None), CFUNCTYPE(UNCHECKED(None), POINTER(None), intptr_t, MlirAffineMap)]
    mlirAffineMapCompressUnusedSymbols.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 26
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeGetTypeID", "cdecl"):
    mlirIntegerTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeGetTypeID", "cdecl")
    mlirIntegerTypeGetTypeID.argtypes = []
    mlirIntegerTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 29
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAInteger", "cdecl"):
    mlirTypeIsAInteger = _libs["MLIRPythonCAPI"].get("mlirTypeIsAInteger", "cdecl")
    mlirTypeIsAInteger.argtypes = [MlirType]
    mlirTypeIsAInteger.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 33
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeGet", "cdecl"):
    mlirIntegerTypeGet = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeGet", "cdecl")
    mlirIntegerTypeGet.argtypes = [MlirContext, c_uint]
    mlirIntegerTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 38
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeSignedGet", "cdecl"):
    mlirIntegerTypeSignedGet = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeSignedGet", "cdecl")
    mlirIntegerTypeSignedGet.argtypes = [MlirContext, c_uint]
    mlirIntegerTypeSignedGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 43
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeUnsignedGet", "cdecl"):
    mlirIntegerTypeUnsignedGet = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeUnsignedGet", "cdecl")
    mlirIntegerTypeUnsignedGet.argtypes = [MlirContext, c_uint]
    mlirIntegerTypeUnsignedGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 47
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeGetWidth", "cdecl"):
    mlirIntegerTypeGetWidth = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeGetWidth", "cdecl")
    mlirIntegerTypeGetWidth.argtypes = [MlirType]
    mlirIntegerTypeGetWidth.restype = c_uint

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 50
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeIsSignless", "cdecl"):
    mlirIntegerTypeIsSignless = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeIsSignless", "cdecl")
    mlirIntegerTypeIsSignless.argtypes = [MlirType]
    mlirIntegerTypeIsSignless.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 53
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeIsSigned", "cdecl"):
    mlirIntegerTypeIsSigned = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeIsSigned", "cdecl")
    mlirIntegerTypeIsSigned.argtypes = [MlirType]
    mlirIntegerTypeIsSigned.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 56
if _libs["MLIRPythonCAPI"].has("mlirIntegerTypeIsUnsigned", "cdecl"):
    mlirIntegerTypeIsUnsigned = _libs["MLIRPythonCAPI"].get("mlirIntegerTypeIsUnsigned", "cdecl")
    mlirIntegerTypeIsUnsigned.argtypes = [MlirType]
    mlirIntegerTypeIsUnsigned.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 63
if _libs["MLIRPythonCAPI"].has("mlirIndexTypeGetTypeID", "cdecl"):
    mlirIndexTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirIndexTypeGetTypeID", "cdecl")
    mlirIndexTypeGetTypeID.argtypes = []
    mlirIndexTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 66
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAIndex", "cdecl"):
    mlirTypeIsAIndex = _libs["MLIRPythonCAPI"].get("mlirTypeIsAIndex", "cdecl")
    mlirTypeIsAIndex.argtypes = [MlirType]
    mlirTypeIsAIndex.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 70
if _libs["MLIRPythonCAPI"].has("mlirIndexTypeGet", "cdecl"):
    mlirIndexTypeGet = _libs["MLIRPythonCAPI"].get("mlirIndexTypeGet", "cdecl")
    mlirIndexTypeGet.argtypes = [MlirContext]
    mlirIndexTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 77
if _libs["MLIRPythonCAPI"].has("mlirFloat8E5M2TypeGetTypeID", "cdecl"):
    mlirFloat8E5M2TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat8E5M2TypeGetTypeID", "cdecl")
    mlirFloat8E5M2TypeGetTypeID.argtypes = []
    mlirFloat8E5M2TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 80
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFloat8E5M2", "cdecl"):
    mlirTypeIsAFloat8E5M2 = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFloat8E5M2", "cdecl")
    mlirTypeIsAFloat8E5M2.argtypes = [MlirType]
    mlirTypeIsAFloat8E5M2.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 84
if _libs["MLIRPythonCAPI"].has("mlirFloat8E5M2TypeGet", "cdecl"):
    mlirFloat8E5M2TypeGet = _libs["MLIRPythonCAPI"].get("mlirFloat8E5M2TypeGet", "cdecl")
    mlirFloat8E5M2TypeGet.argtypes = [MlirContext]
    mlirFloat8E5M2TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 87
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3FNTypeGetTypeID", "cdecl"):
    mlirFloat8E4M3FNTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3FNTypeGetTypeID", "cdecl")
    mlirFloat8E4M3FNTypeGetTypeID.argtypes = []
    mlirFloat8E4M3FNTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 90
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFloat8E4M3FN", "cdecl"):
    mlirTypeIsAFloat8E4M3FN = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFloat8E4M3FN", "cdecl")
    mlirTypeIsAFloat8E4M3FN.argtypes = [MlirType]
    mlirTypeIsAFloat8E4M3FN.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 94
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3FNTypeGet", "cdecl"):
    mlirFloat8E4M3FNTypeGet = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3FNTypeGet", "cdecl")
    mlirFloat8E4M3FNTypeGet.argtypes = [MlirContext]
    mlirFloat8E4M3FNTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 97
if _libs["MLIRPythonCAPI"].has("mlirFloat8E5M2FNUZTypeGetTypeID", "cdecl"):
    mlirFloat8E5M2FNUZTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat8E5M2FNUZTypeGetTypeID", "cdecl")
    mlirFloat8E5M2FNUZTypeGetTypeID.argtypes = []
    mlirFloat8E5M2FNUZTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 100
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFloat8E5M2FNUZ", "cdecl"):
    mlirTypeIsAFloat8E5M2FNUZ = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFloat8E5M2FNUZ", "cdecl")
    mlirTypeIsAFloat8E5M2FNUZ.argtypes = [MlirType]
    mlirTypeIsAFloat8E5M2FNUZ.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 104
if _libs["MLIRPythonCAPI"].has("mlirFloat8E5M2FNUZTypeGet", "cdecl"):
    mlirFloat8E5M2FNUZTypeGet = _libs["MLIRPythonCAPI"].get("mlirFloat8E5M2FNUZTypeGet", "cdecl")
    mlirFloat8E5M2FNUZTypeGet.argtypes = [MlirContext]
    mlirFloat8E5M2FNUZTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 107
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3FNUZTypeGetTypeID", "cdecl"):
    mlirFloat8E4M3FNUZTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3FNUZTypeGetTypeID", "cdecl")
    mlirFloat8E4M3FNUZTypeGetTypeID.argtypes = []
    mlirFloat8E4M3FNUZTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 110
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFloat8E4M3FNUZ", "cdecl"):
    mlirTypeIsAFloat8E4M3FNUZ = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFloat8E4M3FNUZ", "cdecl")
    mlirTypeIsAFloat8E4M3FNUZ.argtypes = [MlirType]
    mlirTypeIsAFloat8E4M3FNUZ.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 114
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3FNUZTypeGet", "cdecl"):
    mlirFloat8E4M3FNUZTypeGet = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3FNUZTypeGet", "cdecl")
    mlirFloat8E4M3FNUZTypeGet.argtypes = [MlirContext]
    mlirFloat8E4M3FNUZTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 117
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3B11FNUZTypeGetTypeID", "cdecl"):
    mlirFloat8E4M3B11FNUZTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3B11FNUZTypeGetTypeID", "cdecl")
    mlirFloat8E4M3B11FNUZTypeGetTypeID.argtypes = []
    mlirFloat8E4M3B11FNUZTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 120
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFloat8E4M3B11FNUZ", "cdecl"):
    mlirTypeIsAFloat8E4M3B11FNUZ = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFloat8E4M3B11FNUZ", "cdecl")
    mlirTypeIsAFloat8E4M3B11FNUZ.argtypes = [MlirType]
    mlirTypeIsAFloat8E4M3B11FNUZ.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 124
if _libs["MLIRPythonCAPI"].has("mlirFloat8E4M3B11FNUZTypeGet", "cdecl"):
    mlirFloat8E4M3B11FNUZTypeGet = _libs["MLIRPythonCAPI"].get("mlirFloat8E4M3B11FNUZTypeGet", "cdecl")
    mlirFloat8E4M3B11FNUZTypeGet.argtypes = [MlirContext]
    mlirFloat8E4M3B11FNUZTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 127
if _libs["MLIRPythonCAPI"].has("mlirBFloat16TypeGetTypeID", "cdecl"):
    mlirBFloat16TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirBFloat16TypeGetTypeID", "cdecl")
    mlirBFloat16TypeGetTypeID.argtypes = []
    mlirBFloat16TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 130
if _libs["MLIRPythonCAPI"].has("mlirTypeIsABF16", "cdecl"):
    mlirTypeIsABF16 = _libs["MLIRPythonCAPI"].get("mlirTypeIsABF16", "cdecl")
    mlirTypeIsABF16.argtypes = [MlirType]
    mlirTypeIsABF16.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 134
if _libs["MLIRPythonCAPI"].has("mlirBF16TypeGet", "cdecl"):
    mlirBF16TypeGet = _libs["MLIRPythonCAPI"].get("mlirBF16TypeGet", "cdecl")
    mlirBF16TypeGet.argtypes = [MlirContext]
    mlirBF16TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 137
if _libs["MLIRPythonCAPI"].has("mlirFloat16TypeGetTypeID", "cdecl"):
    mlirFloat16TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat16TypeGetTypeID", "cdecl")
    mlirFloat16TypeGetTypeID.argtypes = []
    mlirFloat16TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 140
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAF16", "cdecl"):
    mlirTypeIsAF16 = _libs["MLIRPythonCAPI"].get("mlirTypeIsAF16", "cdecl")
    mlirTypeIsAF16.argtypes = [MlirType]
    mlirTypeIsAF16.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 144
if _libs["MLIRPythonCAPI"].has("mlirF16TypeGet", "cdecl"):
    mlirF16TypeGet = _libs["MLIRPythonCAPI"].get("mlirF16TypeGet", "cdecl")
    mlirF16TypeGet.argtypes = [MlirContext]
    mlirF16TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 147
if _libs["MLIRPythonCAPI"].has("mlirFloat32TypeGetTypeID", "cdecl"):
    mlirFloat32TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat32TypeGetTypeID", "cdecl")
    mlirFloat32TypeGetTypeID.argtypes = []
    mlirFloat32TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 150
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAF32", "cdecl"):
    mlirTypeIsAF32 = _libs["MLIRPythonCAPI"].get("mlirTypeIsAF32", "cdecl")
    mlirTypeIsAF32.argtypes = [MlirType]
    mlirTypeIsAF32.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 154
if _libs["MLIRPythonCAPI"].has("mlirF32TypeGet", "cdecl"):
    mlirF32TypeGet = _libs["MLIRPythonCAPI"].get("mlirF32TypeGet", "cdecl")
    mlirF32TypeGet.argtypes = [MlirContext]
    mlirF32TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 157
if _libs["MLIRPythonCAPI"].has("mlirFloat64TypeGetTypeID", "cdecl"):
    mlirFloat64TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloat64TypeGetTypeID", "cdecl")
    mlirFloat64TypeGetTypeID.argtypes = []
    mlirFloat64TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 160
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAF64", "cdecl"):
    mlirTypeIsAF64 = _libs["MLIRPythonCAPI"].get("mlirTypeIsAF64", "cdecl")
    mlirTypeIsAF64.argtypes = [MlirType]
    mlirTypeIsAF64.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 164
if _libs["MLIRPythonCAPI"].has("mlirF64TypeGet", "cdecl"):
    mlirF64TypeGet = _libs["MLIRPythonCAPI"].get("mlirF64TypeGet", "cdecl")
    mlirF64TypeGet.argtypes = [MlirContext]
    mlirF64TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 167
if _libs["MLIRPythonCAPI"].has("mlirFloatTF32TypeGetTypeID", "cdecl"):
    mlirFloatTF32TypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloatTF32TypeGetTypeID", "cdecl")
    mlirFloatTF32TypeGetTypeID.argtypes = []
    mlirFloatTF32TypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 170
if _libs["MLIRPythonCAPI"].has("mlirTypeIsATF32", "cdecl"):
    mlirTypeIsATF32 = _libs["MLIRPythonCAPI"].get("mlirTypeIsATF32", "cdecl")
    mlirTypeIsATF32.argtypes = [MlirType]
    mlirTypeIsATF32.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 174
if _libs["MLIRPythonCAPI"].has("mlirTF32TypeGet", "cdecl"):
    mlirTF32TypeGet = _libs["MLIRPythonCAPI"].get("mlirTF32TypeGet", "cdecl")
    mlirTF32TypeGet.argtypes = [MlirContext]
    mlirTF32TypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 181
if _libs["MLIRPythonCAPI"].has("mlirNoneTypeGetTypeID", "cdecl"):
    mlirNoneTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirNoneTypeGetTypeID", "cdecl")
    mlirNoneTypeGetTypeID.argtypes = []
    mlirNoneTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 184
if _libs["MLIRPythonCAPI"].has("mlirTypeIsANone", "cdecl"):
    mlirTypeIsANone = _libs["MLIRPythonCAPI"].get("mlirTypeIsANone", "cdecl")
    mlirTypeIsANone.argtypes = [MlirType]
    mlirTypeIsANone.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 188
if _libs["MLIRPythonCAPI"].has("mlirNoneTypeGet", "cdecl"):
    mlirNoneTypeGet = _libs["MLIRPythonCAPI"].get("mlirNoneTypeGet", "cdecl")
    mlirNoneTypeGet.argtypes = [MlirContext]
    mlirNoneTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 195
if _libs["MLIRPythonCAPI"].has("mlirComplexTypeGetTypeID", "cdecl"):
    mlirComplexTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirComplexTypeGetTypeID", "cdecl")
    mlirComplexTypeGetTypeID.argtypes = []
    mlirComplexTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 198
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAComplex", "cdecl"):
    mlirTypeIsAComplex = _libs["MLIRPythonCAPI"].get("mlirTypeIsAComplex", "cdecl")
    mlirTypeIsAComplex.argtypes = [MlirType]
    mlirTypeIsAComplex.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 202
if _libs["MLIRPythonCAPI"].has("mlirComplexTypeGet", "cdecl"):
    mlirComplexTypeGet = _libs["MLIRPythonCAPI"].get("mlirComplexTypeGet", "cdecl")
    mlirComplexTypeGet.argtypes = [MlirType]
    mlirComplexTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 205
if _libs["MLIRPythonCAPI"].has("mlirComplexTypeGetElementType", "cdecl"):
    mlirComplexTypeGetElementType = _libs["MLIRPythonCAPI"].get("mlirComplexTypeGetElementType", "cdecl")
    mlirComplexTypeGetElementType.argtypes = [MlirType]
    mlirComplexTypeGetElementType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 212
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAShaped", "cdecl"):
    mlirTypeIsAShaped = _libs["MLIRPythonCAPI"].get("mlirTypeIsAShaped", "cdecl")
    mlirTypeIsAShaped.argtypes = [MlirType]
    mlirTypeIsAShaped.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 215
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeGetElementType", "cdecl"):
    mlirShapedTypeGetElementType = _libs["MLIRPythonCAPI"].get("mlirShapedTypeGetElementType", "cdecl")
    mlirShapedTypeGetElementType.argtypes = [MlirType]
    mlirShapedTypeGetElementType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 218
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeHasRank", "cdecl"):
    mlirShapedTypeHasRank = _libs["MLIRPythonCAPI"].get("mlirShapedTypeHasRank", "cdecl")
    mlirShapedTypeHasRank.argtypes = [MlirType]
    mlirShapedTypeHasRank.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 221
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeGetRank", "cdecl"):
    mlirShapedTypeGetRank = _libs["MLIRPythonCAPI"].get("mlirShapedTypeGetRank", "cdecl")
    mlirShapedTypeGetRank.argtypes = [MlirType]
    mlirShapedTypeGetRank.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 224
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeHasStaticShape", "cdecl"):
    mlirShapedTypeHasStaticShape = _libs["MLIRPythonCAPI"].get("mlirShapedTypeHasStaticShape", "cdecl")
    mlirShapedTypeHasStaticShape.argtypes = [MlirType]
    mlirShapedTypeHasStaticShape.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 227
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeIsDynamicDim", "cdecl"):
    mlirShapedTypeIsDynamicDim = _libs["MLIRPythonCAPI"].get("mlirShapedTypeIsDynamicDim", "cdecl")
    mlirShapedTypeIsDynamicDim.argtypes = [MlirType, intptr_t]
    mlirShapedTypeIsDynamicDim.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 230
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeGetDimSize", "cdecl"):
    mlirShapedTypeGetDimSize = _libs["MLIRPythonCAPI"].get("mlirShapedTypeGetDimSize", "cdecl")
    mlirShapedTypeGetDimSize.argtypes = [MlirType, intptr_t]
    mlirShapedTypeGetDimSize.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 235
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeIsDynamicSize", "cdecl"):
    mlirShapedTypeIsDynamicSize = _libs["MLIRPythonCAPI"].get("mlirShapedTypeIsDynamicSize", "cdecl")
    mlirShapedTypeIsDynamicSize.argtypes = [c_int64]
    mlirShapedTypeIsDynamicSize.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 239
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeGetDynamicSize", "cdecl"):
    mlirShapedTypeGetDynamicSize = _libs["MLIRPythonCAPI"].get("mlirShapedTypeGetDynamicSize", "cdecl")
    mlirShapedTypeGetDynamicSize.argtypes = []
    mlirShapedTypeGetDynamicSize.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 243
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeIsDynamicStrideOrOffset", "cdecl"):
    mlirShapedTypeIsDynamicStrideOrOffset = _libs["MLIRPythonCAPI"].get("mlirShapedTypeIsDynamicStrideOrOffset", "cdecl")
    mlirShapedTypeIsDynamicStrideOrOffset.argtypes = [c_int64]
    mlirShapedTypeIsDynamicStrideOrOffset.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 248
if _libs["MLIRPythonCAPI"].has("mlirShapedTypeGetDynamicStrideOrOffset", "cdecl"):
    mlirShapedTypeGetDynamicStrideOrOffset = _libs["MLIRPythonCAPI"].get("mlirShapedTypeGetDynamicStrideOrOffset", "cdecl")
    mlirShapedTypeGetDynamicStrideOrOffset.argtypes = []
    mlirShapedTypeGetDynamicStrideOrOffset.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 255
if _libs["MLIRPythonCAPI"].has("mlirVectorTypeGetTypeID", "cdecl"):
    mlirVectorTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirVectorTypeGetTypeID", "cdecl")
    mlirVectorTypeGetTypeID.argtypes = []
    mlirVectorTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 258
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAVector", "cdecl"):
    mlirTypeIsAVector = _libs["MLIRPythonCAPI"].get("mlirTypeIsAVector", "cdecl")
    mlirTypeIsAVector.argtypes = [MlirType]
    mlirTypeIsAVector.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 263
if _libs["MLIRPythonCAPI"].has("mlirVectorTypeGet", "cdecl"):
    mlirVectorTypeGet = _libs["MLIRPythonCAPI"].get("mlirVectorTypeGet", "cdecl")
    mlirVectorTypeGet.argtypes = [intptr_t, POINTER(c_int64), MlirType]
    mlirVectorTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 269
if _libs["MLIRPythonCAPI"].has("mlirVectorTypeGetChecked", "cdecl"):
    mlirVectorTypeGetChecked = _libs["MLIRPythonCAPI"].get("mlirVectorTypeGetChecked", "cdecl")
    mlirVectorTypeGetChecked.argtypes = [MlirLocation, intptr_t, POINTER(c_int64), MlirType]
    mlirVectorTypeGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 279
if _libs["MLIRPythonCAPI"].has("mlirTypeIsATensor", "cdecl"):
    mlirTypeIsATensor = _libs["MLIRPythonCAPI"].get("mlirTypeIsATensor", "cdecl")
    mlirTypeIsATensor.argtypes = [MlirType]
    mlirTypeIsATensor.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 282
if _libs["MLIRPythonCAPI"].has("mlirRankedTensorTypeGetTypeID", "cdecl"):
    mlirRankedTensorTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirRankedTensorTypeGetTypeID", "cdecl")
    mlirRankedTensorTypeGetTypeID.argtypes = []
    mlirRankedTensorTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 285
if _libs["MLIRPythonCAPI"].has("mlirTypeIsARankedTensor", "cdecl"):
    mlirTypeIsARankedTensor = _libs["MLIRPythonCAPI"].get("mlirTypeIsARankedTensor", "cdecl")
    mlirTypeIsARankedTensor.argtypes = [MlirType]
    mlirTypeIsARankedTensor.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 288
if _libs["MLIRPythonCAPI"].has("mlirUnrankedTensorTypeGetTypeID", "cdecl"):
    mlirUnrankedTensorTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirUnrankedTensorTypeGetTypeID", "cdecl")
    mlirUnrankedTensorTypeGetTypeID.argtypes = []
    mlirUnrankedTensorTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 291
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAUnrankedTensor", "cdecl"):
    mlirTypeIsAUnrankedTensor = _libs["MLIRPythonCAPI"].get("mlirTypeIsAUnrankedTensor", "cdecl")
    mlirTypeIsAUnrankedTensor.argtypes = [MlirType]
    mlirTypeIsAUnrankedTensor.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 297
if _libs["MLIRPythonCAPI"].has("mlirRankedTensorTypeGet", "cdecl"):
    mlirRankedTensorTypeGet = _libs["MLIRPythonCAPI"].get("mlirRankedTensorTypeGet", "cdecl")
    mlirRankedTensorTypeGet.argtypes = [intptr_t, POINTER(c_int64), MlirType, MlirAttribute]
    mlirRankedTensorTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 304
if _libs["MLIRPythonCAPI"].has("mlirRankedTensorTypeGetChecked", "cdecl"):
    mlirRankedTensorTypeGetChecked = _libs["MLIRPythonCAPI"].get("mlirRankedTensorTypeGetChecked", "cdecl")
    mlirRankedTensorTypeGetChecked.argtypes = [MlirLocation, intptr_t, POINTER(c_int64), MlirType, MlirAttribute]
    mlirRankedTensorTypeGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 310
if _libs["MLIRPythonCAPI"].has("mlirRankedTensorTypeGetEncoding", "cdecl"):
    mlirRankedTensorTypeGetEncoding = _libs["MLIRPythonCAPI"].get("mlirRankedTensorTypeGetEncoding", "cdecl")
    mlirRankedTensorTypeGetEncoding.argtypes = [MlirType]
    mlirRankedTensorTypeGetEncoding.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 314
if _libs["MLIRPythonCAPI"].has("mlirUnrankedTensorTypeGet", "cdecl"):
    mlirUnrankedTensorTypeGet = _libs["MLIRPythonCAPI"].get("mlirUnrankedTensorTypeGet", "cdecl")
    mlirUnrankedTensorTypeGet.argtypes = [MlirType]
    mlirUnrankedTensorTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 319
if _libs["MLIRPythonCAPI"].has("mlirUnrankedTensorTypeGetChecked", "cdecl"):
    mlirUnrankedTensorTypeGetChecked = _libs["MLIRPythonCAPI"].get("mlirUnrankedTensorTypeGetChecked", "cdecl")
    mlirUnrankedTensorTypeGetChecked.argtypes = [MlirLocation, MlirType]
    mlirUnrankedTensorTypeGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 326
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGetTypeID", "cdecl"):
    mlirMemRefTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGetTypeID", "cdecl")
    mlirMemRefTypeGetTypeID.argtypes = []
    mlirMemRefTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 329
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAMemRef", "cdecl"):
    mlirTypeIsAMemRef = _libs["MLIRPythonCAPI"].get("mlirTypeIsAMemRef", "cdecl")
    mlirTypeIsAMemRef.argtypes = [MlirType]
    mlirTypeIsAMemRef.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 332
if _libs["MLIRPythonCAPI"].has("mlirUnrankedMemRefTypeGetTypeID", "cdecl"):
    mlirUnrankedMemRefTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirUnrankedMemRefTypeGetTypeID", "cdecl")
    mlirUnrankedMemRefTypeGetTypeID.argtypes = []
    mlirUnrankedMemRefTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 335
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAUnrankedMemRef", "cdecl"):
    mlirTypeIsAUnrankedMemRef = _libs["MLIRPythonCAPI"].get("mlirTypeIsAUnrankedMemRef", "cdecl")
    mlirTypeIsAUnrankedMemRef.argtypes = [MlirType]
    mlirTypeIsAUnrankedMemRef.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 340
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGet", "cdecl"):
    mlirMemRefTypeGet = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGet", "cdecl")
    mlirMemRefTypeGet.argtypes = [MlirType, intptr_t, POINTER(c_int64), MlirAttribute, MlirAttribute]
    mlirMemRefTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 348
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGetChecked", "cdecl"):
    mlirMemRefTypeGetChecked = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGetChecked", "cdecl")
    mlirMemRefTypeGetChecked.argtypes = [MlirLocation, MlirType, intptr_t, POINTER(c_int64), MlirAttribute, MlirAttribute]
    mlirMemRefTypeGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 357
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeContiguousGet", "cdecl"):
    mlirMemRefTypeContiguousGet = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeContiguousGet", "cdecl")
    mlirMemRefTypeContiguousGet.argtypes = [MlirType, intptr_t, POINTER(c_int64), MlirAttribute]
    mlirMemRefTypeContiguousGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 362
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeContiguousGetChecked", "cdecl"):
    mlirMemRefTypeContiguousGetChecked = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeContiguousGetChecked", "cdecl")
    mlirMemRefTypeContiguousGetChecked.argtypes = [MlirLocation, MlirType, intptr_t, POINTER(c_int64), MlirAttribute]
    mlirMemRefTypeContiguousGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 369
if _libs["MLIRPythonCAPI"].has("mlirUnrankedMemRefTypeGet", "cdecl"):
    mlirUnrankedMemRefTypeGet = _libs["MLIRPythonCAPI"].get("mlirUnrankedMemRefTypeGet", "cdecl")
    mlirUnrankedMemRefTypeGet.argtypes = [MlirType, MlirAttribute]
    mlirUnrankedMemRefTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 373
if _libs["MLIRPythonCAPI"].has("mlirUnrankedMemRefTypeGetChecked", "cdecl"):
    mlirUnrankedMemRefTypeGetChecked = _libs["MLIRPythonCAPI"].get("mlirUnrankedMemRefTypeGetChecked", "cdecl")
    mlirUnrankedMemRefTypeGetChecked.argtypes = [MlirLocation, MlirType, MlirAttribute]
    mlirUnrankedMemRefTypeGetChecked.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 377
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGetLayout", "cdecl"):
    mlirMemRefTypeGetLayout = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGetLayout", "cdecl")
    mlirMemRefTypeGetLayout.argtypes = [MlirType]
    mlirMemRefTypeGetLayout.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 380
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGetAffineMap", "cdecl"):
    mlirMemRefTypeGetAffineMap = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGetAffineMap", "cdecl")
    mlirMemRefTypeGetAffineMap.argtypes = [MlirType]
    mlirMemRefTypeGetAffineMap.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 383
if _libs["MLIRPythonCAPI"].has("mlirMemRefTypeGetMemorySpace", "cdecl"):
    mlirMemRefTypeGetMemorySpace = _libs["MLIRPythonCAPI"].get("mlirMemRefTypeGetMemorySpace", "cdecl")
    mlirMemRefTypeGetMemorySpace.argtypes = [MlirType]
    mlirMemRefTypeGetMemorySpace.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 387
if _libs["MLIRPythonCAPI"].has("mlirUnrankedMemrefGetMemorySpace", "cdecl"):
    mlirUnrankedMemrefGetMemorySpace = _libs["MLIRPythonCAPI"].get("mlirUnrankedMemrefGetMemorySpace", "cdecl")
    mlirUnrankedMemrefGetMemorySpace.argtypes = [MlirType]
    mlirUnrankedMemrefGetMemorySpace.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 394
if _libs["MLIRPythonCAPI"].has("mlirTupleTypeGetTypeID", "cdecl"):
    mlirTupleTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirTupleTypeGetTypeID", "cdecl")
    mlirTupleTypeGetTypeID.argtypes = []
    mlirTupleTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 397
if _libs["MLIRPythonCAPI"].has("mlirTypeIsATuple", "cdecl"):
    mlirTypeIsATuple = _libs["MLIRPythonCAPI"].get("mlirTypeIsATuple", "cdecl")
    mlirTypeIsATuple.argtypes = [MlirType]
    mlirTypeIsATuple.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 401
if _libs["MLIRPythonCAPI"].has("mlirTupleTypeGet", "cdecl"):
    mlirTupleTypeGet = _libs["MLIRPythonCAPI"].get("mlirTupleTypeGet", "cdecl")
    mlirTupleTypeGet.argtypes = [MlirContext, intptr_t, POINTER(MlirType)]
    mlirTupleTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 406
if _libs["MLIRPythonCAPI"].has("mlirTupleTypeGetNumTypes", "cdecl"):
    mlirTupleTypeGetNumTypes = _libs["MLIRPythonCAPI"].get("mlirTupleTypeGetNumTypes", "cdecl")
    mlirTupleTypeGetNumTypes.argtypes = [MlirType]
    mlirTupleTypeGetNumTypes.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 409
if _libs["MLIRPythonCAPI"].has("mlirTupleTypeGetType", "cdecl"):
    mlirTupleTypeGetType = _libs["MLIRPythonCAPI"].get("mlirTupleTypeGetType", "cdecl")
    mlirTupleTypeGetType.argtypes = [MlirType, intptr_t]
    mlirTupleTypeGetType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 416
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGetTypeID", "cdecl"):
    mlirFunctionTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGetTypeID", "cdecl")
    mlirFunctionTypeGetTypeID.argtypes = []
    mlirFunctionTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 419
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAFunction", "cdecl"):
    mlirTypeIsAFunction = _libs["MLIRPythonCAPI"].get("mlirTypeIsAFunction", "cdecl")
    mlirTypeIsAFunction.argtypes = [MlirType]
    mlirTypeIsAFunction.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 422
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGet", "cdecl"):
    mlirFunctionTypeGet = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGet", "cdecl")
    mlirFunctionTypeGet.argtypes = [MlirContext, intptr_t, POINTER(MlirType), intptr_t, POINTER(MlirType)]
    mlirFunctionTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 429
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGetNumInputs", "cdecl"):
    mlirFunctionTypeGetNumInputs = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGetNumInputs", "cdecl")
    mlirFunctionTypeGetNumInputs.argtypes = [MlirType]
    mlirFunctionTypeGetNumInputs.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 432
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGetNumResults", "cdecl"):
    mlirFunctionTypeGetNumResults = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGetNumResults", "cdecl")
    mlirFunctionTypeGetNumResults.argtypes = [MlirType]
    mlirFunctionTypeGetNumResults.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 435
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGetInput", "cdecl"):
    mlirFunctionTypeGetInput = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGetInput", "cdecl")
    mlirFunctionTypeGetInput.argtypes = [MlirType, intptr_t]
    mlirFunctionTypeGetInput.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 439
if _libs["MLIRPythonCAPI"].has("mlirFunctionTypeGetResult", "cdecl"):
    mlirFunctionTypeGetResult = _libs["MLIRPythonCAPI"].get("mlirFunctionTypeGetResult", "cdecl")
    mlirFunctionTypeGetResult.argtypes = [MlirType, intptr_t]
    mlirFunctionTypeGetResult.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 447
if _libs["MLIRPythonCAPI"].has("mlirOpaqueTypeGetTypeID", "cdecl"):
    mlirOpaqueTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirOpaqueTypeGetTypeID", "cdecl")
    mlirOpaqueTypeGetTypeID.argtypes = []
    mlirOpaqueTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 450
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAOpaque", "cdecl"):
    mlirTypeIsAOpaque = _libs["MLIRPythonCAPI"].get("mlirTypeIsAOpaque", "cdecl")
    mlirTypeIsAOpaque.argtypes = [MlirType]
    mlirTypeIsAOpaque.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 455
if _libs["MLIRPythonCAPI"].has("mlirOpaqueTypeGet", "cdecl"):
    mlirOpaqueTypeGet = _libs["MLIRPythonCAPI"].get("mlirOpaqueTypeGet", "cdecl")
    mlirOpaqueTypeGet.argtypes = [MlirContext, MlirStringRef, MlirStringRef]
    mlirOpaqueTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 462
if _libs["MLIRPythonCAPI"].has("mlirOpaqueTypeGetDialectNamespace", "cdecl"):
    mlirOpaqueTypeGetDialectNamespace = _libs["MLIRPythonCAPI"].get("mlirOpaqueTypeGetDialectNamespace", "cdecl")
    mlirOpaqueTypeGetDialectNamespace.argtypes = [MlirType]
    mlirOpaqueTypeGetDialectNamespace.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinTypes.h: 466
if _libs["MLIRPythonCAPI"].has("mlirOpaqueTypeGetData", "cdecl"):
    mlirOpaqueTypeGetData = _libs["MLIRPythonCAPI"].get("mlirOpaqueTypeGetData", "cdecl")
    mlirOpaqueTypeGetData.argtypes = [MlirType]
    mlirOpaqueTypeGetData.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 26
if _libs["MLIRPythonCAPI"].has("mlirAttributeGetNull", "cdecl"):
    mlirAttributeGetNull = _libs["MLIRPythonCAPI"].get("mlirAttributeGetNull", "cdecl")
    mlirAttributeGetNull.argtypes = []
    mlirAttributeGetNull.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 32
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsALocation", "cdecl"):
    mlirAttributeIsALocation = _libs["MLIRPythonCAPI"].get("mlirAttributeIsALocation", "cdecl")
    mlirAttributeIsALocation.argtypes = [MlirAttribute]
    mlirAttributeIsALocation.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 39
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAAffineMap", "cdecl"):
    mlirAttributeIsAAffineMap = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAAffineMap", "cdecl")
    mlirAttributeIsAAffineMap.argtypes = [MlirAttribute]
    mlirAttributeIsAAffineMap.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 43
if _libs["MLIRPythonCAPI"].has("mlirAffineMapAttrGet", "cdecl"):
    mlirAffineMapAttrGet = _libs["MLIRPythonCAPI"].get("mlirAffineMapAttrGet", "cdecl")
    mlirAffineMapAttrGet.argtypes = [MlirAffineMap]
    mlirAffineMapAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 46
if _libs["MLIRPythonCAPI"].has("mlirAffineMapAttrGetValue", "cdecl"):
    mlirAffineMapAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirAffineMapAttrGetValue", "cdecl")
    mlirAffineMapAttrGetValue.argtypes = [MlirAttribute]
    mlirAffineMapAttrGetValue.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 49
if _libs["MLIRPythonCAPI"].has("mlirAffineMapAttrGetTypeID", "cdecl"):
    mlirAffineMapAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirAffineMapAttrGetTypeID", "cdecl")
    mlirAffineMapAttrGetTypeID.argtypes = []
    mlirAffineMapAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 56
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAArray", "cdecl"):
    mlirAttributeIsAArray = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAArray", "cdecl")
    mlirAttributeIsAArray.argtypes = [MlirAttribute]
    mlirAttributeIsAArray.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 60
if _libs["MLIRPythonCAPI"].has("mlirArrayAttrGet", "cdecl"):
    mlirArrayAttrGet = _libs["MLIRPythonCAPI"].get("mlirArrayAttrGet", "cdecl")
    mlirArrayAttrGet.argtypes = [MlirContext, intptr_t, POINTER(MlirAttribute)]
    mlirArrayAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 64
if _libs["MLIRPythonCAPI"].has("mlirArrayAttrGetNumElements", "cdecl"):
    mlirArrayAttrGetNumElements = _libs["MLIRPythonCAPI"].get("mlirArrayAttrGetNumElements", "cdecl")
    mlirArrayAttrGetNumElements.argtypes = [MlirAttribute]
    mlirArrayAttrGetNumElements.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 67
if _libs["MLIRPythonCAPI"].has("mlirArrayAttrGetElement", "cdecl"):
    mlirArrayAttrGetElement = _libs["MLIRPythonCAPI"].get("mlirArrayAttrGetElement", "cdecl")
    mlirArrayAttrGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirArrayAttrGetElement.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 71
if _libs["MLIRPythonCAPI"].has("mlirArrayAttrGetTypeID", "cdecl"):
    mlirArrayAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirArrayAttrGetTypeID", "cdecl")
    mlirArrayAttrGetTypeID.argtypes = []
    mlirArrayAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 78
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADictionary", "cdecl"):
    mlirAttributeIsADictionary = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADictionary", "cdecl")
    mlirAttributeIsADictionary.argtypes = [MlirAttribute]
    mlirAttributeIsADictionary.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 82
if _libs["MLIRPythonCAPI"].has("mlirDictionaryAttrGet", "cdecl"):
    mlirDictionaryAttrGet = _libs["MLIRPythonCAPI"].get("mlirDictionaryAttrGet", "cdecl")
    mlirDictionaryAttrGet.argtypes = [MlirContext, intptr_t, POINTER(MlirNamedAttribute)]
    mlirDictionaryAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 87
if _libs["MLIRPythonCAPI"].has("mlirDictionaryAttrGetNumElements", "cdecl"):
    mlirDictionaryAttrGetNumElements = _libs["MLIRPythonCAPI"].get("mlirDictionaryAttrGetNumElements", "cdecl")
    mlirDictionaryAttrGetNumElements.argtypes = [MlirAttribute]
    mlirDictionaryAttrGetNumElements.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 91
if _libs["MLIRPythonCAPI"].has("mlirDictionaryAttrGetElement", "cdecl"):
    mlirDictionaryAttrGetElement = _libs["MLIRPythonCAPI"].get("mlirDictionaryAttrGetElement", "cdecl")
    mlirDictionaryAttrGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDictionaryAttrGetElement.restype = MlirNamedAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 96
if _libs["MLIRPythonCAPI"].has("mlirDictionaryAttrGetElementByName", "cdecl"):
    mlirDictionaryAttrGetElementByName = _libs["MLIRPythonCAPI"].get("mlirDictionaryAttrGetElementByName", "cdecl")
    mlirDictionaryAttrGetElementByName.argtypes = [MlirAttribute, MlirStringRef]
    mlirDictionaryAttrGetElementByName.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 99
if _libs["MLIRPythonCAPI"].has("mlirDictionaryAttrGetTypeID", "cdecl"):
    mlirDictionaryAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirDictionaryAttrGetTypeID", "cdecl")
    mlirDictionaryAttrGetTypeID.argtypes = []
    mlirDictionaryAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 109
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAFloat", "cdecl"):
    mlirAttributeIsAFloat = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAFloat", "cdecl")
    mlirAttributeIsAFloat.argtypes = [MlirAttribute]
    mlirAttributeIsAFloat.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 113
if _libs["MLIRPythonCAPI"].has("mlirFloatAttrDoubleGet", "cdecl"):
    mlirFloatAttrDoubleGet = _libs["MLIRPythonCAPI"].get("mlirFloatAttrDoubleGet", "cdecl")
    mlirFloatAttrDoubleGet.argtypes = [MlirContext, MlirType, c_double]
    mlirFloatAttrDoubleGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 119
if _libs["MLIRPythonCAPI"].has("mlirFloatAttrDoubleGetChecked", "cdecl"):
    mlirFloatAttrDoubleGetChecked = _libs["MLIRPythonCAPI"].get("mlirFloatAttrDoubleGetChecked", "cdecl")
    mlirFloatAttrDoubleGetChecked.argtypes = [MlirLocation, MlirType, c_double]
    mlirFloatAttrDoubleGetChecked.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 125
if _libs["MLIRPythonCAPI"].has("mlirFloatAttrGetValueDouble", "cdecl"):
    mlirFloatAttrGetValueDouble = _libs["MLIRPythonCAPI"].get("mlirFloatAttrGetValueDouble", "cdecl")
    mlirFloatAttrGetValueDouble.argtypes = [MlirAttribute]
    mlirFloatAttrGetValueDouble.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 128
if _libs["MLIRPythonCAPI"].has("mlirFloatAttrGetTypeID", "cdecl"):
    mlirFloatAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirFloatAttrGetTypeID", "cdecl")
    mlirFloatAttrGetTypeID.argtypes = []
    mlirFloatAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 138
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAInteger", "cdecl"):
    mlirAttributeIsAInteger = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAInteger", "cdecl")
    mlirAttributeIsAInteger.argtypes = [MlirAttribute]
    mlirAttributeIsAInteger.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 142
if _libs["MLIRPythonCAPI"].has("mlirIntegerAttrGet", "cdecl"):
    mlirIntegerAttrGet = _libs["MLIRPythonCAPI"].get("mlirIntegerAttrGet", "cdecl")
    mlirIntegerAttrGet.argtypes = [MlirType, c_int64]
    mlirIntegerAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 147
if _libs["MLIRPythonCAPI"].has("mlirIntegerAttrGetValueInt", "cdecl"):
    mlirIntegerAttrGetValueInt = _libs["MLIRPythonCAPI"].get("mlirIntegerAttrGetValueInt", "cdecl")
    mlirIntegerAttrGetValueInt.argtypes = [MlirAttribute]
    mlirIntegerAttrGetValueInt.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 151
if _libs["MLIRPythonCAPI"].has("mlirIntegerAttrGetValueSInt", "cdecl"):
    mlirIntegerAttrGetValueSInt = _libs["MLIRPythonCAPI"].get("mlirIntegerAttrGetValueSInt", "cdecl")
    mlirIntegerAttrGetValueSInt.argtypes = [MlirAttribute]
    mlirIntegerAttrGetValueSInt.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 155
if _libs["MLIRPythonCAPI"].has("mlirIntegerAttrGetValueUInt", "cdecl"):
    mlirIntegerAttrGetValueUInt = _libs["MLIRPythonCAPI"].get("mlirIntegerAttrGetValueUInt", "cdecl")
    mlirIntegerAttrGetValueUInt.argtypes = [MlirAttribute]
    mlirIntegerAttrGetValueUInt.restype = uint64_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 158
if _libs["MLIRPythonCAPI"].has("mlirIntegerAttrGetTypeID", "cdecl"):
    mlirIntegerAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirIntegerAttrGetTypeID", "cdecl")
    mlirIntegerAttrGetTypeID.argtypes = []
    mlirIntegerAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 165
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsABool", "cdecl"):
    mlirAttributeIsABool = _libs["MLIRPythonCAPI"].get("mlirAttributeIsABool", "cdecl")
    mlirAttributeIsABool.argtypes = [MlirAttribute]
    mlirAttributeIsABool.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 168
if _libs["MLIRPythonCAPI"].has("mlirBoolAttrGet", "cdecl"):
    mlirBoolAttrGet = _libs["MLIRPythonCAPI"].get("mlirBoolAttrGet", "cdecl")
    mlirBoolAttrGet.argtypes = [MlirContext, c_int]
    mlirBoolAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 171
if _libs["MLIRPythonCAPI"].has("mlirBoolAttrGetValue", "cdecl"):
    mlirBoolAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirBoolAttrGetValue", "cdecl")
    mlirBoolAttrGetValue.argtypes = [MlirAttribute]
    mlirBoolAttrGetValue.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 178
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAIntegerSet", "cdecl"):
    mlirAttributeIsAIntegerSet = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAIntegerSet", "cdecl")
    mlirAttributeIsAIntegerSet.argtypes = [MlirAttribute]
    mlirAttributeIsAIntegerSet.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 181
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetAttrGetTypeID", "cdecl"):
    mlirIntegerSetAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirIntegerSetAttrGetTypeID", "cdecl")
    mlirIntegerSetAttrGetTypeID.argtypes = []
    mlirIntegerSetAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 188
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAOpaque", "cdecl"):
    mlirAttributeIsAOpaque = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAOpaque", "cdecl")
    mlirAttributeIsAOpaque.argtypes = [MlirAttribute]
    mlirAttributeIsAOpaque.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 194
if _libs["MLIRPythonCAPI"].has("mlirOpaqueAttrGet", "cdecl"):
    mlirOpaqueAttrGet = _libs["MLIRPythonCAPI"].get("mlirOpaqueAttrGet", "cdecl")
    mlirOpaqueAttrGet.argtypes = [MlirContext, MlirStringRef, intptr_t, String, MlirType]
    mlirOpaqueAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 200
if _libs["MLIRPythonCAPI"].has("mlirOpaqueAttrGetDialectNamespace", "cdecl"):
    mlirOpaqueAttrGetDialectNamespace = _libs["MLIRPythonCAPI"].get("mlirOpaqueAttrGetDialectNamespace", "cdecl")
    mlirOpaqueAttrGetDialectNamespace.argtypes = [MlirAttribute]
    mlirOpaqueAttrGetDialectNamespace.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 204
if _libs["MLIRPythonCAPI"].has("mlirOpaqueAttrGetData", "cdecl"):
    mlirOpaqueAttrGetData = _libs["MLIRPythonCAPI"].get("mlirOpaqueAttrGetData", "cdecl")
    mlirOpaqueAttrGetData.argtypes = [MlirAttribute]
    mlirOpaqueAttrGetData.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 207
if _libs["MLIRPythonCAPI"].has("mlirOpaqueAttrGetTypeID", "cdecl"):
    mlirOpaqueAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirOpaqueAttrGetTypeID", "cdecl")
    mlirOpaqueAttrGetTypeID.argtypes = []
    mlirOpaqueAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 214
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAString", "cdecl"):
    mlirAttributeIsAString = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAString", "cdecl")
    mlirAttributeIsAString.argtypes = [MlirAttribute]
    mlirAttributeIsAString.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 218
if _libs["MLIRPythonCAPI"].has("mlirStringAttrGet", "cdecl"):
    mlirStringAttrGet = _libs["MLIRPythonCAPI"].get("mlirStringAttrGet", "cdecl")
    mlirStringAttrGet.argtypes = [MlirContext, MlirStringRef]
    mlirStringAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 223
if _libs["MLIRPythonCAPI"].has("mlirStringAttrTypedGet", "cdecl"):
    mlirStringAttrTypedGet = _libs["MLIRPythonCAPI"].get("mlirStringAttrTypedGet", "cdecl")
    mlirStringAttrTypedGet.argtypes = [MlirType, MlirStringRef]
    mlirStringAttrTypedGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 228
if _libs["MLIRPythonCAPI"].has("mlirStringAttrGetValue", "cdecl"):
    mlirStringAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirStringAttrGetValue", "cdecl")
    mlirStringAttrGetValue.argtypes = [MlirAttribute]
    mlirStringAttrGetValue.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 231
if _libs["MLIRPythonCAPI"].has("mlirStringAttrGetTypeID", "cdecl"):
    mlirStringAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirStringAttrGetTypeID", "cdecl")
    mlirStringAttrGetTypeID.argtypes = []
    mlirStringAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 238
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsASymbolRef", "cdecl"):
    mlirAttributeIsASymbolRef = _libs["MLIRPythonCAPI"].get("mlirAttributeIsASymbolRef", "cdecl")
    mlirAttributeIsASymbolRef.argtypes = [MlirAttribute]
    mlirAttributeIsASymbolRef.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 244
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGet", "cdecl"):
    mlirSymbolRefAttrGet = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGet", "cdecl")
    mlirSymbolRefAttrGet.argtypes = [MlirContext, MlirStringRef, intptr_t, POINTER(MlirAttribute)]
    mlirSymbolRefAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 250
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGetRootReference", "cdecl"):
    mlirSymbolRefAttrGetRootReference = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGetRootReference", "cdecl")
    mlirSymbolRefAttrGetRootReference.argtypes = [MlirAttribute]
    mlirSymbolRefAttrGetRootReference.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 255
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGetLeafReference", "cdecl"):
    mlirSymbolRefAttrGetLeafReference = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGetLeafReference", "cdecl")
    mlirSymbolRefAttrGetLeafReference.argtypes = [MlirAttribute]
    mlirSymbolRefAttrGetLeafReference.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 260
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGetNumNestedReferences", "cdecl"):
    mlirSymbolRefAttrGetNumNestedReferences = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGetNumNestedReferences", "cdecl")
    mlirSymbolRefAttrGetNumNestedReferences.argtypes = [MlirAttribute]
    mlirSymbolRefAttrGetNumNestedReferences.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 264
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGetNestedReference", "cdecl"):
    mlirSymbolRefAttrGetNestedReference = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGetNestedReference", "cdecl")
    mlirSymbolRefAttrGetNestedReference.argtypes = [MlirAttribute, intptr_t]
    mlirSymbolRefAttrGetNestedReference.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 267
if _libs["MLIRPythonCAPI"].has("mlirSymbolRefAttrGetTypeID", "cdecl"):
    mlirSymbolRefAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirSymbolRefAttrGetTypeID", "cdecl")
    mlirSymbolRefAttrGetTypeID.argtypes = []
    mlirSymbolRefAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 274
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAFlatSymbolRef", "cdecl"):
    mlirAttributeIsAFlatSymbolRef = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAFlatSymbolRef", "cdecl")
    mlirAttributeIsAFlatSymbolRef.argtypes = [MlirAttribute]
    mlirAttributeIsAFlatSymbolRef.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 278
if _libs["MLIRPythonCAPI"].has("mlirFlatSymbolRefAttrGet", "cdecl"):
    mlirFlatSymbolRefAttrGet = _libs["MLIRPythonCAPI"].get("mlirFlatSymbolRefAttrGet", "cdecl")
    mlirFlatSymbolRefAttrGet.argtypes = [MlirContext, MlirStringRef]
    mlirFlatSymbolRefAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 284
if _libs["MLIRPythonCAPI"].has("mlirFlatSymbolRefAttrGetValue", "cdecl"):
    mlirFlatSymbolRefAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirFlatSymbolRefAttrGetValue", "cdecl")
    mlirFlatSymbolRefAttrGetValue.argtypes = [MlirAttribute]
    mlirFlatSymbolRefAttrGetValue.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 291
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAType", "cdecl"):
    mlirAttributeIsAType = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAType", "cdecl")
    mlirAttributeIsAType.argtypes = [MlirAttribute]
    mlirAttributeIsAType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 295
if _libs["MLIRPythonCAPI"].has("mlirTypeAttrGet", "cdecl"):
    mlirTypeAttrGet = _libs["MLIRPythonCAPI"].get("mlirTypeAttrGet", "cdecl")
    mlirTypeAttrGet.argtypes = [MlirType]
    mlirTypeAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 298
if _libs["MLIRPythonCAPI"].has("mlirTypeAttrGetValue", "cdecl"):
    mlirTypeAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirTypeAttrGetValue", "cdecl")
    mlirTypeAttrGetValue.argtypes = [MlirAttribute]
    mlirTypeAttrGetValue.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 301
if _libs["MLIRPythonCAPI"].has("mlirTypeAttrGetTypeID", "cdecl"):
    mlirTypeAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirTypeAttrGetTypeID", "cdecl")
    mlirTypeAttrGetTypeID.argtypes = []
    mlirTypeAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 308
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAUnit", "cdecl"):
    mlirAttributeIsAUnit = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAUnit", "cdecl")
    mlirAttributeIsAUnit.argtypes = [MlirAttribute]
    mlirAttributeIsAUnit.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 311
if _libs["MLIRPythonCAPI"].has("mlirUnitAttrGet", "cdecl"):
    mlirUnitAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnitAttrGet", "cdecl")
    mlirUnitAttrGet.argtypes = [MlirContext]
    mlirUnitAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 314
if _libs["MLIRPythonCAPI"].has("mlirUnitAttrGetTypeID", "cdecl"):
    mlirUnitAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirUnitAttrGetTypeID", "cdecl")
    mlirUnitAttrGetTypeID.argtypes = []
    mlirUnitAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 321
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAElements", "cdecl"):
    mlirAttributeIsAElements = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAElements", "cdecl")
    mlirAttributeIsAElements.argtypes = [MlirAttribute]
    mlirAttributeIsAElements.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 324
if _libs["MLIRPythonCAPI"].has("mlirElementsAttrGetValue", "cdecl"):
    mlirElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirElementsAttrGetValue", "cdecl")
    mlirElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t, POINTER(uint64_t)]
    mlirElementsAttrGetValue.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 331
if _libs["MLIRPythonCAPI"].has("mlirElementsAttrIsValidIndex", "cdecl"):
    mlirElementsAttrIsValidIndex = _libs["MLIRPythonCAPI"].get("mlirElementsAttrIsValidIndex", "cdecl")
    mlirElementsAttrIsValidIndex.argtypes = [MlirAttribute, intptr_t, POINTER(uint64_t)]
    mlirElementsAttrIsValidIndex.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 336
if _libs["MLIRPythonCAPI"].has("mlirElementsAttrGetNumElements", "cdecl"):
    mlirElementsAttrGetNumElements = _libs["MLIRPythonCAPI"].get("mlirElementsAttrGetNumElements", "cdecl")
    mlirElementsAttrGetNumElements.argtypes = [MlirAttribute]
    mlirElementsAttrGetNumElements.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 342
if _libs["MLIRPythonCAPI"].has("mlirDenseArrayAttrGetTypeID", "cdecl"):
    mlirDenseArrayAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirDenseArrayAttrGetTypeID", "cdecl")
    mlirDenseArrayAttrGetTypeID.argtypes = []
    mlirDenseArrayAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 345
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseBoolArray", "cdecl"):
    mlirAttributeIsADenseBoolArray = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseBoolArray", "cdecl")
    mlirAttributeIsADenseBoolArray.argtypes = [MlirAttribute]
    mlirAttributeIsADenseBoolArray.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 346
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseI8Array", "cdecl"):
    mlirAttributeIsADenseI8Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseI8Array", "cdecl")
    mlirAttributeIsADenseI8Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseI8Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 347
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseI16Array", "cdecl"):
    mlirAttributeIsADenseI16Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseI16Array", "cdecl")
    mlirAttributeIsADenseI16Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseI16Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 348
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseI32Array", "cdecl"):
    mlirAttributeIsADenseI32Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseI32Array", "cdecl")
    mlirAttributeIsADenseI32Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseI32Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 349
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseI64Array", "cdecl"):
    mlirAttributeIsADenseI64Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseI64Array", "cdecl")
    mlirAttributeIsADenseI64Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseI64Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 350
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseF32Array", "cdecl"):
    mlirAttributeIsADenseF32Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseF32Array", "cdecl")
    mlirAttributeIsADenseF32Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseF32Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 351
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseF64Array", "cdecl"):
    mlirAttributeIsADenseF64Array = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseF64Array", "cdecl")
    mlirAttributeIsADenseF64Array.argtypes = [MlirAttribute]
    mlirAttributeIsADenseF64Array.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 354
if _libs["MLIRPythonCAPI"].has("mlirDenseBoolArrayGet", "cdecl"):
    mlirDenseBoolArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseBoolArrayGet", "cdecl")
    mlirDenseBoolArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_int)]
    mlirDenseBoolArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 357
if _libs["MLIRPythonCAPI"].has("mlirDenseI8ArrayGet", "cdecl"):
    mlirDenseI8ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseI8ArrayGet", "cdecl")
    mlirDenseI8ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_int8)]
    mlirDenseI8ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 360
if _libs["MLIRPythonCAPI"].has("mlirDenseI16ArrayGet", "cdecl"):
    mlirDenseI16ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseI16ArrayGet", "cdecl")
    mlirDenseI16ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_int16)]
    mlirDenseI16ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 363
if _libs["MLIRPythonCAPI"].has("mlirDenseI32ArrayGet", "cdecl"):
    mlirDenseI32ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseI32ArrayGet", "cdecl")
    mlirDenseI32ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_int32)]
    mlirDenseI32ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 366
if _libs["MLIRPythonCAPI"].has("mlirDenseI64ArrayGet", "cdecl"):
    mlirDenseI64ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseI64ArrayGet", "cdecl")
    mlirDenseI64ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_int64)]
    mlirDenseI64ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 369
if _libs["MLIRPythonCAPI"].has("mlirDenseF32ArrayGet", "cdecl"):
    mlirDenseF32ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseF32ArrayGet", "cdecl")
    mlirDenseF32ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_float)]
    mlirDenseF32ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 372
if _libs["MLIRPythonCAPI"].has("mlirDenseF64ArrayGet", "cdecl"):
    mlirDenseF64ArrayGet = _libs["MLIRPythonCAPI"].get("mlirDenseF64ArrayGet", "cdecl")
    mlirDenseF64ArrayGet.argtypes = [MlirContext, intptr_t, POINTER(c_double)]
    mlirDenseF64ArrayGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 377
if _libs["MLIRPythonCAPI"].has("mlirDenseArrayGetNumElements", "cdecl"):
    mlirDenseArrayGetNumElements = _libs["MLIRPythonCAPI"].get("mlirDenseArrayGetNumElements", "cdecl")
    mlirDenseArrayGetNumElements.argtypes = [MlirAttribute]
    mlirDenseArrayGetNumElements.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 380
if _libs["MLIRPythonCAPI"].has("mlirDenseBoolArrayGetElement", "cdecl"):
    mlirDenseBoolArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseBoolArrayGetElement", "cdecl")
    mlirDenseBoolArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseBoolArrayGetElement.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 382
if _libs["MLIRPythonCAPI"].has("mlirDenseI8ArrayGetElement", "cdecl"):
    mlirDenseI8ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseI8ArrayGetElement", "cdecl")
    mlirDenseI8ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseI8ArrayGetElement.restype = c_int8

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 384
if _libs["MLIRPythonCAPI"].has("mlirDenseI16ArrayGetElement", "cdecl"):
    mlirDenseI16ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseI16ArrayGetElement", "cdecl")
    mlirDenseI16ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseI16ArrayGetElement.restype = c_int16

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 386
if _libs["MLIRPythonCAPI"].has("mlirDenseI32ArrayGetElement", "cdecl"):
    mlirDenseI32ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseI32ArrayGetElement", "cdecl")
    mlirDenseI32ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseI32ArrayGetElement.restype = c_int32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 388
if _libs["MLIRPythonCAPI"].has("mlirDenseI64ArrayGetElement", "cdecl"):
    mlirDenseI64ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseI64ArrayGetElement", "cdecl")
    mlirDenseI64ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseI64ArrayGetElement.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 390
if _libs["MLIRPythonCAPI"].has("mlirDenseF32ArrayGetElement", "cdecl"):
    mlirDenseF32ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseF32ArrayGetElement", "cdecl")
    mlirDenseF32ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseF32ArrayGetElement.restype = c_float

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 392
if _libs["MLIRPythonCAPI"].has("mlirDenseF64ArrayGetElement", "cdecl"):
    mlirDenseF64ArrayGetElement = _libs["MLIRPythonCAPI"].get("mlirDenseF64ArrayGetElement", "cdecl")
    mlirDenseF64ArrayGetElement.argtypes = [MlirAttribute, intptr_t]
    mlirDenseF64ArrayGetElement.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 404
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseElements", "cdecl"):
    mlirAttributeIsADenseElements = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseElements", "cdecl")
    mlirAttributeIsADenseElements.argtypes = [MlirAttribute]
    mlirAttributeIsADenseElements.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 405
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseIntElements", "cdecl"):
    mlirAttributeIsADenseIntElements = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseIntElements", "cdecl")
    mlirAttributeIsADenseIntElements.argtypes = [MlirAttribute]
    mlirAttributeIsADenseIntElements.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 406
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsADenseFPElements", "cdecl"):
    mlirAttributeIsADenseFPElements = _libs["MLIRPythonCAPI"].get("mlirAttributeIsADenseFPElements", "cdecl")
    mlirAttributeIsADenseFPElements.argtypes = [MlirAttribute]
    mlirAttributeIsADenseFPElements.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 409
if _libs["MLIRPythonCAPI"].has("mlirDenseIntOrFPElementsAttrGetTypeID", "cdecl"):
    mlirDenseIntOrFPElementsAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirDenseIntOrFPElementsAttrGetTypeID", "cdecl")
    mlirDenseIntOrFPElementsAttrGetTypeID.argtypes = []
    mlirDenseIntOrFPElementsAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 413
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGet", "cdecl"):
    mlirDenseElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGet", "cdecl")
    mlirDenseElementsAttrGet.argtypes = [MlirType, intptr_t, POINTER(MlirAttribute)]
    mlirDenseElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 430
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrRawBufferGet", "cdecl"):
    mlirDenseElementsAttrRawBufferGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrRawBufferGet", "cdecl")
    mlirDenseElementsAttrRawBufferGet.argtypes = [MlirType, c_size_t, POINTER(None)]
    mlirDenseElementsAttrRawBufferGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 436
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrSplatGet", "cdecl"):
    mlirDenseElementsAttrSplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrSplatGet", "cdecl")
    mlirDenseElementsAttrSplatGet.argtypes = [MlirType, MlirAttribute]
    mlirDenseElementsAttrSplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 438
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrBoolSplatGet", "cdecl"):
    mlirDenseElementsAttrBoolSplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrBoolSplatGet", "cdecl")
    mlirDenseElementsAttrBoolSplatGet.argtypes = [MlirType, c_bool]
    mlirDenseElementsAttrBoolSplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 440
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt8SplatGet", "cdecl"):
    mlirDenseElementsAttrUInt8SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt8SplatGet", "cdecl")
    mlirDenseElementsAttrUInt8SplatGet.argtypes = [MlirType, uint8_t]
    mlirDenseElementsAttrUInt8SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 442
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt8SplatGet", "cdecl"):
    mlirDenseElementsAttrInt8SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt8SplatGet", "cdecl")
    mlirDenseElementsAttrInt8SplatGet.argtypes = [MlirType, c_int8]
    mlirDenseElementsAttrInt8SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 444
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt32SplatGet", "cdecl"):
    mlirDenseElementsAttrUInt32SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt32SplatGet", "cdecl")
    mlirDenseElementsAttrUInt32SplatGet.argtypes = [MlirType, uint32_t]
    mlirDenseElementsAttrUInt32SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 446
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt32SplatGet", "cdecl"):
    mlirDenseElementsAttrInt32SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt32SplatGet", "cdecl")
    mlirDenseElementsAttrInt32SplatGet.argtypes = [MlirType, c_int32]
    mlirDenseElementsAttrInt32SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 448
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt64SplatGet", "cdecl"):
    mlirDenseElementsAttrUInt64SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt64SplatGet", "cdecl")
    mlirDenseElementsAttrUInt64SplatGet.argtypes = [MlirType, uint64_t]
    mlirDenseElementsAttrUInt64SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 450
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt64SplatGet", "cdecl"):
    mlirDenseElementsAttrInt64SplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt64SplatGet", "cdecl")
    mlirDenseElementsAttrInt64SplatGet.argtypes = [MlirType, c_int64]
    mlirDenseElementsAttrInt64SplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 452
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrFloatSplatGet", "cdecl"):
    mlirDenseElementsAttrFloatSplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrFloatSplatGet", "cdecl")
    mlirDenseElementsAttrFloatSplatGet.argtypes = [MlirType, c_float]
    mlirDenseElementsAttrFloatSplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 454
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrDoubleSplatGet", "cdecl"):
    mlirDenseElementsAttrDoubleSplatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrDoubleSplatGet", "cdecl")
    mlirDenseElementsAttrDoubleSplatGet.argtypes = [MlirType, c_double]
    mlirDenseElementsAttrDoubleSplatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 459
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrBoolGet", "cdecl"):
    mlirDenseElementsAttrBoolGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrBoolGet", "cdecl")
    mlirDenseElementsAttrBoolGet.argtypes = [MlirType, intptr_t, POINTER(c_int)]
    mlirDenseElementsAttrBoolGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 461
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt8Get", "cdecl"):
    mlirDenseElementsAttrUInt8Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt8Get", "cdecl")
    mlirDenseElementsAttrUInt8Get.argtypes = [MlirType, intptr_t, POINTER(uint8_t)]
    mlirDenseElementsAttrUInt8Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 463
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt8Get", "cdecl"):
    mlirDenseElementsAttrInt8Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt8Get", "cdecl")
    mlirDenseElementsAttrInt8Get.argtypes = [MlirType, intptr_t, POINTER(c_int8)]
    mlirDenseElementsAttrInt8Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 465
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt16Get", "cdecl"):
    mlirDenseElementsAttrUInt16Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt16Get", "cdecl")
    mlirDenseElementsAttrUInt16Get.argtypes = [MlirType, intptr_t, POINTER(uint16_t)]
    mlirDenseElementsAttrUInt16Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 467
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt16Get", "cdecl"):
    mlirDenseElementsAttrInt16Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt16Get", "cdecl")
    mlirDenseElementsAttrInt16Get.argtypes = [MlirType, intptr_t, POINTER(c_int16)]
    mlirDenseElementsAttrInt16Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 469
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt32Get", "cdecl"):
    mlirDenseElementsAttrUInt32Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt32Get", "cdecl")
    mlirDenseElementsAttrUInt32Get.argtypes = [MlirType, intptr_t, POINTER(uint32_t)]
    mlirDenseElementsAttrUInt32Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 471
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt32Get", "cdecl"):
    mlirDenseElementsAttrInt32Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt32Get", "cdecl")
    mlirDenseElementsAttrInt32Get.argtypes = [MlirType, intptr_t, POINTER(c_int32)]
    mlirDenseElementsAttrInt32Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 473
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrUInt64Get", "cdecl"):
    mlirDenseElementsAttrUInt64Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrUInt64Get", "cdecl")
    mlirDenseElementsAttrUInt64Get.argtypes = [MlirType, intptr_t, POINTER(uint64_t)]
    mlirDenseElementsAttrUInt64Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 475
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrInt64Get", "cdecl"):
    mlirDenseElementsAttrInt64Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrInt64Get", "cdecl")
    mlirDenseElementsAttrInt64Get.argtypes = [MlirType, intptr_t, POINTER(c_int64)]
    mlirDenseElementsAttrInt64Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 477
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrFloatGet", "cdecl"):
    mlirDenseElementsAttrFloatGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrFloatGet", "cdecl")
    mlirDenseElementsAttrFloatGet.argtypes = [MlirType, intptr_t, POINTER(c_float)]
    mlirDenseElementsAttrFloatGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 479
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrDoubleGet", "cdecl"):
    mlirDenseElementsAttrDoubleGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrDoubleGet", "cdecl")
    mlirDenseElementsAttrDoubleGet.argtypes = [MlirType, intptr_t, POINTER(c_double)]
    mlirDenseElementsAttrDoubleGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 481
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrBFloat16Get", "cdecl"):
    mlirDenseElementsAttrBFloat16Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrBFloat16Get", "cdecl")
    mlirDenseElementsAttrBFloat16Get.argtypes = [MlirType, intptr_t, POINTER(uint16_t)]
    mlirDenseElementsAttrBFloat16Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 483
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrFloat16Get", "cdecl"):
    mlirDenseElementsAttrFloat16Get = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrFloat16Get", "cdecl")
    mlirDenseElementsAttrFloat16Get.argtypes = [MlirType, intptr_t, POINTER(uint16_t)]
    mlirDenseElementsAttrFloat16Get.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 488
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrStringGet", "cdecl"):
    mlirDenseElementsAttrStringGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrStringGet", "cdecl")
    mlirDenseElementsAttrStringGet.argtypes = [MlirType, intptr_t, POINTER(MlirStringRef)]
    mlirDenseElementsAttrStringGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 495
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrReshapeGet", "cdecl"):
    mlirDenseElementsAttrReshapeGet = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrReshapeGet", "cdecl")
    mlirDenseElementsAttrReshapeGet.argtypes = [MlirAttribute, MlirType]
    mlirDenseElementsAttrReshapeGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 499
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrIsSplat", "cdecl"):
    mlirDenseElementsAttrIsSplat = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrIsSplat", "cdecl")
    mlirDenseElementsAttrIsSplat.argtypes = [MlirAttribute]
    mlirDenseElementsAttrIsSplat.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 504
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetSplatValue", "cdecl"):
    mlirDenseElementsAttrGetSplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetSplatValue", "cdecl")
    mlirDenseElementsAttrGetSplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetSplatValue.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 506
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetBoolSplatValue", "cdecl"):
    mlirDenseElementsAttrGetBoolSplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetBoolSplatValue", "cdecl")
    mlirDenseElementsAttrGetBoolSplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetBoolSplatValue.restype = c_int

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 508
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt8SplatValue", "cdecl"):
    mlirDenseElementsAttrGetInt8SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt8SplatValue", "cdecl")
    mlirDenseElementsAttrGetInt8SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetInt8SplatValue.restype = c_int8

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 510
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt8SplatValue", "cdecl"):
    mlirDenseElementsAttrGetUInt8SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt8SplatValue", "cdecl")
    mlirDenseElementsAttrGetUInt8SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetUInt8SplatValue.restype = uint8_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 512
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt32SplatValue", "cdecl"):
    mlirDenseElementsAttrGetInt32SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt32SplatValue", "cdecl")
    mlirDenseElementsAttrGetInt32SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetInt32SplatValue.restype = c_int32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 514
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt32SplatValue", "cdecl"):
    mlirDenseElementsAttrGetUInt32SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt32SplatValue", "cdecl")
    mlirDenseElementsAttrGetUInt32SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetUInt32SplatValue.restype = uint32_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 516
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt64SplatValue", "cdecl"):
    mlirDenseElementsAttrGetInt64SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt64SplatValue", "cdecl")
    mlirDenseElementsAttrGetInt64SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetInt64SplatValue.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 518
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt64SplatValue", "cdecl"):
    mlirDenseElementsAttrGetUInt64SplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt64SplatValue", "cdecl")
    mlirDenseElementsAttrGetUInt64SplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetUInt64SplatValue.restype = uint64_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 520
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetFloatSplatValue", "cdecl"):
    mlirDenseElementsAttrGetFloatSplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetFloatSplatValue", "cdecl")
    mlirDenseElementsAttrGetFloatSplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetFloatSplatValue.restype = c_float

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 522
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetDoubleSplatValue", "cdecl"):
    mlirDenseElementsAttrGetDoubleSplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetDoubleSplatValue", "cdecl")
    mlirDenseElementsAttrGetDoubleSplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetDoubleSplatValue.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 524
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetStringSplatValue", "cdecl"):
    mlirDenseElementsAttrGetStringSplatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetStringSplatValue", "cdecl")
    mlirDenseElementsAttrGetStringSplatValue.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetStringSplatValue.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 528
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetBoolValue", "cdecl"):
    mlirDenseElementsAttrGetBoolValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetBoolValue", "cdecl")
    mlirDenseElementsAttrGetBoolValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetBoolValue.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 530
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt8Value", "cdecl"):
    mlirDenseElementsAttrGetInt8Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt8Value", "cdecl")
    mlirDenseElementsAttrGetInt8Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetInt8Value.restype = c_int8

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 533
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt8Value", "cdecl"):
    mlirDenseElementsAttrGetUInt8Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt8Value", "cdecl")
    mlirDenseElementsAttrGetUInt8Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetUInt8Value.restype = uint8_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 535
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt16Value", "cdecl"):
    mlirDenseElementsAttrGetInt16Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt16Value", "cdecl")
    mlirDenseElementsAttrGetInt16Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetInt16Value.restype = c_int16

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 537
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt16Value", "cdecl"):
    mlirDenseElementsAttrGetUInt16Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt16Value", "cdecl")
    mlirDenseElementsAttrGetUInt16Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetUInt16Value.restype = uint16_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 539
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt32Value", "cdecl"):
    mlirDenseElementsAttrGetInt32Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt32Value", "cdecl")
    mlirDenseElementsAttrGetInt32Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetInt32Value.restype = c_int32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 541
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt32Value", "cdecl"):
    mlirDenseElementsAttrGetUInt32Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt32Value", "cdecl")
    mlirDenseElementsAttrGetUInt32Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetUInt32Value.restype = uint32_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 543
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetInt64Value", "cdecl"):
    mlirDenseElementsAttrGetInt64Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetInt64Value", "cdecl")
    mlirDenseElementsAttrGetInt64Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetInt64Value.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 545
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetUInt64Value", "cdecl"):
    mlirDenseElementsAttrGetUInt64Value = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetUInt64Value", "cdecl")
    mlirDenseElementsAttrGetUInt64Value.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetUInt64Value.restype = uint64_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 546
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetFloatValue", "cdecl"):
    mlirDenseElementsAttrGetFloatValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetFloatValue", "cdecl")
    mlirDenseElementsAttrGetFloatValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetFloatValue.restype = c_float

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 549
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetDoubleValue", "cdecl"):
    mlirDenseElementsAttrGetDoubleValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetDoubleValue", "cdecl")
    mlirDenseElementsAttrGetDoubleValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetDoubleValue.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 551
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetStringValue", "cdecl"):
    mlirDenseElementsAttrGetStringValue = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetStringValue", "cdecl")
    mlirDenseElementsAttrGetStringValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseElementsAttrGetStringValue.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 554
if _libs["MLIRPythonCAPI"].has("mlirDenseElementsAttrGetRawData", "cdecl"):
    mlirDenseElementsAttrGetRawData = _libs["MLIRPythonCAPI"].get("mlirDenseElementsAttrGetRawData", "cdecl")
    mlirDenseElementsAttrGetRawData.argtypes = [MlirAttribute]
    mlirDenseElementsAttrGetRawData.restype = POINTER(c_ubyte)
    mlirDenseElementsAttrGetRawData.errcheck = lambda v,*a : cast(v, c_void_p)

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 561
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseBoolResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseBoolResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseBoolResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseBoolResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_int)]
    mlirUnmanagedDenseBoolResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 564
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseUInt8ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseUInt8ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseUInt8ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseUInt8ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(uint8_t)]
    mlirUnmanagedDenseUInt8ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 567
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseInt8ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseInt8ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseInt8ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseInt8ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_int8)]
    mlirUnmanagedDenseInt8ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 571
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseUInt16ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseUInt16ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseUInt16ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseUInt16ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(uint16_t)]
    mlirUnmanagedDenseUInt16ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 575
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseInt16ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseInt16ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseInt16ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseInt16ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_int16)]
    mlirUnmanagedDenseInt16ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 579
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseUInt32ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseUInt32ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseUInt32ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseUInt32ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(uint32_t)]
    mlirUnmanagedDenseUInt32ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 583
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseInt32ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseInt32ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseInt32ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseInt32ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_int32)]
    mlirUnmanagedDenseInt32ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 587
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseUInt64ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseUInt64ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseUInt64ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseUInt64ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(uint64_t)]
    mlirUnmanagedDenseUInt64ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 591
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseInt64ResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseInt64ResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseInt64ResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseInt64ResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_int64)]
    mlirUnmanagedDenseInt64ResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 594
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseFloatResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseFloatResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseFloatResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseFloatResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_float)]
    mlirUnmanagedDenseFloatResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 598
if _libs["MLIRPythonCAPI"].has("mlirUnmanagedDenseDoubleResourceElementsAttrGet", "cdecl"):
    mlirUnmanagedDenseDoubleResourceElementsAttrGet = _libs["MLIRPythonCAPI"].get("mlirUnmanagedDenseDoubleResourceElementsAttrGet", "cdecl")
    mlirUnmanagedDenseDoubleResourceElementsAttrGet.argtypes = [MlirType, MlirStringRef, intptr_t, POINTER(c_double)]
    mlirUnmanagedDenseDoubleResourceElementsAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 606
if _libs["MLIRPythonCAPI"].has("mlirDenseBoolResourceElementsAttrGetValue", "cdecl"):
    mlirDenseBoolResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseBoolResourceElementsAttrGetValue", "cdecl")
    mlirDenseBoolResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseBoolResourceElementsAttrGetValue.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 608
if _libs["MLIRPythonCAPI"].has("mlirDenseInt8ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseInt8ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseInt8ResourceElementsAttrGetValue", "cdecl")
    mlirDenseInt8ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseInt8ResourceElementsAttrGetValue.restype = c_int8

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 610
if _libs["MLIRPythonCAPI"].has("mlirDenseUInt8ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseUInt8ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseUInt8ResourceElementsAttrGetValue", "cdecl")
    mlirDenseUInt8ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseUInt8ResourceElementsAttrGetValue.restype = uint8_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 612
if _libs["MLIRPythonCAPI"].has("mlirDenseInt16ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseInt16ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseInt16ResourceElementsAttrGetValue", "cdecl")
    mlirDenseInt16ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseInt16ResourceElementsAttrGetValue.restype = c_int16

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 614
if _libs["MLIRPythonCAPI"].has("mlirDenseUInt16ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseUInt16ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseUInt16ResourceElementsAttrGetValue", "cdecl")
    mlirDenseUInt16ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseUInt16ResourceElementsAttrGetValue.restype = uint16_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 616
if _libs["MLIRPythonCAPI"].has("mlirDenseInt32ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseInt32ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseInt32ResourceElementsAttrGetValue", "cdecl")
    mlirDenseInt32ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseInt32ResourceElementsAttrGetValue.restype = c_int32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 618
if _libs["MLIRPythonCAPI"].has("mlirDenseUInt32ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseUInt32ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseUInt32ResourceElementsAttrGetValue", "cdecl")
    mlirDenseUInt32ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseUInt32ResourceElementsAttrGetValue.restype = uint32_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 620
if _libs["MLIRPythonCAPI"].has("mlirDenseInt64ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseInt64ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseInt64ResourceElementsAttrGetValue", "cdecl")
    mlirDenseInt64ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseInt64ResourceElementsAttrGetValue.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 622
if _libs["MLIRPythonCAPI"].has("mlirDenseUInt64ResourceElementsAttrGetValue", "cdecl"):
    mlirDenseUInt64ResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseUInt64ResourceElementsAttrGetValue", "cdecl")
    mlirDenseUInt64ResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseUInt64ResourceElementsAttrGetValue.restype = uint64_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 624
if _libs["MLIRPythonCAPI"].has("mlirDenseFloatResourceElementsAttrGetValue", "cdecl"):
    mlirDenseFloatResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseFloatResourceElementsAttrGetValue", "cdecl")
    mlirDenseFloatResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseFloatResourceElementsAttrGetValue.restype = c_float

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 626
if _libs["MLIRPythonCAPI"].has("mlirDenseDoubleResourceElementsAttrGetValue", "cdecl"):
    mlirDenseDoubleResourceElementsAttrGetValue = _libs["MLIRPythonCAPI"].get("mlirDenseDoubleResourceElementsAttrGetValue", "cdecl")
    mlirDenseDoubleResourceElementsAttrGetValue.argtypes = [MlirAttribute, intptr_t]
    mlirDenseDoubleResourceElementsAttrGetValue.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 633
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsASparseElements", "cdecl"):
    mlirAttributeIsASparseElements = _libs["MLIRPythonCAPI"].get("mlirAttributeIsASparseElements", "cdecl")
    mlirAttributeIsASparseElements.argtypes = [MlirAttribute]
    mlirAttributeIsASparseElements.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 640
if _libs["MLIRPythonCAPI"].has("mlirSparseElementsAttribute", "cdecl"):
    mlirSparseElementsAttribute = _libs["MLIRPythonCAPI"].get("mlirSparseElementsAttribute", "cdecl")
    mlirSparseElementsAttribute.argtypes = [MlirType, MlirAttribute, MlirAttribute]
    mlirSparseElementsAttribute.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 646
if _libs["MLIRPythonCAPI"].has("mlirSparseElementsAttrGetIndices", "cdecl"):
    mlirSparseElementsAttrGetIndices = _libs["MLIRPythonCAPI"].get("mlirSparseElementsAttrGetIndices", "cdecl")
    mlirSparseElementsAttrGetIndices.argtypes = [MlirAttribute]
    mlirSparseElementsAttrGetIndices.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 651
if _libs["MLIRPythonCAPI"].has("mlirSparseElementsAttrGetValues", "cdecl"):
    mlirSparseElementsAttrGetValues = _libs["MLIRPythonCAPI"].get("mlirSparseElementsAttrGetValues", "cdecl")
    mlirSparseElementsAttrGetValues.argtypes = [MlirAttribute]
    mlirSparseElementsAttrGetValues.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 654
if _libs["MLIRPythonCAPI"].has("mlirSparseElementsAttrGetTypeID", "cdecl"):
    mlirSparseElementsAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirSparseElementsAttrGetTypeID", "cdecl")
    mlirSparseElementsAttrGetTypeID.argtypes = []
    mlirSparseElementsAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 661
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsAStridedLayout", "cdecl"):
    mlirAttributeIsAStridedLayout = _libs["MLIRPythonCAPI"].get("mlirAttributeIsAStridedLayout", "cdecl")
    mlirAttributeIsAStridedLayout.argtypes = [MlirAttribute]
    mlirAttributeIsAStridedLayout.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 665
if _libs["MLIRPythonCAPI"].has("mlirStridedLayoutAttrGet", "cdecl"):
    mlirStridedLayoutAttrGet = _libs["MLIRPythonCAPI"].get("mlirStridedLayoutAttrGet", "cdecl")
    mlirStridedLayoutAttrGet.argtypes = [MlirContext, c_int64, intptr_t, POINTER(c_int64)]
    mlirStridedLayoutAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 669
if _libs["MLIRPythonCAPI"].has("mlirStridedLayoutAttrGetOffset", "cdecl"):
    mlirStridedLayoutAttrGetOffset = _libs["MLIRPythonCAPI"].get("mlirStridedLayoutAttrGetOffset", "cdecl")
    mlirStridedLayoutAttrGetOffset.argtypes = [MlirAttribute]
    mlirStridedLayoutAttrGetOffset.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 673
if _libs["MLIRPythonCAPI"].has("mlirStridedLayoutAttrGetNumStrides", "cdecl"):
    mlirStridedLayoutAttrGetNumStrides = _libs["MLIRPythonCAPI"].get("mlirStridedLayoutAttrGetNumStrides", "cdecl")
    mlirStridedLayoutAttrGetNumStrides.argtypes = [MlirAttribute]
    mlirStridedLayoutAttrGetNumStrides.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 676
if _libs["MLIRPythonCAPI"].has("mlirStridedLayoutAttrGetStride", "cdecl"):
    mlirStridedLayoutAttrGetStride = _libs["MLIRPythonCAPI"].get("mlirStridedLayoutAttrGetStride", "cdecl")
    mlirStridedLayoutAttrGetStride.argtypes = [MlirAttribute, intptr_t]
    mlirStridedLayoutAttrGetStride.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/BuiltinAttributes.h: 680
if _libs["MLIRPythonCAPI"].has("mlirStridedLayoutAttrGetTypeID", "cdecl"):
    mlirStridedLayoutAttrGetTypeID = _libs["MLIRPythonCAPI"].get("mlirStridedLayoutAttrGetTypeID", "cdecl")
    mlirStridedLayoutAttrGetTypeID.argtypes = []
    mlirStridedLayoutAttrGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/RegisterEverything.h: 26
if _libs["MLIRPythonCAPI"].has("mlirRegisterAllDialects", "cdecl"):
    mlirRegisterAllDialects = _libs["MLIRPythonCAPI"].get("mlirRegisterAllDialects", "cdecl")
    mlirRegisterAllDialects.argtypes = [MlirDialectRegistry]
    mlirRegisterAllDialects.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/RegisterEverything.h: 29
if _libs["MLIRPythonCAPI"].has("mlirRegisterAllLLVMTranslations", "cdecl"):
    mlirRegisterAllLLVMTranslations = _libs["MLIRPythonCAPI"].get("mlirRegisterAllLLVMTranslations", "cdecl")
    mlirRegisterAllLLVMTranslations.argtypes = [MlirContext]
    mlirRegisterAllLLVMTranslations.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/RegisterEverything.h: 32
if _libs["MLIRPythonCAPI"].has("mlirRegisterAllPasses", "cdecl"):
    mlirRegisterAllPasses = _libs["MLIRPythonCAPI"].get("mlirRegisterAllPasses", "cdecl")
    mlirRegisterAllPasses.argtypes = []
    mlirRegisterAllPasses.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 43
class struct_MlirPass(Structure):
    pass

struct_MlirPass.__slots__ = [
    'ptr',
]
struct_MlirPass._fields_ = [
    ('ptr', POINTER(None)),
]

MlirPass = struct_MlirPass# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 43

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 44
class struct_MlirExternalPass(Structure):
    pass

struct_MlirExternalPass.__slots__ = [
    'ptr',
]
struct_MlirExternalPass._fields_ = [
    ('ptr', POINTER(None)),
]

MlirExternalPass = struct_MlirExternalPass# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 44

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 45
class struct_MlirPassManager(Structure):
    pass

struct_MlirPassManager.__slots__ = [
    'ptr',
]
struct_MlirPassManager._fields_ = [
    ('ptr', POINTER(None)),
]

MlirPassManager = struct_MlirPassManager# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 45

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 46
class struct_MlirOpPassManager(Structure):
    pass

struct_MlirOpPassManager.__slots__ = [
    'ptr',
]
struct_MlirOpPassManager._fields_ = [
    ('ptr', POINTER(None)),
]

MlirOpPassManager = struct_MlirOpPassManager# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 46

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 55
if _libs["MLIRPythonCAPI"].has("mlirPassManagerCreate", "cdecl"):
    mlirPassManagerCreate = _libs["MLIRPythonCAPI"].get("mlirPassManagerCreate", "cdecl")
    mlirPassManagerCreate.argtypes = [MlirContext]
    mlirPassManagerCreate.restype = MlirPassManager

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 59
if _libs["MLIRPythonCAPI"].has("mlirPassManagerCreateOnOperation", "cdecl"):
    mlirPassManagerCreateOnOperation = _libs["MLIRPythonCAPI"].get("mlirPassManagerCreateOnOperation", "cdecl")
    mlirPassManagerCreateOnOperation.argtypes = [MlirContext, MlirStringRef]
    mlirPassManagerCreateOnOperation.restype = MlirPassManager

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 62
if _libs["MLIRPythonCAPI"].has("mlirPassManagerDestroy", "cdecl"):
    mlirPassManagerDestroy = _libs["MLIRPythonCAPI"].get("mlirPassManagerDestroy", "cdecl")
    mlirPassManagerDestroy.argtypes = [MlirPassManager]
    mlirPassManagerDestroy.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 71
if _libs["MLIRPythonCAPI"].has("mlirPassManagerGetAsOpPassManager", "cdecl"):
    mlirPassManagerGetAsOpPassManager = _libs["MLIRPythonCAPI"].get("mlirPassManagerGetAsOpPassManager", "cdecl")
    mlirPassManagerGetAsOpPassManager.argtypes = [MlirPassManager]
    mlirPassManagerGetAsOpPassManager.restype = MlirOpPassManager

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 75
if _libs["MLIRPythonCAPI"].has("mlirPassManagerRunOnOp", "cdecl"):
    mlirPassManagerRunOnOp = _libs["MLIRPythonCAPI"].get("mlirPassManagerRunOnOp", "cdecl")
    mlirPassManagerRunOnOp.argtypes = [MlirPassManager, MlirOperation]
    mlirPassManagerRunOnOp.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 79
if _libs["MLIRPythonCAPI"].has("mlirPassManagerEnableIRPrinting", "cdecl"):
    mlirPassManagerEnableIRPrinting = _libs["MLIRPythonCAPI"].get("mlirPassManagerEnableIRPrinting", "cdecl")
    mlirPassManagerEnableIRPrinting.argtypes = [MlirPassManager]
    mlirPassManagerEnableIRPrinting.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 83
if _libs["MLIRPythonCAPI"].has("mlirPassManagerEnableVerifier", "cdecl"):
    mlirPassManagerEnableVerifier = _libs["MLIRPythonCAPI"].get("mlirPassManagerEnableVerifier", "cdecl")
    mlirPassManagerEnableVerifier.argtypes = [MlirPassManager, c_bool]
    mlirPassManagerEnableVerifier.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 90
if _libs["MLIRPythonCAPI"].has("mlirPassManagerGetNestedUnder", "cdecl"):
    mlirPassManagerGetNestedUnder = _libs["MLIRPythonCAPI"].get("mlirPassManagerGetNestedUnder", "cdecl")
    mlirPassManagerGetNestedUnder.argtypes = [MlirPassManager, MlirStringRef]
    mlirPassManagerGetNestedUnder.restype = MlirOpPassManager

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 96
if _libs["MLIRPythonCAPI"].has("mlirOpPassManagerGetNestedUnder", "cdecl"):
    mlirOpPassManagerGetNestedUnder = _libs["MLIRPythonCAPI"].get("mlirOpPassManagerGetNestedUnder", "cdecl")
    mlirOpPassManagerGetNestedUnder.argtypes = [MlirOpPassManager, MlirStringRef]
    mlirOpPassManagerGetNestedUnder.restype = MlirOpPassManager

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 102
if _libs["MLIRPythonCAPI"].has("mlirPassManagerAddOwnedPass", "cdecl"):
    mlirPassManagerAddOwnedPass = _libs["MLIRPythonCAPI"].get("mlirPassManagerAddOwnedPass", "cdecl")
    mlirPassManagerAddOwnedPass.argtypes = [MlirPassManager, MlirPass]
    mlirPassManagerAddOwnedPass.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 110
if _libs["MLIRPythonCAPI"].has("mlirOpPassManagerAddOwnedPass", "cdecl"):
    mlirOpPassManagerAddOwnedPass = _libs["MLIRPythonCAPI"].get("mlirOpPassManagerAddOwnedPass", "cdecl")
    mlirOpPassManagerAddOwnedPass.argtypes = [MlirOpPassManager, MlirPass]
    mlirOpPassManagerAddOwnedPass.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 115
if _libs["MLIRPythonCAPI"].has("mlirOpPassManagerAddPipeline", "cdecl"):
    mlirOpPassManagerAddPipeline = _libs["MLIRPythonCAPI"].get("mlirOpPassManagerAddPipeline", "cdecl")
    mlirOpPassManagerAddPipeline.argtypes = [MlirOpPassManager, MlirStringRef, MlirStringCallback, POINTER(None)]
    mlirOpPassManagerAddPipeline.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 122
if _libs["MLIRPythonCAPI"].has("mlirPrintPassPipeline", "cdecl"):
    mlirPrintPassPipeline = _libs["MLIRPythonCAPI"].get("mlirPrintPassPipeline", "cdecl")
    mlirPrintPassPipeline.argtypes = [MlirOpPassManager, MlirStringCallback, POINTER(None)]
    mlirPrintPassPipeline.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 130
if _libs["MLIRPythonCAPI"].has("mlirParsePassPipeline", "cdecl"):
    mlirParsePassPipeline = _libs["MLIRPythonCAPI"].get("mlirParsePassPipeline", "cdecl")
    mlirParsePassPipeline.argtypes = [MlirOpPassManager, MlirStringRef, MlirStringCallback, POINTER(None)]
    mlirParsePassPipeline.restype = MlirLogicalResult

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 143
class struct_MlirExternalPassCallbacks(Structure):
    pass

struct_MlirExternalPassCallbacks.__slots__ = [
    'construct',
    'destruct',
    'initialize',
    'clone',
    'run',
]
struct_MlirExternalPassCallbacks._fields_ = [
    ('construct', CFUNCTYPE(UNCHECKED(None), POINTER(None))),
    ('destruct', CFUNCTYPE(UNCHECKED(None), POINTER(None))),
    ('initialize', CFUNCTYPE(UNCHECKED(MlirLogicalResult), MlirContext, POINTER(None))),
    ('clone', CFUNCTYPE(UNCHECKED(POINTER(c_ubyte)), POINTER(None))),
    ('run', CFUNCTYPE(UNCHECKED(None), MlirOperation, MlirExternalPass, POINTER(None))),
]

MlirExternalPassCallbacks = struct_MlirExternalPassCallbacks# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 166

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 171
if _libs["MLIRPythonCAPI"].has("mlirCreateExternalPass", "cdecl"):
    mlirCreateExternalPass = _libs["MLIRPythonCAPI"].get("mlirCreateExternalPass", "cdecl")
    mlirCreateExternalPass.argtypes = [MlirTypeID, MlirStringRef, MlirStringRef, MlirStringRef, MlirStringRef, intptr_t, POINTER(MlirDialectHandle), MlirExternalPassCallbacks, POINTER(None)]
    mlirCreateExternalPass.restype = MlirPass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 180
if _libs["MLIRPythonCAPI"].has("mlirExternalPassSignalFailure", "cdecl"):
    mlirExternalPassSignalFailure = _libs["MLIRPythonCAPI"].get("mlirExternalPassSignalFailure", "cdecl")
    mlirExternalPassSignalFailure.argtypes = [MlirExternalPass]
    mlirExternalPassSignalFailure.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 26
class struct_MlirDiagnostic(Structure):
    pass

struct_MlirDiagnostic.__slots__ = [
    'ptr',
]
struct_MlirDiagnostic._fields_ = [
    ('ptr', POINTER(None)),
]

MlirDiagnostic = struct_MlirDiagnostic# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 29

enum_MlirDiagnosticSeverity = c_int# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 32

MlirDiagnosticError = 0# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 32

MlirDiagnosticWarning = (MlirDiagnosticError + 1)# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 32

MlirDiagnosticNote = (MlirDiagnosticWarning + 1)# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 32

MlirDiagnosticRemark = (MlirDiagnosticNote + 1)# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 32

MlirDiagnosticSeverity = enum_MlirDiagnosticSeverity# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 38

MlirDiagnosticHandlerID = uint64_t# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 41

MlirDiagnosticHandler = CFUNCTYPE(UNCHECKED(MlirLogicalResult), MlirDiagnostic, POINTER(None))# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 49

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 53
if _libs["MLIRPythonCAPI"].has("mlirDiagnosticPrint", "cdecl"):
    mlirDiagnosticPrint = _libs["MLIRPythonCAPI"].get("mlirDiagnosticPrint", "cdecl")
    mlirDiagnosticPrint.argtypes = [MlirDiagnostic, MlirStringCallback, POINTER(None)]
    mlirDiagnosticPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 59
if _libs["MLIRPythonCAPI"].has("mlirDiagnosticGetLocation", "cdecl"):
    mlirDiagnosticGetLocation = _libs["MLIRPythonCAPI"].get("mlirDiagnosticGetLocation", "cdecl")
    mlirDiagnosticGetLocation.argtypes = [MlirDiagnostic]
    mlirDiagnosticGetLocation.restype = MlirLocation

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 63
if _libs["MLIRPythonCAPI"].has("mlirDiagnosticGetSeverity", "cdecl"):
    mlirDiagnosticGetSeverity = _libs["MLIRPythonCAPI"].get("mlirDiagnosticGetSeverity", "cdecl")
    mlirDiagnosticGetSeverity.argtypes = [MlirDiagnostic]
    mlirDiagnosticGetSeverity.restype = MlirDiagnosticSeverity

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 67
if _libs["MLIRPythonCAPI"].has("mlirDiagnosticGetNumNotes", "cdecl"):
    mlirDiagnosticGetNumNotes = _libs["MLIRPythonCAPI"].get("mlirDiagnosticGetNumNotes", "cdecl")
    mlirDiagnosticGetNumNotes.argtypes = [MlirDiagnostic]
    mlirDiagnosticGetNumNotes.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 72
if _libs["MLIRPythonCAPI"].has("mlirDiagnosticGetNote", "cdecl"):
    mlirDiagnosticGetNote = _libs["MLIRPythonCAPI"].get("mlirDiagnosticGetNote", "cdecl")
    mlirDiagnosticGetNote.argtypes = [MlirDiagnostic, intptr_t]
    mlirDiagnosticGetNote.restype = MlirDiagnostic

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 82
if _libs["MLIRPythonCAPI"].has("mlirContextAttachDiagnosticHandler", "cdecl"):
    mlirContextAttachDiagnosticHandler = _libs["MLIRPythonCAPI"].get("mlirContextAttachDiagnosticHandler", "cdecl")
    mlirContextAttachDiagnosticHandler.argtypes = [MlirContext, MlirDiagnosticHandler, POINTER(None), CFUNCTYPE(UNCHECKED(None), POINTER(None))]
    mlirContextAttachDiagnosticHandler.restype = MlirDiagnosticHandlerID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 89
if _libs["MLIRPythonCAPI"].has("mlirContextDetachDiagnosticHandler", "cdecl"):
    mlirContextDetachDiagnosticHandler = _libs["MLIRPythonCAPI"].get("mlirContextDetachDiagnosticHandler", "cdecl")
    mlirContextDetachDiagnosticHandler.argtypes = [MlirContext, MlirDiagnosticHandlerID]
    mlirContextDetachDiagnosticHandler.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 94
if _libs["MLIRPythonCAPI"].has("mlirEmitError", "cdecl"):
    mlirEmitError = _libs["MLIRPythonCAPI"].get("mlirEmitError", "cdecl")
    mlirEmitError.argtypes = [MlirLocation, String]
    mlirEmitError.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 38
class struct_MlirIntegerSet(Structure):
    pass

struct_MlirIntegerSet.__slots__ = [
    'ptr',
]
struct_MlirIntegerSet._fields_ = [
    ('ptr', POINTER(None)),
]

MlirIntegerSet = struct_MlirIntegerSet# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 38

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 43
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetContext", "cdecl"):
    mlirIntegerSetGetContext = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetContext", "cdecl")
    mlirIntegerSetGetContext.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetContext.restype = MlirContext

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 54
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetEqual", "cdecl"):
    mlirIntegerSetEqual = _libs["MLIRPythonCAPI"].get("mlirIntegerSetEqual", "cdecl")
    mlirIntegerSetEqual.argtypes = [MlirIntegerSet, MlirIntegerSet]
    mlirIntegerSetEqual.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 60
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetPrint", "cdecl"):
    mlirIntegerSetPrint = _libs["MLIRPythonCAPI"].get("mlirIntegerSetPrint", "cdecl")
    mlirIntegerSetPrint.argtypes = [MlirIntegerSet, MlirStringCallback, POINTER(None)]
    mlirIntegerSetPrint.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 65
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetDump", "cdecl"):
    mlirIntegerSetDump = _libs["MLIRPythonCAPI"].get("mlirIntegerSetDump", "cdecl")
    mlirIntegerSetDump.argtypes = [MlirIntegerSet]
    mlirIntegerSetDump.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 69
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetEmptyGet", "cdecl"):
    mlirIntegerSetEmptyGet = _libs["MLIRPythonCAPI"].get("mlirIntegerSetEmptyGet", "cdecl")
    mlirIntegerSetEmptyGet.argtypes = [MlirContext, intptr_t, intptr_t]
    mlirIntegerSetEmptyGet.restype = MlirIntegerSet

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 79
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGet", "cdecl"):
    mlirIntegerSetGet = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGet", "cdecl")
    mlirIntegerSetGet.argtypes = [MlirContext, intptr_t, intptr_t, intptr_t, POINTER(MlirAffineExpr), POINTER(c_bool)]
    mlirIntegerSetGet.restype = MlirIntegerSet

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 89
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetReplaceGet", "cdecl"):
    mlirIntegerSetReplaceGet = _libs["MLIRPythonCAPI"].get("mlirIntegerSetReplaceGet", "cdecl")
    mlirIntegerSetReplaceGet.argtypes = [MlirIntegerSet, POINTER(MlirAffineExpr), POINTER(MlirAffineExpr), intptr_t, intptr_t]
    mlirIntegerSetReplaceGet.restype = MlirIntegerSet

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 96
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetIsCanonicalEmpty", "cdecl"):
    mlirIntegerSetIsCanonicalEmpty = _libs["MLIRPythonCAPI"].get("mlirIntegerSetIsCanonicalEmpty", "cdecl")
    mlirIntegerSetIsCanonicalEmpty.argtypes = [MlirIntegerSet]
    mlirIntegerSetIsCanonicalEmpty.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 99
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumDims", "cdecl"):
    mlirIntegerSetGetNumDims = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumDims", "cdecl")
    mlirIntegerSetGetNumDims.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumDims.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 102
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumSymbols", "cdecl"):
    mlirIntegerSetGetNumSymbols = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumSymbols", "cdecl")
    mlirIntegerSetGetNumSymbols.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumSymbols.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 105
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumInputs", "cdecl"):
    mlirIntegerSetGetNumInputs = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumInputs", "cdecl")
    mlirIntegerSetGetNumInputs.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumInputs.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 109
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumConstraints", "cdecl"):
    mlirIntegerSetGetNumConstraints = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumConstraints", "cdecl")
    mlirIntegerSetGetNumConstraints.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumConstraints.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 112
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumEqualities", "cdecl"):
    mlirIntegerSetGetNumEqualities = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumEqualities", "cdecl")
    mlirIntegerSetGetNumEqualities.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumEqualities.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 116
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetNumInequalities", "cdecl"):
    mlirIntegerSetGetNumInequalities = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetNumInequalities", "cdecl")
    mlirIntegerSetGetNumInequalities.argtypes = [MlirIntegerSet]
    mlirIntegerSetGetNumInequalities.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 120
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetGetConstraint", "cdecl"):
    mlirIntegerSetGetConstraint = _libs["MLIRPythonCAPI"].get("mlirIntegerSetGetConstraint", "cdecl")
    mlirIntegerSetGetConstraint.argtypes = [MlirIntegerSet, intptr_t]
    mlirIntegerSetGetConstraint.restype = MlirAffineExpr

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 124
if _libs["MLIRPythonCAPI"].has("mlirIntegerSetIsConstraintEq", "cdecl"):
    mlirIntegerSetIsConstraintEq = _libs["MLIRPythonCAPI"].get("mlirIntegerSetIsConstraintEq", "cdecl")
    mlirIntegerSetIsConstraintEq.argtypes = [MlirIntegerSet, intptr_t]
    mlirIntegerSetIsConstraintEq.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SCF.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__scf__", "cdecl"):
        continue
    mlirGetDialectHandle__scf__ = _lib.get("mlirGetDialectHandle__scf__", "cdecl")
    mlirGetDialectHandle__scf__.argtypes = []
    mlirGetDialectHandle__scf__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 19
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__quant__", "cdecl"):
    mlirGetDialectHandle__quant__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__quant__", "cdecl")
    mlirGetDialectHandle__quant__.argtypes = []
    mlirGetDialectHandle__quant__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 26
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAQuantizedType", "cdecl"):
    mlirTypeIsAQuantizedType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAQuantizedType", "cdecl")
    mlirTypeIsAQuantizedType.argtypes = [MlirType]
    mlirTypeIsAQuantizedType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 29
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetSignedFlag", "cdecl"):
    mlirQuantizedTypeGetSignedFlag = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetSignedFlag", "cdecl")
    mlirQuantizedTypeGetSignedFlag.argtypes = []
    mlirQuantizedTypeGetSignedFlag.restype = c_uint

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 32
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetDefaultMinimumForInteger", "cdecl"):
    mlirQuantizedTypeGetDefaultMinimumForInteger = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetDefaultMinimumForInteger", "cdecl")
    mlirQuantizedTypeGetDefaultMinimumForInteger.argtypes = [c_bool, c_uint]
    mlirQuantizedTypeGetDefaultMinimumForInteger.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 36
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetDefaultMaximumForInteger", "cdecl"):
    mlirQuantizedTypeGetDefaultMaximumForInteger = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetDefaultMaximumForInteger", "cdecl")
    mlirQuantizedTypeGetDefaultMaximumForInteger.argtypes = [c_bool, c_uint]
    mlirQuantizedTypeGetDefaultMaximumForInteger.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 40
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetExpressedType", "cdecl"):
    mlirQuantizedTypeGetExpressedType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetExpressedType", "cdecl")
    mlirQuantizedTypeGetExpressedType.argtypes = [MlirType]
    mlirQuantizedTypeGetExpressedType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 43
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetFlags", "cdecl"):
    mlirQuantizedTypeGetFlags = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetFlags", "cdecl")
    mlirQuantizedTypeGetFlags.argtypes = [MlirType]
    mlirQuantizedTypeGetFlags.restype = c_uint

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 46
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeIsSigned", "cdecl"):
    mlirQuantizedTypeIsSigned = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeIsSigned", "cdecl")
    mlirQuantizedTypeIsSigned.argtypes = [MlirType]
    mlirQuantizedTypeIsSigned.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 49
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetStorageType", "cdecl"):
    mlirQuantizedTypeGetStorageType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetStorageType", "cdecl")
    mlirQuantizedTypeGetStorageType.argtypes = [MlirType]
    mlirQuantizedTypeGetStorageType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 53
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetStorageTypeMin", "cdecl"):
    mlirQuantizedTypeGetStorageTypeMin = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetStorageTypeMin", "cdecl")
    mlirQuantizedTypeGetStorageTypeMin.argtypes = [MlirType]
    mlirQuantizedTypeGetStorageTypeMin.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 57
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetStorageTypeMax", "cdecl"):
    mlirQuantizedTypeGetStorageTypeMax = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetStorageTypeMax", "cdecl")
    mlirQuantizedTypeGetStorageTypeMax.argtypes = [MlirType]
    mlirQuantizedTypeGetStorageTypeMax.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 62
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetStorageTypeIntegralWidth", "cdecl"):
    mlirQuantizedTypeGetStorageTypeIntegralWidth = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetStorageTypeIntegralWidth", "cdecl")
    mlirQuantizedTypeGetStorageTypeIntegralWidth.argtypes = [MlirType]
    mlirQuantizedTypeGetStorageTypeIntegralWidth.restype = c_uint

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 67
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeIsCompatibleExpressedType", "cdecl"):
    mlirQuantizedTypeIsCompatibleExpressedType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeIsCompatibleExpressedType", "cdecl")
    mlirQuantizedTypeIsCompatibleExpressedType.argtypes = [MlirType, MlirType]
    mlirQuantizedTypeIsCompatibleExpressedType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 72
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeGetQuantizedElementType", "cdecl"):
    mlirQuantizedTypeGetQuantizedElementType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeGetQuantizedElementType", "cdecl")
    mlirQuantizedTypeGetQuantizedElementType.argtypes = [MlirType]
    mlirQuantizedTypeGetQuantizedElementType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 78
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeCastFromStorageType", "cdecl"):
    mlirQuantizedTypeCastFromStorageType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeCastFromStorageType", "cdecl")
    mlirQuantizedTypeCastFromStorageType.argtypes = [MlirType, MlirType]
    mlirQuantizedTypeCastFromStorageType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 82
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeCastToStorageType", "cdecl"):
    mlirQuantizedTypeCastToStorageType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeCastToStorageType", "cdecl")
    mlirQuantizedTypeCastToStorageType.argtypes = [MlirType]
    mlirQuantizedTypeCastToStorageType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 88
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeCastFromExpressedType", "cdecl"):
    mlirQuantizedTypeCastFromExpressedType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeCastFromExpressedType", "cdecl")
    mlirQuantizedTypeCastFromExpressedType.argtypes = [MlirType, MlirType]
    mlirQuantizedTypeCastFromExpressedType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 92
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeCastToExpressedType", "cdecl"):
    mlirQuantizedTypeCastToExpressedType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeCastToExpressedType", "cdecl")
    mlirQuantizedTypeCastToExpressedType.argtypes = [MlirType]
    mlirQuantizedTypeCastToExpressedType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 97
if _libs["MLIRPythonCAPI"].has("mlirQuantizedTypeCastExpressedToStorageType", "cdecl"):
    mlirQuantizedTypeCastExpressedToStorageType = _libs["MLIRPythonCAPI"].get("mlirQuantizedTypeCastExpressedToStorageType", "cdecl")
    mlirQuantizedTypeCastExpressedToStorageType.argtypes = [MlirType, MlirType]
    mlirQuantizedTypeCastExpressedToStorageType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 104
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAAnyQuantizedType", "cdecl"):
    mlirTypeIsAAnyQuantizedType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAAnyQuantizedType", "cdecl")
    mlirTypeIsAAnyQuantizedType.argtypes = [MlirType]
    mlirTypeIsAAnyQuantizedType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 109
if _libs["MLIRPythonCAPI"].has("mlirAnyQuantizedTypeGet", "cdecl"):
    mlirAnyQuantizedTypeGet = _libs["MLIRPythonCAPI"].get("mlirAnyQuantizedTypeGet", "cdecl")
    mlirAnyQuantizedTypeGet.argtypes = [c_uint, MlirType, MlirType, c_int64, c_int64]
    mlirAnyQuantizedTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 120
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAUniformQuantizedType", "cdecl"):
    mlirTypeIsAUniformQuantizedType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAUniformQuantizedType", "cdecl")
    mlirTypeIsAUniformQuantizedType.argtypes = [MlirType]
    mlirTypeIsAUniformQuantizedType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 125
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedTypeGet", "cdecl"):
    mlirUniformQuantizedTypeGet = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedTypeGet", "cdecl")
    mlirUniformQuantizedTypeGet.argtypes = [c_uint, MlirType, MlirType, c_double, c_int64, c_int64, c_int64]
    mlirUniformQuantizedTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 130
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedTypeGetScale", "cdecl"):
    mlirUniformQuantizedTypeGetScale = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedTypeGetScale", "cdecl")
    mlirUniformQuantizedTypeGetScale.argtypes = [MlirType]
    mlirUniformQuantizedTypeGetScale.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 133
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedTypeGetZeroPoint", "cdecl"):
    mlirUniformQuantizedTypeGetZeroPoint = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedTypeGetZeroPoint", "cdecl")
    mlirUniformQuantizedTypeGetZeroPoint.argtypes = [MlirType]
    mlirUniformQuantizedTypeGetZeroPoint.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 136
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedTypeIsFixedPoint", "cdecl"):
    mlirUniformQuantizedTypeIsFixedPoint = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedTypeIsFixedPoint", "cdecl")
    mlirUniformQuantizedTypeIsFixedPoint.argtypes = [MlirType]
    mlirUniformQuantizedTypeIsFixedPoint.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 143
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAUniformQuantizedPerAxisType", "cdecl"):
    mlirTypeIsAUniformQuantizedPerAxisType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAUniformQuantizedPerAxisType", "cdecl")
    mlirTypeIsAUniformQuantizedPerAxisType.argtypes = [MlirType]
    mlirTypeIsAUniformQuantizedPerAxisType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 149
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeGet", "cdecl"):
    mlirUniformQuantizedPerAxisTypeGet = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeGet", "cdecl")
    mlirUniformQuantizedPerAxisTypeGet.argtypes = [c_uint, MlirType, MlirType, intptr_t, POINTER(c_double), POINTER(c_int64), c_int32, c_int64, c_int64]
    mlirUniformQuantizedPerAxisTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 156
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeGetNumDims", "cdecl"):
    mlirUniformQuantizedPerAxisTypeGetNumDims = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeGetNumDims", "cdecl")
    mlirUniformQuantizedPerAxisTypeGetNumDims.argtypes = [MlirType]
    mlirUniformQuantizedPerAxisTypeGetNumDims.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 159
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeGetScale", "cdecl"):
    mlirUniformQuantizedPerAxisTypeGetScale = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeGetScale", "cdecl")
    mlirUniformQuantizedPerAxisTypeGetScale.argtypes = [MlirType, intptr_t]
    mlirUniformQuantizedPerAxisTypeGetScale.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 164
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeGetZeroPoint", "cdecl"):
    mlirUniformQuantizedPerAxisTypeGetZeroPoint = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeGetZeroPoint", "cdecl")
    mlirUniformQuantizedPerAxisTypeGetZeroPoint.argtypes = [MlirType, intptr_t]
    mlirUniformQuantizedPerAxisTypeGetZeroPoint.restype = c_int64

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 169
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeGetQuantizedDimension", "cdecl"):
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeGetQuantizedDimension", "cdecl")
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension.argtypes = [MlirType]
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension.restype = c_int32

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 173
if _libs["MLIRPythonCAPI"].has("mlirUniformQuantizedPerAxisTypeIsFixedPoint", "cdecl"):
    mlirUniformQuantizedPerAxisTypeIsFixedPoint = _libs["MLIRPythonCAPI"].get("mlirUniformQuantizedPerAxisTypeIsFixedPoint", "cdecl")
    mlirUniformQuantizedPerAxisTypeIsFixedPoint.argtypes = [MlirType]
    mlirUniformQuantizedPerAxisTypeIsFixedPoint.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 180
if _libs["MLIRPythonCAPI"].has("mlirTypeIsACalibratedQuantizedType", "cdecl"):
    mlirTypeIsACalibratedQuantizedType = _libs["MLIRPythonCAPI"].get("mlirTypeIsACalibratedQuantizedType", "cdecl")
    mlirTypeIsACalibratedQuantizedType.argtypes = [MlirType]
    mlirTypeIsACalibratedQuantizedType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 186
if _libs["MLIRPythonCAPI"].has("mlirCalibratedQuantizedTypeGet", "cdecl"):
    mlirCalibratedQuantizedTypeGet = _libs["MLIRPythonCAPI"].get("mlirCalibratedQuantizedTypeGet", "cdecl")
    mlirCalibratedQuantizedTypeGet.argtypes = [MlirType, c_double, c_double]
    mlirCalibratedQuantizedTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 189
if _libs["MLIRPythonCAPI"].has("mlirCalibratedQuantizedTypeGetMin", "cdecl"):
    mlirCalibratedQuantizedTypeGetMin = _libs["MLIRPythonCAPI"].get("mlirCalibratedQuantizedTypeGetMin", "cdecl")
    mlirCalibratedQuantizedTypeGetMin.argtypes = [MlirType]
    mlirCalibratedQuantizedTypeGetMin.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Quant.h: 192
if _libs["MLIRPythonCAPI"].has("mlirCalibratedQuantizedTypeGetMax", "cdecl"):
    mlirCalibratedQuantizedTypeGetMax = _libs["MLIRPythonCAPI"].get("mlirCalibratedQuantizedTypeGetMax", "cdecl")
    mlirCalibratedQuantizedTypeGetMax.argtypes = [MlirType]
    mlirCalibratedQuantizedTypeGetMax.restype = c_double

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Shape.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__shape__", "cdecl"):
        continue
    mlirGetDialectHandle__shape__ = _lib.get("mlirGetDialectHandle__shape__", "cdecl")
    mlirGetDialectHandle__shape__.argtypes = []
    mlirGetDialectHandle__shape__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__llvm__", "cdecl"):
        continue
    mlirGetDialectHandle__llvm__ = _lib.get("mlirGetDialectHandle__llvm__", "cdecl")
    mlirGetDialectHandle__llvm__.argtypes = []
    mlirGetDialectHandle__llvm__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 22
for _lib in _libs.values():
    if not _lib.has("mlirLLVMPointerTypeGet", "cdecl"):
        continue
    mlirLLVMPointerTypeGet = _lib.get("mlirLLVMPointerTypeGet", "cdecl")
    mlirLLVMPointerTypeGet.argtypes = [MlirType, c_uint]
    mlirLLVMPointerTypeGet.restype = MlirType
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 26
for _lib in _libs.values():
    if not _lib.has("mlirLLVMVoidTypeGet", "cdecl"):
        continue
    mlirLLVMVoidTypeGet = _lib.get("mlirLLVMVoidTypeGet", "cdecl")
    mlirLLVMVoidTypeGet.argtypes = [MlirContext]
    mlirLLVMVoidTypeGet.restype = MlirType
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 29
for _lib in _libs.values():
    if not _lib.has("mlirLLVMArrayTypeGet", "cdecl"):
        continue
    mlirLLVMArrayTypeGet = _lib.get("mlirLLVMArrayTypeGet", "cdecl")
    mlirLLVMArrayTypeGet.argtypes = [MlirType, c_uint]
    mlirLLVMArrayTypeGet.restype = MlirType
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 34
for _lib in _libs.values():
    if not _lib.has("mlirLLVMFunctionTypeGet", "cdecl"):
        continue
    mlirLLVMFunctionTypeGet = _lib.get("mlirLLVMFunctionTypeGet", "cdecl")
    mlirLLVMFunctionTypeGet.argtypes = [MlirType, intptr_t, POINTER(MlirType), c_bool]
    mlirLLVMFunctionTypeGet.restype = MlirType
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/LLVM.h: 39
for _lib in _libs.values():
    if not _lib.has("mlirLLVMStructTypeLiteralGet", "cdecl"):
        continue
    mlirLLVMStructTypeLiteralGet = _lib.get("mlirLLVMStructTypeLiteralGet", "cdecl")
    mlirLLVMStructTypeLiteralGet.argtypes = [MlirContext, intptr_t, POINTER(MlirType), c_bool]
    mlirLLVMStructTypeLiteralGet.restype = MlirType
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/MLProgram.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__ml_program__", "cdecl"):
        continue
    mlirGetDialectHandle__ml_program__ = _lib.get("mlirGetDialectHandle__ml_program__", "cdecl")
    mlirGetDialectHandle__ml_program__.argtypes = []
    mlirGetDialectHandle__ml_program__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Async.h: 20
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__async__", "cdecl"):
    mlirGetDialectHandle__async__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__async__", "cdecl")
    mlirGetDialectHandle__async__.argtypes = []
    mlirGetDialectHandle__async__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 20
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__sparse_tensor__", "cdecl"):
    mlirGetDialectHandle__sparse_tensor__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__sparse_tensor__", "cdecl")
    mlirGetDialectHandle__sparse_tensor__.argtypes = []
    mlirGetDialectHandle__sparse_tensor__.restype = MlirDialectHandle

enum_MlirSparseTensorDimLevelType = c_int# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 4# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 8# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU = 9# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO = 10# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO = 11# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 16# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU = 17# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO = 18# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO = 19# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI = 32# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU = 33# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NO = 34# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU_NO = 35# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 28

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 50
if _libs["MLIRPythonCAPI"].has("mlirAttributeIsASparseTensorEncodingAttr", "cdecl"):
    mlirAttributeIsASparseTensorEncodingAttr = _libs["MLIRPythonCAPI"].get("mlirAttributeIsASparseTensorEncodingAttr", "cdecl")
    mlirAttributeIsASparseTensorEncodingAttr.argtypes = [MlirAttribute]
    mlirAttributeIsASparseTensorEncodingAttr.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 53
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingAttrGet", "cdecl"):
    mlirSparseTensorEncodingAttrGet = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingAttrGet", "cdecl")
    mlirSparseTensorEncodingAttrGet.argtypes = [MlirContext, intptr_t, POINTER(enum_MlirSparseTensorDimLevelType), MlirAffineMap, c_int, c_int]
    mlirSparseTensorEncodingAttrGet.restype = MlirAttribute

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 60
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingGetLvlRank", "cdecl"):
    mlirSparseTensorEncodingGetLvlRank = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingGetLvlRank", "cdecl")
    mlirSparseTensorEncodingGetLvlRank.argtypes = [MlirAttribute]
    mlirSparseTensorEncodingGetLvlRank.restype = intptr_t

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 64
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingAttrGetLvlType", "cdecl"):
    mlirSparseTensorEncodingAttrGetLvlType = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingAttrGetLvlType", "cdecl")
    mlirSparseTensorEncodingAttrGetLvlType.argtypes = [MlirAttribute, intptr_t]
    mlirSparseTensorEncodingAttrGetLvlType.restype = enum_MlirSparseTensorDimLevelType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 69
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingAttrGetDimToLvl", "cdecl"):
    mlirSparseTensorEncodingAttrGetDimToLvl = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingAttrGetDimToLvl", "cdecl")
    mlirSparseTensorEncodingAttrGetDimToLvl.argtypes = [MlirAttribute]
    mlirSparseTensorEncodingAttrGetDimToLvl.restype = MlirAffineMap

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 73
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingAttrGetPosWidth", "cdecl"):
    mlirSparseTensorEncodingAttrGetPosWidth = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingAttrGetPosWidth", "cdecl")
    mlirSparseTensorEncodingAttrGetPosWidth.argtypes = [MlirAttribute]
    mlirSparseTensorEncodingAttrGetPosWidth.restype = c_int

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/SparseTensor.h: 77
if _libs["MLIRPythonCAPI"].has("mlirSparseTensorEncodingAttrGetCrdWidth", "cdecl"):
    mlirSparseTensorEncodingAttrGetCrdWidth = _libs["MLIRPythonCAPI"].get("mlirSparseTensorEncodingAttrGetCrdWidth", "cdecl")
    mlirSparseTensorEncodingAttrGetCrdWidth.argtypes = [MlirAttribute]
    mlirSparseTensorEncodingAttrGetCrdWidth.restype = c_int

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/GPU.h: 20
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__gpu__", "cdecl"):
    mlirGetDialectHandle__gpu__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__gpu__", "cdecl")
    mlirGetDialectHandle__gpu__.argtypes = []
    mlirGetDialectHandle__gpu__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 20
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__transform__", "cdecl"):
    mlirGetDialectHandle__transform__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__transform__", "cdecl")
    mlirGetDialectHandle__transform__.argtypes = []
    mlirGetDialectHandle__transform__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 26
if _libs["MLIRPythonCAPI"].has("mlirTypeIsATransformAnyOpType", "cdecl"):
    mlirTypeIsATransformAnyOpType = _libs["MLIRPythonCAPI"].get("mlirTypeIsATransformAnyOpType", "cdecl")
    mlirTypeIsATransformAnyOpType.argtypes = [MlirType]
    mlirTypeIsATransformAnyOpType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 28
if _libs["MLIRPythonCAPI"].has("mlirTransformAnyOpTypeGet", "cdecl"):
    mlirTransformAnyOpTypeGet = _libs["MLIRPythonCAPI"].get("mlirTransformAnyOpTypeGet", "cdecl")
    mlirTransformAnyOpTypeGet.argtypes = [MlirContext]
    mlirTransformAnyOpTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 34
if _libs["MLIRPythonCAPI"].has("mlirTypeIsATransformOperationType", "cdecl"):
    mlirTypeIsATransformOperationType = _libs["MLIRPythonCAPI"].get("mlirTypeIsATransformOperationType", "cdecl")
    mlirTypeIsATransformOperationType.argtypes = [MlirType]
    mlirTypeIsATransformOperationType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 36
if _libs["MLIRPythonCAPI"].has("mlirTransformOperationTypeGetTypeID", "cdecl"):
    mlirTransformOperationTypeGetTypeID = _libs["MLIRPythonCAPI"].get("mlirTransformOperationTypeGetTypeID", "cdecl")
    mlirTransformOperationTypeGetTypeID.argtypes = []
    mlirTransformOperationTypeGetTypeID.restype = MlirTypeID

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 39
if _libs["MLIRPythonCAPI"].has("mlirTransformOperationTypeGet", "cdecl"):
    mlirTransformOperationTypeGet = _libs["MLIRPythonCAPI"].get("mlirTransformOperationTypeGet", "cdecl")
    mlirTransformOperationTypeGet.argtypes = [MlirContext, MlirStringRef]
    mlirTransformOperationTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Transform.h: 42
if _libs["MLIRPythonCAPI"].has("mlirTransformOperationTypeGetOperationName", "cdecl"):
    mlirTransformOperationTypeGetOperationName = _libs["MLIRPythonCAPI"].get("mlirTransformOperationTypeGetOperationName", "cdecl")
    mlirTransformOperationTypeGetOperationName.argtypes = [MlirType]
    mlirTransformOperationTypeGetOperationName.restype = MlirStringRef

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Linalg.h: 23
if _libs["MLIRPythonCAPI"].has("mlirLinalgFillBuiltinNamedOpRegion", "cdecl"):
    mlirLinalgFillBuiltinNamedOpRegion = _libs["MLIRPythonCAPI"].get("mlirLinalgFillBuiltinNamedOpRegion", "cdecl")
    mlirLinalgFillBuiltinNamedOpRegion.argtypes = [MlirOperation]
    mlirLinalgFillBuiltinNamedOpRegion.restype = None

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Linalg.h: 25
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__linalg__", "cdecl"):
    mlirGetDialectHandle__linalg__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__linalg__", "cdecl")
    mlirGetDialectHandle__linalg__.argtypes = []
    mlirGetDialectHandle__linalg__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Func.h: 27
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__func__", "cdecl"):
    mlirGetDialectHandle__func__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__func__", "cdecl")
    mlirGetDialectHandle__func__.argtypes = []
    mlirGetDialectHandle__func__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/ControlFlow.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__cf__", "cdecl"):
        continue
    mlirGetDialectHandle__cf__ = _lib.get("mlirGetDialectHandle__cf__", "cdecl")
    mlirGetDialectHandle__cf__.argtypes = []
    mlirGetDialectHandle__cf__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/Tensor.h: 19
for _lib in _libs.values():
    if not _lib.has("mlirGetDialectHandle__tensor__", "cdecl"):
        continue
    mlirGetDialectHandle__tensor__ = _lib.get("mlirGetDialectHandle__tensor__", "cdecl")
    mlirGetDialectHandle__tensor__.argtypes = []
    mlirGetDialectHandle__tensor__.restype = MlirDialectHandle
    break

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 19
if _libs["MLIRPythonCAPI"].has("mlirGetDialectHandle__pdl__", "cdecl"):
    mlirGetDialectHandle__pdl__ = _libs["MLIRPythonCAPI"].get("mlirGetDialectHandle__pdl__", "cdecl")
    mlirGetDialectHandle__pdl__.argtypes = []
    mlirGetDialectHandle__pdl__.restype = MlirDialectHandle

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 25
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLType", "cdecl"):
    mlirTypeIsAPDLType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLType", "cdecl")
    mlirTypeIsAPDLType.argtypes = [MlirType]
    mlirTypeIsAPDLType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 31
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLAttributeType", "cdecl"):
    mlirTypeIsAPDLAttributeType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLAttributeType", "cdecl")
    mlirTypeIsAPDLAttributeType.argtypes = [MlirType]
    mlirTypeIsAPDLAttributeType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 33
if _libs["MLIRPythonCAPI"].has("mlirPDLAttributeTypeGet", "cdecl"):
    mlirPDLAttributeTypeGet = _libs["MLIRPythonCAPI"].get("mlirPDLAttributeTypeGet", "cdecl")
    mlirPDLAttributeTypeGet.argtypes = [MlirContext]
    mlirPDLAttributeTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 39
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLOperationType", "cdecl"):
    mlirTypeIsAPDLOperationType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLOperationType", "cdecl")
    mlirTypeIsAPDLOperationType.argtypes = [MlirType]
    mlirTypeIsAPDLOperationType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 41
if _libs["MLIRPythonCAPI"].has("mlirPDLOperationTypeGet", "cdecl"):
    mlirPDLOperationTypeGet = _libs["MLIRPythonCAPI"].get("mlirPDLOperationTypeGet", "cdecl")
    mlirPDLOperationTypeGet.argtypes = [MlirContext]
    mlirPDLOperationTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 47
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLRangeType", "cdecl"):
    mlirTypeIsAPDLRangeType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLRangeType", "cdecl")
    mlirTypeIsAPDLRangeType.argtypes = [MlirType]
    mlirTypeIsAPDLRangeType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 49
if _libs["MLIRPythonCAPI"].has("mlirPDLRangeTypeGet", "cdecl"):
    mlirPDLRangeTypeGet = _libs["MLIRPythonCAPI"].get("mlirPDLRangeTypeGet", "cdecl")
    mlirPDLRangeTypeGet.argtypes = [MlirType]
    mlirPDLRangeTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 51
if _libs["MLIRPythonCAPI"].has("mlirPDLRangeTypeGetElementType", "cdecl"):
    mlirPDLRangeTypeGetElementType = _libs["MLIRPythonCAPI"].get("mlirPDLRangeTypeGetElementType", "cdecl")
    mlirPDLRangeTypeGetElementType.argtypes = [MlirType]
    mlirPDLRangeTypeGetElementType.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 57
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLTypeType", "cdecl"):
    mlirTypeIsAPDLTypeType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLTypeType", "cdecl")
    mlirTypeIsAPDLTypeType.argtypes = [MlirType]
    mlirTypeIsAPDLTypeType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 59
if _libs["MLIRPythonCAPI"].has("mlirPDLTypeTypeGet", "cdecl"):
    mlirPDLTypeTypeGet = _libs["MLIRPythonCAPI"].get("mlirPDLTypeTypeGet", "cdecl")
    mlirPDLTypeTypeGet.argtypes = [MlirContext]
    mlirPDLTypeTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 65
if _libs["MLIRPythonCAPI"].has("mlirTypeIsAPDLValueType", "cdecl"):
    mlirTypeIsAPDLValueType = _libs["MLIRPythonCAPI"].get("mlirTypeIsAPDLValueType", "cdecl")
    mlirTypeIsAPDLValueType.argtypes = [MlirType]
    mlirTypeIsAPDLValueType.restype = c_bool

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Dialect/PDL.h: 67
if _libs["MLIRPythonCAPI"].has("mlirPDLValueTypeGet", "cdecl"):
    mlirPDLValueTypeGet = _libs["MLIRPythonCAPI"].get("mlirPDLValueTypeGet", "cdecl")
    mlirPDLValueTypeGet.argtypes = [MlirContext]
    mlirPDLValueTypeGet.restype = MlirType

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 154
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 155
for _lib in _libs.values():
    try:
        expr = (MlirAffineExpr).in_dll(_lib, "expr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 173
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 174
for _lib in _libs.values():
    try:
        attr = (MlirAttribute).in_dll(_lib, "attr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 191
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 192
for _lib in _libs.values():
    try:
        context = (MlirContext).in_dll(_lib, "context")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 212
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 214
for _lib in _libs.values():
    try:
        registry = (MlirDialectRegistry).in_dll(_lib, "registry")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 231
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 232
for _lib in _libs.values():
    try:
        loc = (MlirLocation).in_dll(_lib, "loc")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 249
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 250
for _lib in _libs.values():
    try:
        module = (MlirModule).in_dll(_lib, "module")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 267
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 268
for _lib in _libs.values():
    try:
        pm = (MlirPassManager).in_dll(_lib, "pm")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 285
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 286
for _lib in _libs.values():
    try:
        op = (MlirOperation).in_dll(_lib, "op")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 304
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 305
for _lib in _libs.values():
    try:
        typeID = (MlirTypeID).in_dll(_lib, "typeID")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 323
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 324
for _lib in _libs.values():
    try:
        type = (MlirType).in_dll(_lib, "type")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 342
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 343
for _lib in _libs.values():
    try:
        affineMap = (MlirAffineMap).in_dll(_lib, "affineMap")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 361
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 362
for _lib in _libs.values():
    try:
        integerSet = (MlirIntegerSet).in_dll(_lib, "integerSet")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 381
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 383
for _lib in _libs.values():
    try:
        jit = (MlirExecutionEngine).in_dll(_lib, "jit")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 401
for _lib in _libs.values():
    try:
        ptr = (POINTER(None)).in_dll(_lib, "ptr")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 402
for _lib in _libs.values():
    try:
        value = (MlirValue).in_dll(_lib, "value")
        break
    except:
        pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 54
def MLIR_PYTHON_STRINGIZE(s):
    return s

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 55
def MLIR_PYTHON_STRINGIZE_ARG(arg):
    return (MLIR_PYTHON_STRINGIZE (arg))

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 95
try:
    MLIR_PYTHON_CAPI_PTR_ATTR = '_CAPIPtr'
except:
    pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 108
try:
    MLIR_PYTHON_CAPI_FACTORY_ATTR = '_CAPICreate'
except:
    pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 116
try:
    MLIR_PYTHON_MAYBE_DOWNCAST_ATTR = 'maybe_downcast'
except:
    pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 125
try:
    MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR = 'register_type_caster'
except:
    pass

# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Bindings/Python/Interop.h: 133
def MLIR_PYTHON_GET_WRAPPED_POINTER(object):
    return cast((object.ptr), POINTER(None))

MlirLlvmThreadPool = struct_MlirLlvmThreadPool# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 60

MlirTypeID = struct_MlirTypeID# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 61

MlirTypeIDAllocator = struct_MlirTypeIDAllocator# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 62

MlirStringRef = struct_MlirStringRef# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 73

MlirLogicalResult = struct_MlirLogicalResult# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Support.h: 116

MlirBytecodeWriterConfig = struct_MlirBytecodeWriterConfig# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 51

MlirContext = struct_MlirContext# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 52

MlirDialect = struct_MlirDialect# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 53

MlirDialectRegistry = struct_MlirDialectRegistry# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 54

MlirOperation = struct_MlirOperation# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 55

MlirOpOperand = struct_MlirOpOperand# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 56

MlirOpPrintingFlags = struct_MlirOpPrintingFlags# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 57

MlirBlock = struct_MlirBlock# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 58

MlirRegion = struct_MlirRegion# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 59

MlirSymbolTable = struct_MlirSymbolTable# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 60

MlirAttribute = struct_MlirAttribute# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 62

MlirIdentifier = struct_MlirIdentifier# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 63

MlirLocation = struct_MlirLocation# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 64

MlirModule = struct_MlirModule# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 65

MlirType = struct_MlirType# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 66

MlirValue = struct_MlirValue# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 67

MlirNamedAttribute = struct_MlirNamedAttribute# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 76

MlirDialectHandle = struct_MlirDialectHandle# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 201

MlirOperationState = struct_MlirOperationState# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IR.h: 340

MlirExecutionEngine = struct_MlirExecutionEngine# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/ExecutionEngine.h: 31

MlirAffineExpr = struct_MlirAffineExpr# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineExpr.h: 38

MlirAffineMap = struct_MlirAffineMap# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/AffineMap.h: 39

MlirPass = struct_MlirPass# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 43

MlirExternalPass = struct_MlirExternalPass# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 44

MlirPassManager = struct_MlirPassManager# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 45

MlirOpPassManager = struct_MlirOpPassManager# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 46

MlirExternalPassCallbacks = struct_MlirExternalPassCallbacks# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Pass.h: 143

MlirDiagnostic = struct_MlirDiagnostic# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/Diagnostics.h: 26

MlirIntegerSet = struct_MlirIntegerSet# /Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include/mlir-c/IntegerSet.h: 38

# No inserted files

# No prefix-stripping

