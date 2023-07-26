import re
from textwrap import dedent, indent

from yaml import safe_load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import multiprocess as multiprocessing
import os

from .._mlir._mlir_libs._nelli_mlir import print_help


def capture_help():
    def method(w):
        os.dup2(w.fileno(), 1)
        print_help()

    r, w = multiprocessing.Pipe()

    reader = os.fdopen(r.fileno(), "r")

    process = multiprocessing.Process(None, method, "print_help", (w,))
    process.start()

    lines = []

    capturing = False

    for i in range(1000):
        line = reader.readline()
        if "Passes:" in line:
            capturing = True
            continue
        if "Pass Pipelines:" in line:
            break
        if capturing:
            lines.append(line)

    # you don't need these because help already exits
    # reader.close()
    # process.join()

    # print("".join(lines))
    return lines


def fixup_lines_into_yaml(lines):
    i = 0
    while i < len(lines):
        l = lines[i]
        prev_line = lines[i - 1]
        leading_spaces = len(prev_line) - len(prev_line.strip())
        while l.strip().startswith("="):
            lines[i] = " " * leading_spaces + l
            i += 1
            l = lines[i]

        i += 1

    lines = dedent("".join(lines)).splitlines(keepends=True)
    for i, l in enumerate(lines):
        l = l.replace(" - ", " # ")
        if l.startswith("--"):
            l = re.sub(r"^--((\w|-)+)", r"\1:", l, re.M)
        l = re.sub(r"^ {2}--", " - ", l, re.M)
        l = re.sub(r"^ {7}=", "     - ", l, re.M)
        l = l.replace("=<value>", "=<value>:")
        lines[i] = l

    # with open("passes_new.txt", "w") as f:
    #     f.write("".join(lines))
    yml = safe_load("".join(lines))
    return yml


def parse_passes(yml):
    illegal_names = {"global": "global_"}
    illegal_names_back = {"global_": "global"}
    ident = 4
    all_types = set()
    for pass_name, maybe_args in yml.items():
        # print(k, v)
        py_args = []
        if maybe_args is None:
            pass
        else:
            for a in maybe_args:
                if isinstance(a, str):
                    if "=" in a:
                        name, typ = a.split("=")
                        all_types.add(typ)
                        match typ:
                            case item if item in {
                                "<long>",
                                "<ulong>",
                                "<number>",
                                "<uint>",
                                "<int>",
                            }:
                                arg_typ = "int"
                            case "<string>" | "<pass-manager>":
                                arg_typ = "str"
                            case _:
                                raise RuntimeError(a)
                    else:
                        name = a
                        arg_typ = "bool"
                    name = illegal_names.get(name, name)
                    arg_name = f"{name.replace('-', '_')}"
                    py_args.append((arg_name, arg_typ))
                else:
                    assert isinstance(a, dict)
                    assert len(a) == 1
                    name, _typ = list(a.keys())[0].split("=")
                    arg_name = f"{name.replace('-', '_')}"
                    py_args.append((arg_name, "str"))

        if py_args:
            py_args_str = ", ".join([f"{n}=None" for n, t in py_args])
            print(
                indent(
                    f"def {pass_name.replace('-', '_')}(self, {py_args_str}):",
                    prefix=" " * ident,
                )
            )
            mlir_args = []
            for n, t in py_args:
                if t in {"int", "str"}:
                    print(
                        indent(
                            f"if {n} is not None and isinstance({n}, (list, tuple)):",
                            prefix=" " * ident * 2,
                        )
                    )
                    print(
                        indent(f"{n} = ','.join(map(str, {n}))", prefix=" " * ident * 3)
                    )
                mlir_args.append(f"{n}={n}")
            print(
                indent(
                    dedent(
                        f"""\
               self._add_pass("{pass_name}", {', '.join(mlir_args)})
               return self
        """
                    ),
                    prefix=" " * ident * 2,
                )
            )

        else:
            print(
                indent(
                    dedent(
                        f"""\
            def {pass_name.replace('-', '_')}(self):
               self._add_pass("{pass_name}")
               return self
        """
                    ),
                    prefix=" " * ident,
                )
            )


if __name__ == "__main__":
    lines = capture_help()
    yml = fixup_lines_into_yaml(lines)
    parse_passes(yml)
