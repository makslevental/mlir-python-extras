import glob
import json
import keyword
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE
from textwrap import dedent, indent

from mlir._mlir_libs import include

include_path = Path(include.__path__[0])


def dump_json(td_path: Path):
    llvm_tblgen_name = "llvm-tblgen"
    if platform.system() == "Windows":
        llvm_tblgen_name += ".exe"

    # try from mlir-native-tools
    llvm_tblgen_path = Path(sys.prefix) / "bin" / llvm_tblgen_name
    # try to find using which
    if not llvm_tblgen_path.exists():
        llvm_tblgen_path = shutil.which(llvm_tblgen_name)
    assert Path(llvm_tblgen_path).exists() is not None, "couldn't find llvm-tblgen"

    args = [f"-I{include_path}", f"-I{td_path.parent}", str(td_path), "-dump-json"]
    res = subprocess.run(
        [llvm_tblgen_path] + args,
        cwd=Path(".").cwd(),
        check=True,
        stdout=PIPE,
        stderr=subprocess.DEVNULL,
    )
    res_json = json.loads(res.stdout.decode("utf-8"))

    return res_json


@dataclass
class Option:
    argument: str
    description: str
    type: str
    additional_opt_flags: str
    default_value: str
    list_option: bool = False


@dataclass
class Pass:
    name: str
    argument: str
    options: list[Option]
    description: str
    summary: str


TYPE_MAP = {
    "::mlir::gpu::amd::Runtime": '"gpu::amd::Runtime"',
    "OpPassManager": '"OpPassManager"',
    "bool": "bool",
    "double": "float",
    "enum FusionMode": '"FusionMode"',
    "int": "int",
    "int32_t": "int",
    "int64_t": "int",
    "mlir::SparseParallelizationStrategy": '"SparseParallelizationStrategy"',
    "mlir::arm_sme::ArmStreaming": '"arm_sme::ArmStreaming"',
    "std::string": "str",
    "uint64_t": "int",
    "unsigned": "int",
}


def generate_pass_method(pass_: Pass):
    ident = 4
    py_args = []
    for o in pass_.options:
        argument = o.argument.replace("-", "_")
        if keyword.iskeyword(argument):
            argument += "_"
        type = TYPE_MAP[o.type]
        if o.list_option:
            type = f"list[{type}]"
        py_args.append((argument, type))

    def print_options_doc_string(pass_):
        print(
            indent(
                f"'''{pass_.summary}",
                prefix=" " * ident * 2,
            )
        )
        if pass_.description:
            for l in pass_.description.split("\n"):
                print(
                    indent(
                        f"{l}",
                        prefix=" " * ident,
                    )
                )
        if pass_.options:
            print(
                indent(
                    f"Args:",
                    prefix=" " * ident * 2,
                )
            )
            for o in pass_.options:
                print(
                    indent(
                        f"{o.argument}: {o.description}",
                        prefix=" " * ident * 3,
                    )
                )
        print(
            indent(
                f"'''",
                prefix=" " * ident * 2,
            )
        )

    pass_name = pass_.argument
    if py_args:
        py_args_str = ", ".join([f"{n}: {t} = None" for n, t in py_args])
        print(
            indent(
                f"def {pass_name.replace('-', '_')}(self, {py_args_str}):",
                prefix=" " * ident,
            )
        )
        print_options_doc_string(pass_)

        mlir_args = []
        for n, t in py_args:
            if "list" in t:
                print(
                    indent(
                        f"if {n} is not None and isinstance({n}, (list, tuple)):",
                        prefix=" " * ident * 2,
                    )
                )
                print(indent(f"{n} = ','.join(map(str, {n}))", prefix=" " * ident * 3))
            mlir_args.append(f"{n}={n}")
        print(
            indent(
                dedent(
                    f"""\
            self.add_pass("{pass_name}", {', '.join(mlir_args)})
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
        def {pass_name.replace('-', '_')}(self):"""
                ),
                prefix=" " * ident,
            )
        )
        print_options_doc_string(pass_)
        print(
            indent(
                dedent(
                    f"""\
            self.add_pass("{pass_name}")
            return self
    """
                ),
                prefix=" " * ident * 2,
            )
        )


def gather_passes_from_td_json(j):
    passes = []
    for pass_ in j["!instanceof"]["Pass"]:
        pass_ = j[pass_]
        options = []
        for o in pass_["options"]:
            option = j[o["def"]]
            option = Option(
                argument=option["argument"],
                description=option["description"],
                type=option["type"],
                additional_opt_flags=option["additionalOptFlags"],
                default_value=option["defaultValue"],
                list_option="ListOption" in option["!superclasses"],
            )
            options.append(option)
        pass_ = Pass(
            name=pass_["!name"],
            argument=pass_["argument"],
            options=options,
            description=pass_["description"],
            summary=pass_["summary"],
        )
        passes.append(pass_)

    return passes


if __name__ == "__main__":
    passes = []
    for td in glob.glob(str(include_path / "**" / "*.td"), recursive=True):
        try:
            j = dump_json(Path(td))
            if j["!instanceof"]["Pass"]:
                passes.extend(gather_passes_from_td_json(j))
        except:
            continue

    for p in sorted(passes, key=lambda p: p.argument):
        generate_pass_method(p)

    for p in sorted(passes, key=lambda p: p.argument):
        argument = p.argument.replace("-", "_")
        print(f"{argument} = Pipeline().{argument}")
