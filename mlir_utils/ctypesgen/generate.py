import os
from pathlib import Path

from ctypesgen.main import main

INCLUDE_DIR = "/Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs/include"
LIB_DIR = "/Users/mlevental/miniforge3/envs/mlir-utils/lib/python3.11/site-packages/mlir/mlir/_mlir_libs"

paths = []
for root, dirs, files in os.walk(INCLUDE_DIR):
    p = Path(root).relative_to(INCLUDE_DIR)
    for f in files:
        paths.append(f"{INCLUDE_DIR}/{p}/{f}")

args = [
    "ctypesgen",
    "-lMLIRPythonCAPI",
    f"-L{LIB_DIR}",
    f"-I{INCLUDE_DIR}",
    *paths,
    "-o",
    "mlir_capi.py",
]
main(args)
