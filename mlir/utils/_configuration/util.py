import hashlib
import sys
from base64 import urlsafe_b64encode
from importlib.metadata import distribution, packages_distributions
from importlib.resources import files
from pathlib import Path

if sys.version_info.minor > 10:
    from importlib.resources.readers import MultiplexedPath
else:
    from importlib.readers import MultiplexedPath


def add_file_to_sources_txt_file(file_path: Path):
    package = __package__.split(".")[0]
    return
    package_root_path = files(package)
    if isinstance(package_root_path, MultiplexedPath):
        package_root_path = package_root_path._paths[0]
    dist = distribution(packages_distributions()[package][0])

    assert file_path.exists(), f"file being added doesn't exist at {file_path}"
    relative_file_path = Path(package) / file_path.relative_to(package_root_path)
    if dist._read_files_egginfo() is not None:
        with open(dist._path / "SOURCES.txt", "a") as sources_file:
            sources_file.write(f"\n{relative_file_path}")
    if dist._read_files_distinfo():
        with open(file_path, "rb") as file, open(
            dist._path / "RECORD", "a"
        ) as sources_file:
            # https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file
            m = hashlib.sha256()
            file = file.read()
            m.update(file)
            encoded = urlsafe_b64encode(m.digest())
            sources_file.write(
                f"{relative_file_path},sha256={encoded[:-1].decode()},{len(file)}\n"
            )
