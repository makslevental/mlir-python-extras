import argparse

from pip._internal.cli import cmdoptions
from pip._internal.commands import index


def get_available_package_versions(self, options, args):
    target_python = cmdoptions.make_target_python(options)
    query = args[0]

    with self._build_session(options) as session:
        finder = self._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
            ignore_requires_python=options.ignore_requires_python,
        )

        versions = set(
            candidate.version for candidate in finder.find_all_candidates(query)
        )

        return list(versions)


def get_latest_gpu_version(all_versions, platform: str):
    bindings_versions = list(
        filter(lambda x: x.local and platform in x.local, all_versions)
    )
    assert len(bindings_versions), "couldn't find any bindings versions"
    bindings_versions.sort(key=lambda x: x.release)
    return bindings_versions[0]


cmd = index.IndexCommand("blah", "")


def get_latest_gpu_version_name(platform):
    options, _args = cmd.parse_args(
        [
            "index",
            "versions",
            "mlir-python-bindings",
            "--find-links",
            "https://makslevental.github.io/wheels",
        ]
    )
    options.no_index = True
    all_versions = get_available_package_versions(
        cmd, options, ["mlir-python-bindings"]
    )
    return get_latest_gpu_version(all_versions, platform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", nargs="?", default="cuda")
    platform = parser.parse_args().platform
    print(get_latest_gpu_version_name(platform))


if __name__ == "__main__":
    main()
