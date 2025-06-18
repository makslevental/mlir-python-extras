import argparse

from pip._internal.cli import cmdoptions
from pip._internal.commands import index

cmd = index.IndexCommand("blah", "")


def get_available_package_candidates():
    query = "mlir-python-bindings"
    options, _args = cmd.parse_args(
        [
            "index",
            "versions",
            query,
            "--find-links",
            "https://makslevental.github.io/wheels",
        ]
    )
    options.no_index = True
    target_python = cmdoptions.make_target_python(options)
    with cmd._build_session(options) as session:
        finder = cmd._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
            ignore_requires_python=options.ignore_requires_python,
        )

        return list(finder.find_all_candidates(query))


def get_latest_bindings_cand(all_candidates, gpu_platform: str, no_gpu=False):
    if gpu_platform == "none":
        bindings_cands = list(
            filter(
                lambda x: x.version.local
                and "cuda" not in x.version.local
                and "amdgpu" not in x.version.local,
                all_candidates,
            )
        )
    else:
        bindings_cands = list(
            filter(
                lambda x: x.version.local and gpu_platform in x.version.local,
                all_candidates,
            )
        )

    assert len(bindings_cands), "couldn't find any bindings versions"
    bindings_cands.sort(key=lambda x: x.version.release)
    return bindings_cands[-1]


def get_latest_bindings_version_name(gpu_platform, no_gpu=False):
    return get_latest_bindings_cand(
        get_available_package_candidates(), gpu_platform, no_gpu
    ).version


def get_latest_bindings_version_plat(gpu_platform, no_gpu=False):
    latest_cand = get_latest_bindings_cand(
        get_available_package_candidates(), gpu_platform, no_gpu
    )
    _, plat = latest_cand.link.filename.rsplit("-", 1)
    return plat.split(".")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_platform", nargs="?", default="cuda")
    parser.add_argument("--only-plat", action="store_true")
    args = parser.parse_args()
    if args.only_plat:
        print(get_latest_bindings_version_plat(args.gpu_platform))
    else:
        print(get_latest_bindings_version_name(args.gpu_platform))


if __name__ == "__main__":
    main()
