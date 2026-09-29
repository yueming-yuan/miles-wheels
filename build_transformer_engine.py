"""Build the Transformer Engine wheel triplet from radixark/TransformerEngine.

All three dists come from one fork checkout, versioned "<VERSION.txt>+miles"
so they can never be mistaken for NVIDIA's PyPI wheels. The build has two
phases because they need different environments:

  sources  Docker only: NVIDIA's manylinux release recipe builds the
           transformer_engine metapackage, the transformer_engine_cu<N> core
           and the transformer_engine_torch sdist.
  torch    The target image's Python 3.12 / torch / nvcc: compiles
           transformer_engine_torch from that sdist against the image's torch.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from email.parser import BytesParser

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

TE_REPO = "https://github.com/radixark/TransformerEngine.git"
TE_REF_DEFAULT = "miles-main"
TE_LOCAL_LABEL = "miles"
PHASES = ("all", "sources", "torch")

# Records which fork commit the wheels in the wheel directory were built from.
# It is uploaded next to them, so a pipeline can tell whether the ref has moved.
SOURCE_MANIFEST = "transformer_engine-source.json"
# Hands the torch sdist from the sources phase to the torch phase. A
# subdirectory, so upload (top-level files only) never publishes it.
SDIST_SUBDIR = "te-sdist"


def _arch(args):
    return "x86_64" if args.arch == "x86" else args.arch


def _core_dist(args):
    return f"transformer_engine_cu{int(args.cuda[:2])}"


def _expected_wheels(args, version):
    arch = _arch(args)
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return [
        f"transformer_engine-{version}-py3-none-any.whl",
        f"{_core_dist(args)}-{version}-py3-none-manylinux_2_28_{arch}.whl",
        f"transformer_engine_torch-{version}-{python_tag}-{python_tag}-linux_{arch}.whl",
    ]


def _validate_arch(args):
    expected_arch = _arch(args)
    machine = platform.machine()
    if machine != expected_arch:
        raise RuntimeError(
            f"Transformer Engine target arch is {expected_arch}, running on {machine}"
        )


def _validate_torch_environment(args):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Transformer Engine release wheels require Python 3.12, "
            f"running on {sys.version_info.major}.{sys.version_info.minor}"
        )

    expected_cuda = f"{args.cuda[:2]}.{args.cuda[2:]}"
    torch_cuda = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.version.cuda)"],
        text=True,
    ).strip()
    if torch_cuda != expected_cuda:
        raise RuntimeError(
            f"Transformer Engine target CUDA is {expected_cuda}, "
            f"but torch was built for CUDA {torch_cuda}"
        )

    nvcc_output = subprocess.check_output(
        ["nvcc", "--version"],
        stderr=subprocess.STDOUT,
        text=True,
    )
    if f"release {expected_cuda}," not in nvcc_output:
        raise RuntimeError(
            f"Transformer Engine target CUDA is {expected_cuda}, "
            "but nvcc reports a different toolkit"
        )


def _stamp_version(repo_dir):
    path = os.path.join(repo_dir, "build_tools", "VERSION.txt")
    with open(path) as f:
        base = f.readline().strip()
    if "+" in base:
        raise RuntimeError(f"{path} already carries a local version: {base}")
    version = f"{base}+{TE_LOCAL_LABEL}"
    with open(path, "w") as f:
        f.write(version + "\n")
    return version


def _remove_docker_image(image_tag):
    try:
        result = subprocess.run(["docker", "image", "rm", image_tag], check=False)
    except OSError as exc:
        print(f"WARNING: Failed to remove Docker image {image_tag}: {exc}")
    else:
        if result.returncode != 0:
            print(
                f"WARNING: Failed to remove Docker image {image_tag} "
                f"(exit code {result.returncode})"
            )


def _build_sources(args, wheel_dir, run):
    _validate_arch(args)
    arch = _arch(args)
    sdist_dir = os.path.join(wheel_dir, SDIST_SUBDIR)

    for pattern in (
        "transformer_engine-*.whl",
        "transformer_engine_cu1[23]-*.whl",
        "transformer_engine_torch-*.whl",
        SOURCE_MANIFEST,
    ):
        for path in glob.glob(os.path.join(wheel_dir, pattern)):
            os.remove(path)
    shutil.rmtree(sdist_dir, ignore_errors=True)
    os.makedirs(sdist_dir)

    repo_dir = tempfile.mkdtemp(prefix="transformer-engine-")
    wheelhouse = tempfile.mkdtemp(prefix="te-wheelhouse-")
    image_tag = f"miles-wheels-transformer-engine:{arch}-{uuid.uuid4().hex}"
    image_built = False

    try:
        run(["git", "clone", TE_REPO, repo_dir])
        run(["git", "checkout", args.te_ref], cwd=repo_dir)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
        ).strip()
        version = _stamp_version(repo_dir)

        dockerfile = "Dockerfile.x86" if args.arch == "x86" else "Dockerfile.aarch"
        run([
            "docker", "build", "--no-cache",
            "--network", "host",
            "--build-arg", f"CUDA_MAJOR={args.cuda[:2]}",
            "--build-arg", f"CUDA_MINOR={args.cuda[2:]}",
            "--build-arg", "BUILD_METAPACKAGE=true",
            "--build-arg", "BUILD_COMMON=true",
            "--build-arg", "BUILD_PYTORCH=true",
            "--build-arg", "BUILD_JAX=false",
            "--tag", image_tag,
            "--file", os.path.join(repo_dir, "build_tools/wheel_utils", dockerfile),
            repo_dir,
        ])
        image_built = True
        # The recipe checks out TARGET_BRANCH first. Pin it to the commit that is
        # already checked out, so the stamped VERSION.txt survives.
        run([
            "docker", "run", "--rm",
            "--network", "host",
            "--env", f"TARGET_BRANCH={commit}",
            "--mount", f"type=bind,source={wheelhouse},target=/wheelhouse",
            image_tag,
        ])

        for name in _expected_wheels(args, version)[:2]:
            src = os.path.join(wheelhouse, name)
            if not os.path.isfile(src):
                raise RuntimeError(
                    f"Transformer Engine recipe did not produce {name}; "
                    f"got {sorted(os.listdir(wheelhouse))}"
                )
            shutil.move(src, wheel_dir)
        sdists = glob.glob(os.path.join(wheelhouse, "transformer_engine_torch-*.tar.gz"))
        if len(sdists) != 1:
            raise RuntimeError(f"Expected one transformer_engine_torch sdist, found {sdists}")
        shutil.move(sdists[0], sdist_dir)

        with open(os.path.join(wheel_dir, SOURCE_MANIFEST), "w") as f:
            json.dump(
                {"repo": TE_REPO, "ref": args.te_ref, "commit": commit, "version": version},
                f, indent=2,
            )
            f.write("\n")
        print(f"Transformer Engine {version} sources built from {TE_REPO}@{commit}")
    finally:
        if image_built:
            _remove_docker_image(image_tag)
        for path in (repo_dir, wheelhouse):
            try:
                shutil.rmtree(path)
            except OSError as exc:
                print(f"WARNING: Failed to remove {path}: {exc}")


def _validate_te_torch_wheel(path, core_dist, version):
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = [
            name for name in wheel.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise RuntimeError(
                f"Expected one METADATA file in {path}, found {metadata_paths}"
            )
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))

    expected_name = canonicalize_name("transformer_engine_torch")
    if canonicalize_name(metadata["Name"]) != expected_name:
        raise RuntimeError(
            f"Unexpected Transformer Engine torch wheel name: {metadata['Name']}"
        )
    if metadata["Version"] != version:
        raise RuntimeError(
            f"Unexpected Transformer Engine torch wheel version: {metadata['Version']}"
        )

    requirements = [
        Requirement(value)
        for value in metadata.get_all("Requires-Dist", [])
    ]
    core_requirements = [
        requirement for requirement in requirements
        if canonicalize_name(requirement.name).startswith("transformer-engine-cu")
    ]
    expected_core = canonicalize_name(core_dist)
    if (
        len(core_requirements) != 1
        or canonicalize_name(core_requirements[0].name) != expected_core
        or str(core_requirements[0].specifier) != f"=={version}"
    ):
        raise RuntimeError(
            f"Expected {core_dist}=={version} in {path}, found {core_requirements}"
        )


def _build_torch(args, wheel_dir, run):
    manifest_path = os.path.join(wheel_dir, SOURCE_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"{manifest_path} is missing; run the sources phase first")
    with open(manifest_path) as f:
        version = json.load(f)["version"]

    sdists = glob.glob(os.path.join(wheel_dir, SDIST_SUBDIR, "transformer_engine_torch-*.tar.gz"))
    if len(sdists) != 1:
        raise RuntimeError(f"Expected one transformer_engine_torch sdist, found {sdists}")
    for path in glob.glob(os.path.join(wheel_dir, "transformer_engine_torch-*.whl")):
        os.remove(path)

    expected = _expected_wheels(args, version)
    run([sys.executable, "-m", "pip", "install", "nvidia-mathdx==25.6.0"])
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", os.path.join(wheel_dir, expected[1]),
    ])
    run(
        [sys.executable, "-m", "pip", "wheel",
         "--no-cache-dir",
         sdists[0],
         "-v", "--no-build-isolation", "--no-deps",
         "-w", wheel_dir],
        env={
            "NVTE_NO_LOCAL_VERSION": "1",
            "NVTE_PYTORCH_FORCE_BUILD": "TRUE",
        },
    )

    missing = [
        name for name in expected
        if not os.path.isfile(os.path.join(wheel_dir, name))
    ]
    if missing:
        raise RuntimeError(f"Missing Transformer Engine wheel(s): {missing}")
    _validate_te_torch_wheel(os.path.join(wheel_dir, expected[2]), _core_dist(args), version)


def build(args, wheel_dir, run):
    _validate_arch(args)
    if args.te_phase in ("all", "torch"):
        # Fail before the hours-long sources phase, not after it.
        _validate_torch_environment(args)
    if args.te_phase in ("all", "sources"):
        _build_sources(args, wheel_dir, run)
    if args.te_phase in ("all", "torch"):
        _build_torch(args, wheel_dir, run)
