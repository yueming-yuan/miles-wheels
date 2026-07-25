"""Build a complete Transformer Engine wheel triplet."""

import glob
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

TE_VERSION = "2.17.0"
TE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"


def _build_te_core_aarch64(wheel_dir, run):
    repo_dir = tempfile.mkdtemp(prefix="transformer-engine-")
    image_tag = (
        f"miles-wheels-transformer-engine:"
        f"{TE_VERSION}-aarch64-{uuid.uuid4().hex}"
    )
    image_built = False

    try:
        run(["git", "clone", "https://github.com/NVIDIA/TransformerEngine.git", repo_dir])
        run(["git", "checkout", TE_COMMIT], cwd=repo_dir)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)

        # PyPI has no CUDA 13 aarch64 core wheel. Use NVIDIA's fixed
        # manylinux_2_28_aarch64 common-only release recipe.
        run([
            "docker", "build", "--no-cache",
            "--network", "host",
            "--build-arg", "CUDA_MAJOR=13",
            "--build-arg", "CUDA_MINOR=0",
            "--build-arg", "BUILD_METAPACKAGE=false",
            "--build-arg", "BUILD_COMMON=true",
            "--build-arg", "BUILD_PYTORCH=false",
            "--build-arg", "BUILD_JAX=false",
            "--tag", image_tag,
            "--file", os.path.join(repo_dir, "build_tools/wheel_utils/Dockerfile.aarch"),
            repo_dir,
        ])
        image_built = True
        run([
            "docker", "run", "--rm",
            "--network", "host",
            "--env", f"TARGET_BRANCH={TE_COMMIT}",
            "--mount", f"type=bind,source={wheel_dir},target=/wheelhouse",
            image_tag,
        ])
    finally:
        if image_built:
            try:
                result = subprocess.run(
                    ["docker", "image", "rm", image_tag],
                    check=False,
                )
            except OSError as exc:
                print(f"WARNING: Failed to remove Docker image {image_tag}: {exc}")
            else:
                if result.returncode != 0:
                    print(
                        f"WARNING: Failed to remove Docker image {image_tag} "
                        f"(exit code {result.returncode})"
                    )
        try:
            shutil.rmtree(repo_dir)
        except OSError as exc:
            print(f"WARNING: Failed to remove Transformer Engine source {repo_dir}: {exc}")


def _validate_te_build_environment(args):
    expected_arch = "x86_64" if args.arch == "x86" else "aarch64"
    machine = platform.machine()
    if machine != expected_arch:
        raise RuntimeError(
            f"Transformer Engine target arch is {expected_arch}, running on {machine}"
        )

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


def _validate_te_torch_wheel(path, core_dist):
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
    if metadata["Version"] != TE_VERSION:
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
        or str(core_requirements[0].specifier) != f"=={TE_VERSION}"
    ):
        raise RuntimeError(
            f"Expected {core_dist}=={TE_VERSION} in {path}, found {core_requirements}"
        )


def build(args, wheel_dir, run):
    _validate_te_build_environment(args)
    cuda_major = int(args.cuda[:2])
    core_dist = f"transformer_engine_cu{cuda_major}"

    for pattern in (
        "transformer_engine-*.whl",
        "transformer_engine_cu1[23]-*.whl",
        "transformer_engine_torch-*.whl",
    ):
        for path in glob.glob(os.path.join(wheel_dir, pattern)):
            os.remove(path)

    run([sys.executable, "-m", "pip", "install", "nvidia-mathdx==25.6.0"])
    run([
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:", "--no-deps",
        f"transformer_engine=={TE_VERSION}",
        "--dest", wheel_dir,
    ])

    if cuda_major < 13 or args.arch == "x86":
        run([
            sys.executable, "-m", "pip", "download",
            "--only-binary=:all:", "--no-deps",
            f"{core_dist}=={TE_VERSION}",
            "--dest", wheel_dir,
        ])
    else:
        _build_te_core_aarch64(wheel_dir, run)

    core_wheels = glob.glob(
        os.path.join(wheel_dir, f"{core_dist}-{TE_VERSION}-*.whl")
    )
    if len(core_wheels) != 1:
        raise RuntimeError(
            f"Expected one {core_dist} {TE_VERSION} wheel, found {core_wheels}"
        )
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", core_wheels[0],
    ])
    run(
        [sys.executable, "-m", "pip", "wheel",
         "--no-cache-dir",
         f"transformer_engine_torch=={TE_VERSION}",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", wheel_dir],
        env={
            "NVTE_NO_LOCAL_VERSION": "1",
            "NVTE_PYTORCH_FORCE_BUILD": "TRUE",
        },
    )

    arch = "x86_64" if args.arch == "x86" else args.arch
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    expected = [
        f"transformer_engine-{TE_VERSION}-py3-none-any.whl",
        f"{core_dist}-{TE_VERSION}-py3-none-manylinux_2_28_{arch}.whl",
        f"transformer_engine_torch-{TE_VERSION}-{python_tag}-{python_tag}-linux_{arch}.whl",
    ]
    missing = [
        name for name in expected
        if not os.path.isfile(os.path.join(wheel_dir, name))
    ]
    if missing:
        raise RuntimeError(f"Missing Transformer Engine wheel(s): {missing}")
    _validate_te_torch_wheel(os.path.join(wheel_dir, expected[2]), core_dist)
