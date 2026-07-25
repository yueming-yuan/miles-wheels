#!/usr/bin/env python3
"""Build GPU wheels (flash-attn, apex, transformer_engine, etc.)."""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import build_sglang_gateway

WHEEL_DIR = os.environ.get("WHEEL_DIR", "/tmp/wheels")
REPO = "yueming-yuan/miles-wheels"
TE_VERSION = "2.17.0"
TE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"


def run(cmd, *, env=None, cwd=None):
    """Run a command, streaming output. Exit on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def _setup_env(args):
    cuda_major, cuda_minor = args.cuda[:2], args.cuda[2:]
    print(f"CUDA  : {cuda_major}.{cuda_minor}  (cu{args.cuda})")
    print(f"Arch  : {args.arch}")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        "8.0;8.6;8.9;9.0;10.0;10.3" if args.arch == "x86" else "9.0;10.0;10.3",
    )
    os.environ["CUDA_VERSION"] = f"{cuda_major}.{cuda_minor}"
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")


# ── build steps ──────────────────────────────────────────────

def _build_flash_attn(args):
    run(
        [sys.executable, "-m", "pip", "wheel",
         "flash-attn==2.7.4.post1",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"MAX_JOBS": "64"},
    )


def _build_flash_attn_hopper(args):
    repo_dir = "/tmp/flash-attention"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    run(["git", "clone", "https://github.com/Dao-AILab/flash-attention.git", repo_dir])
    run(["git", "checkout", "fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89"], cwd=repo_dir)
    run(["git", "submodule", "update", "--init"], cwd=repo_dir)
    run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=os.path.join(repo_dir, "hopper"),
        env={"MAX_JOBS": "96"},
    )

    hopper_dist = os.path.join(repo_dir, "hopper", "dist")
    for f in os.listdir(hopper_dist):
        if f.endswith(".whl"):
            shutil.copy2(os.path.join(hopper_dist, f), WHEEL_DIR)

    shutil.rmtree(repo_dir)


def _build_apex(args):
    run(
        [sys.executable, "-m", "pip", "wheel",
         "-v", "--no-build-isolation", "--no-deps",
         "--config-settings", "--build-option=--cpp_ext --cuda_ext --parallel 8",
         "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4",
         "-w", WHEEL_DIR],
        env={"NVCC_APPEND_FLAGS": "--threads 4"},
    )


def _build_int4_qat(args):
    miles_dir = "/tmp/miles"
    if os.path.exists(miles_dir):
        shutil.rmtree(miles_dir)

    run(["git", "clone", "https://github.com/radixark/miles.git", miles_dir])
    run(
        [sys.executable, "-m", "pip", "wheel", ".",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        cwd=os.path.join(miles_dir, "miles/backends/megatron_utils/kernels/int4_qat"),
    )


def _build_te_core_aarch64():
    repo_dir = tempfile.mkdtemp(prefix="transformer-engine-")
    image_tag = f"miles-wheels-transformer-engine:{TE_VERSION}-aarch64-{os.getpid()}"

    run(["git", "clone", "https://github.com/NVIDIA/TransformerEngine.git", repo_dir])
    run(["git", "checkout", TE_COMMIT], cwd=repo_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)

    # PyPI has no CUDA 13 aarch64 core wheel. Use NVIDIA's fixed
    # manylinux_2_28_aarch64 common-only release recipe.
    run([
        "docker", "build", "--no-cache",
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
    run([
        "docker", "run", "--rm",
        "--env", f"TARGET_BRANCH={TE_COMMIT}",
        "--mount", f"type=bind,source={WHEEL_DIR},target=/wheelhouse",
        image_tag,
    ])
    run(["docker", "image", "rm", image_tag])
    shutil.rmtree(repo_dir)


def _build_transformer_engine(args):
    cuda_major = int(args.cuda[:2])
    version = TE_VERSION if cuda_major >= 13 else "2.10.0"
    core_dist = f"transformer_engine_cu{cuda_major}"

    for pattern in (
        "transformer_engine-*.whl",
        "transformer_engine_cu1[23]-*.whl",
        "transformer_engine_torch-*.whl",
    ):
        for path in glob.glob(os.path.join(WHEEL_DIR, pattern)):
            os.remove(path)

    if cuda_major >= 13:
        run([sys.executable, "-m", "pip", "install", "nvidia-mathdx==25.6.0"])
    run([
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:", "--no-deps",
        f"transformer_engine=={version}",
        "--dest", WHEEL_DIR,
    ])

    if cuda_major < 13 or args.arch == "x86":
        run([
            sys.executable, "-m", "pip", "download",
            "--only-binary=:all:", "--no-deps",
            f"{core_dist}=={version}",
            "--dest", WHEEL_DIR,
        ])
    else:
        _build_te_core_aarch64()

    core_wheels = glob.glob(
        os.path.join(WHEEL_DIR, f"{core_dist}-{version}-*.whl")
    )
    if len(core_wheels) != 1:
        raise RuntimeError(
            f"Expected one {core_dist} {version} wheel, found {core_wheels}"
        )
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", core_wheels[0],
    ])
    run(
        [sys.executable, "-m", "pip", "wheel",
         f"transformer_engine_torch=={version}",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"NVTE_PYTORCH_FORCE_BUILD": "TRUE"},
    )

    arch = "x86_64" if args.arch == "x86" else args.arch
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    expected = [
        f"transformer_engine-{version}-py3-none-any.whl",
        f"{core_dist}-{version}-py3-none-manylinux_2_28_{arch}.whl",
        f"transformer_engine_torch-{version}-{python_tag}-{python_tag}-linux_{arch}.whl",
    ]
    missing = [
        name for name in expected
        if not os.path.isfile(os.path.join(WHEEL_DIR, name))
    ]
    if missing:
        raise RuntimeError(f"Missing Transformer Engine wheel(s): {missing}")


def _build_causal_conv1d(args):
    # FORCE_BUILD: upstream setup.py otherwise downloads its own prebuilt wheel
    # when one matches, which never exists for aarch64/cu13 and may mismatch
    # this image's torch build on x86.
    run(
        [sys.executable, "-m", "pip", "wheel",
         "causal-conv1d==1.6.1",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAX_JOBS": "64"},
    )


def _build_mamba_ssm(args):
    # FORCE_BUILD: same reason as causal-conv1d.
    run(
        [sys.executable, "-m", "pip", "wheel",
         "mamba-ssm==2.3.1",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"MAMBA_FORCE_BUILD": "TRUE", "MAX_JOBS": "64"},
    )


def _build_fast_hadamard(args):
    run(
        [sys.executable, "-m", "pip", "wheel",
         "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"MAX_JOBS": "64"},
    )


def _build_sgl_router(args):
    """Build sgl-router Python wheel and standalone binary from source."""
    cfg = build_sglang_gateway.BuildConfig(bootstrap_rust=args.bootstrap_rust)
    build_sglang_gateway.build(cfg, WHEEL_DIR)


# Pinned past v0.3.12: the structured object store API miles' mooncake
# object-store backend imports (FieldSchema, export_ref/import_ref, unified
# put/get, release_result; kvcache-ai/Mooncake#2907/#3013/#3023) missed the
# v0.3.12 release cut. Drop this step for a plain pip pin once a release
# ships the API.
MOONCAKE_COMMIT = "4dbe5a4c194669850e9abad61172e9878b245b15"
MOONCAKE_VERSION = "0.3.13.dev0+g4dbe5a4c"


def _build_mooncake(args):
    """Build the mooncake wheel from source, mirroring upstream release-cuda13.yaml."""
    cuda_major = int(args.cuda[:2])
    repo_dir = "/tmp/mooncake"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    run(["git", "clone", "https://github.com/kvcache-ai/Mooncake.git", repo_dir])
    run(["git", "checkout", MOONCAKE_COMMIT], cwd=repo_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)
    run(["bash", "dependencies.sh", "-y"], cwd=repo_dir)

    # The wheel version comes from pyproject.toml, not the VERSION env
    # (upstream bumps it on release branches); stamp the pinned commit in.
    run(["sed", "-i", "-E", f's/^version = ".*"$/version = "{MOONCAKE_VERSION}"/',
         "mooncake-wheel/pyproject.toml"], cwd=repo_dir)
    run(["grep", "-q", f'version = "{MOONCAKE_VERSION}"',
         "mooncake-wheel/pyproject.toml"], cwd=repo_dir)

    torch_version = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__.split('+')[0])"],
        text=True,
    ).strip()

    build_env = {
        "BUILD_WITH_EP": "1",
        "CUDA_HOME": "/usr/local/cuda",
        # Bound the nested Ninja build used by torch.utils.cpp_extension.
        "MAX_JOBS": "2",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:" + os.environ.get("LIBRARY_PATH", ""),
        "PATH": os.environ["PATH"] + ":/usr/local/go/bin",
    }
    build_dir = os.path.join(repo_dir, "build")
    os.makedirs(build_dir)
    run(
        ["cmake", "..",
         "-DBUILD_UNIT_TESTS=OFF", "-DUSE_HTTP=ON", "-DUSE_ETCD=ON", "-DUSE_CUDA=ON",
         "-DWITH_EP=ON", "-DSTORE_USE_ETCD=ON", "-DCMAKE_BUILD_TYPE=Release",
         # EP only for this image's torch; extend when the image ships more.
         f"-DEP_TORCH_VERSIONS={torch_version}",
         f"-DPython3_EXECUTABLE={sys.executable}"],
        cwd=build_dir, env=build_env,
    )
    run(["cmake", "--build", ".", f"-j{os.cpu_count()}"], cwd=build_dir, env=build_env)
    run(["cmake", "--install", "."], cwd=build_dir, env=build_env)

    if args.arch == "x86":
        allocator_out = os.path.join(build_dir, "mooncake-transfer-engine/nvlink-allocator/")
        os.makedirs(allocator_out, exist_ok=True)
        run(["bash", "build.sh", allocator_out],
            cwd=os.path.join(repo_dir, "mooncake-transfer-engine/nvlink-allocator"),
            env=build_env)

    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    wheel_env = {
        **build_env,
        "PYTHON_VERSION": pyver,
        "OUTPUT_DIR": "dist",
    }
    if cuda_major >= 13:
        wheel_env["CU13_BUILD"] = "1"
    run(["./scripts/build_wheel.sh"], cwd=repo_dir, env=wheel_env)

    for f in glob.glob(os.path.join(repo_dir, "mooncake-wheel/dist/*.whl")):
        shutil.copy2(f, WHEEL_DIR)
    shutil.rmtree(repo_dir)


STEPS = {
    "flash-attn": _build_flash_attn,
    "flash-attn-hopper": _build_flash_attn_hopper,
    "apex": _build_apex,
    "int4_qat": _build_int4_qat,
    "te": _build_transformer_engine,
    "causal-conv1d": _build_causal_conv1d,
    "mamba-ssm": _build_mamba_ssm,
    "fast-hadamard": _build_fast_hadamard,
    "sgl-router": _build_sgl_router,
    "mooncake": _build_mooncake,
}

STEP_NAMES = ", ".join(STEPS)


# ── commands ─────────────────────────────────────────────────

def cmd_build(args):
    """Build all GPU wheels into the wheel output directory."""
    assert args.cuda in ("129", "130"), "currently only cu129 and cu130 are supported"
    _setup_env(args)
    os.makedirs(WHEEL_DIR, exist_ok=True)

    selected = {s.lower() for s in (args.only or [])}
    for name, fn in STEPS.items():
        if selected and name not in selected:
            print(f"\nSkipping {name}")
            continue
        print(f"\n>>> Building {name} ...")
        fn(args)

    print(f"\nDone. Wheels in {WHEEL_DIR}:")
    run(["ls", "-lh", WHEEL_DIR])


def _gh_json(gh_args):
    out = subprocess.check_output(["gh", *gh_args], text=True)
    return json.loads(out)


def _latest_versioned_release(tag):
    """Newest legacy cu<cuda>-<arch>-vX.Y.Z release, used to seed a fresh rolling tag."""
    releases = _gh_json(["release", "list", "--repo", REPO, "--limit", "100",
                         "--json", "tagName,createdAt"])
    candidates = [r for r in releases if r["tagName"].startswith(tag + "-v")]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["createdAt"])["tagName"]


def cmd_upload(args):
    """Sync the wheel output directory into the rolling cu<cuda>-<arch> release.

    The release is never deleted: unchanged assets stay, new packages are
    added, and a wheel whose version changed replaces its old asset. A fresh
    tag is seeded from the newest legacy versioned release so the set stays
    complete (the miles Dockerfile downloads every asset of one tag).
    """
    assert args.cuda in ("129", "130"), "currently only cu129 and cu130 are supported"

    cuda_major, cuda_minor = args.cuda[:2], args.cuda[2:]
    arch_str = "x86_64" if args.arch == "x86" else args.arch
    tag = f"cu{args.cuda}-{arch_str}"
    title = f"CUDA {cuda_major}.{cuda_minor} + {arch_str}"

    local = (sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl")))
             + sorted(glob.glob(os.path.join(WHEEL_DIR, "*.tar.gz"))))
    if not local:
        print(f"No .whl or .tar.gz files found in {WHEEL_DIR}")
        sys.exit(1)

    exists = subprocess.run(
        ["gh", "release", "view", tag, "--repo", REPO],
        capture_output=True,
    ).returncode == 0

    if not exists:
        run(["gh", "release", "create", tag, "--repo", REPO, "--title", title,
             "--notes", f"Rolling wheel set for CUDA {cuda_major}.{cuda_minor} / {arch_str}."])
        seed = _latest_versioned_release(tag)
        if seed:
            print(f"Seeding {tag} from legacy release {seed}")
            seed_dir = tempfile.mkdtemp(prefix="seed-wheels-")
            run(["gh", "release", "download", seed, "--repo", REPO, "--dir", seed_dir])
            for f in sorted(os.listdir(seed_dir)):
                run(["gh", "release", "upload", tag, os.path.join(seed_dir, f), "--repo", REPO])
            shutil.rmtree(seed_dir)

    remote = {a["name"] for a in
              _gh_json(["release", "view", tag, "--repo", REPO, "--json", "assets"])["assets"]}

    print(f"\nSyncing {len(local)} local assets into release '{tag}'")
    for path in local:
        name = os.path.basename(path)
        if name.endswith(".whl"):
            # A version bump changes the wheel filename; drop the superseded
            # asset so the Dockerfile's <dist>-*.whl glob stays unambiguous.
            dist = name.split("-")[0]
            for stale in [r for r in remote
                          if r.endswith(".whl") and r.split("-")[0] == dist and r != name]:
                run(["gh", "release", "delete-asset", tag, stale, "--repo", REPO, "--yes"])
                remote.discard(stale)
        run(["gh", "release", "upload", tag, path, "--repo", REPO, "--clobber"])
        remote.add(name)

    names = sorted({r.split("-")[0] for r in remote if r.endswith(".whl")})
    run(["gh", "release", "edit", tag, "--repo", REPO, "--title", title,
         "--notes", "Pre-built wheels: " + ", ".join(names)])

    print(f"\nRelease synced: https://github.com/{REPO}/releases/tag/{tag}")


def main():
    parser = argparse.ArgumentParser(description="Build and upload GPU wheels.")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build", help="Build all GPU wheels into the wheel output directory")
    p_build.add_argument("--cuda", default="129", help="CUDA version, e.g. 129, 130")
    p_build.add_argument("--arch", default="x86", choices=["x86", "aarch64"], help="Architecture")
    p_build.add_argument("--only", nargs="+", help=f"Only run specific steps ({STEP_NAMES})")
    p_build.add_argument("--no-bootstrap-rust", dest="bootstrap_rust", action="store_false",
                         help="Don't auto-install Rust toolchain")
    p_build.set_defaults(func=cmd_build, bootstrap_rust=True)

    # ── upload ───────────────────────────────────────────────
    p_upload = sub.add_parser(
        "upload",
        help="Sync the wheel output directory into the rolling cu<cuda>-<arch> release",
    )
    p_upload.add_argument("--cuda", default="129", help="CUDA version, e.g. 129, 130")
    p_upload.add_argument("--arch", default="x86", choices=["x86", "aarch64"], help="Architecture")
    p_upload.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
