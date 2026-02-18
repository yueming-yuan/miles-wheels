#!/usr/bin/env python3
"""Build GPU wheels (flash-attn, apex, transformer_engine, etc.)."""

import glob
import os
import shutil
import subprocess
import sys
from enum import Enum
from typing import Annotated, Optional

import typer

WHEEL_DIR = "/tmp/wheels"
REPO = "yueming-yuan/miles-wheels"

app = typer.Typer(help="Build and upload GPU wheels.")


class Arch(str, Enum):
    x86 = "x86"
    aarch64 = "aarch64"


def run(cmd, *, env=None, cwd=None):
    """Run a command, streaming output. Exit on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode}): {cmd}")
        raise typer.Exit(result.returncode)


def _setup_env(cuda: str, arch: Arch):
    cuda_major, cuda_minor = cuda[:2], cuda[2:]
    print(f"CUDA  : {cuda_major}.{cuda_minor}  (cu{cuda})")
    print(f"Arch  : {arch.value}")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        "8.0;8.6;8.9;9.0;10.0;10.3" if arch == Arch.x86 else "9.0;10.0;10.3",
    )
    os.environ["CUDA_VERSION"] = f"{cuda_major}.{cuda_minor}"
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")


# ── build steps ──────────────────────────────────────────────

def _build_flash_attn(cuda: str):
    run(
        [sys.executable, "-m", "pip", "wheel",
         "flash-attn==2.7.4.post1",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"MAX_JOBS": "64"},
    )


def _build_flash_attn_hopper(cuda: str):
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


def _build_apex(cuda: str):
    run(
        [sys.executable, "-m", "pip", "wheel",
         "-v", "--no-build-isolation", "--no-deps",
         "--config-settings", "--build-option=--cpp_ext --cuda_ext --parallel 8",
         "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4",
         "-w", WHEEL_DIR],
        env={"NVCC_APPEND_FLAGS": "--threads 4"},
    )


def _build_int4_qat(cuda: str):
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


def _build_transformer_engine(cuda: str):
    cuda_major = cuda[:2]
    extras = "core_cu13,pytorch" if cuda_major >= "13" else "pytorch"
    run(
        [sys.executable, "-m", "pip", "wheel",
         f"transformer_engine[{extras}]==2.10.0",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
    )


STEPS = {
    "flash-attn": _build_flash_attn,
    "flash-attn-hopper": _build_flash_attn_hopper,
    "apex": _build_apex,
    "int4_qat": _build_int4_qat,
    "te": _build_transformer_engine,
}

STEP_NAMES = ", ".join(STEPS)


# ── commands ─────────────────────────────────────────────────

@app.command()
def build(
    cuda: Annotated[str, typer.Option(help="CUDA version, e.g. 129, 130")] = "129",
    arch: Annotated[Arch, typer.Option(help="Architecture")] = Arch.x86,
    only: Annotated[Optional[list[str]], typer.Option(
        help=f"Only run specific steps ({STEP_NAMES})",
    )] = None,
):
    """Build all GPU wheels into /tmp/wheels."""
    assert cuda in ("129", "130"), "currently only cu129 and cu130 are supported"
    _setup_env(cuda, arch)
    os.makedirs(WHEEL_DIR, exist_ok=True)

    selected = {s.lower() for s in (only or [])}
    for name, fn in STEPS.items():
        if selected and name not in selected:
            print(f"\nSkipping {name}")
            continue
        print(f"\n>>> Building {name} ...")
        fn(cuda)

    print(f"\nDone. Wheels in {WHEEL_DIR}:")
    run(["ls", "-lh", WHEEL_DIR])


@app.command()
def upload(
    cuda: Annotated[str, typer.Option(help="CUDA version, e.g. 129, 130")] = "129",
    arch: Annotated[Arch, typer.Option(help="Architecture")] = Arch.x86,
):
    """Upload all wheels in /tmp/wheels as a GitHub release."""
    assert cuda in ("129", "130"), "currently only cu129 and cu130 are supported"

    cuda_major, cuda_minor = cuda[:2], cuda[2:]
    arch_str = "x86_64" if arch == Arch.x86 else arch.value
    tag = f"cu{cuda}-{arch_str}"
    title = f"CUDA {cuda_major}.{cuda_minor} + {arch_str}"

    wheels = sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl")))
    if not wheels:
        print(f"No .whl files found in {WHEEL_DIR}")
        raise typer.Exit(1)

    names = [os.path.splitext(os.path.basename(w))[0].split("-")[0] for w in wheels]
    body = "Pre-built wheels: " + ", ".join(names)

    print(f"\nUploading {len(wheels)} wheels as release '{tag}'")
    for w in wheels:
        print(f"  {os.path.basename(w)}")

    # Delete existing release with same tag (if any)
    subprocess.run(
        ["gh", "release", "delete", tag, "--repo", REPO, "--yes", "--cleanup-tag"],
        capture_output=True,
    )

    run(["gh", "release", "create", tag,
         "--repo", REPO,
         "--title", title,
         "--notes", body,
         *wheels])

    print(f"\nRelease created: https://github.com/{REPO}/releases/tag/{tag}")


if __name__ == "__main__":
    app()
