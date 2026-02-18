#!/usr/bin/env python3
"""Build GPU wheels (flash-attn, apex, transformer_engine, etc.)."""

import argparse
import os
import shutil
import subprocess
import sys


WHEEL_DIR = "/tmp/wheels"


def run(cmd, *, env=None, cwd=None, shell=False):
    """Run a command, streaming output. Exit on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}")
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd, env=merged_env, cwd=cwd, shell=shell,
    )
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def build_flash_attn(args):
    """1. flash-attn"""
    run(
        [sys.executable, "-m", "pip", "wheel",
         "flash-attn==2.7.4.post1",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
        env={"MAX_JOBS": "64"},
    )


def build_flash_attn_hopper(args):
    """2. flash-attn hopper (build from source)"""
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

    # copy wheels
    hopper_dist = os.path.join(repo_dir, "hopper", "dist")
    for f in os.listdir(hopper_dist):
        if f.endswith(".whl"):
            shutil.copy2(os.path.join(hopper_dist, f), WHEEL_DIR)

    shutil.rmtree(repo_dir)


def build_apex(args):
    """3. apex"""
    run(
        [sys.executable, "-m", "pip", "wheel",
         "-v", "--no-build-isolation", "--no-deps",
         "--config-settings", "--build-option=--cpp_ext --cuda_ext --parallel 8",
         "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4",
         "-w", WHEEL_DIR],
        env={"NVCC_APPEND_FLAGS": "--threads 4"},
    )


def build_int4_qat(args):
    """4. int4_qat (needs miles source)"""
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


def build_transformer_engine(args):
    """5. transformer_engine"""
    cuda_major = args.cuda[:2]
    extras = "core_cu13,pytorch" if cuda_major >= "13" else "pytorch"
    run(
        [sys.executable, "-m", "pip", "wheel",
         f"transformer_engine[{extras}]==2.10.0",
         "-v", "--no-build-isolation", "--no-deps",
         "-w", WHEEL_DIR],
    )


STEPS = [
    ("flash-attn", build_flash_attn),
    ("flash-attn-hopper", build_flash_attn_hopper),
    ("apex", build_apex),
    ("int4_qat", build_int4_qat),
]


def main():
    parser = argparse.ArgumentParser(description="Build GPU wheels")
    parser.add_argument("--cuda", default="129",
                        help="CUDA version")
    parser.add_argument("--arch", default="x86",
                        help="Architecture: x86 or aarch64 (default: x86)")
    parser.add_argument("--only", nargs="*", metavar="STEP",
                        help=f"Only run these steps: {', '.join(n for n, _ in STEPS)}")
    args = parser.parse_args()

    assert args.cuda in ["129", "130"], "currently only cu129 and cu130 are supported"

    cuda_major = args.cuda[:2]   # e.g. "12"
    cuda_minor = args.cuda[2:]   # e.g. "9"
    print(f"CUDA  : {cuda_major}.{cuda_minor}  (cu{args.cuda})")
    print(f"Arch  : {args.arch}")

    # Set CUDA-related env vars
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0"
                          if args.arch == "x86" else "9.0")
    os.environ["CUDA_VERSION"] = f"{cuda_major}.{cuda_minor}"

    os.makedirs(WHEEL_DIR, exist_ok=True)

    selected = {s.lower() for s in (args.only or [])}
    for name, fn in STEPS:
        if selected and name not in selected:
            print(f"\nSkipping {name}")
            continue
        print(f"\n>>> Building {name} ...")
        fn(args)

    print(f"\nDone. Wheels in {WHEEL_DIR}:")
    run(["ls", "-lh", WHEEL_DIR])


if __name__ == "__main__":
    main()
