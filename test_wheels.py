#!/usr/bin/env python3
"""Install and test GPU wheels built by build_wheels.py."""

import glob
import os
import subprocess
import sys
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Install and test GPU wheels.")

STEP_NAMES = "flash-attn, flash-attn-hopper, apex, int4_qat, te"


def run(cmd, *, env=None):
    """Run a command, streaming output. Exit on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode}): {cmd}")
        raise typer.Exit(result.returncode)


def _find_wheel(wheel_dir: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(wheel_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No wheel matching {pattern!r} in {wheel_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple wheels matching {pattern!r}: {matches}")
    return matches[0]


# ── install steps ─────────────────────────────────────────────

def _install_flash_attn(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "flash_attn-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_flash_attn_hopper(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "flash_attn_3-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])

    # Install the hopper Python interface (not included in the wheel)
    python_path = subprocess.check_output(
        [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    interface_dir = os.path.join(python_path, "flash_attn_3")
    os.makedirs(interface_dir, exist_ok=True)
    run([
        "curl", "-fSL",
        "https://raw.githubusercontent.com/Dao-AILab/flash-attention/"
        "fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89/hopper/flash_attn_interface.py",
        "-o", os.path.join(interface_dir, "flash_attn_interface.py"),
    ])


def _install_apex(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "apex-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_int4_qat(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "fake_int4_quant_cuda-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_te(wheel_dir: str):
    for whl in glob.glob(os.path.join(wheel_dir, "transformer_engine*.whl")):
        run([sys.executable, "-m", "pip", "install", whl])


INSTALL_STEPS = {
    "flash-attn": _install_flash_attn,
    "flash-attn-hopper": _install_flash_attn_hopper,
    "apex": _install_apex,
    "int4_qat": _install_int4_qat,
    "te": _install_te,
}


# ── test steps ────────────────────────────────────────────────

def _test_flash_attn():
    import torch
    from flash_attn import flash_attn_func
    q = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v)
    out.sum().backward()
    print("flash-attn backward: OK")


def _test_flash_attn_hopper():
    import torch
    from flash_attn_3 import flash_attn_interface
    q = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_interface.flash_attn_func(q, k, v)
    out.sum().backward()
    print("flash-attn-hopper backward: OK")


def _test_apex():
    import torch
    from apex.optimizers import FusedAdam
    model = torch.nn.Linear(1024, 1024).cuda().to(torch.bfloat16)
    opt = FusedAdam(model.parameters())
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    model(x).sum().backward()
    opt.step()
    print("apex FusedAdam step: OK")


def _test_int4_qat():
    import fake_int4_quant_cuda  # noqa: F401
    print("int4_qat import: OK")


def _test_te():
    import transformer_engine.pytorch as te  # noqa: F401
    print("transformer_engine import: OK")


TEST_STEPS = {
    "flash-attn": _test_flash_attn,
    "flash-attn-hopper": _test_flash_attn_hopper,
    "apex": _test_apex,
    "int4_qat": _test_int4_qat,
}


# ── commands ──────────────────────────────────────────────────

@app.command()
def install(
    wheel_dir: Annotated[str, typer.Argument(help="Directory containing .whl files")] = "/tmp/wheels",
    only: Annotated[Optional[list[str]], typer.Option(
        help=f"Only install specific wheels ({STEP_NAMES})",
    )] = None,
):
    """Install all wheels from WHEEL_DIR."""
    selected = {s.lower() for s in (only or [])}
    for name, fn in INSTALL_STEPS.items():
        if selected and name not in selected:
            print(f"\nSkipping {name}")
            continue
        print(f"\n>>> Installing {name} ...")
        try:
            fn(wheel_dir)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping")
    print("\nInstall done.")


@app.command()
def test(
    only: Annotated[Optional[list[str]], typer.Option(
        help=f"Only test specific wheels ({STEP_NAMES})",
    )] = None,
):
    """Test installed wheels (forward + backward pass). Each step runs in an isolated subprocess."""
    selected = {s.lower() for s in (only or [])}
    passed, failed, skipped = [], [], []

    for name in TEST_STEPS:
        if selected and name not in selected:
            skipped.append(name)
            continue
        print(f"\n>>> Testing {name} ...")
        result = subprocess.run([sys.executable, __file__, "--run-step", name])
        if result.returncode == 0:
            passed.append(name)
        else:
            print(f"FAILED (exit code {result.returncode})")
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if passed:
        print(f"  PASSED : {', '.join(passed)}")
    if failed:
        print(f"  FAILED : {', '.join(failed)}")
    if skipped:
        print(f"  SKIPPED: {', '.join(skipped)}")
    print(f"{'='*60}")

    if failed:
        raise typer.Exit(1)


@app.command()
def install_and_test(
    wheel_dir: Annotated[str, typer.Argument(help="Directory containing .whl files")] = "/tmp/wheels",
    only: Annotated[Optional[list[str]], typer.Option(
        help=f"Only run specific steps ({STEP_NAMES})",
    )] = None,
):
    """Install all wheels from WHEEL_DIR, then test them."""
    install(wheel_dir=wheel_dir, only=only)
    test(only=only)


if __name__ == "__main__":
    # Internal subprocess dispatch: python test_wheels.py --run-step <name>
    if len(sys.argv) == 3 and sys.argv[1] == "--run-step":
        import torch
        step = sys.argv[2]
        print(f"GPU: {torch.cuda.get_device_name(0)}, SM: {torch.cuda.get_device_capability()}")
        TEST_STEPS[step]()
        sys.exit(0)

    app()
