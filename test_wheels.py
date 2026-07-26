#!/usr/bin/env python3
"""Install and test GPU wheels built by build_wheels.py."""

import glob
import importlib.metadata
import os
import subprocess
import sys
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Install and test GPU wheels.")

STEP_NAMES = ("flash-attn, flash-attn-hopper, apex, int4_qat, te, "
              "causal-conv1d, mamba-ssm, fast-hadamard, sgl-model-gateway, mooncake")


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

    # The wheel now ships both flash_attn_interface (top-level) and
    # flash_attn_3.flash_attn_interface (a re-export shim), so nothing extra to
    # install. Do not drop a copy of the full module at the shim's path: it
    # re-runs @torch.library.custom_op("flash_attn_3::...") and double-registers.


def _install_apex(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "apex-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_int4_qat(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "fake_int4_quant_cuda-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_te(wheel_dir: str):
    patterns = (
        "transformer_engine-[0-9]*.whl",
        "transformer_engine_cu1[23]-*.whl",
        "transformer_engine_torch-*.whl",
    )
    matches = {
        pattern: glob.glob(os.path.join(wheel_dir, pattern))
        for pattern in patterns
    }
    invalid = {pattern: found for pattern, found in matches.items() if len(found) != 1}
    if invalid:
        raise RuntimeError(
            f"Expected exactly one wheel for each Transformer Engine dist: {invalid}"
        )
    wheels = [matches[pattern][0] for pattern in patterns]
    versions = {os.path.basename(whl).split("-")[1] for whl in wheels}
    if len(versions) != 1:
        raise RuntimeError(f"Transformer Engine wheel versions do not match: {wheels}")
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", *wheels,
    ])


def _install_causal_conv1d(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "causal_conv1d-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_mamba_ssm(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "mamba_ssm-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_fast_hadamard(wheel_dir: str):
    whl = _find_wheel(wheel_dir, "fast_hadamard_transform-*.whl")
    run([sys.executable, "-m", "pip", "install", whl])


def _install_sgl_model_gateway(wheel_dir: str):
    # Install the Python wheel (package name: sglang-router, wheel: sglang_router-*.whl)
    whl = _find_wheel(wheel_dir, "sglang_router-*.whl")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", whl])

    # Extract and install the binary from tarball
    import platform
    import tarfile

    machine = platform.machine()
    tarball = os.path.join(wheel_dir, f"sgl-model-gateway-linux-{machine}.tar.gz")
    if not os.path.exists(tarball):
        raise FileNotFoundError(f"Binary tarball not found: {tarball}")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extract("sgl-model-gateway", path="/usr/local/bin", filter="data")
    os.chmod("/usr/local/bin/sgl-model-gateway", 0o755)
    print("Installed sgl-model-gateway binary to /usr/local/bin/")


def _install_mooncake(wheel_dir: str):
    # Same dist name as the base image's mooncake; force-reinstall to replace it.
    whl = _find_wheel(wheel_dir, "mooncake_transfer_engine*-*.whl")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", whl])


INSTALL_STEPS = {
    "flash-attn": _install_flash_attn,
    "flash-attn-hopper": _install_flash_attn_hopper,
    "apex": _install_apex,
    "int4_qat": _install_int4_qat,
    "te": _install_te,
    "causal-conv1d": _install_causal_conv1d,
    "mamba-ssm": _install_mamba_ssm,
    "fast-hadamard": _install_fast_hadamard,
    "sgl-model-gateway": _install_sgl_model_gateway,
    "mooncake": _install_mooncake,
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
    import torch
    import transformer_engine.pytorch as te

    cuda_major = torch.version.cuda.split(".", maxsplit=1)[0]
    packages = (
        "transformer-engine",
        f"transformer-engine-cu{cuda_major}",
        "transformer-engine-torch",
    )
    versions = {package: importlib.metadata.version(package) for package in packages}
    assert set(versions.values()) == {"2.17.0"}, (
        f"Expected Transformer Engine 2.17.0, found {versions}"
    )

    model = te.Linear(64, 64, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    print(f"transformer_engine {next(iter(versions.values()))} Linear forward+backward: OK")


def _test_causal_conv1d():
    import torch
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = causal_conv1d_fn(x, w)
    out.sum().backward()
    print("causal-conv1d backward: OK")


def _test_mamba_ssm():
    import torch
    from mamba_ssm import Mamba
    model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda().to(torch.bfloat16)
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    out.sum().backward()
    print("mamba-ssm Mamba forward+backward: OK")


def _test_fast_hadamard():
    import torch
    from fast_hadamard_transform import hadamard_transform
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = hadamard_transform(x)
    out.sum().backward()
    print("fast-hadamard-transform backward: OK")


def _test_sgl_model_gateway():
    import sglang_router  # noqa: F401
    print("sglang_router import: OK")
    import subprocess as _sp
    result = _sp.run(["sgl-model-gateway", "--help"], capture_output=True)
    assert result.returncode == 0, "sgl-model-gateway binary --help failed"
    print("sgl-model-gateway binary: OK")


def _test_mooncake():
    import shutil

    from mooncake.store import MooncakeDistributedStore  # noqa: F401
    from mooncake.structured_object_store import (  # noqa: F401
        FieldSchema,
        MooncakeBundleTransfer,
        export_ref,
        import_ref,
    )
    for method in ("put", "get", "release_result", "cleanup_dataproto"):
        assert hasattr(MooncakeBundleTransfer, method), f"MooncakeBundleTransfer.{method} missing"
    assert shutil.which("mooncake_master"), "mooncake_master binary not on PATH"
    print("mooncake structured object store API + master binary: OK")


TEST_STEPS = {
    "flash-attn": _test_flash_attn,
    "flash-attn-hopper": _test_flash_attn_hopper,
    "apex": _test_apex,
    "int4_qat": _test_int4_qat,
    "te": _test_te,
    "causal-conv1d": _test_causal_conv1d,
    "mamba-ssm": _test_mamba_ssm,
    "fast-hadamard": _test_fast_hadamard,
    "sgl-model-gateway": _test_sgl_model_gateway,
    "mooncake": _test_mooncake,
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
