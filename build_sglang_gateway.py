#!/usr/bin/env python3
"""Build sgl-model-gateway Python wheel and standalone binary from source.

This module is self-contained: it handles Rust toolchain setup, source checkout,
source patching, and compilation. It is called by build_wheels.py but can also
be used standalone for debugging.

Usage (standalone):
    python build_sglang_gateway.py --out /tmp/wheels
    python build_sglang_gateway.py --repo https://github.com/sgl-project/sglang.git --ref main
"""

import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass

SGLANG_REPO_DEFAULT = "https://github.com/sgl-project/sglang.git"
SGLANG_REF_DEFAULT = "main"


@dataclass
class BuildConfig:
    repo: str = SGLANG_REPO_DEFAULT
    ref: str = SGLANG_REF_DEFAULT
    bootstrap_rust: bool = True


# ── helpers ─────────────────────────────────────────────────


def _run(cmd, *, env=None, cwd=None):
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _prepend_user_bin_paths():
    path_entries = os.environ.get("PATH", "").split(":")
    for bin_dir in (os.path.expanduser("~/.cargo/bin"), os.path.expanduser("~/.local/bin")):
        if os.path.isdir(bin_dir) and bin_dir not in path_entries:
            path_entries.insert(0, bin_dir)
    os.environ["PATH"] = ":".join(path_entries)


def _patch_file(path: str, pattern: str, replacement: str, *, flags: int = 0):
    """Regex-replace in a file. Warns if pattern doesn't match (upstream may have fixed it)."""
    text = open(path).read()
    new_text, n = re.subn(pattern, replacement, text, flags=flags)
    if n == 0:
        print(f"WARNING: patch pattern not found in {path} — already fixed upstream?")
    else:
        open(path, "w").write(new_text)
        print(f"Patched {path} ({n} replacement(s))")


# ── prerequisites ───────────────────────────────────────────


def _ensure_rust_and_maturin(bootstrap_rust: bool):
    _prepend_user_bin_paths()
    if not (_command_exists("rustc") and _command_exists("cargo")):
        if not bootstrap_rust:
            raise RuntimeError(
                "Rust toolchain is required (missing rustc/cargo). "
                "Re-run with bootstrap_rust=True."
            )
        _run([
            "bash", "-lc",
            "curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf "
            "https://sh.rustup.rs | sh -s -- -y",
        ])
        _prepend_user_bin_paths()

    if not (_command_exists("rustc") and _command_exists("cargo")):
        raise RuntimeError("Rust installation finished but rustc/cargo are still unavailable.")

    _run(["rustc", "--version"])
    _run(["cargo", "--version"])

    if not _command_exists("maturin"):
        _run([sys.executable, "-m", "pip", "install", "maturin"])
        _prepend_user_bin_paths()
    _run([sys.executable, "-m", "maturin", "--version"])


def _ensure_protoc():
    if not _command_exists("protoc"):
        raise RuntimeError(
            "protoc is required for sgl-model-gateway build. "
            "Install protobuf-compiler or set PROTOC to your protoc binary."
        )
    _run(["protoc", "--version"])


def _checkout_git_ref(repo: str, ref: str, dest: str):
    _run(["git", "clone", "--depth=1", repo, dest])
    _run(["git", "fetch", "--depth=1", "origin", ref], cwd=dest)
    _run(["git", "checkout", "-f", "FETCH_HEAD"], cwd=dest)
    _run(["git", "log", "--oneline", "-1"], cwd=dest)


# ── patches ─────────────────────────────────────────────────


def _patch_sources(gw_dir: str):
    """Apply source patches to fix Rust compilation errors in sgl-model-gateway.

    These patches work around type mismatches between sgl-model-gateway and its
    dependency smg-wasm (<1.0.1). The gateway code assumes interfaces that were
    changed in smg-wasm, causing compilation failures. We patch the gateway side
    rather than pinning smg-wasm because we don't want to fork or modify the
    upstream sglang repo — just build it as-is with minimal fixups.

    If upstream fixes these issues, _patch_file will log a warning ("pattern not
    found — already fixed upstream?") and continue without error.
    """
    # Patch 1: app_context.rs — error type mismatch in WasmModuleManager init.
    #
    # WasmModuleManager::new() returns Result<_, E> where E is not String.
    # The gateway code uses `format!(...)` in map_err, which produces a String,
    # but the compiler expects E (or something convertible via From<String>).
    # Fix: use .to_string() which satisfies the Display-based conversion.
    _patch_file(
        os.path.join(gw_dir, "src", "app_context.rs"),
        r'\.map_err\(\|e\| format!\("Failed to initialize WASM module manager: \{\}", e\)\)',
        '.map_err(|e| e.to_string())',
        flags=re.DOTALL,
    )
    # Patch 2: wasm_module_registration.rs — field type mismatch for wasm_bytes.
    #
    # The struct field expects a different type (e.g. Bytes) than the local
    # variable (e.g. Vec<u8>). Rust's field shorthand `wasm_bytes,` only works
    # when types match exactly. Fix: use explicit `.into()` conversion.
    _patch_file(
        os.path.join(gw_dir, "src", "core", "steps", "wasm_module_registration.rs"),
        r'^(\s*)wasm_bytes,$',
        r'\1wasm_bytes: wasm_bytes.into(),',
        flags=re.MULTILINE,
    )


# ── main build ──────────────────────────────────────────────


def build(cfg: BuildConfig, out_dir: str):
    """Build sgl-model-gateway wheel + binary, writing artifacts to *out_dir*."""
    repo_dir = "/tmp/sglang"
    gw_dir = os.path.join(repo_dir, "sgl-model-gateway")

    _ensure_rust_and_maturin(cfg.bootstrap_rust)
    _ensure_protoc()

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    os.makedirs(out_dir, exist_ok=True)

    try:
        _checkout_git_ref(cfg.repo, cfg.ref, repo_dir)
        _patch_sources(gw_dir)

        # Raise open-file limit for Rust parallel compilation
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))

        # Build Python wheel
        _run(
            [sys.executable, "-m", "maturin", "build", "--release",
             "--features", "vendored-openssl", "--out", out_dir],
            cwd=os.path.join(gw_dir, "bindings", "python"),
        )

        # Build standalone binary
        _run(
            ["cargo", "build", "--release", "--bin", "sgl-model-gateway",
             "--features", "vendored-openssl"],
            cwd=gw_dir,
        )

        # Package binary as tarball
        binary = os.path.join(gw_dir, "target", "release", "sgl-model-gateway")
        tarball = os.path.join(out_dir, f"sgl-model-gateway-linux-{platform.machine()}.tar.gz")
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(binary, arcname="sgl-model-gateway")
        print(f"Packaged binary: {tarball}")
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)


# ── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/tmp/wheels", help="Output directory for artifacts")
    p.add_argument("--repo", default=SGLANG_REPO_DEFAULT, help="sglang git repository")
    p.add_argument("--ref", default=SGLANG_REF_DEFAULT, help="sglang git ref (branch/tag/commit)")
    p.add_argument("--no-bootstrap-rust", action="store_true", help="Don't auto-install Rust")
    args = p.parse_args()

    cfg = BuildConfig(repo=args.repo, ref=args.ref, bootstrap_rust=not args.no_bootstrap_rust)
    build(cfg, args.out)


if __name__ == "__main__":
    main()
