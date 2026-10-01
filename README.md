# miles-wheels

Each release is the complete, rolling wheel set for one (CUDA, arch) pair;
the tag is just `cu<cuda>-<arch>` (e.g. `cu130-aarch64`). `upload` syncs
`WHEEL_DIR` into that release in place: new packages are added, a wheel
whose version changed replaces its old asset, unchanged assets are left
alone — no full re-upload. The first `upload` to a fresh tag seeds it from
the newest legacy `cu<cuda>-<arch>-vX.Y.Z` release.

`WHEEL_DIR` defaults to `/tmp/wheels`; any override must be an absolute path.

`build --only <step> ...` writes only that step to `WHEEL_DIR`; `upload` touches only those assets.

CUDA 12.9 supports only x86_64.

The `te` step builds all three Transformer Engine wheels from
[radixark/TransformerEngine](https://github.com/radixark/TransformerEngine)
`miles-main` (override with `--te-ref`), versioned `<VERSION.txt>+miles`, and
writes `transformer_engine-source.json` recording the commit. It runs NVIDIA's
manylinux release recipe in Docker, so every target needs a Docker daemon with
host-network support, but not the NVIDIA container runtime.

### cu12.9 + x86_64
```shell
python build_wheels.py build --cuda 129 --arch x86
python build_wheels.py upload --cuda 129 --arch x86
```

### cu13.0 + aarch64

```shell
python build_wheels.py build --cuda 130 --arch aarch64
python build_wheels.py upload --cuda 130 --arch aarch64
```

### cu13.0 + x86_64 (B300, sm_103a)
```shell
python build_wheels.py build --cuda 130 --arch x86 --only flash-attn flash-attn-hopper apex
python build_wheels.py upload --cuda 130 --arch x86
```

Note:
- `te`: not needed here — install via PyPI: `pip install --no-build-isolation "transformer_engine[core_cu13,pytorch]==2.12.0"`
- `int4_qat`: not yet supported for this platform (pending PTX fix merge).

### test wheels
```shell
python test_wheels.py install-and-test "${WHEEL_DIR:-/tmp/wheels}"
```

### Update only the router

Build the router wheel and standalone binary from the same source revision, then
upload them into the existing CUDA/Torch/architecture release. Other assets stay unchanged;
router updates do not need a separate release tag.

```shell
WHEEL_DIR=/tmp/router-wheels python build_wheels.py build --cuda 130 --arch x86 --only sgl-router --router-ref <commit>
WHEEL_DIR=/tmp/router-wheels python build_wheels.py upload --cuda 130 --arch x86 --torch 213
```

Repeat on aarch64 with `--arch aarch64`.

### Update only Transformer Engine

`--te-phase all` needs Docker and the target torch in one environment. To split
them, build the sources on a host with Docker, then compile the torch extension
inside the image the wheels are for, sharing the same absolute `WHEEL_DIR`:

```shell
WHEEL_DIR=/tmp/te-wheels python build_wheels.py build --cuda 130 --arch x86 --only te --te-phase sources
docker run --rm -v /tmp/te-wheels:/tmp/te-wheels -v "$PWD":/miles-wheels -e WHEEL_DIR=/tmp/te-wheels \
  lmsysorg/sglang:v0.5.20 python3 /miles-wheels/build_wheels.py build --cuda 130 --arch x86 --only te --te-phase torch
WHEEL_DIR=/tmp/te-wheels python build_wheels.py upload --cuda 130 --arch x86 --torch 213
```

Pass `--keep-superseded` to `upload` while a consumer is still pinned to the
previous Transformer Engine version, then delete those assets once it has moved.
