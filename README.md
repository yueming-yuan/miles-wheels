# miles-wheels

Each release is the complete, rolling wheel set for one (CUDA, arch) pair;
the tag is just `cu<cuda>-<arch>` (e.g. `cu130-aarch64`). `upload` syncs
`WHEEL_DIR` into that release in place: new packages are added, a wheel
whose version changed replaces its old asset, unchanged assets are left
alone — no full re-upload. The first `upload` to a fresh tag seeds it from
the newest legacy `cu<cuda>-<arch>-vX.Y.Z` release.

`WHEEL_DIR` defaults to `/tmp/wheels`; set it to an absolute path on another disk.

`build --only <step> ...` writes only that step to `WHEEL_DIR`; `upload` touches only those assets.

### cu12.9 + x86_64
```shell
python build_wheels.py build --cuda 129 --arch x86
python build_wheels.py upload --cuda 129 --arch x86
```

### cu13.0 + aarch64

The aarch64 core build runs NVIDIA's pinned manylinux recipe in Docker.

It needs a Docker daemon but not the NVIDIA container runtime.

```shell
python build_wheels.py build --cuda 130 --arch aarch64
python build_wheels.py upload --cuda 130 --arch aarch64
```

### test wheels
```shell
python test_wheels.py install-and-test "${WHEEL_DIR:-/tmp/wheels}"
```
