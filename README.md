# miles-wheels

### cu12.9 + x86_64
```shell
python build_wheels.py build --cuda 129 --arch x86
python build_wheels.py upload --cuda 129 --arch x86 --version 0.5.12
```

### cu13.0 + aarch64
```shell
python build_wheels.py build --cuda 130 --arch aarch64
python build_wheels.py upload --cuda 130 --arch aarch64 --version 0.5.12
```

`--version` is required on `upload`; the release tag becomes `cu<cuda>-<arch>-vX.Y.Z`
(e.g. `cu130-x86_64-v0.5.12`).

### test wheels
```shell
python test_wheels.py install-and-test /tmp/wheels
```