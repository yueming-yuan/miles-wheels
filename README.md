# miles-wheels

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
python test_wheels.py install-and-test /tmp/wheels
```