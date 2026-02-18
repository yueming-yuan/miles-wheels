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