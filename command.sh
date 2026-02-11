mkdir -p /tmp/wheels

# 1. flash-attn
MAX_JOBS=64 pip wheel flash-attn==2.7.4.post1 --no-build-isolation --no-deps -w /tmp/wheels/

# 2. flash-attn hopper (no standard wheel — build manually)
git clone https://github.com/Dao-AILab/flash-attention.git && \
  cd flash-attention && git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && \
  git submodule update --init && cd hopper && \
  MAX_JOBS=96 python setup.py bdist_wheel && \
  cp dist/*.whl /tmp/wheels/ && \
  cd /root && rm -rf flash-attention

# 3. apex
NVCC_APPEND_FLAGS="--threads 4" pip wheel \
  --no-build-isolation --no-deps \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 \
  -w /tmp/wheels/

# 4. int4_qat (need miles source)
git clone https://github.com/radixark/miles.git /tmp/miles && \
  cd /tmp/miles/miles/backends/megatron_utils/kernels/int4_qat && \
  pip wheel . --no-build-isolation --no-deps -w /tmp/wheels/

ls -lh /tmp/wheels/
