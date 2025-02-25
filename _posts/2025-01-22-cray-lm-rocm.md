# Building ROCm Containers for ScalarLM: A Comprehensive Guide

ROCm (Radeon Open Compute) provides an open-source software foundation for GPU computing on AMD hardware. In this guide, we'll walk through the process of building ROCm-enabled containers for ScalarLM, enabling you to leverage AMD GPUs for large language model training and inference.

## Prerequisites

Before we begin, ensure you have:

- Docker installed on your system
- An AMD GPU that supports ROCm (check the [ROCm Hardware Compatibility List](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html))
- ROCm drivers installed on your host system
- Access to the ScalarLM repository

## Base Container Configuration

First, let's create a Dockerfile that sets up the ROCm environment. Create a new file called `Dockerfile.rocm`:

```dockerfile
# Start with the ROCm vLLM base image
FROM rocm/vllm-dev:20250124

# Setup the working directory
ENV PATH="/opt/conda/envs/py_3.10/bin:$PATH"
ENV CONDA_PREFIX=/opt/conda/envs/py_3.10

ARG MAX_JOBS=4

# Install additional dependencies
RUN pip install uv

```

## Building the ScalarLM vLLM Components

Next, we'll build the vLLM components for ScalarLM. Add these sections to your Dockerfile:

```dockerfile

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
COPY ./test/requirements-pytest.txt ${INSTALL_ROOT}/requirements-pytest.txt
COPY ./infra/requirements-vllm-build.txt ${INSTALL_ROOT}/requirements-vllm-build.txt

RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-build.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-pytest.txt

WORKDIR ${INSTALL_ROOT}

COPY ./infra/cray_infra/vllm ${INSTALL_ROOT}/infra/cray_infra/vllm
COPY ./infra/setup.py ${INSTALL_ROOT}/infra/cray_infra/setup.py

COPY ./infra/CMakeLists.txt ${INSTALL_ROOT}/infra/cray_infra/CMakeLists.txt
COPY ./infra/cmake ${INSTALL_ROOT}/infra/cray_infra/cmake
COPY ./infra/csrc ${INSTALL_ROOT}/infra/cray_infra/csrc

COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/infra/cray_infra/requirements.txt

WORKDIR ${INSTALL_ROOT}/infra/cray_infra

ARG VLLM_TARGET_DEVICE=rocm
ARG TORCH_CUDA_ARCH_LIST=gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100

# Build vllm python package
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} \
    python ${INSTALL_ROOT}/infra/cray_infra/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/infra/cray_infra/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/infra/cray_infra/dist

WORKDIR ${INSTALL_ROOT}

```

## ScalarLM Components and Configuration

The final section includes craylm components and adds SLURM support for distributed training:

```dockerfile
RUN apt-get update -y  \
    && apt-get install -y slurm-wlm libslurm-dev \
    build-essential \
    less curl wget net-tools vim iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Build SLURM
COPY ./infra/slurm_src ${INSTALL_ROOT}/infra/slurm_src
RUN /app/cray/infra/slurm_src/compile.sh

# Copy slurm config templates
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"

ENV SLURM_CONF=${INSTALL_ROOT}/infra/slurm_configs/slurm.conf

RUN mkdir -p ${INSTALL_ROOT}/jobs

COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./cray ${INSTALL_ROOT}/cray
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts
```

## Building the Container

You can build the container using Docker:

```bash
docker build -f Dockerfile.rocm \
    --build-arg VLLM_TARGET_DEVICE=rocm \
    -t cray-rocm:latest .
```

## Running ScalarLM with ROCm

To start a development server on a system with AMD GPUs:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -p 8000:8000 \
    --entrypoint /app/cray/scripts/start_one_server.sh \
    gdiamos/cray-rocm:latest
```

The key differences from the CPU or NVIDIA containers are:

1. The `--device=/dev/kfd --device=/dev/dri` flags expose the AMD GPU devices
2. Adding the container to the `video` group for GPU access
3. Setting `seccomp=unconfined` for ROCm compatibility

## Performance Considerations

When running ScalarLM on AMD GPUs, consider these optimization tips:

1. Ensure you're using the latest ROCm drivers for optimal performance
2. Set appropriate memory limits based on your GPU's VRAM
3. Use hip-specific environmental variables for fine-tuning:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3  # Specify which GPUs to use
```

### vLLM engine performance settings
vLLM provides a number of engine options which can be changed to improve performance. Refer to the vLLM Engine Args documentation for the complete list of vLLM engine options.

Below is a list of a few of the key vLLM engine arguments for performance; these can be passed to the vLLM benchmark scripts:

--max-model-len : Maximum context length supported by the model instance. Can be set to a lower value than model configuration value to improve performance and gpu memory utilization.
--max-num-batched-tokens : The maximum prefill size, i.e., how many prompt tokens can be packed together in a single prefill. Set to a higher value to improve prefill performance at the cost of higher gpu memory utilization. 65536 works well for LLama models.
--max-num-seqs : The maximum decode batch size (default 256). Using larger values will allow more prompts to be processed concurrently, resulting in increased throughput (possibly at the expense of higher latency). If the value is too large, there may not be enough GPU memory for the KV cache, resulting in requests getting preempted. The optimal value will depend on the GPU memory, model size, and maximum context length.
--max-seq-len-to-capture : Maximum sequence length for which Hip-graphs are captured and utilized. It's recommended to use Hip-graphs for the best decode performance. The default value of this parameter is 8K, which is lower than the large context lengths supported by recent models such as LLama. Set this parameter to max-model-len or maximum context length supported by the model for best performance.
--gpu-memory-utilization : The ratio of GPU memory reserved by a vLLM instance. Default value is 0.9. Increasing the value (potentially as high as 0.99) will increase the amount of memory available for KV cache. When running in graph mode (i.e. not using --enforce-eager), it may be necessary to use a slightly smaller value of 0.92 - 0.95 to ensure adequate memory is available for the HIP graph.

## Troubleshooting

Common issues and solutions:

1. If you encounter "GPU device not found" errors, verify that:
   - ROCm is properly installed on the host
   - The container has access to GPU devices
   - The user has proper permissions

2. For memory-related issues:
   - Check GPU memory usage with `rocm-smi`
   - Adjust batch sizes and model configurations accordingly
   - Monitor system memory usage alongside GPU memory

## Conclusion

Building ROCm containers for ScalarLM enables efficient use of AMD GPUs for machine learning workloads. By following this guide, you can create and deploy containers that leverage the full potential of AMD hardware for both training and inference tasks.

For more information about ScalarLM and its capabilities, visit our [documentation](https://docs.scalarlm.com) or join our community on [GitHub](https://github.com/scalarlm/scalarlm).

Remember to check for updates and new releases of both ROCm and ScalarLM to ensure you're using the latest features and optimizations.
