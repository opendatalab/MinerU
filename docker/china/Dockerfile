# Use DaoCloud mirrored vllm image for China region for gpu with Volta、Turing、Ampere、Ada Lovelace、Hopper、Blackwell architecture (7.0 <= Compute Capability <= 12.1)
# The default base image uses vLLM 0.21.0 with CUDA 13.0. For CUDA 12.9 environments, switch to the commented cu129 image below.
# Compute Capability version query (https://developer.nvidia.com/cuda-gpus)
# support x86_64 architecture and ARM(AArch64) architecture
FROM docker.m.daocloud.io/vllm/vllm-openai:v0.21.0
# FROM docker.m.daocloud.io/vllm/vllm-openai:v0.21.0-cu129

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mineru latest
RUN python3 -m pip install -U 'mineru[core]>=3.4.0' -i https://mirrors.aliyun.com/pypi/simple --break-system-packages && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
