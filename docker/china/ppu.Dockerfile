# 基础镜像配置 vLLM 或 LMDeploy 推理环境，请根据实际需要选择其中一个，要求 amd64(x86-64) CPU + t-head PPU。
# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + t-head PPU.
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/ppu:ppu-pytorch2.6.0-ubuntu24.04-cuda12.6-vllm0.8.5-py312
# Base image containing the LMDeploy inference environment, requiring amd64(x86-64) CPU + t-head PPU.
# FROM crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ppu:mineru-ppu

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
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install 'mineru[core]>=2.6.5' \
                            numpy==1.26.4 \
                            opencv-python==4.11.0.86 \
                            -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]