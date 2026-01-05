# 基础镜像配置 vLLM 或 LMDeploy ，请根据实际需要选择其中一个，要求 ARM(AArch64) CPU + Ascend NPU。
# Base image containing the vLLM inference environment, requiring ARM(AArch64) CPU + Ascend NPU.
FROM quay.m.daocloud.io/ascend/vllm-ascend:v0.11.0
# Base image containing the LMDeploy inference environment, requiring ARM(AArch64) CPU + Ascend NPU.
# FROM crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:mineru-a2


# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libglib2.0-0 && \
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
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]