# 基础镜像配置 vLLM 或 LMDeploy ，请根据实际需要选择其中一个，要求 amd64(x86-64) CPU + Cambricon MLU.
# Base image containing the LMDEPLOY inference environment, requiring amd64(x86-64) CPU + Cambricon MLU.
FROM crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/camb:qwen_vl2.5
ARG BACKEND=lmdeploy
# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + Cambricon MLU.
# FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/mlu:vllm0.8.3-torch2.6.0-torchmlu1.26.1-ubuntu22.04-py310
# ARG BACKEND=vllm

# Install Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mineru latest
RUN /bin/bash -c '\
    if [ "$BACKEND" = "vllm" ]; then \
        source /torch/venv3/pytorch_infer/bin/activate; \
    fi && \
    python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install "mineru[core]>=2.7.4" \
                            numpy==1.26.4 \
                            opencv-python==4.11.0.86 \
                            -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install $(if [ "$BACKEND" = "lmdeploy" ]; then echo "accelerate==1.2.0"; else echo "transformers==4.50.3"; fi) && \
    python3 -m pip cache purge'

# Download models and update the configuration file
RUN /bin/bash -c '\
    if [ "$BACKEND" = "vllm" ]; then \
        source /torch/venv3/pytorch_infer/bin/activate; \
    fi && \
    mineru-models-download -s modelscope -m all'

WORKDIR /workspace

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]