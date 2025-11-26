# 基础镜像配置 vLLM 或 LMDeploy 推理环境，请根据实际需要选择其中一个，要求 amd64(x86-64) CPU + metax GPU。
# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + metax GPU.
FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64
# Base image containing the LMDeploy inference environment, requiring amd64(x86-64) CPU + metax GPU.
# FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/maca:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-lmdeploy0.10.2-amd64

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

# mod torchvision to be compatible with torch 2.6
RUN sed -i '3s/^Version: 0.15.1+metax3\.1\.0\.4$/Version: 0.21.0+metax3.1.0.4/' /opt/conda/lib/python3.10/site-packages/torchvision-0.15.1+metax3.1.0.4.dist-info/METADATA && \
    mv /opt/conda/lib/python3.10/site-packages/torchvision-0.15.1+metax3.1.0.4.dist-info /opt/conda/lib/python3.10/site-packages/torchvision-0.21.0+metax3.1.0.4.dist-info

# Install mineru latest
RUN /opt/conda/bin/python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    /opt/conda/bin/python3 -m pip install 'mineru[core]>=2.6.5' \
                                           numpy==1.26.4 \
                                           opencv-python==4.11.0.86 \
                                           -i https://mirrors.aliyun.com/pypi/simple && \
    /opt/conda/bin/python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "/opt/conda/bin/mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]