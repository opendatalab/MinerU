# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + Kunlun XPU.
FROM docker.1ms.run/wjie520/vllm_kunlun:v0.10.1.1rc1


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
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install "mineru[api,gradio]>=2.7.6" \
                            "matplotlib>=3.10,<4" \
                            "ultralytics>=8.3.48,<9" \
                            "doclayout_yolo==0.0.4" \
                            "ftfy>=6.3.1,<7" \
                            "shapely>=2.0.7,<3" \
                            "pyclipper>=1.3.0,<2" \
                            "omegaconf>=2.3.0,<3" \
                            -i https://mirrors.aliyun.com/pypi/simple && \
    sed -i '1,200{s/self\.act = act_layer()/self.act = nn.GELU()/;t;b};' /root/miniconda/envs/vllm_kunlun_0.10.1.1/lib/python3.10/site-packages/vllm_kunlun/models/qwen2_vl.py && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]