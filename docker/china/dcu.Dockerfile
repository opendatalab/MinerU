# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + Hygon DCU.
FROM harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.2-1226-das1.7-py3.10-20251226


# Install libgl for opencv support & Noto fonts for Chinese characters
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
    python3 -m pip install mineru[api,gradio] \
                            "matplotlib>=3.10,<4" \
                            "ultralytics>=8.3.48,<9" \
                            "doclayout_yolo==0.0.4" \
                            "ftfy>=6.3.1,<7" \
                            "shapely>=2.0.7,<3" \
                            "pyclipper>=1.3.0,<2" \
                            "omegaconf>=2.3.0,<3" \
                            numpy==1.25.0 \
                            opencv-python==4.11.0.86 \
                            -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]