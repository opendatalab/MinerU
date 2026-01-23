# Base image containing the vLLM inference environment, requiring amd64(x86-64) CPU + Enflame GCU.
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/gcu:docker_images_topsrider_i3x_3.6.20260106_vllm0.11_pytorch2.8.0


# Install libgl for opencv support & Noto fonts for Chinese characters
RUN echo 'deb http://mirrors.aliyun.com/ubuntu/ noble main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ noble-updates main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ noble-backports main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ noble-security main restricted universe multiverse' > /tmp/aliyun-sources.list && \
    apt-get -o Dir::Etc::SourceList=/tmp/aliyun-sources.list update && \
    apt-get -o Dir::Etc::SourceList=/tmp/aliyun-sources.list install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/aliyun-sources.list

# Install mineru latest
RUN python3 -m pip install "mineru[core]>=2.7.2" \
                            numpy==1.26.4 \
                            opencv-python==4.11.0.86 \
                            -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]