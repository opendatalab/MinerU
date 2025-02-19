# Use the official Ubuntu base image
FROM swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.2_mindspore2.3:20240722

USER root

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3.10-dev \
        python3-pip \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a virtual environment for MinerU
RUN python3 -m venv /opt/mineru_venv

# Activate the virtual environment and install necessary Python packages
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/ascend_npu/requirements.txt -O requirements.txt && \
    pip3 install -r requirements.txt --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple && \
    wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch2.3.1/torch_npu-2.3.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
    pip3 install torch_npu-2.3.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
    pip3 install https://gcore.jsdelivr.net/gh/myhloli/wheels@main/assets/whl/paddle-custom-npu/paddle_custom_npu-0.0.0-cp310-cp310-linux_aarch64.whl"

# Copy the configuration file template and install magic-pdf latest
RUN /bin/bash -c "wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json && \
    cp magic-pdf.template.json /root/magic-pdf.json && \
    source /opt/mineru_venv/bin/activate && \
    pip3 install -U magic-pdf -i https://mirrors.aliyun.com/pypi/simple"

# Download models and update the configuration file
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install modelscope -i https://mirrors.aliyun.com/pypi/simple && \
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py && \
    python3 download_models.py && \
    sed -i 's|cpu|npu|g' /root/magic-pdf.json"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && exec \"$@\"", "--"]
