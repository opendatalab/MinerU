# Use the official Ubuntu base image
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-infer:24.0.RC3-ubuntu20.04

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
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
    pip3 install --upgrade pip && \
    wget https://gitee.com/myhloli/MinerU/raw/dev/docker/huawei_npu/requirements.txt && \
    pip3 install -r requirements.txt --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple"

# Copy the configuration file template and install magic-pdf latest
RUN /bin/bash -c "wget https://gitee.com/myhloli/MinerU/raw/master/magic-pdf.template.json && \
    cp magic-pdf.template.json /root/magic-pdf.json && \
    source /opt/mineru_venv/bin/activate && \
    pip3 install git+https://gitee.com/myhloli/MinerU.git@dev"

# Download models and update the configuration file
RUN /bin/bash -c "pip3 install modelscope && \
    wget https://gitee.com/myhloli/MinerU/raw/master/scripts/download_models.py && \
    python3 download_models.py && \
    sed -i 's|cpu|npu|g' /root/magic-pdf.json"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && exec \"$@\"", "--"]
