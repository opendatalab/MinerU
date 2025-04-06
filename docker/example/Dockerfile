# Use the official Ubuntu base image
FROM ubuntu:22.04

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
    wget https://github.com/jianjungki/MinerU/raw/master/docker/global/requirements.txt -O requirements.txt && \
    pip3 install -r requirements.txt "

# Copy the configuration file template and install magic-pdf latest
RUN /bin/bash -c "wget https://github.com/jianjungki/MinerU/raw/master/magic-pdf.template.json && \
    cp magic-pdf.template.json /root/magic-pdf.json && \
    source /opt/mineru_venv/bin/activate && \
    pip3 install -U magic-pdf"

# Download models and update the configuration file
RUN /bin/bash -c "pip3 install huggingface_hub && \
    wget https://github.com/jianjungki/MinerU/raw/master/scripts/download_models_hf.py -O download_models.py && \
    python3 download_models.py && \
    sed -i 's|cpu|cuda|g' /root/magic-pdf.json"

    


COPY api /tmp/api


WORKDIR /tmp/api
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install -r requirements.txt"

RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    nohup uvicorn api:app --host 0.0.0.0 --port 3000 --workers 1 --reload &"

COPY example /tmp/example

ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /tmp/example
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install -r requirements.txt"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && python3 app.py"]
