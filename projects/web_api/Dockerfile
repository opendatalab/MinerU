# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Update the package list and install necessary packages
RUN apt-get -q update && \
    apt-get -q install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        # gpg \
    # && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a virtual environment for MinerU and install packages
RUN python3 -m venv /opt/mineru_venv && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip install --upgrade pip && \
    pip install magic-pdf[full] --extra-index-url https://myhloli.github.io/wheels/ --no-cache-dir && \
    pip install fastapi uvicorn python-multipart --no-cache-dir && \
    pip uninstall paddlepaddle -y && \
    pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --no-cache-dir"

# Copy the configuration file template and set up the model directory
COPY models/models /opt/models
COPY layoutreader /opt/layoutreader
COPY .paddleocr /root/.paddleocr 
COPY app.py /root/app.py
COPY magic-pdf.json /root/magic-pdf.json

WORKDIR /root


# Create the models directory
# RUN mkdir -p /opt/models

# Set the entry point to activate the virtual environment and run the command line tool
# ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && exec \"$@\" && python3 app.py", "--"]


# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn, pointing to app.py and binding to 0.0.0.0:8000
CMD ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000"]