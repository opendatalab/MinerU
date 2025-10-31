FROM vllm/vllm-openai:v0.10.1.1

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    fonts-noto-core \
    fonts-noto-cjk \
    fontconfig \
    libgl1 \
    vim \
    tmux \
    htop && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /data /opt/modelscope/hub 

# Set up environment variables
ENV MODELSCOPE_CACHE="/opt/modelscope/hub"
ENV HUGGINGFACE_HUB_CACHE="/opt/modelscope/hub"
ENV PYTHONPATH="/data/MinerU"

# Copy source code
COPY ./ /data/MinerU/

WORKDIR /data/MinerU

# Install all Python dependencies in a single layer
RUN pip install --no-cache-dir flash_attn && \
    pip install --no-cache-dir -e .[all] && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu==1.22.0

# Download all models in a single layer
RUN /bin/bash -c "export MINERU_MODEL_SOURCE=huggingface && \
    export MODELSCOPE_CACHE=/opt/modelscope/hub && \
    export HUGGINGFACE_HUB_CACHE=/opt/modelscope/hub && \
    python3 -m mineru.cli.models_download -s huggingface -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]

