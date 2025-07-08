# Local Deployment

## Install MinerU

### Install via pip or uv

```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[core]"
```

### Install from source

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e .[core]
```

> [!NOTE]  
> Linux and macOS systems automatically support CUDA/MPS acceleration after installation. For Windows users who want to use CUDA acceleration, 
> please visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to install PyTorch with the appropriate CUDA version.

### Install Full Version (supports sglang acceleration) (requires device with Turing or newer architecture and at least 8GB GPU memory)

If you need to use **sglang to accelerate VLM model inference**, you can choose any of the following methods to install the full version:

- Install using uv or pip:
  ```bash
  uv pip install -U "mineru[all]"
  ```
- Install from source:
  ```bash
  uv pip install -e .[all]
  ```

> [!TIP]  
> If any exceptions occur during the installation of `sglang`, please refer to the [official sglang documentation](https://docs.sglang.ai/start/install.html) for troubleshooting and solutions, or directly use Docker-based installation.

- Build image using Dockerfile:
  ```bash
  wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/global/Dockerfile
  docker build -t mineru-sglang:latest -f Dockerfile .
  ```
  Start Docker container:
  ```bash
  docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ipc=host \
    mineru-sglang:latest \
    mineru-sglang-server --host 0.0.0.0 --port 30000
  ```
  Or start using Docker Compose:
  ```bash
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
    docker compose -f compose.yaml up -d
  ```
  
> [!TIP]
> The Dockerfile uses `lmsysorg/sglang:v0.4.8.post1-cu126` as the default base image, which supports the Turing/Ampere/Ada Lovelace/Hopper platforms.  
> If you are using the newer Blackwell platform, please change the base image to `lmsysorg/sglang:v0.4.8.post1-cu128-b200`.

### Install client  (for connecting to sglang-server on edge devices that require only CPU and network connectivity)

```bash
uv pip install -U mineru
mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://<host_ip>:<port>
```

---