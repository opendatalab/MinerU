# Deploying MinerU with Docker

MinerU provides a convenient Docker deployment method, which helps quickly set up the environment and solve some tricky environment compatibility issues.

## Build Docker Image using Dockerfile

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/global/Dockerfile
docker build -t mineru:latest -f Dockerfile .
```

> [!TIP]
> The [Dockerfile](https://github.com/opendatalab/MinerU/blob/master/docker/global/Dockerfile) uses `vllm/vllm-openai:v0.10.1.1` as the base image by default. This version of vLLM v1 engine has limited support for GPU models. 
> This version supports a limited range of GPU models and may only function on Ampere, Ada Lovelace, and Hopper architectures. If you cannot use vLLM for accelerated inference on Volta, Turing, or Blackwell GPUs, you can resolve this issue by changing the base image to `vllm/vllm-openai:v0.11.0`.

## Docker Description

MinerU's Docker uses `vllm/vllm-openai` as the base image, so it includes the `vllm` inference acceleration framework and necessary dependencies by default. Therefore, on compatible devices, you can directly use `vllm` to accelerate VLM model inference.

> [!NOTE]
> Requirements for using `vllm` to accelerate VLM model inference:
> 
> - Device must have Volta architecture or later graphics cards with 8GB+ available VRAM.
> - The host machine's graphics driver should support CUDA 12.8 or higher; You can check the driver version using the `nvidia-smi` command.
> - Docker container must have access to the host machine's graphics devices.

## Start Docker Container

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 -p 7860:7860 -p 8000:8000 \
  --ipc=host \
  -it mineru:latest \
  /bin/bash
```

After executing this command, you will enter the Docker container's interactive terminal with some ports mapped for potential services. You can directly run MinerU-related commands within the container to use MinerU's features.
You can also directly start MinerU services by replacing `/bin/bash` with service startup commands. For detailed instructions, please refer to the [Start the service via command](https://opendatalab.github.io/MinerU/usage/quick_usage/#advanced-usage-via-api-webui-http-clientserver).

## Start Services Directly with Docker Compose

We provide a [compose.yaml](https://github.com/opendatalab/MinerU/blob/master/docker/compose.yaml) file that you can use to quickly start MinerU services.

```bash
# Download compose.yaml file
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
```

>[!NOTE]
>
>- The `compose.yaml` file contains configurations for multiple services of MinerU, you can choose to start specific services as needed.
>- Different services might have additional parameter configurations, which you can view and edit in the `compose.yaml` file.
>- Due to the pre-allocation of GPU memory by the `vllm` inference acceleration framework, you may not be able to run multiple `vllm` services simultaneously on the same machine. Therefore, ensure that other services that might use GPU memory have been stopped before starting the `vlm-openai-server` service or using the `vlm-vllm-engine` backend.

---

### Start OpenAI-compatible server service
connect to `openai-server` via `vlm-http-client` backend
  ```bash
  docker compose -f compose.yaml --profile openai-server up -d
  ```
  >[!TIP]
  >In another terminal, connect to openai server via http client (only requires CPU and network, no vllm environment needed)
  > ```bash
  > mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://<server_ip>:30000
  > ```

---

### Start Web API service
  ```bash
  docker compose -f compose.yaml --profile api up -d
  ```
  >[!TIP]
  >Access `http://<server_ip>:8000/docs` in your browser to view the API documentation.

---

### Start Gradio WebUI service
  ```bash
  docker compose -f compose.yaml --profile gradio up -d
  ```
  >[!TIP]
  >
  >- Access `http://<server_ip>:7860` in your browser to use the Gradio WebUI.
