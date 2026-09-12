# Docker Deployment (MinerU 4.0 / NVIDIA)

The general Docker setup targets Linux with NVIDIA GPUs (WSL2 on Windows). Install directly on macOS for Apple Silicon; this Docker workflow does not provide MPS/MLX acceleration. AMD and vendor accelerators use the [legacy guides](../usage/compatibility.md), whose dedicated Dockerfiles stay on `mineru<4`.

## Build the image

Run from a repository root containing the 4.0 Dockerfile:

```bash
docker build -t mineru:4 -f docker/global/Dockerfile .
```

The base image keeps vLLM 0.21.0. The Dockerfile installs `mineru[torch]>=4.0,<5` and reuses the bundled vLLM. The default image uses CUDA 13.0; enable the commented `v0.21.0-cu129` base image for CUDA 12.9 and ensure the host driver supports the selected runtime.

Build-time downloads explicitly select Torch small models and original vLLM weights. Runtime selection is also fixed with `MINERU_MODEL_SMALL_BACKEND=torch` and `MINERU_MODEL_VLM_ENGINE=vllm`. A build machine without a GPU can therefore prepare the correct weights, while inference still requires matching GPU hardware. The China Dockerfile uses ModelScope; the global version uses Hugging Face.

## Interactive container

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 7860:7860 -p 8000:8000 -p 8002:8002 -p 30000:30000 \
  -it mineru:4 /bin/bash
```

Start the WebUI inside the container:

```bash
mineru-kit webui --server-name 0.0.0.0 --server-port 7860
```

## Docker Compose

These commands use the Compose configuration in the same checkout and the locally built `mineru:4` image. Choose a service profile as needed; running multiple model services together requires planning GPU and memory allocation.

```bash
docker compose -f docker/compose.yaml --profile webui up -d
docker compose -f docker/compose.yaml --profile api up -d
docker compose -f docker/compose.yaml --profile router up -d
docker compose -f docker/compose.yaml --profile openai-server up -d
```

| Profile | Address / health check | Purpose |
| --- | --- | --- |
| `webui` | `http://127.0.0.1:7860` | Document UI; service name `mineru-webui` |
| `api` | `http://127.0.0.1:8000/v1/health` | V1 document parsing API |
| `router` | `http://127.0.0.1:8002/v1/health` | Multi-service / multi-GPU V1 entrypoint |
| `openai-server` | `http://127.0.0.1:30000/health` | OpenAI-compatible VLM inference, not the parsing API |

Inspect configuration and logs:

```bash
docker compose -f docker/compose.yaml --profile webui config
docker compose -f docker/compose.yaml logs mineru-webui
```

When upgrading from the old Compose file, change the `gradio` profile to `webui` and the service name to `mineru-webui`. Adjust mounts, GPU `device_ids`, and ports for your deployment. See [SDK and API](../usage/sdk_api.md).
