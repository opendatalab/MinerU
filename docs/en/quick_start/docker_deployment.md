# Docker Deployment (MinerU 4.0 / NVIDIA)

The general Docker setup targets Linux with NVIDIA GPUs (WSL2 on Windows). Install directly on macOS for Apple Silicon; this Docker workflow does not provide MPS/MLX acceleration. AMD and vendor accelerators use the [legacy guides](../usage/compatibility.md), whose dedicated Dockerfiles stay on `mineru<4`.

## Build the image

Run from a repository root containing the 4.0 Dockerfile:

```bash
docker build -t mineru:4 -f docker/global/Dockerfile .
```

The base image keeps vLLM 0.21.0. The Dockerfile installs `mineru[torch]>=4.0,<5` and reuses the bundled vLLM. The default image uses CUDA 13.0; enable the commented `v0.21.0-cu129` base image for CUDA 12.9 and ensure the host driver supports the selected runtime.

Build-time downloads explicitly select Torch small models and original vLLM weights. Runtime selection is also fixed with `MINERU_MODEL_SMALL_BACKEND=torch` and `MINERU_MODEL_VLM_ENGINE=vllm`. A build machine without a GPU can therefore prepare the correct weights, while inference still requires matching GPU hardware. The China Dockerfile uses ModelScope; the global version uses Hugging Face.

### Image provenance and version checks

The image installs the package-index range `mineru[torch]>=4.0,<5`; it does **not** install the repository checkout used as the build context, and a version-range install is not locked by the image tag. Building from a `next` or patched checkout still produces a package-index image, and rebuilding the same tag later may install a different 4.x. Two paths cover the different needs:

| Path | Version source | Use it for |
| --- | --- | --- |
| Release image (`docker/global/Dockerfile`, `docker/china/Dockerfile`) | Package-index range `mineru[torch]>=4.0,<5` | Production deployments |
| Source-debug container (release image + mounted checkout) | Your working copy / commit | Verifying `next`, debugging, reproducing a patch |

Tag images with the actual version instead of only `mineru:4`, and assert the installed version after building — this detects a tag/version mismatch; it does not make a range build reproducible:

```bash
docker build -t mineru:4.0.0 -f docker/global/Dockerfile .
docker run --rm mineru:4.0.0 python3 -c '
from mineru.version import __version__
expected = "4.0.0"
print(f"actual={__version__}, expected={expected}")
raise SystemExit(0 if __version__ == expected else 1)'
```

To test a specific source revision, build the release image once, then install your checkout inside a **source-debug container**. It depends on the host mount and is not distributable:

```bash
docker run --gpus all --shm-size 32g --ipc=host -it \
  -v "$PWD":/src mineru:4.0.0 /bin/bash -c \
  "python3 -m pip install -e '/src[torch]' && mineru-kit parse /src/document.pdf -o /src/document.md"
```

A distributable source image instead copies the source at build time; keep the commit SHA in the tag:

```dockerfile
FROM mineru:4.0.0
COPY . /src
RUN python3 -m pip install "/src[torch]"
```

After replacing the installed package with a different source revision, re-check that the image's pre-downloaded models still satisfy it: `mineru-kit models verify --tier standard --small-backend torch --vlm-engine vllm` exits non-zero when model files are incomplete. `verify` is a file check, not an inference acceptance test.

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
