# AMD GPU installation

This guide covers installing and using MinerU **4.x (`>=4.0,<5`)** on AMD GPUs. The [On Linux](#on-linux) example uses an AMD **Radeon GPU**, Ubuntu 24.04, Python 3.12 and ROCm 7.2.3 in a container; [On Windows](#on-windows) runs through the base package with llama.cpp Vulkan.

## On Linux

### 1. Prepare the image and container

Use the following ROCm/PyTorch base image; MinerU is installed after entering the container:

```text
rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

The host needs a working AMD GPU driver and Docker. The following command preserves the test container's permission flags and mounts a workspace under the current directory. Elevated options such as `--privileged` have not been established as requirements; restrict permissions as appropriate for shared or production environments.

```bash
mkdir -p mineru-workspace
docker run -d --name mineru-amd \
  --network host --ipc host \
  --privileged --cap-add CAP_SYS_ADMIN --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size 10g --device /dev/kfd --device /dev/dri \
  -v "$PWD/mineru-workspace:/workspace" -w /workspace \
  -e MINERU_HOME=/workspace/.mineru \
  -e MINERU_MODEL_BASE_DIR=/workspace/models \
  -e MINERU_MODEL_SMALL_BACKEND=onnx \
  rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  sleep infinity
docker exec -it mineru-amd bash
```

Run the remaining installation and server commands inside the container.

### 2. Install MinerU

Install into an isolated virtual environment without changing the image's existing PyTorch installation:

```bash
cd /workspace
python3 -m venv .uv-tools
.uv-tools/bin/python -m pip install -U uv
export PATH="$PWD/.uv-tools/bin:$PATH"
uv venv --python python3 .venv
source .venv/bin/activate
uv pip install -U --default-index https://pypi.org/simple "mineru>=4.0,<5"
uv pip check
```

The base package includes ONNX Runtime and llama.cpp, with versions resolved from MinerU's dependency constraints. This guide keeps layout/OCR small models on **ONNX CPU**, while the VLM uses the GPU inference engine selected below.

### 3. Configure the inference engine

Choose **vLLM** or **llama.cpp** and follow only the installation and configuration steps for that engine.

#### vLLM

In the activated virtual environment, install the vLLM wheel for ROCm 7.2.3 together with its matching dependencies:

```bash
uv pip install -U \
  --index https://wheels.vllm.ai/rocm/0.28.0/rocm723 \
  --default-index https://pypi.org/simple \
  "mineru[full]>=4.0,<5" 'vllm==0.28.0+rocm723' \
  'torch==2.12.0+git6bbd260' 'torchvision==0.27.1+df56172' \
  'torchaudio==2.11.0+34c52a6' 'triton==3.7.1+gitf0b55c07'
uv pip check
export MINERU_MODEL_VLM_ENGINE=vllm
```

MinerU uses a 4.x version range; vLLM and its matching Torch/Triton wheels retain the validated version combination. This vLLM wheel uses Torch 2.12 inside the virtual environment rather than the image's Torch 2.10. Keep the version-specific ROCm index, not ordinary CUDA wheels or a floating latest index. When upgrading the engine, check both MinerU's dependency constraints and the matching ROCm wheel requirements.

#### llama.cpp

No additional Python inference package is needed; GPU acceleration uses **Vulkan**. The container needs a Vulkan loader and an ICD that recognizes the AMD GPU. If these are missing, install them inside this dedicated Ubuntu 24.04 container:

```bash
apt-get update
apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers vulkan-tools
```

Skip installation if Vulkan already works. These commands install container system packages, not the host kernel driver. The validated Mesa RADV version was 25.2.8; the current repository version may differ.

Configure a headless environment and check the devices:

```bash
export XDG_RUNTIME_DIR="$PWD/.xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
unset DISPLAY WAYLAND_DISPLAY GGML_VK_VISIBLE_DEVICES
vulkaninfo --summary
mineru-kit vlm-server --engine llama-cpp --list-devices
export MINERU_MODEL_VLM_ENGINE=llama-cpp
```

The output must include a physical AMD GPU, not only software devices such as `llvmpipe`. If the installed Mesa version does not recognize the card, configure a compatible Vulkan driver before proceeding.

### 4. Download models

In the same terminal where you selected the engine, download the ONNX small models and the VLM weights required by that engine. Choose either of the following model sources:

```bash
unset HF_HUB_OFFLINE
mineru-kit models download --tier standard --small-backend onnx \
  --vlm-engine "$MINERU_MODEL_VLM_ENGINE" --source huggingface
```

If Hugging Face is unreachable, use ModelScope instead:

```bash
unset HF_HUB_OFFLINE
mineru-kit models download --tier standard --small-backend onnx \
  --vlm-engine "$MINERU_MODEL_VLM_ENGINE" --source modelscope
```

After downloading successfully from the selected source, verify the local models and enable offline mode only if verification succeeds:

```bash
mineru-kit models verify --tier standard --small-backend onnx \
  --vlm-engine "$MINERU_MODEL_VLM_ENGINE" && \
  export MINERU_MODEL_SOURCE=local HF_HUB_OFFLINE=1
```

vLLM uses the original model weights; llama.cpp uses GGUF and a vision projector. The download command fetches the model repositories' current default revisions. See [Model Source](../usage/model_source.md) for source selection and offline deployment.

### 5. Start the VLM server

Keep the current terminal and virtual environment, and start only the selected engine. Both examples listen on `127.0.0.1:30000`, so do not run them simultaneously.

> **Additional information: GPU selection and device numbering**
>
> The commands below use device index `0` as an example. On multi-GPU systems, use `amd-smi` to inspect usage and select an idle GPU as needed. HIP, Vulkan and AMD-SMI indices do not necessarily match.
>
> - **vLLM**: Select the target GPU through `HIP_VISIBLE_DEVICES`. ROCm PyTorch uses `cuda` as its device/API name; this is expected.
> - **llama.cpp**: Set `GGML_VK_VISIBLE_DEVICES` to the target physical device index from Vulkan enumeration. When selecting a single GPU, llama.cpp renumbers it to `Vulkan0` within its own process. For example, `GGML_VK_VISIBLE_DEVICES=7` still requires `--device Vulkan0`, not `Vulkan7`. This does not change HIP or AMD-SMI indices.
>
> After changing Vulkan device selection, run `mineru-kit vlm-server --engine llama-cpp --list-devices` under the same environment before starting the server to confirm the target AMD GPU. `MINERU_DEVICE_MODE=cpu` does not disable the explicitly requested Vulkan acceleration.

#### Start with vLLM

```bash
export HIP_VISIBLE_DEVICES=0
export MINERU_DEVICE_MODE=cuda
export VLLM_ROCM_USE_AITER=0 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=1
mineru-kit vlm-server --engine vllm \
  --host 127.0.0.1 --port 30000 \
  --served-model-name mineru-vlm --allowed-origins '[]' \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.35 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA \
  --enforce-eager
```

Wait for `Application startup complete` in the logs.

#### Start with llama.cpp

Run this in the same terminal where you configured Vulkan:

```bash
export GGML_VK_VISIBLE_DEVICES=0
export MINERU_DEVICE_MODE=cpu
mineru-kit vlm-server --engine llama-cpp \
  --host 127.0.0.1 --port 30000 \
  --n-gpu-layers 99 --mmproj-offload --device Vulkan0 --split-mode none \
  --parallel 1 --ctx-size 8192 --threads 8 --threads-batch 8
```

### 6. Use MinerU

Once the server is ready, open another terminal on the host and enter the same container:

```bash
docker exec -it mineru-amd bash
```

In the new container terminal, activate the environment and connect to the running VLM server. The following settings are shared by the CLI and WebUI; configure them first, then choose either interface:

```bash
cd /workspace
source .venv/bin/activate
export MINERU_DEVICE_MODE=cpu MINERU_MODEL_SMALL_BACKEND=onnx
export MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000
export MINERU_MODEL_VLM_MAX_CONCURRENCY=1
export MINERU_MODEL_SOURCE=local HF_HUB_OFFLINE=1
curl --fail --silent --show-error "$MINERU_MODEL_VLM_SERVER_URL/v1/models"
```

#### Command-line parsing

Place a PDF in the host's `mineru-workspace` directory and run this in the configured terminal:

```bash
mineru-kit parse document.pdf -o document.md --tier standard
```

Replace `document.pdf` with your input filename. The result is written to `document.md` in the same directory and is also accessible on the host. Use the same command for scanned PDFs. A model-list response only confirms connectivity; inspect the text, tables and formulas after parsing.

For batch conversion with complete result bundles:

```bash
mineru-kit parse ./documents -o ./output --format zip --tier standard
```

#### WebUI

The MinerU base package includes the WebUI; no additional installation is needed. Start it in the same terminal after applying the environment settings above. Running a CLI conversion first is not required:

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860 \
  --api-server-tier standard
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in a browser on the host, upload a PDF, select the `standard` tier and start parsing to preview and download results. For a remote server, forward port **7860** to your local machine using VS Code port forwarding or an SSH tunnel. The container above uses host networking, so no additional `-p` mapping is needed.

The WebUI manages a document parsing API and reuses the existing vLLM or llama.cpp service through `MINERU_MODEL_VLM_SERVER_URL`. Do not set `--api-url` here: that option connects to a separate MinerU document parsing API, not the VLM endpoint on port `30000`.

To stop, press Ctrl+C in the WebUI and VLM server terminals separately. See [Quick Usage](../usage/quick_usage.md) and [Python SDK](../usage/sdk_api.md) for additional API and parsing options.

## On Windows

On Windows, the ROCm vLLM path is not available because ROCm vLLM only ships Linux wheels. MinerU 4.x still runs on AMD GPUs through the base package: the small models use ONNX on CPU, and the VLM runs llama.cpp in Vulkan mode on the Radeon GPU. No container or ROCm driver is required.

### 1. Install uv and Python 3.12

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
```

### 2. Create the environment and install the base package

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -U "mineru>=4.0,<5"
```

The base package already includes ONNX Runtime and llama.cpp. Do not install the `[full]` extra on Windows: it pulls CUDA/LMDeploy engines and does not help on AMD.

### 3. Confirm the AMD GPU is visible to the llama.cpp Vulkan backend

```powershell
mineru-kit vlm-server --engine llama-cpp --list-devices
```

The output must list a physical AMD device, for example `Vulkan0: AMD Radeon(TM) 8060S Graphics`. If it only shows software devices such as `llvmpipe`, the Vulkan driver is missing.

### 4. Download the standard-tier models

```powershell
mineru-kit models download --tier standard --small-backend onnx --vlm-engine llama-cpp --source modelscope
```

Use `--source huggingface` if ModelScope is unreachable.

### 5. Parse a document

```powershell
$env:MINERU_DEVICE_MODE = 'cpu'   # small models stay on CPU; the VLM still uses Vulkan
mineru-kit parse document.pdf -o document.md --tier standard
```

The WebUI is included in the base package:

```powershell
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
```

Open <http://127.0.0.1:7860> in a browser, upload a PDF and select the `standard` tier.