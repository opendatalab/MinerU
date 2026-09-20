# AMD GPU 安装

本页介绍在 AMD GPU 容器中安装和使用 MinerU **4.x（`>=4.0,<5`）**。示例环境为 AMD **gfx1201（约 32 GB 显存）**、Ubuntu 24.04、Python 3.12 和 ROCm 7.2.3。

## 1. 准备镜像和容器

使用以下 ROCm/PyTorch 基础镜像，MinerU 在进入容器后安装：

```text
rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

宿主机需要已安装可用的 AMD GPU 驱动和 Docker。以下命令保留测试容器的权限参数，并将当前目录下的工作目录挂载到容器；其中 `--privileged` 等高权限选项不是已验证的必要条件，共享或生产环境应按需收紧。

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

后续安装和启动命令均在容器中执行。可先运行 `amd-smi` 查看 GPU 状态，并选择空闲设备。

## 2. 安装 MinerU

在独立虚拟环境中安装，不修改镜像自带的 PyTorch：

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

基础包已包含 ONNX Runtime 和 llama.cpp，依赖版本由 MinerU 的包约束自动解析。本文将版面分析/OCR 小模型固定为 **ONNX CPU**，VLM 使用下方选择的 GPU 推理引擎。

## 3. 配置推理引擎

根据需要选择 **vLLM** 或 **llama.cpp**，只执行所选引擎的安装和配置步骤即可。

### vLLM

在已激活的虚拟环境中安装 ROCm 7.2.3 对应的 vLLM wheel 及配套依赖：

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

MinerU 使用 4.x 版本范围；vLLM 和配套 Torch/Triton wheel 保留已验证的版本组合。该 vLLM wheel 使用虚拟环境内的 Torch 2.12，不复用镜像中的 Torch 2.10。请保留版本专用 ROCm 索引，不要改用普通 CUDA wheel 或浮动的 latest 索引；升级引擎时需同时核对 MinerU 的依赖范围及 ROCm wheel 的配套要求。

### llama.cpp

无需额外安装 Python 推理包；GPU 加速使用 **Vulkan**。容器内需要 Vulkan 加载器和可识别 AMD GPU 的 ICD。若尚未安装，可在这个专用 Ubuntu 24.04 容器中执行：

```bash
apt-get update
apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers vulkan-tools
```

已有可用 Vulkan 驱动时跳过安装。上面的命令会安装容器内系统包，不是对宿主机内核驱动的安装；验证过的 Mesa RADV 版本为 25.2.8，软件源当前版本可能不同。

配置无显示服务环境并检查设备：

```bash
export XDG_RUNTIME_DIR="$PWD/.xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
unset DISPLAY WAYLAND_DISPLAY
vulkaninfo --summary
mineru-kit vlm-server --engine llama-cpp --list-devices
export MINERU_MODEL_VLM_ENGINE=llama-cpp
```

输出应包含实际 AMD GPU，而不只是 `llvmpipe` 等软件设备。若当前 Mesa 无法识别显卡，请先配置与该 GPU 匹配的 Vulkan 驱动。

## 4. 下载模型

在刚才选择引擎的同一终端中，下载 ONNX 小模型和该引擎所需的 VLM 权重：

```bash
unset HF_HUB_OFFLINE
mineru-kit models download --tier standard --small-backend onnx \
  --vlm-engine "$MINERU_MODEL_VLM_ENGINE" --source huggingface
mineru-kit models verify --tier standard --small-backend onnx \
  --vlm-engine "$MINERU_MODEL_VLM_ENGINE"
export MINERU_MODEL_SOURCE=local HF_HUB_OFFLINE=1
```

vLLM 使用原始模型权重，llama.cpp 使用 GGUF 和视觉投影器。下载命令获取模型仓库的当前默认版本；模型源及离线部署配置见[模型源配置](../usage/model_source.md)。

## 5. 启动 VLM 服务

保持当前终端和虚拟环境，只启动已选择的引擎。两个示例均使用 `127.0.0.1:30000`，不要同时启动。设备编号 `0` 仅作示例，请按实际空闲 GPU 调整；HIP、Vulkan 和 AMD-SMI 的编号不一定相同。

### 使用 vLLM 启动

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

等待日志出现 `Application startup complete`。ROCm PyTorch 使用 `cuda` 作为设备/API 名称，这是正常行为。

### 使用 llama.cpp 启动

在完成 Vulkan 配置的同一终端执行：

```bash
export GGML_VK_VISIBLE_DEVICES=0
export MINERU_DEVICE_MODE=cpu
mineru-kit vlm-server --engine llama-cpp \
  --host 127.0.0.1 --port 30000 \
  --n-gpu-layers 99 --mmproj-offload --device Vulkan0 --split-mode none \
  --parallel 1 --ctx-size 8192 --threads 8 --threads-batch 8
```

根据设备列表调整 `GGML_VK_VISIBLE_DEVICES` 和 `--device`。日志应显示模型层卸载到 GPU，以及 `CLIP using Vulkan0 backend`；这里的 `MINERU_DEVICE_MODE=cpu` 不会关闭显式指定的 Vulkan 加速。

## 6. 使用 MinerU

服务启动后，在宿主机的另一个终端进入同一容器：

```bash
docker exec -it mineru-amd bash
```

在新开的容器终端中激活环境，并连接已启动的 VLM 服务。以下配置供命令行和 WebUI 共用，完成后选择一种使用方式即可：

```bash
cd /workspace
source .venv/bin/activate
export MINERU_DEVICE_MODE=cpu MINERU_MODEL_SMALL_BACKEND=onnx
export MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000
export MINERU_MODEL_VLM_MAX_CONCURRENCY=1
export MINERU_MODEL_SOURCE=local HF_HUB_OFFLINE=1
curl --fail --silent --show-error "$MINERU_MODEL_VLM_SERVER_URL/v1/models"
```

### 命令行解析

将待解析 PDF 放到宿主机的 `mineru-workspace` 目录，在上述终端中执行：

```bash
mineru-kit parse document.pdf -o document.md --tier standard
```

将 `document.pdf` 替换为实际文件名。结果写入同一目录下的 `document.md`，宿主机可直接读取；扫描 PDF 也使用相同命令。接口返回模型列表仅表示服务可连接，解析后还应检查正文、表格和公式内容。

批量转换并保存完整结果包：

```bash
mineru-kit parse ./documents -o ./output --format zip --tier standard
```

### WebUI

MinerU 基础包已包含 WebUI，无需额外安装。在完成上述环境配置的同一终端中启动，无需先执行命令行解析：

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860 \
  --api-server-tier standard
```

在宿主机浏览器打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)，上传 PDF，选择 `standard` 档位并开始解析，即可预览和下载结果。若运行在远程服务器上，通过 VS Code 端口转发或 SSH 隧道将 **7860** 转发到本机后访问；上述容器使用 host 网络，无需额外添加 `-p` 映射。

WebUI 会自动托管文档解析 API，并通过 `MINERU_MODEL_VLM_SERVER_URL` 复用前面启动的 vLLM 或 llama.cpp 服务。这里不需要 `--api-url`；该参数用于连接独立的 MinerU 文档解析 API，不能填写 VLM 的 `30000` 端口。

停止使用时，在 WebUI 和 VLM 服务各自的终端按 Ctrl+C。更多 API 和解析选项见[基础使用](../usage/quick_usage.md)和[Python SDK](../usage/sdk_api.md)。

## 使用说明

- 实际验证版本为 MinerU **4.0.4**；`>=4.0,<5` 是安装版本范围，不代表所有 4.x 版本或 AMD GPU 均已验证。测试基于已有容器，未进行全新镜像重放。
- 保持 `MINERU_MODEL_SMALL_BACKEND=onnx`；当前测试环境的 Torch 默认 FP16 OCR 存在识别异常，不建议直接切换。
- 如果 RADV 提示 `not a conformant Vulkan implementation, testing use only`，推理可运行并不代表驱动通过规范认证或适合生产部署。