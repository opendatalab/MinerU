# Docker 部署（MinerU 4.0 / NVIDIA）

通用 Docker 面向 Linux NVIDIA 环境（Windows 使用 WSL2）。Apple Silicon 使用 macOS 原生安装；此 Docker 流程不提供 MPS/MLX 加速。AMD 和国产卡使用[旧平台指南](../usage/compatibility.md)，对应专用 Docker 保持 `mineru<4`。

## 构建镜像

在包含本次 4.0 Dockerfile 的仓库根目录执行：

```bash
docker build -t mineru:4 -f docker/china/Dockerfile .
```

基础镜像保留 vLLM 0.21.0，Dockerfile 安装 `mineru[torch]>=4.0,<5`，复用基础镜像中的 vLLM。默认镜像使用 CUDA 13.0；需要 CUDA 12.9 时启用文件中注释的 `v0.21.0-cu129` 基础镜像，并确保主机驱动支持所选运行时。

构建时明确下载 Torch 小模型和 vLLM 原始权重，运行时也固定 `MINERU_MODEL_SMALL_BACKEND=torch`、`MINERU_MODEL_VLM_ENGINE=vllm`。这允许无 GPU 构建机准备正确的权重，实际推理仍需要匹配的 GPU。国内 Dockerfile 使用 ModelScope，国际版本使用 Hugging Face。

### 镜像版本来源与确认

镜像从包索引安装 `mineru[torch]>=4.0,<5`，**不会**安装作为构建上下文的仓库源码；即使从 `next` 或打过补丁的 checkout 构建，得到的仍是 PyPI 版本镜像。两条路径对应不同需求：

| 路径 | 版本来源 | 适用场景 |
| --- | --- | --- |
| 正式版镜像（`docker/global/Dockerfile`、`docker/china/Dockerfile`） | 包索引固定的 `mineru[torch]>=4.0,<5` | 生产部署 |
| 源码镜像（正式版镜像 + 本地安装） | 你的 checkout / 提交 | 验证 `next`、调试、复现补丁 |

建议给镜像打上实际版本号的 tag，而不是只用 `mineru:4`，并在构建后确认镜像内容：

```bash
docker build -t mineru:4.0.0 -f docker/china/Dockerfile .
docker run --rm mineru:4.0.0 python3 -c "from mineru.version import __version__; print(__version__)"
```

要测试特定源码版本，先构建正式版镜像，再在容器内安装 checkout：

```bash
docker run --gpus all --shm-size 32g --ipc=host -it \
  -v "$PWD":/src mineru:4.0.0 /bin/bash -c \
  "python3 -m pip install -e '/src[torch]' && mineru-kit parse /src/document.pdf -o /src/document.md"
```

可编辑安装会把容器内的 MinerU 换成你的 checkout，且不重建模型层；分发此类镜像时把提交 SHA 一并记录到 tag 中。

## 交互式容器

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 7860:7860 -p 8000:8000 -p 8002:8002 -p 30000:30000 \
  -it mineru:4 /bin/bash
```

容器内启动 WebUI：

```bash
mineru-kit webui --server-name 0.0.0.0 --server-port 7860
```

## Docker Compose

以下命令使用同一仓库中的 Compose 配置和本地构建的 `mineru:4`。按需要选择一个服务 profile；同时启动多个模型服务时需规划显存和 GPU 分配。

```bash
docker compose -f docker/compose.yaml --profile webui up -d
docker compose -f docker/compose.yaml --profile api up -d
docker compose -f docker/compose.yaml --profile router up -d
docker compose -f docker/compose.yaml --profile openai-server up -d
```

| Profile | 地址 / 健康检查 | 用途 |
| --- | --- | --- |
| `webui` | `http://127.0.0.1:7860` | 文档解析界面，服务名 `mineru-webui` |
| `api` | `http://127.0.0.1:8000/v1/health` | V1 文档解析 API |
| `router` | `http://127.0.0.1:8002/v1/health` | 多服务 / 多 GPU V1 入口 |
| `openai-server` | `http://127.0.0.1:30000/health` | OpenAI 兼容 VLM 推理，不是文档解析 API |

检查配置与日志：

```bash
docker compose -f docker/compose.yaml --profile webui config
docker compose -f docker/compose.yaml logs mineru-webui
```

从旧 Compose 升级时，将 `gradio` profile 改为 `webui`，服务名改为 `mineru-webui`。容器挂载目录、GPU `device_ids` 和端口按部署环境调整。更多接口用法见[SDK 与 API](../usage/sdk_api.md)。
