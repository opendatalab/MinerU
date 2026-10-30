# 扩展模块安装

所有示例面向 MinerU 4.x，在已激活的虚拟环境中执行。

| 安装方式 | 包含能力 | 适用场景 |
| --- | --- | --- |
| `mineru>=4.0,<5` | ONNX、llama.cpp、WebUI、SDK/API；Apple Silicon 自动包含 Torch | 原生文档、ONNX 小模型（4.1 支持自动 GPU/CPU 选型）、llama.cpp 或连接已有服务 |
| `mineru[torch]>=4.0,<5` | 基础包 + Torch、Torchvision、Transformers、Accelerate | 使用 Torch 小模型，VLM 引擎独立选择 |
| `mineru[full]>=4.0,<5` | Torch extra + Linux 上的 vLLM / Windows 上的 LMDeploy | 支持设备上的高吞吐部署 |

```bash
uv pip install -U "mineru[torch]>=4.0,<5"
```

或安装平台对应的推理引擎：

```bash
uv pip install -U "mineru[full]>=4.0,<5"
```

`all` 是 `full` 的组合别名。`core`、`pipeline`、`vlm`、`vllm`、`lmdeploy`、`gradio` 不再是 MinerU 4.0 的 extras。档位名称也不是 extra。

## 可选引擎约束

当前声明 `torch>=2.7.0,<3`、`transformers>=5.10.1,<6`、Linux `vllm>=0.19.1,<0.31.0`、Windows `lmdeploy>=0.17.0,<0.19`。Python 支持范围还应与这些依赖实际提供的 wheel 取交集。升级已有厂商环境前，先确认它是否属于[旧平台适配](../usage/compatibility.md)。

Apple Silicon 的默认 VLM 是 llama.cpp。

WebUI 使用 Gradio 6，包含本地 PDF.js 预览资源，无需额外安装 `gradio-pdf`。独立 UI 环境可用基础包，通过 `mineru-kit webui --api-url http://127.0.0.1:8000` 连接推理服务。

## 升级已有环境

先停止该环境的 MinerU 服务，再用原安装工具升级并保留需要的 extras，完成后确认版本、依赖与模型配置。模型下载与推理使用相同的 `model.small_backend` 和 `model.vlm.engine`，详见[模型源说明](../usage/model_source.md)。

## Apple Silicon 显式使用 MLX

MLX 是可选引擎，必须显式选择。独立服务入口要求 Apple Silicon、macOS 14 或更新版本，以及 `mlx-vlm>=0.7.0,<0.8.0`；不存在 `mineru[mlx]` extra。

```bash
uv pip install "mlx-vlm>=0.7.0,<0.8.0"
mineru-kit vlm-server --engine mlx --port 30000
```

进程内解析配置 `model.vlm.engine: mlx`；macOS 的 auto 仍选择 llama.cpp。
