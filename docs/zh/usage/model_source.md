# 模型下载与配置

MinerU 4.0 独立配置小模型后端和 VLM 引擎。配置文件默认位于 `$MINERU_HOME/config.yaml`（`MINERU_HOME` 默认 `~/.mineru`），可通过 `MINERU_CONFIG` 指定其他文件。

```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
  small_backend: auto
  vlm:
    engine: auto
```

## 模型源

`model.source` 支持 `auto`、`huggingface`、`modelscope`、`local`。`auto` 优先探测 Hugging Face，不可访问时选择 ModelScope；来自默认值或配置文件的自动选择可写回配置。环境变量优先于文件配置：

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru-kit parse document.pdf -o document.md --tier standard
```

Windows PowerShell：

```powershell
$env:MINERU_MODEL_SOURCE = "modelscope"
```

小模型包 `MinerU-4_models_torch` 和 `MinerU-4_models_onnx` 在 Hugging Face 与 ModelScope 均已登记；VLM 原始权重与 llama.cpp 的 GGUF/mmproj 使用不同模型仓库，由所选引擎决定。Basic 需要对应小模型包，Standard 还需要 VLM 模型，并同时支持 Advanced 请求。

## 下载、检查和离线使用

按当前配置下载并验证 Standard 部署模型：

```bash
mineru-kit models download --tier standard --source modelscope
mineru-kit models verify --tier standard
mineru-kit models show
```

CPU 小模型和 llama.cpp 的显式组合：

```bash
mineru-kit models download --tier standard --small-backend onnx --vlm-engine llama-cpp --source huggingface
mineru-kit models verify --tier standard --small-backend onnx --vlm-engine llama-cpp
```

NVIDIA / vLLM 部署可以在无 GPU 的构建机下载目标模型；下载时必须显式指定目标后端，不能依赖构建机自动选择：

```bash
mineru-kit models download --tier standard --small-backend torch --vlm-engine vllm --source modelscope
mineru-kit models verify --tier standard --small-backend torch --vlm-engine vllm
```

这些命令选项只覆盖本次操作，不修改持久配置。运行解析前使用相同组合，例如：

```bash
export MINERU_MODEL_SMALL_BACKEND=torch
export MINERU_MODEL_VLM_ENGINE=vllm
export MINERU_MODEL_SOURCE=local
mineru-kit parse document.pdf -o document.md --tier standard
```

只需 Basic 时将下载档位改为 `basic`；Advanced 不使用独立下载档位。`model.base_dir` 控制模型根目录，下载前设置；完成标记和必需文件共同决定模型是否就绪。重复下载会利用 provider 缓存；不应手动创建完成标记。

`local` 模式只使用就绪的本地模型，缺失时报错，不自动下载。显式执行 `models download` 是下载操作，即使配置为 `local`，该命令也会临时采用自动远端模型源。

## VLM 服务与手动 MLX

`model.vlm.server_url` 配置已有 VLM 服务时，优先使用该服务，不要求本地 VLM 权重；它是模型推理接口，不是 MinerU V1 文档解析 API。小模型依赖仍由所选档位决定。

MLX 需手动安装 `mlx-vlm>=0.7.0,<0.8.0` 并显式设置 `model.vlm.engine: mlx`。`model.stack`、`MINERU_MODEL_STACK` 和 `--stack` 已移除。修改配置后重启相关服务。

## 下载问题

Hugging Face 默认使用 `hf_xet`。如果网络无法访问 Xet CAS，可在下载前设置 `HF_HUB_DISABLE_XET=1` 使用普通 HTTP；也可切换 ModelScope。离线部署先完成下载和验证，再设置 `local`。更多安装组合见[扩展模块](../quick_start/extension_modules.md)。
