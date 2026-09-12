# 档位与运行环境

档位表达解析质量和成本；小模型后端与 VLM 引擎是独立配置，不是档位名称。

| 档位 | 主要用途 | 本地执行 |
| --- | --- | --- |
| `flash` | 快速预览、索引、原生文档解析 | 原生 PDF 文本 / 文档解析；扫描 PDF 与图片使用 Flash OCR |
| `basic` | 基础 OCR、公式和表格解析 | ONNX 或 Torch 小模型，不需要本地 VLM |
| `standard` | 复杂版面与高质量解析 | 小模型 + VLM |
| `advanced` | 更高质量需求 | 与 Standard 共用模型与运行环境，投入更多推理计算 |

PDF 和图片可选四档。Office、OpenDocument、RTF、EPUB、OFD、HTML、CSV/TSV 固定走本地 Flash，整本解析；Python 本地 SDK 调用这些格式时显式传 `tier="flash"`。纯文本不进入解析，但可由文档库索引和读取。

## 默认选择

- `mineru-kit parse` 和本地 Python `parse()` 的 PDF 默认档位为 Standard。
- 文档库发现服务能力后，PDF/图片默认优先 Standard，其次 Basic；不可用时报 `quality_tier_unavailable`，不会静默降为 Flash 或上传到官网。
- Advanced 需要显式选择；本地托管服务按 Standard 准备模型和启动能力，不单独下载 Advanced 模型。
- 文档库 `read` 读取已有缓存，不发起新解析；未指定档位时优先已有的 Advanced、Standard、Basic 结果。

## 后端与引擎

| 环境 | 自动选择的小模型 | 自动选择的 VLM |
| --- | --- | --- |
| Apple Silicon / MPS | Torch / MPS | llama.cpp |
| Linux / Windows，基础包 | ONNX / CPU（没有可用 Torch 加速环境时） | llama.cpp |
| Linux，有加速器并安装 `full` | Torch | vLLM；否则尝试已安装的 LMDeploy |
| Windows，有加速器并安装 `full` | Torch | LMDeploy |
| CPU 环境 | ONNX / CPU | llama.cpp |

自动选择依据已安装依赖和可用设备；显式指定的后端缺依赖时会报错，不会静默改用另一种。ONNX 小模型在 CPU 上执行，VLM 的设备选择独立。

```yaml
model:
  small_backend: auto
  vlm:
    engine: auto
```

`small_backend` 可选 `auto/onnx/torch`；`engine` 可选 `auto/llama-cpp/vllm/lmdeploy/mlx`。macOS 不自动选择 MLX，需手动安装并显式配置。

## 资源与平台边界

原生文本解析不需要推理模型。Basic 可在 CPU 运行；Standard / Advanced 的速度和内存占用取决于 VLM 引擎、输入规模和并发。高吞吐本地部署建议至少 16 GB 系统内存；NVIDIA 部署还需要所选引擎支持的显卡、驱动和足够可用显存，8 GB 仅作为规划起点，不是所有任务的保证。

Apple Silicon 建议使用至少 16 GB 统一内存并直接在 macOS 安装。既有 AMD 和国产卡适配继续使用 [MinerU <4](compatibility.md)，不属于本文的 4.0 平台兼容承诺。

安装组合见[扩展模块](../quick_start/extension_modules.md)，下载与配置见[模型源](model_source.md)。
