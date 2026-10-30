# 档位与运行环境

档位表达解析质量和成本；小模型后端与 VLM 引擎是独立配置，不是档位名称。

本文包含 4.1 设备选择与服务档位行为；已有部署的升级步骤见 [4.1 迁移说明](../reference/migration_4_1.md)。

| 档位 | 主要用途 | 本地执行 |
| --- | --- | --- |
| `flash` | 快速预览、索引、原生文档解析 | 原生 PDF 文本 / 文档解析；扫描 PDF 与图片使用 Flash OCR |
| `basic` | 基础 OCR、公式和表格解析 | ONNX 或 Torch 小模型，不需要本地 VLM |
| `standard` | 复杂版面与高质量解析 | 小模型 + VLM |
| `advanced` | 更高质量需求 | 与 Standard 共用模型与运行环境，投入更多推理计算 |

PDF 和图片可选四档。Office、OpenDocument、RTF、EPUB、OFD、HTML/MHTML、CSV/TSV 固定走本地 Flash，整本解析；Python 本地 SDK 调用这些格式时显式传 `tier="flash"`。MHTML 网页归档支持 `.mhtml`、`.mht`，按整份文档解析。纯文本不进入解析，但可由文档库索引和读取。

解析保存的网页归档：

```bash
mineru-kit parse saved-page.mhtml -o saved-page.md --tier flash
```

## 默认选择

- `mineru-kit parse` 和本地 Python `parse()` 的 PDF 默认档位为 Standard。
- 文档库发现服务能力后，PDF/图片默认优先 Standard，其次 Basic；不可用时报 `quality_tier_unavailable`，不会静默降为 Flash 或上传到官网。
- Tasks 与 Gradio 默认按 Standard → Basic → Flash → Advanced 选档，仅开放 Advanced 时自动选择它；Advanced 与 Standard 共用模型，不单独下载一套 Advanced 权重。服务需开放 Advanced 才能接受相应 PDF/图片请求。
- 文档库 `read` 读取已有缓存，不发起新解析；未指定档位时优先已有的 Advanced、Standard、Basic 结果。

API 启动 `--tier` 表示开放上限，包含自身和较低档位；`--tier standard` 不开放 Advanced。省略或 `--tier auto` 按实际后端和硬件推荐，ONNX 路径可能在启动时执行设备校准。HTTP 请求省略 tier 时按 Standard → Basic → Flash → Advanced 选择；原生文档固定 Flash，不受 PDF/图片开放档位限制。完整规则见 [Tasks](tasks.md)。

## 后端与引擎

| 环境 | 自动选择的小模型 | 自动选择的 VLM |
| --- | --- | --- |
| Apple Silicon / MPS | Torch / MPS | llama.cpp / Metal |
| Windows x64，基础包 | ONNX / DirectML 或 CPU（没有可用 Torch 加速环境时） | llama.cpp |
| Linux x86_64，基础包 | ONNX / WebGPU 或 CPU（没有可用 Torch 加速环境时） | llama.cpp |
| Linux，有加速器并安装 `full` | Torch | vLLM；否则尝试已安装的 LMDeploy |
| Windows，有加速器并安装 `full` | Torch | LMDeploy |
| CPU 环境 | ONNX / CPU | llama.cpp |

自动选择依据已安装依赖和可用设备；显式指定的后端缺依赖时会报错，不会静默改用另一种。ONNX 设备由 `model.onnx_device=auto|cpu|gpu` 控制，VLM 设备独立选择。环境变量 `MINERU_MODEL_ONNX_DEVICE` 高于文件配置，且不会将覆盖值写回文件。

4.1 基础包在 Windows x64 安装 DirectML 运行时，在 Linux x86_64 安装 WebGPU 插件（经 Vulkan 执行）。兼容且确认的独显优先；auto 下的集显或类型未知设备按需校准，收益不足或运行不可靠时使用 CPU。Linux 的旧 Intel/Xe-LP 及未识别架构使用 CPU；显式 gpu 也遵守兼容性限制。其他系统、GPU 不可用或初始化失败时记录 CPU 回退原因。

有效自动选择可保存到配置；重新测速需将 `model.onnx_device` 改回 auto 并重启。设备变化后旧校准证据不再视为当前性能证据，显式或已保存的 cpu 不会因为后来接入独显而自动改写。首次校准可能下载识别模型；`hardware info` 不触发此过程。

ONNX 栈的 layout、OCR rec、MFR 和 UNet 共用设备选择；普通 OCR det 在 Windows 走 CPU、Linux 可走 WebGPU，印章检测、表格分类和 SLANet Plus 走 CPU。MFR 的 GPU 路径仅处理满 8 批次，其余尾批使用 CPU。UNet 不受旧 `MINERU_TABLE_DEVICE` 覆盖；Torch 栈的表格保留其已有 CPU/CUDA 策略。无需为启用这条 ONNX 路径手动替换为 `onnxruntime-gpu`。

Intel XPU 不会自动选择 LMDeploy。Linux 上安装了支持 XPU 的 vLLM 时优先使用 vLLM，否则回退 llama.cpp；Windows 上使用 llama.cpp。

```yaml
model:
  small_backend: auto
  onnx_device: auto
  vlm:
    engine: auto
```

`small_backend` 可选 `auto/onnx/torch`；`engine` 可选 `auto/llama-cpp/vllm/lmdeploy/mlx`。macOS 默认使用 llama.cpp。

```bash
mineru-kit hardware detect --json
mineru-kit hardware info --json
```

detect 与 API 自动启动共用推荐逻辑，可能执行适用的 ONNX 校准；info 只读采集硬件、已安装运行时、有效配置和校准记录，不测速、不下载权重、不写配置。

API/WebUI 的自动 tier 采用小模型后端的设备结论：GPU 开放全部四档；CPU 可用逻辑核至少 12 时开放到 Standard，4–11 核开放到 Basic，不足 4 核只开放 Flash。自动推荐不查询显存容量。开放 Standard 时默认请求 Standard，CPU 不自动开放 Advanced。显式启动上限跳过自动推荐。

## 资源与平台边界

原生文本解析不需要推理模型。Basic 可在 CPU 运行；Standard / Advanced 的速度和内存占用取决于 VLM 引擎、输入规模和并发。高吞吐本地部署建议至少 16 GB 系统内存；NVIDIA 部署还需要所选引擎支持的显卡、驱动和足够可用显存，8 GB 仅作为规划起点，不是所有任务的保证。

Apple Silicon 建议使用至少 16 GB 统一内存并直接在 macOS 安装。4.1 的 Windows/Linux ONNX 路径支持兼容的 Intel、AMD、NVIDIA GPU；国产卡的旧适配教程仍以 [MinerU <4](compatibility.md) 为适用范围。

安装组合见[扩展模块](../quick_start/extension_modules.md)，下载与配置见[模型源](model_source.md)。
