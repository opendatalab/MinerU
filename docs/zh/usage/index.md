# 使用 MinerU

## 命令行使用方式

### 基础用法

最简单的命令行调用方式如下：

```bash
mineru -p <input_path> -o <output_path>
```

- `<input_path>`：本地 PDF/图片 文件或目录（支持 pdf/png/jpg/jpeg/webp/gif）
- `<output_path>`：输出目录

### 查看帮助信息

获取所有可用参数说明：

```bash
mineru --help
```

### 参数详解

```text
Usage: mineru [OPTIONS]

Options:
  -v, --version                   显示版本并退出
  -p, --path PATH                 输入文件路径或目录（必填）
  -o, --output PATH               输出目录（必填）
  -m, --method [auto|txt|ocr]     解析方法：auto（默认）、txt、ocr（仅用于 pipeline 后端）
  -b, --backend [pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client]
                                  解析后端（默认为 pipeline）
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|latin|arabic|east_slavic|cyrillic|devanagari]
                                  指定文档语言（可提升 OCR 准确率，仅用于 pipeline 后端）
  -u, --url TEXT                  当使用 sglang-client 时，需指定服务地址
  -s, --start INTEGER             开始解析的页码（从 0 开始）
  -e, --end INTEGER               结束解析的页码（从 0 开始）
  -f, --formula BOOLEAN           是否启用公式解析（默认开启）
  -t, --table BOOLEAN             是否启用表格解析（默认开启）
  -d, --device TEXT               推理设备（如 cpu/cuda/cuda:0/npu/mps，仅 pipeline 后端）
  --vram INTEGER                  单进程最大 GPU 显存占用(GB)（仅 pipeline 后端）
  --source [huggingface|modelscope|local]
                                  模型来源，默认 huggingface
  --help                          显示帮助信息
```

---

## 模型源配置

MinerU 默认在首次运行时自动从 HuggingFace 下载所需模型。若无法访问 HuggingFace，可通过以下方式切换模型源：

### 切换至 ModelScope 源

```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

或设置环境变量：

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```

### 使用本地模型

#### 1. 下载模型到本地

```bash
mineru-models-download --help
```

或使用交互式命令行工具选择模型下载：

```bash
mineru-models-download
```

下载完成后，模型路径会在当前终端窗口输出，并自动写入用户目录下的 `mineru.json`。

#### 2. 使用本地模型进行解析

```bash
mineru -p <input_path> -o <output_path> --source local
```

或通过环境变量启用：

```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```

---

## 使用 sglang 加速 VLM 模型推理

### 通过 sglang-engine 模式

```bash
mineru -p <input_path> -o <output_path> -b vlm-sglang-engine
```

### 通过 sglang-server/client 模式

1. 启动 Server：

```bash
mineru-sglang-server --port 30000
```

2. 在另一个终端中使用 Client 调用：

```bash
mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://127.0.0.1:30000
```

> [!TIP]
> 更多关于输出文件的信息，请参考 [输出文件说明](../output_file.md)

---
