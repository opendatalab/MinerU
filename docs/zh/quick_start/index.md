# 快速入门

本指南面向 MinerU **4.0 正式版**。AMD 和国产加速卡的既有适配保持在 `<4`，请使用[旧平台指南](../usage/compatibility.md)。

## 安装 MinerU

MinerU 包支持 Python `>=3.10,<3.15`。建议新建 Python 3.12 环境；vLLM、LMDeploy、Torch 等可选引擎还受各自 wheel、操作系统和驱动约束，包支持的 Python 范围不等于所有引擎均支持该范围。

```bash
uv venv --python 3.12 .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装基础包：

```bash
uv pip install -U "mineru>=4.0,<5"
```

也可在已激活的环境中使用 `python -m pip install -U "mineru>=4.0,<5"`。中国大陆可给安装命令添加 `-i https://mirrors.aliyun.com/pypi/simple`。

基础包包含 WebUI、ONNX 小模型运行时和 llama.cpp VLM；Apple Silicon 上会自动安装 Torch 依赖。需要 Torch 小模型或高吞吐 VLM 时，按[扩展模块](extension_modules.md)安装 `torch` 或 `full`，不要使用 3.x 的 extras。

## 第一次解析

无状态解析单个文件，将 Markdown 写入指定文件：

```bash
mineru-kit parse document.pdf -o document.md --tier standard
```

首次使用模型可能需要下载权重。无模型的原生 PDF 文本提取可显式选择：

```bash
mineru-kit parse document.pdf -o document.md --tier flash --ocr-mode txt
```

该命令使用 PDF 文本层，不会为扫描页追加 OCR 回退。扫描 PDF 应使用 `--ocr-mode ocr` 并准备对应模型。

原生文档解析与批量转换：

```bash
mineru-kit parse report.docx -o report.md --tier flash
mineru-kit parse ./documents -o ./output --format zip
```

目录和多文件输入的 `-o` 必须是目录；单文件 Markdown 输出使用文件路径。`mineru-kit parse` 默认处理全部 PDF 页。

## 文档库与 Agent

```bash
mineru parse document.pdf --json
mineru parse document.pdf --pages all -o document.md
mineru search "关键词" --json
```

`mineru` 使用文档库和缓存。`mineru parse` 默认读取 PDF 前 10 页，可能按输出长度返回继续阅读请求；按返回的 locator 和 `next_request` 继续。它与直接输出完整文件的 `mineru-kit parse` 是不同工作流，详见[基础使用](../usage/quick_usage.md)。

## WebUI 与 API

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
```

在浏览器打开 [本地 WebUI](http://127.0.0.1:7860)。`mineru-webui` 是相同入口的独立命令。未指定 `--api-url` 时，WebUI 托管本地 V1 API 服务；连接已有服务见[SDK 与 API](../usage/sdk_api.md)。

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

在 [OpenAPI 文档](http://127.0.0.1:8000/docs)查看当前服务接口。

## 源码安装

在包含 4.0 源码的仓库根目录执行：

```bash
uv pip install -e .
mineru version --json
```

确认输出版本为 4.x；源码开发环境可显示预发布版本。NVIDIA 容器部署见[Docker 部署](docker_deployment.md)。

## 继续阅读

- [档位与运行环境](../usage/tiers.md)
- [模型下载与配置](../usage/model_source.md)
- [3.x → 4.0 迁移](../reference/migration_4.md)
- [官网](https://mineru.net/)与[在线演示](../demo/index.md)
