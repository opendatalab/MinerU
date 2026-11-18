# 快速入门

安装 MinerU 4.x 后，用一个小文件验证本地解析并查看生成的 Markdown。

## 安装 MinerU

MinerU 包支持 Python `>=3.11,<3.15`。建议新建 Python 3.12 环境；vLLM、LMDeploy、Torch 等可选引擎还受各自 wheel、操作系统和驱动约束，包支持的 Python 范围不等于所有引擎均支持该范围。

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

安装最新已发布的 4.x 基础包：

```bash
uv pip install -U "mineru>=4.0,<5"
```

也可在已激活的环境中使用 `python -m pip install -U "mineru>=4.0,<5"`。中国大陆可给安装命令添加 `-i https://mirrors.aliyun.com/pypi/simple`。

基础包包含 WebUI、ONNX 小模型运行时和 llama.cpp VLM；Apple Silicon 上会自动安装 Torch 依赖。需要 Torch 小模型或高吞吐 VLM 时，按[扩展模块](extension_modules.md)安装 `torch` 或 `full`，不要使用 3.x 的 extras。

需要确认设备或准备推理模型时，参见[部署与运维](../deployment/index.md)。

## 第一次解析

在一个空目录中创建一份小文档，用无状态 CLI 将它转换为 Markdown。这个示例使用原生 HTML 解析，无需下载模型。

```bash
python -c "from pathlib import Path; Path('report.html').write_text('<h1>MinerU</h1><p>First conversion.</p>', encoding='utf-8')"
mineru-kit parse report.html -o report.md --tier flash
```

打开当前目录的 `report.md`，可以看到标题 `MinerU` 和正文 `First conversion.`。转换自己的 PDF 时，在同一条命令中换用 PDF 路径和所需档位。

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

想保存结果、搜索或继续阅读时，可用上面创建的 `report.html` 跟随[文档库与 Agent 阅读](../usage/quick_usage.md)启动本地服务并完成首次入库。文档库命令 `mineru parse` 默认请求 PDF 的前 10 页。

### 文档库遥测

Doclib 提供使用与诊断遥测，初始设置为启用。可通过 `mineru telemetry status`、`preview` 查看，使用 `mineru telemetry disable` 停止采集和发送；也可在服务启动前设置 `MINERU_TELEMETRY_DISABLED=1`。遥测不包含文档内容、文件名和路径，范围与设置见[遥测说明](../../next/telemetry-minimum.md)。

## WebUI 与 API

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
```

打开[本地 WebUI](http://127.0.0.1:7860)，按 [WebUI 操作指南](../usage/webui.md)上传文件、对照原文和下载结果。若要在应用中调用解析服务，可单独启动 API。

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

在 [OpenAPI 文档](http://127.0.0.1:8000/docs)查看当前服务接口。按[第一次 HTTP 请求](../usage/http_api.md)下载示例结果，异步批次的字段见 [Tasks](../usage/tasks.md)；需要上传单文件并直接返回附件时，另行启动 [file-parse 服务](../usage/file_parse.md)。

## 源码安装

在对应版本源码的仓库根目录执行：

```bash
uv pip install -e .
mineru version --json
```

确认输出版本为 4.x；源码开发环境可显示预发布版本。NVIDIA 容器部署见[Docker 部署](docker_deployment.md)。

## 继续阅读

- [档位与运行环境](../usage/tiers.md)
- [模型下载与配置](../usage/model_source.md)
- [3.x → 4.0 迁移](../reference/migration_4.md)
- [4.0.x → 4.1 迁移](../reference/migration_4_1.md)
- [官网](https://mineru.net/)与[在线演示](https://www.modelscope.cn/studios/OpenDataLab/MinerU)
