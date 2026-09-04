# 命令行工具

MinerU 提供两个主要命令树：`mineru` 是面向交互和 Agent 工作流的文档库客户端，`mineru-kit` 提供无状态解析、服务、模型、Router 和 WebUI 工具。

## 文档库 CLI

使用 `mineru --help` 查看全部文档库命令。最常见的解析方式是：

```bash
mineru parse report.pdf --pages all -o report.md
```

未指定 `-o/--output` 时，渲染结果写入标准输出。PDF 默认解析前 10 页；使用 `--pages all` 解析整份文档。文档库负责入库、缓存、后台解析、阅读、搜索和清理。

使用以下命令管理本地文档库服务：

```bash
mineru server start
mineru server status
mineru server stop
```

各命令的权威参数以 `mineru <command> --help` 为准。

## 无状态与服务工具

使用 `mineru-kit --help` 查看所有工具。

### 批量解析

```bash
mineru-kit parse report.pdf -o report.md --tier standard
mineru-kit parse ./documents -o ./output --format zip
```

`mineru-kit parse` 不使用文档库数据库或缓存，支持本地解析和显式 V1 远程解析。tier、backend、页范围和输出参数以 `mineru-kit parse --help` 为准。

### V1 API Server

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

在浏览器打开 `http://127.0.0.1:8000/docs` 查看自动生成的 OpenAPI 文档。当前服务接口统一位于 `/v1/*`；已删除的 `/file_parse` 和 `/tasks` 不再提供。

### Gradio WebUI

```bash
mineru-kit gradio --server-name 127.0.0.1 --server-port 7860
```

未传 `--api-url` 时，Gradio 会托管 loopback `mineru-kit api-server`；传入后只连接指定的 V1 服务。`mineru-gradio` 保留为命令名兼容别名，接受相同的新版参数，不恢复旧 Gradio 参数或 HTTP 路由。

### Router 与 VLM Server

```bash
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus auto
mineru-kit vlm-server --engine auto --port 30000
```

`mineru-router` 保留为 `mineru-kit router` 的命令名别名。Router 只暴露 V1 API，并且只接受文档中声明的 worker 参数。

## 环境变量

- `MINERU_HOME`：MinerU 配置、缓存和文档库状态的根目录。
- `MINERU_CONFIG`：显式指定 `config.yaml` 路径。
- `MINERU_MODEL_SOURCE`：模型源，例如 `huggingface`、`modelscope` 或 `local`。
- `MINERU_API_URL` / `MINERU_API_KEY`：API 客户端默认使用的 V1 地址和 Bearer Key。
- `MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS`：Gradio 托管本地 V1 服务的启动超时，默认 `300` 秒。
- `MINERU_API_ENABLE_FASTAPI_DOCS`：是否为 V1 API Server 启用 `/docs`、`/openapi.json` 和 `/redoc`，默认 `true`。
- `MINERU_PDF_RENDER_TIMEOUT` / `MINERU_PDF_RENDER_THREADS`：PDF 渲染超时和 worker 数量。
- `MINERU_PROCESSING_WINDOW_SIZE`：大文档处理窗口大小。
- `MINERU_INTRA_OP_NUM_THREADS` / `MINERU_INTER_OP_NUM_THREADS`：ONNX 算子线程配置。

当前默认值以各命令的 `--help` 和[模型源说明](./model_source.md)为准。

## PDF 页码选择

使用 `--pages "1-5,8,r3-r1"`：页码从 1 开始，包含区间两端，`r1` 表示最后一页，`all` 表示全部。
结果去重并按原页序排列；部分越界取有效交集，倒序或选不到页面时返回 `page_range_invalid`。
省略页码时 `mineru parse` 默认前 10 页，`mineru-kit parse`、Python 和 Gradio 默认全部。
不再接受 `~` 和负数页码；升级调用方和服务端后，在新 Doclib 数据目录重建解析缓存。
完整说明见 [PDF 页码范围规范](../../next/page-ranges.md)。
