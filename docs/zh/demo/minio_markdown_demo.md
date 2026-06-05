# MinIO Markdown Demo 执行指南

本文说明如何执行 [demo/minio_markdown_demo.py](/Users/perryhe/Projects/MinerU/demo/minio_markdown_demo.py)，以及如何为图片解释环节接入 VLM。

## 1. 这份 Demo 做什么

脚本执行链路如下：

1. 本地文件上传到 MinIO。
2. 从 MinIO 回读原文件字节，交给 MinerU 解析。
3. 将解析产物重新上传到 MinIO。
4. 将 Markdown/JSON 中的相对图片路径改写成 MinIO HTTP URL。
5. 可选：把 Markdown 中的图片再从 MinIO 下载下来，发送给 OpenAI-compatible VLM 生成图片解释。
6. 将图片解释以引用块形式拼接回 Markdown。
7. 将增强后的 Markdown 再写回 MinIO，并下载到本地目录。

## 2. 两类“VLM”不要混淆

这个 Demo 里存在两层能力：

- 文档解析 backend
  - 由 `--backend` 控制。
  - 默认是 `hybrid-auto-engine`。
  - 这是 MinerU 自己的解析阶段。

- 图片解释 VLM
  - 由 `--vlm-base-url`、`--vlm-model` 等参数控制。
  - 这是 Markdown 生成之后的二次增强阶段。
  - 作用是“从 MinIO 下载图片，再生成解释并回填 Markdown”。

也就是说，即使 `--backend` 仍然是 `hybrid-auto-engine`，你依然可以单独接入一个外部 VLM 来做图片解释。

## 3. 前置条件

需要满足以下条件：

- 已进入项目根目录：
  - `/Users/perryhe/Projects/MinerU`
- 已准备好 Python 环境。
- 可访问 MinIO。
- 输入文件是 MinerU 支持的类型之一：
  - PDF
  - 图片
  - DOCX

如果要启用图片解释，还需要：

- 当前环境已安装 `openai` Python 包。
- 有一个 OpenAI-compatible VLM 服务可访问。
- 知道该服务的：
  - `base_url`
  - `model`
  - `api_key`（如果服务不校验，可传任意占位值，例如 `EMPTY`）

## 4. MinIO 配置方式

脚本按以下优先级读取 MinIO 配置：

1. CLI 参数
2. 环境变量
3. `mineru.json` 中对应 bucket 的配置

可用 CLI 参数：

- `--minio-url`
- `--minio-ak`
- `--minio-sk`
- `--bucket`
- `--input-prefix`
- `--output-prefix`

可用环境变量：

- `MINIO_URL`
- `MINIO_AK`
- `MINIO_SK`

示例：

```bash
export MINIO_URL=http://127.0.0.1:9000
export MINIO_AK=minioadmin
export MINIO_SK=minioadmin
```

## 5. 图片解释 VLM 接入方式

脚本支持通过 CLI 或环境变量配置图片解释 VLM。

CLI 参数：

- `--vlm-base-url`
- `--vlm-api-key`
- `--vlm-model`
- `--vlm-prompt`
- `--vlm-timeout`
- `--vlm-max-tokens`

环境变量：

- `VLM_BASE_URL`
- `VLM_API_KEY`
- `VLM_MODEL`

兼容兜底环境变量：

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

脚本只会在同时拿到以下两个信息时启用图片解释：

- `base_url`
- `model`

否则会直接跳过图片解释阶段。

## 6. 推荐接法

### 方案 A：接 MinerU 自带 OpenAI-compatible 服务

如果你已经在另一台机器或本机启动了兼容服务，例如：

```bash
mineru-openai-server --port 30000
```

则图片解释阶段可按 OpenAI-compatible 方式接入。通常建议显式传 `/v1` 路径，例如：

```bash
export VLM_BASE_URL=http://127.0.0.1:30000/v1
export VLM_API_KEY=EMPTY
export VLM_MODEL=qwen2.5-vl
```

注意：

- `VLM_MODEL` 必须与你服务端实际加载的模型名一致。
- `VLM_BASE_URL` 应该填写 OpenAI SDK 可直接访问的根地址。

### 方案 B：接第三方 OpenAI-compatible 多模态服务

例如你的服务已经兼容 OpenAI Chat Completions，并支持 `image_url` 输入，则可以直接配置：

```bash
export VLM_BASE_URL=https://your-vlm-host.example.com/v1
export VLM_API_KEY=your_api_key
export VLM_MODEL=your_multimodal_model
```

这个 Demo 在请求时会把图片转成 `data:` URL，然后通过 `chat.completions.create(...)` 发送给模型。

## 7. 最小运行示例

### 7.1 不启用图片解释

这时只做：

- 上传 MinIO
- MinIO 回读解析
- 产物上传 MinIO
- Markdown 下载到本地

```bash
.venv/bin/python demo/minio_markdown_demo.py \
  demo/pdfs/demo1.pdf \
  --bucket mineru-bucket \
  --minio-url http://127.0.0.1:9000 \
  --minio-ak minioadmin \
  --minio-sk minioadmin \
  --backend hybrid-auto-engine \
  --language ch \
  --print-md
```

### 7.2 启用图片解释

```bash
export MINIO_URL=http://127.0.0.1:9000
export MINIO_AK=minioadmin
export MINIO_SK=minioadmin

export VLM_BASE_URL=http://127.0.0.1:30000/v1
export VLM_API_KEY=EMPTY
export VLM_MODEL=qwen2.5-vl

.venv/bin/python demo/minio_markdown_demo.py \
  demo/pdfs/demo1.pdf \
  --bucket mineru-bucket \
  --backend hybrid-auto-engine \
  --language ch \
  --vlm-prompt "请用中文解释这张文档图片的关键信息，控制在2句内。" \
  --vlm-max-tokens 200 \
  --print-md
```

### 7.3 全部通过 CLI 传入

```bash
.venv/bin/python demo/minio_markdown_demo.py \
  demo/pdfs/demo1.pdf \
  --bucket mineru-bucket \
  --minio-url http://127.0.0.1:9000 \
  --minio-ak minioadmin \
  --minio-sk minioadmin \
  --backend hybrid-auto-engine \
  --language ch \
  --vlm-base-url http://127.0.0.1:30000/v1 \
  --vlm-api-key EMPTY \
  --vlm-model qwen2.5-vl \
  --vlm-prompt "请结合上下文解释图片，不要复述无关内容。" \
  --vlm-timeout 120 \
  --vlm-max-tokens 200
```

## 8. 输出结果怎么看

脚本运行后会输出几类路径：

- 上传的原始文件 S3 URI
- 原始文件 HTTP URL
- 生成后的 Markdown S3 URI
- 生成后的 Markdown HTTP URL
- 本地 Markdown 路径

本地 Markdown 默认保存在：

```text
demo/minio_output/<task_id>/<doc_stem>.md
```

如果启用了图片解释，则图片后面会多出类似内容：

```md
![](http://127.0.0.1:9000/mineru-bucket/output/xxx/images/page-1.png)
> 图片解释：这张图片展示了系统整体流程，包括解析、上传、回读和回填四个阶段。
```

## 9. 常见问题

### 9.1 为什么没有生成图片解释

通常是以下原因之一：

- 没有传 `--vlm-base-url`
- 没有传 `--vlm-model`
- 对应环境变量未设置

脚本会直接跳过这一阶段。

### 9.2 为什么报 `ModuleNotFoundError: openai`

说明当前环境没有安装 `openai`，但你启用了图片解释。

先在当前环境安装依赖，再重新执行。

### 9.3 为什么图片解释请求失败

常见原因：

- `base_url` 不正确
- 漏了 `/v1`
- `model` 名称与服务端不一致
- 服务端不支持 `image_url`
- API key 不正确
- 服务端虽然兼容 OpenAI，但不支持 `data:` URL 图片输入

### 9.4 为什么同一张图会多次出现在 Markdown 中

脚本会对同一个图片 URL 做缓存，避免重复下载和重复请求 VLM；但如果同一张图在 Markdown 中出现多次，解释会按出现位置分别插回。

## 10. 建议的排查顺序

建议按这个顺序检查：

1. 先不启用 VLM，确认 MinIO 上传、解析、Markdown 下载链路正常。
2. 再加上 `VLM_BASE_URL` 和 `VLM_MODEL`。
3. 如果失败，先确认服务端是否支持 OpenAI-compatible 多模态输入。
4. 再检查 `base_url`、`model`、`api_key` 是否正确。

## 11. 当前脚本适用边界

当前 [demo/minio_markdown_demo.py](/Users/perryhe/Projects/MinerU/demo/minio_markdown_demo.py) 已支持：

- MinIO 输入/输出
- Markdown 图片 URL 改写
- 从 MinIO 回读图片
- 使用 OpenAI-compatible VLM 生成图片解释
- 将解释回填到 Markdown

当前脚本没有额外暴露文档解析阶段的 `server_url` 参数；因此它更适合：

- 本地解析
- 或使用当前默认 backend 做解析，再单独接外部 VLM 做图片解释

如果你后续希望“文档解析 backend”本身也走远程 `vlm-http-client` 或 `hybrid-http-client`，可以继续在这个脚本上增加 `--server-url` 透传。
