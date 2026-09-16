# ADR-0018: MinerU Kit VLM Server Command

状态: Accepted
日期: 2026-06-17
相关文档:
- ../cli/mineru-kit.md
- ../cli/mineru-kit-vlm-server.md
- ../api/chat.md

## 背景

`mineru-kit vlm-server` 负责本地部署 VLM 模型服务。它与 `mineru-kit api-server` 的边界需要保持清楚：

- `mineru-kit api-server` 面向完整文档解析 API，负责 Files / Uploads / Parse Jobs 等资源语义。
- `mineru-kit vlm-server` 面向 VLM 推理能力本身，可作为 `mineru-kit api-server` 的后端，尤其服务于 `vlm-http-client` / `hybrid-http-client`。

现有仓库里还保留了 `mineru-vllm-server`、`mineru-lmdeploy-server`、`mineru-openai-server` 等旧入口，需要收敛到一个正式命令。

同时，需要明确它提供哪些 path、它是否承诺普通聊天能力，以及它如何处理不同 engine 的参数。

## 决策

### 1. 命令定位

`mineru-kit vlm-server` 是未来唯一正式的本地 VLM 服务启动入口。

它部署的是与 `mineru.net` chat API 同类的 VLM 模型，但该模型主要面向文档理解，经过专用微调，因此：

- 擅长 OCR、布局理解、页面理解、文档局部问答
- 不承担通用聊天能力承诺

### 2. 与 api-server 的边界

#### `mineru-kit api-server`

- 面向完整文档解析
- 提供 v1 parse API
- 管理文件、任务和解析产物
- 可把 VLM 推理委托给 `vlm-server`

#### `mineru-kit vlm-server`

- 面向 VLM 推理本身
- 提供 OpenAI-compatible chat 接口
- 不处理 Parse Jobs / Files / Uploads
- 不处理 doclib API

### 3. 协议范围

`mineru-kit vlm-server` 当前稳定提供：

- `GET /v1/health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`/v1/responses` 不作为当前稳定承诺。

原因：

- 部分底层 engine 可能支持 `/v1/responses`
- 但 `mineru-kit vlm-server` 不把它作为统一对外契约

### 4. 协议兼容语义

`mineru-kit vlm-server` 兼容 OpenAI Chat Completions 协议。

但这种兼容只表示：

- 请求/响应形态兼容
- 可复用现有 OpenAI-style client 和 serving stack

不表示：

- 提供通用大语言模型聊天体验
- 覆盖完整 OpenAI 产品语义

### 5. 命令参数

`mineru-kit vlm-server` 只稳定一个统一参数：

```bash
--engine auto|vllm|lmdeploy|sglang|mlx
```

规则：

1. `--engine` 决定使用哪类底层 serving engine
2. 其它参数不在 `mineru-kit vlm-server` 这一层统一定义
3. 除统一参数外，其余参数原样透传到底层 engine server

### 6. 与旧入口的迁移关系

`mineru-kit vlm-server` 替代以下旧入口：

- `mineru-vllm-server`
- `mineru-lmdeploy-server`
- `mineru-openai-server`

未来正式入口统一为：

```bash
mineru-kit vlm-server
```

## 替代方案

### 1. 把 `vlm-server` 设计成通用聊天服务

没有采用。当前模型主要面向文档理解，基础通用聊天能力较弱，不应在产品语义上承诺通用聊天。

### 2. 在 `mineru-kit vlm-server` 这一层重新定义完整稳定参数面

没有采用。不同底层 engine 的参数差异较大，当前阶段只统一 `--engine` 更清楚，也更容易保持和底层 runtime 对齐。

### 3. 把 `/v1/responses` 一并纳入稳定承诺

没有采用。不同 engine 的支持程度不一致，当前只把 `/v1/chat/completions` 作为稳定协议面。

## 影响

- `vlm-server` 与 `api-server` 的职责分层更清楚。
- 本地 VLM 服务的正式入口收敛为单一命令。
- `api-server` 可稳定把 `vlm-server` 作为 http backend。
- 未来如需扩大协议面（如稳定支持 `/v1/responses`），应作为增量设计处理。

## 后续动作

- 新增 `docs/next/cli/mineru-kit-vlm-server.md`
- 更新 `docs/next/cli/mineru-kit.md`
- 更新 `docs/next/cli/README.md`
- 更新 `docs/next/cli.md`
- 更新 `docs/next/cli/`
- 更新 `docs/next/decisions/README.md`
