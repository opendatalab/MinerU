# mineru-kit vlm-server

状态: Draft
读者: 服务部署者、核心开发者、VLM backend 集成开发者
范围: `mineru-kit vlm-server` 的定位、协议范围、与 `api-server` 的边界和命令参数
非目标: 完整 Parse API；通用聊天能力说明；底层 engine 私有参数表
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit vlm-server` 是未来唯一正式的本地 VLM 服务启动入口。

它部署的是与 `mineru.net` chat API 同类的 VLM 模型，但模型主要面向文档理解，经过专用微调，因此更适合：

- OCR
- 布局理解
- 页面理解
- 文档局部问答 / 提取

它不承担通用聊天能力承诺。

## 2. 与 api-server 的边界

`mineru-kit vlm-server` 可作为 `mineru-kit api-server` 的后端，尤其服务于：

- `vlm-http-client`
- `hybrid-http-client`

边界:

| 命令 | 主要职责 |
|------|----------|
| `mineru-kit api-server` | 完整文档解析 API，处理 Files / Uploads / Parse Jobs |
| `mineru-kit vlm-server` | VLM 推理服务，处理 OpenAI-compatible chat 请求 |

`vlm-server` 不处理：

- Parse Jobs
- Files / Uploads
- doclib API

## 3. 协议范围

当前稳定提供：

- `GET /v1/health`
- `GET /v1/models`
- `POST /v1/chat/completions`

当前不把 `/v1/responses` 作为稳定承诺。

某些底层 engine 版本可能支持 `/v1/responses`，但 `mineru-kit vlm-server` 不以它作为统一契约。

## 4. 协议兼容语义

`mineru-kit vlm-server` 兼容 OpenAI Chat Completions 协议。

这里的“兼容”表示：

- 请求 / 响应结构兼容
- 可以复用 OpenAI-style client 和 serving stack

不表示：

- 提供通用聊天产品语义
- 提供开放域通用助手能力

## 5. Usage

统一参数只有一个：

```bash
mineru-kit vlm-server --engine auto
mineru-kit vlm-server --engine vllm
mineru-kit vlm-server --engine lmdeploy
mineru-kit vlm-server --engine mlx
```

`--engine` 决定使用哪类底层 serving engine。当前合法值为 `auto`、`vllm`、`lmdeploy`、`mlx`；`auto` 按 vLLM、LMDeploy、MLX-VLM 的顺序选择已安装且可用的 engine。

除统一参数外，其余参数原样透传到底层 engine server。

## 6. 与旧入口的迁移关系

`mineru-kit vlm-server` 替代：

- `mineru-vllm-server`
- `mineru-lmdeploy-server`
- `mineru-openai-server`

未来正式入口统一为：

```bash
mineru-kit vlm-server
```

完整设计背景见 [ADR-0018](../decisions/0018-mineru-kit-vlm-server-command.md)。
