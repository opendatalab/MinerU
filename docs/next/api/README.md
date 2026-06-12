# API 总览

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 统一 API 的资源模型、端点地图、认证和通用约定
底稿: `../../../NEXT-API.md`

## 设计目标

MinerU v1 API 以 `mineru.net` 官方远程 API 为主线，同时让 Local Parse Server 尽量复用同一套客户端代码。官方 API 提供完整的公网能力，本地 server 负责在可信环境中提供近似一致的解析入口。

本目录是可实现规格。服务端开发者应能根据每个 endpoint 的请求字段、响应字段、错误和本地差异实现 API；客户端和 SDK 开发者应能据此构造请求、解析响应并处理失败。

## 资源模型

| 资源 | 含义 |
|------|------|
| `file` | 平台中的文件，由不透明 `file_id` 标识。可表示用户上传的源文件、Chat/Responses 的图片输入，或解析产物。 |
| `upload` | OpenAI Uploads API 风格的上传会话，用来管理上传生命周期。 |
| `job` | 一次解析任务，可包含一个或多个输入文件。 |

ID 使用 OpenAI 风格的前缀加随机串。官方 API 使用 24 字符 base62 随机串；本地 server 可以用同长度 hex 随机串降低实现复杂度。ID 是不透明标识，客户端不得从中推断时间、租户或存储路径。

| 资源 | 示例 |
|------|------|
| `file` | `file-r9NSmHLJE6flShV5vQ0Y60Rd` |
| `upload` | `upload_r9NSmHLJE6flShV5vQ0Y60Rd` |
| `job` | `job_r9NSmHLJE6flShV5vQ0Y60Rd` |

## Base URL

| 部署形态 | Base URL |
|----------|----------|
| 官方远程 API | `https://mineru.net/api` |
| Local Parse Server | `http://localhost:8000/api` |

所有 endpoint 都带 `/v1` 版本前缀，例如 `GET https://mineru.net/api/v1/health`。

官方远程 API 在相当长时间内只提供 `pro` 解析。Local Parse Server 默认监听 loopback；如需对局域网或公网暴露，必须由部署者显式配置 host、防火墙和认证。

## Endpoint 地图

| Method | Path | 用途 |
|--------|------|------|
| `GET` | `/v1/health` | 健康检查 |
| `GET` | `/v1/models` | 列出模型 |
| `GET` | `/v1/models/{model}` | 查询单个模型 |
| `GET` | `/v1/tiers` | 列出解析档位 |
| `POST` | `/v1/uploads` | 创建 upload |
| `POST` | `/v1/uploads/{upload_id}/complete` | 完成 upload |
| `POST` | `/v1/uploads/{upload_id}/cancel` | 取消 upload |
| `GET` | `/v1/uploads/{upload_id}` | 查询 upload |
| `GET` | `/v1/files` | 列出文件 |
| `GET` | `/v1/files/{file_id}` | 查询文件元信息 |
| `GET` | `/v1/files/{file_id}/content` | 下载解析产物 |
| `DELETE` | `/v1/files/{file_id}` | 删除文件 |
| `POST` | `/v1/parse/jobs` | 创建解析任务 |
| `GET` | `/v1/parse/jobs/{job_id}` | 查询任务 |
| `GET` | `/v1/parse/jobs/{job_id}/events` | SSE 任务事件 |
| `GET` | `/v1/parse/jobs` | 列出任务 |
| `DELETE` | `/v1/parse/jobs/{job_id}` | 取消任务 |
| `POST` | `/v1/chat/completions` | Chat Completions 文档对话 |
| `POST` | `/v1/responses` | Responses 文档对话 |
| `GET` | `/v1/usage` | 查询用量 |

## 认证

官方 API 使用 Bearer Token:

```http
Authorization: Bearer <MINERU_API_KEY>
```

不传 API Key 时视为匿名访问。匿名访问不是鉴权失败，而是进入 anonymous access level。传入无效 API Key 时返回 `401 invalid_api_key`。

Local Parse Server 默认监听 loopback 且不启用鉴权。启用 `--api-key` 后，除 `GET /v1/health`、`GET /v1/models`、`GET /v1/models/{model}`、`GET /v1/tiers` 外，其余 endpoint 都要求 Bearer Token。通过鉴权的本地请求视为 registered 能力，不再按公网 access level 做差异化售卖。

## 通用约定

- 成功响应直接返回资源对象，没有外层 envelope。
- 错误响应使用 OpenAI-compatible envelope，详见 [响应与错误](responses.md)。
- `Content-Type` 默认为 `application/json; charset=utf-8`，上传字节流除外。
- 时间字段按 endpoint 语义选择 ISO-8601 UTC 字符串或 Unix 秒级时间戳。
- 成功和失败响应都应携带 `X-Request-Id`。
- `POST` endpoint 支持 `Idempotency-Key`，相同 key 在有效期内返回首次请求结果。
- 未知响应字段必须被客户端忽略，以支持服务端向后兼容扩展。
- 请求中的可选字段可以省略；省略时按该字段默认值处理。
- 请求中的未知字段由各 endpoint 决定是忽略还是返回 `400 invalid_request`；安全相关字段必须显式校验。

## 本地 Server 差异

Local Parse Server 的目标是复用客户端协议，而不是完整模拟公网平台。它可以简化以下能力:

- 不支持 Webhook。
- 不支持 Parse Job SSE 事件流时，客户端使用轮询。
- 上传 URL 可以指向本地 server 的临时上传 endpoint，而不是 OSS。
- 文件产物下载可以直接返回 `200 + body`，不必返回 CDN 重定向。
- 支持 `local` source，由 server 直接读取 allowlist 内的本地路径。
