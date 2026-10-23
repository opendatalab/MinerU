# 响应与错误

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 成功响应、错误 envelope、请求追踪和幂等
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 成功响应

成功响应直接返回接口定义的对象，不增加统一外层 envelope。资源对象、分页列表及二进制附件是不同的返回类型，不能统一按 Task 对象解析。

| 接口 | 成功 HTTP | Content-Type | 正文 |
|---|---|---|---|
| `POST /v1/tasks` | 202 | `application/json` | `CreateTaskResponse`：task_id、status、files 创建回执。 |
| `GET /v1/tasks/{task_id}` | 200 | `application/json` | `TaskDetailResponse`，包括失败或取消后的完整详情。 |
| `POST /v1/tasks/{task_id}/cancel` | 200 | `application/json` | `CancelTaskResponse`：task_id、status、finished_at 和全部文件终态，保留成功产物链接。 |
| 本地 `PUT /v1/tasks/{task_id}/uploads/{file_id}` | 200 | `application/json` | `UploadFileResponse`：task_id、file_id 和该文件 status。对象存储上传接受其成功 2xx 响应。 |
| `GET /v1/files/{file_id}` | 200 | `application/json` | 完整 `FileObject`。 |
| `GET /v1/files/{file_id}/content` | 200（完整下载） | 随产物格式 | ZIP、DOCX 或 HTML 文件字节。 |
| 独立 `POST /v1/file_parse` | 200 / 202 | 附件 MIME / `application/json` | 等待成功返回附件；等待到期或零等待返回包装层 `StatusResponse`。 |
| 独立 `GET /v1/file_parse/{request_id}` | 200 | `application/json` | 包装层 `StatusResponse`，独立于 Tasks 的响应模型。 |
| 独立 `GET /v1/file_parse/{request_id}/result` | 200 | 随产物格式 | ZIP、DOCX 或 HTML 文件字节。 |

Tasks 各响应模型、字段省略规则及完整 JSON 见 [Tasks](tasks.md#响应模型)；Files 和下载响应见 [Files](files.md)。
Health、Models、Tiers 见 [能力发现接口](health-models.md)，用量见 [Usage](usage-limits.md)，独立 Chat/Responses 协议见 [文档对话](chat.md)。
独立包装服务的附件、202、状态查询和错误结构见 [file_parse 响应说明](file-parse.md#响应结构与格式)。
JSON 正文使用 UTF-8。Tasks 成功响应省略无值可选字段，保留 0、false、[] 和 {}；其他接口及非成功错误 envelope 沿用各自的 null 规则。任务取消使用 `POST /v1/tasks/{task_id}/cancel`；目前不提供任务记录或产物的公开删除接口。

当前 Python API Server 和 Router 不统一生成或返回 `X-Request-Id`；网关或公网部署提供请求追踪时，客户端可记录该响应头，不能把它作为上述共同响应的必带头。
独立 file_parse 的成功附件和 202 响应明确携带 `X-Request-ID`；其状态查询和错误响应不统一附加该头，以专页列出的条件为准。

列表响应使用 OpenAI 风格分页 envelope:

```json
{
  "object": "list",
  "data": [],
  "first_id": null,
  "last_id": null,
  "has_more": false
}
```

资源创建、状态变更和查询 endpoint 可以返回不同 HTTP 状态码表达异步状态:

| HTTP | 含义 |
|------|------|
| 200 | HTTP 操作成功，响应体是当前资源、列表或附件；不表示异步 Task 或所有文件均解析成功。 |
| 202 | Tasks 创建请求已接受，或 file_parse 等待结束/零等待后返回状态 JSON；具体任务状态以正文为准。 |
| 302 | 文件内容下载重定向到 CDN 或对象存储。 |

## 错误响应

主 API Server 和 Router 的错误响应采用下面的 OpenAI-compatible 结构。
独立 file_parse 使用 `error.code/message/upstream_status`，已有恢复记录时还返回状态字段；其错误格式以 [file_parse 错误响应](file-parse.md#错误响应) 为准。

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "task_already_terminal",
    "message": "Task is canceled",
    "param": null
  }
}
```

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `type` | string | 是 | 错误大类。 |
| `code` | string 或 null | 是 | 机器可读的细分错误码，无具体错误码时为 `null`。 |
| `message` | string | 是 | 人类可读描述，应包含必要上下文和修复建议。 |
| `param` | string 或 null | 否 | 导致错误的参数名；本地 API 无具体参数时返回 `null`，Router 代理层自身错误也可能省略此字段。 |

上例对应 `409 Conflict`；HTTP 错误的正文类型为 `application/json`。当前 Python API Server 和 Router 的错误对象均包含 `type/code/message`，客户端将缺省的 `param` 按 `null` 处理。
请求级错误位于顶层 `error`；异步解析失败则通常通过查询接口的 `200` 响应及 `files[].error` 表达，详见 [Tasks 错误响应](tasks.md#各接口错误响应)。

## 请求追踪

请求追踪属于部署扩展。提供此能力的网关或服务可以在成功和失败响应中返回：

```http
X-Request-Id: req_01HXYZ123ABCDEF
```

该部署可以接收客户端传入的 request id，并返回生成或确认后的 `X-Request-Id`。当前 Python Tasks 响应没有 `request_id` 字段，也不承诺自动回传该头；不要据此推断 Task 创建幂等性。

## 幂等

公网部署可以为 `POST` endpoint 提供 `Idempotency-Key`。以下为该能力的契约；当前自部署 tasks 不实现创建请求幂等存储，客户端不自动重试创建，取得 `task_id` 后应继续原任务。

```http
Idempotency-Key: 7f5b7f7c-5c5a-4b2f-bb89-0c901c9a2d6a
```

同一租户在有效期内使用相同 key 和相同请求语义时，服务端返回首次请求的结果，不重复创建 upload、job 或其他资源。建议有效期为 24 小时。

如果同一 key 被用于不同请求体，服务端应返回 `400 invalid_request`，避免客户端误以为复用了同一操作。

## 错误类型

| `type` | 含义 |
|--------|------|
| `invalid_request_error` | 请求参数非法、资源不存在或状态不允许。 |
| `authentication_error` | API Key 无效或过期。 |
| `permission_error` | 权限不足、配额耗尽或功能需要更高 access level。 |
| `rate_limit_error` | 请求触发限流。 |
| `engine_error` | 解析引擎不可用、崩溃或解析失败。 |
| `timeout_error` | 请求等待或处理超时。 |
| `internal_error` | 未归类异常的服务端内部错误；作为 `api_error` type 下的 code 出现（兜底 handler `api_server.py:2295-2305`），不是独立 type。 |
| `api_error` | 服务端 API 层错误（存储、元数据、远端依赖等副作用失败），同时也是未归类异常兜底 handler 返回的 type。 |

## 常用错误码

| HTTP | `type` | `code` |
|------|--------|--------|
| 400 | `invalid_request_error` | `invalid_request` |
| 400 | `invalid_request_error` | `invalid_sha256sum` |
| 400 | `invalid_request_error` | `file_hash_mismatch` |
| 400 | `invalid_request_error` | `bytes_mismatch` |
| 400 | `invalid_request_error` | `unsupported_parameter` |
| 400 | `invalid_request_error` | `unsupported_output_format` |
| 400 | `invalid_request_error` | `unsupported_source` |
| 401 | `authentication_error` | `invalid_api_key` |
| 403 | `permission_error` | `quota_exceeded` |
| 403 | `permission_error` | `feature_requires_api_key` |
| 403 | `permission_error` | `list_requires_api_key` |
| 404 | `invalid_request_error` | `task_not_found`、`file_not_found`、`upload_not_found`、`model_not_found` |
| 409 | `invalid_request_error` | `task_already_terminal`、`file_already_terminal` |
| 410 | `invalid_request_error` | `upload_expired` |
| 413 | `invalid_request_error` | `file_too_large` |
| 429 | `rate_limit_error` | `rate_limit_exceeded` |
| 500 | `api_error` | `internal_error` |
| 503 | `api_error` | `service_unavailable` |
| 503 | `api_error` 或 `engine_error` | `quality_tier_unavailable`，以具体入口返回的错误大类为准。 |

完整错误分类见 [错误码体系](../errors.md)。

## 本地 Server 差异

Local Parse Server 使用同一套错误 envelope。差异主要体现在错误来源:

- 未启用 API Key 时，不会产生 `invalid_api_key`。
- PDF/image 省略 `tier` 或传 `null` 且无法解析到非 `flash` 质量 tier 时，应返回 tier 相关错误，不能降级到 `flash`。OFD/EPUB/Office/HTML/MHTML/CSV 按 [ADR-0024](../decisions/0024-file-type-tier-normalization.md) 与 [ADR-0028](../decisions/0028-csv-structured-flash-parsing.md) 归一为 `flash`，其它 text 不作为解析输入。
- 解析能力缺失时，应优先返回可操作的 `engine_error`，而不是泛化为 `internal_error`。
