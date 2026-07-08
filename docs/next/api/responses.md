# 响应与错误

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 成功响应、错误 envelope、请求追踪和幂等
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 成功响应

成功响应直接返回资源对象，不增加统一外层 envelope。

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "queued"
}
```

所有成功响应都应携带 `X-Request-Id`，便于客户端日志和服务端日志关联。

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
| 200 | 请求已完成，响应体是最终资源或当前资源。 |
| 202 | 请求已接受，但任务仍在排队或执行。 |
| 302 | 文件内容下载重定向到 CDN 或对象存储。 |

## 错误响应

错误响应采用 OpenAI-compatible 结构:

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_too_large",
    "message": "File exceeds the 200MB limit.",
    "param": "file"
  }
}
```

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `type` | string | 是 | 错误大类。 |
| `code` | string 或 null | 否 | 机器可读的细分错误码。 |
| `message` | string | 是 | 人类可读描述，应包含必要上下文和修复建议。 |
| `param` | string 或 null | 否 | 导致错误的参数名。 |

所有错误响应都应携带 `X-Request-Id`。客户端应把该值记录到日志中；排查问题时，优先向服务端提供 request id。

## 请求追踪

服务端必须在所有 API 响应中返回:

```http
X-Request-Id: req_01HXYZ123ABCDEF
```

客户端可以在请求中传入自己的 request id，但服务端仍应返回服务端生成或确认后的 `X-Request-Id`。同一个外部请求在内部产生异步任务时，job 对象也应保留创建请求的 request id 以便追踪。

## 幂等

`POST` endpoint 支持 `Idempotency-Key`:

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
| `api_error` | 服务端内部错误。 |

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
| 404 | `invalid_request_error` | `job_not_found`、`file_not_found`、`upload_not_found`、`model_not_found` |
| 409 | `invalid_request_error` | `job_already_terminal`、`upload_not_ready`、`upload_already_terminal` |
| 413 | `invalid_request_error` | `file_too_large`、`content_too_large` |
| 429 | `rate_limit_error` | `rate_limit_exceeded` |
| 500 | `api_error` | `internal_error` |
| 503 | `api_error` | `service_unavailable` |
| 503 | `engine_error` | `quality_tier_unavailable` |

完整错误分类见 [错误码体系](../errors.md)。

## 本地 Server 差异

Local Parse Server 使用同一套错误 envelope。差异主要体现在错误来源:

- 未启用 API Key 时，不会产生 `invalid_api_key`。
- 请求 `local` source 但路径不在 allowlist 内，应返回 `invalid_request_error`。
- 省略 `tier` 或传 `null` 且无法解析到 `medium` 或 `high` 能力时，应返回 tier 相关错误，不能降级到 `flash`。
- 解析能力缺失时，应优先返回可操作的 `engine_error`，而不是泛化为 `internal_error`。
