# Usage 与 Limits

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: access level、限流、用量查询
来源: 由根目录旧 Unified API 底稿迁移整理而来

## Access Level

官方 API 的 access level 由 API Key 决定，但响应结构保持一致。不传 API Key 是 anonymous；传入有效 API Key 是 registered；传入无效 API Key 返回 `401 invalid_api_key`。

| 能力 | anonymous | registered |
|------|-----------|------------|
| 单文件页数 | 1000 | 1000 |
| 单文件大小 | 200 MB | 200 MB |
| 单任务文件数 | 100 | 100 |
| 并发任务 | 10+ | 10+ |
| 队列优先级 | 默认 | 优先 |
| `tier` | 当前服务可用 tier。`mineru.net/api` 当前为 `standard`；可省略或传 `null` | 当前服务可用 tier。`mineru.net/api` 当前为 `standard`；可省略或传 `null` |
| 高级输出格式 | 不支持 `html`、`latex`、`docx` | 支持 |
| `callback` | 不支持 | 支持 |
| 产物保留期 | 30 天 | 30 天 |

anonymous 与 registered 的主要差异是限流、高级输出格式和 callback。请求高级输出格式或 callback 但没有 API Key 时，返回 `403 feature_requires_api_key`。

## 限流

官方 API 按 endpoint 类别做 per-minute 限流。registered 按 API Key 计数，anonymous 按请求 IP 计数。

| 类别 | 范围 | anonymous | registered |
|------|------|-----------|------------|
| `parse` | `POST /v1/parse/jobs` | 5/min | 30/min |
| `upload` | `POST /v1/uploads`、complete、cancel | 10/min | 60/min |
| `chat` | `POST /v1/chat/completions`、`POST /v1/responses` | 10/min | 60/min |
| `read` | 其他 GET/DELETE endpoint | 60/min | 300/min |

响应头携带当前 endpoint 类别的限流状态:

```http
X-RateLimit-Limit-Parse: 30
X-RateLimit-Remaining-Parse: 28
X-RateLimit-Reset-Parse: 1719184920
Retry-After: 0
```

Header 说明:

| Header | 说明 |
|--------|------|
| `X-RateLimit-Limit-<Category>` | 当前类别每分钟上限。 |
| `X-RateLimit-Remaining-<Category>` | 当前窗口剩余次数。 |
| `X-RateLimit-Reset-<Category>` | 窗口重置时间，Unix 秒级时间戳。 |
| `Retry-After` | 超限时建议等待秒数；未超限为 `0`。 |

超限响应:

```json
{
  "error": {
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded",
    "message": "Parse rate limit (5/min) exceeded. Retry in 12s.",
    "param": null
  }
}
```

## GET `/v1/usage`

查询当前计费周期的用量与配额。官方 API 不传 API Key 时返回 anonymous 视角；传 API Key 时返回该 key 的 registered 视角。

请求:

```http
GET /v1/usage HTTP/1.1
Authorization: Bearer <MINERU_API_KEY>
```

响应:

```json
{
  "object": "usage",
  "access_level": "registered",
  "billing_period": {
    "start": "2026-05-27T00:00:00Z",
    "end": "2026-05-28T00:00:00Z"
  },
  "current": {
    "pages_processed": 247,
    "files_processed": 18,
    "jobs_created": 12
  },
  "limits": {
    "max_pages_per_file": 1000,
    "max_file_size_bytes": 209715200,
    "max_files_per_job": 100,
    "max_concurrent_jobs": 10,
    "max_file_retention_days": 30
  }
}
```

字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `object` | string | 固定为 `"usage"`。 |
| `access_level` | string | `anonymous` 或 `registered`。 |
| `billing_period.start` | string | 当前计费周期开始，ISO-8601 UTC。 |
| `billing_period.end` | string 或 null | 当前计费周期结束。 |
| `current.pages_processed` | integer | 当前周期累计处理页数。 |
| `current.files_processed` | integer | 当前周期累计处理文件数。 |
| `current.jobs_created` | integer | 当前周期累计创建任务数。 |
| `limits.max_pages_per_file` | integer | 单文件最大页数。 |
| `limits.max_file_size_bytes` | integer | 单文件最大字节数。 |
| `limits.max_files_per_job` | integer | 单任务最大文件数。 |
| `limits.max_concurrent_jobs` | integer | 最大并发任务数。 |
| `limits.max_file_retention_days` | integer 或 null | 文件保留天数。 |

官方 API 的 `billing_period` 以 UTC 自然日对齐。未来如果加入月度配额，可以在 `limits` 中新增字段，不改变已有字段语义。

错误:

| HTTP | code | 场景 |
|------|------|------|
| 401 | `invalid_api_key` | 传入无效 API Key。 |
| 429 | `rate_limit_exceeded` | 触发 read 类限流。 |

## 本地 Server 差异

Local Parse Server 的 usage 不表达计费周期，而是表达本地进程视角:

- `billing_period.start` 是服务进程启动时间。
- `billing_period.end` 为 `null`。
- `current.*` 是自启动以来的累计值。
- `limits.max_file_retention_days` 为 `null`，除非本地实现了自动清理策略。
- `access_level` 取决于是否启用并通过 API Key 鉴权。

本地 server 可以不实现公网同等级别的限流。若实现限流，也应复用同一错误 envelope、`Retry-After` 和 `X-RateLimit-*` 响应头。
