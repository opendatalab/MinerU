# Uploads 与 Files

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 上传生命周期、文件对象、文件列表、产物下载
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 上传模型

官方 API 使用三段式上传生命周期:

1. `POST /v1/uploads` 创建 upload。
2. 客户端向返回的 `upload_url` 上传字节。
3. `POST /v1/uploads/{upload_id}/complete` 完成 upload 并生成 file。

这套流程对齐 OpenAI Uploads API 的生命周期，但字节传输不经过 API 网关。官方 API 的 `upload_url` 指向对象存储预签名 URL；本地 server 的 `upload_url` 可以指向 server 自身的临时上传 endpoint。

## Upload 对象

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | `upload_<random>`，创建时分配。 |
| `object` | string | 是 | 固定为 `"upload"`。 |
| `bytes` | integer | 是 | 创建时声明的预期上传字节数。 |
| `created_at` | integer | 是 | Unix 秒级时间戳。 |
| `expires_at` | integer | 是 | Upload 过期时间。 |
| `filename` | string | 是 | 用户传入的文件名。 |
| `purpose` | string | 是 | `"parse"` 或 `"input_image"`。 |
| `mime_type` | string | 是 | MIME 类型。 |
| `sha256sum` | string 或 null | 是 | 64 位小写 hex；未提供时为 `null`。 |
| `status` | string | 是 | `pending`、`completed`、`cancelled`、`expired`。 |
| `upload_url` | string 或 null | 条件 | `pending` 时返回；秒传或 complete 后为 `null`。 |
| `upload_method` | string | 条件 | `pending` 时返回，当前为 `"PUT"`。 |
| `upload_headers` | object | 条件 | 上传字节时必须携带的 header。 |
| `file` | object | 条件 | `completed` 时返回 File 对象。 |

Upload 状态机:

| 当前状态 | 可执行操作 | 下一状态 |
|----------|------------|----------|
| `pending` | 上传字节后 complete | `completed` |
| `pending` | cancel | `cancelled` |
| `pending` | 过期任务清理 | `expired` |
| `completed` | 无 | 终态 |
| `cancelled` | 无 | 终态 |
| `expired` | 无 | 终态 |

## File 对象

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | `file-<random>`，全局唯一不透明标识。 |
| `object` | string | 是 | 固定为 `"file"`。 |
| `bytes` | integer | 是 | 文件字节数。 |
| `created_at` | integer | 是 | Unix 秒级时间戳。 |
| `expires_at` | integer 或 null | 是 | 文件过期时间；`null` 表示不自动过期。 |
| `filename` | string | 是 | 文件名。 |
| `purpose` | string | 是 | `"parse"`、`"input_image"` 或 `"parse_output"`。 |
| `sha256sum` | string 或 null | 是 | 64 位小写 hex；未知时为 `null`。 |

`purpose:"parse"` 表示待解析源文件。`purpose:"input_image"` 表示 Chat/Responses 输入图片。`purpose:"parse_output"` 表示解析产物，如 markdown、middle_json、content_list、structured_content、docx、zip。解析图片 sidecar 不作为独立 parse output；需要图片时从 zip 中读取。

## POST `/v1/uploads`

创建上传会话。请求体:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `filename` | string | 是 | 文件名。 |
| `bytes` | integer | 是 | 预期字节数，服务端用它做配额预校验。 |
| `mime_type` | string | 是 | MIME 类型，如 `application/pdf`。 |
| `purpose` | string | 是 | `"parse"` 或 `"input_image"`。 |
| `sha256sum` | string | 否 | 64 位小写 hex。提供时启用秒传和完整性校验。 |
| `expires_after` | object | 否 | Upload 过期策略。 |
| `expires_after.anchor` | string | 否 | 当前仅支持 `"created_at"`。 |
| `expires_after.seconds` | integer | 否 | 过期秒数，范围 `[3600, 2592000]`。 |

示例:

```json
{
  "filename": "report.pdf",
  "bytes": 1048576,
  "mime_type": "application/pdf",
  "purpose": "parse",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "expires_after": {
    "anchor": "created_at",
    "seconds": 3600
  }
}
```

### 秒传响应

如果传入 `sha256sum` 且命中当前租户已有文件，返回 `200`，`status` 为 `completed`，并内嵌 `file`。客户端无需上传字节，也无需调用 complete。

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "expires_at": 1719188511,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "completed",
  "upload_url": null,
  "file": {
    "id": "file-01HXYZ123ABCDEF",
    "object": "file",
    "bytes": 1048576,
    "created_at": 1719184911,
    "expires_at": 1719188511,
    "filename": "report.pdf",
    "purpose": "parse",
    "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef"
  }
}
```

秒传作用域是租户级。不同租户之间即使内容相同，也不能互相看到对方文件。

### 待上传响应

需要上传字节时返回 `200`，`status` 为 `pending`。

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "expires_at": 1719188511,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "pending",
  "upload_url": "https://oss.example/mineru/uploads/upload_01HXYZ...?Signature=...",
  "upload_method": "PUT",
  "upload_headers": {
    "Content-Type": "application/pdf",
    "x-amz-content-sha256": "a1b2c3d4e5f6..."
  }
}
```

客户端必须使用响应中的 `upload_method`、`upload_url` 和 `upload_headers` 上传字节。预签名 URL 自带授权，不需要 Bearer Token。

错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_request` | 缺少必填字段或字段类型错误。 |
| 400 | `invalid_sha256sum` | `sha256sum` 不是 64 位小写 hex。 |
| 413 | `file_too_large` | 文件超过当前 access level 的大小限制。 |
| 429 | `rate_limit_exceeded` | 触发 upload 类限流。 |

## PUT `{upload_url}`

上传文件字节。此请求不属于 `/v1` API envelope，由 `POST /v1/uploads` 返回的 URL 和 headers 决定。

```http
PUT https://oss.example/mineru/uploads/upload_01HXYZ...?Signature=...
Content-Type: application/pdf
x-amz-content-sha256: a1b2c3d4e5f6...

<binary file bytes>
```

成功时对象存储或本地临时上传 endpoint 返回 `200 OK`。客户端不应假设响应体存在。

## POST `/v1/uploads/{upload_id}/complete`

标记 upload 完成并生成 File 对象。请求体可为空；如果提供 `sha256sum`，它只用于校验，不能在 complete 阶段触发秒传。

```json
{
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef"
}
```

响应为 `status:"completed"` 的 Upload 对象，并内嵌 File。

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "expires_at": 1719188511,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "completed",
  "upload_url": null,
  "file": {
    "id": "file-01HXYZ123ABCDEF",
    "object": "file",
    "bytes": 1048576,
    "created_at": 1719184911,
    "expires_at": 1719188511,
    "filename": "report.pdf",
    "purpose": "parse",
    "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef"
  }
}
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_sha256sum` | complete 请求中的 `sha256sum` 格式错误。 |
| 400 | `file_hash_mismatch` | 实际字节 SHA-256 与声明不一致。 |
| 400 | `bytes_mismatch` | 实际字节数与 create 时的 `bytes` 不一致。 |
| 404 | `upload_not_found` | upload 不存在或不属于当前租户。 |
| 409 | `upload_not_ready` | 字节尚未上传完成。 |
| 409 | `upload_already_terminal` | upload 已是终态。 |

## POST `/v1/uploads/{upload_id}/cancel`

取消 `pending` 状态的 upload。响应:

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "expires_at": 1719188511,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": null,
  "status": "cancelled"
}
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 404 | `upload_not_found` | upload 不存在或不属于当前租户。 |
| 409 | `upload_already_terminal` | upload 已完成、已取消或已过期。 |

## GET `/v1/uploads/{upload_id}`

查询 upload 当前状态。响应结构与 create、complete、cancel 返回的 Upload 对象一致。`completed` 状态必须包含 `file`。

错误:

| HTTP | code |
|------|------|
| 404 | `upload_not_found` |

## GET `/v1/files`

列出当前租户文件，使用游标分页。anonymous 用户调用返回 `403 list_requires_api_key`。

查询参数:

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `after` | string | `null` | 游标，上一页响应的 `last_id`。 |
| `limit` | integer | `100` | 返回数量，范围 `[1, 1000]`。 |
| `order` | string | `desc` | `asc` 或 `desc`，按 `created_at` 排序。 |
| `purpose` | string | `null` | 可选 `parse`、`input_image`、`parse_output`。 |

响应:

```json
{
  "object": "list",
  "data": [
    {
      "id": "file-01HXYZ123ABCDEF",
      "object": "file",
      "bytes": 1048576,
      "created_at": 1719184911,
      "expires_at": 1719188511,
      "filename": "report.pdf",
      "purpose": "parse",
      "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef"
    }
  ],
  "first_id": "file-01HXYZ123ABCDEF",
  "last_id": "file-01HXYZ123ABCDEF",
  "has_more": false
}
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_request` | 分页参数非法。 |
| 403 | `list_requires_api_key` | anonymous 用户访问列表。 |

## GET `/v1/files/{file_id}`

查询单个文件元信息。响应为 File 对象。

错误:

| HTTP | code |
|------|------|
| 404 | `file_not_found` |

## GET `/v1/files/{file_id}/content`

下载 `purpose:"parse_output"` 的解析产物。源文件 `purpose:"parse"` 不提供内容下载。

官方 API 返回 302，客户端跟随重定向到 CDN 或对象存储签名 URL。

```http
HTTP/1.1 302 Found
Location: https://cdn.mineru.net/files/file-01HXYZ.../signed?Signature=...
```

客户端示例:

```bash
curl -L "https://mineru.net/api/v1/files/$MD_FILE_ID/content" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -o report.md
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 404 | `file_not_found` | 文件不存在或不属于当前租户。 |
| 400 | `invalid_request` | 文件不是可下载的 `parse_output`。 |

## DELETE `/v1/files/{file_id}`

从当前租户视图删除文件。已关联的 job 历史结果不受影响。

响应:

```json
{
  "id": "file-01HXYZ123ABCDEF",
  "object": "file",
  "deleted": true
}
```

错误:

| HTTP | code |
|------|------|
| 404 | `file_not_found` |

## 本地 Server 差异

Local Parse Server 保持相同 endpoint 和对象结构，但传输实现可以简化:

- `upload_url` 指向本地 server 的临时上传 endpoint，例如 `/v1/uploads/{upload_id}/content`。
- 字节上传仍由客户端使用 `upload_method`、`upload_url` 和 `upload_headers` 完成；客户端不应硬编码 OSS 行为。
- `GET /v1/files/{file_id}/content` 直接返回 `200 + body`，不做 CDN 重定向。
- 本地 server 支持 parse job 的 `local` source；只有启动时开启 `--allow-local-source` 并在 `features.sources` 返回 `local` 时，用户才可跳过 upload 直接引用 server 进程权限范围内的本地路径。
- 官方 API 必须拒绝 `local` source，返回 `400 invalid_request`。
- 本地文件默认可以不设置保留期，File 对象的 `expires_at` 可为 `null`。
