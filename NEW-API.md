# MinerU Unified API (v1)

> 一套 REST API,覆盖当前 ParseApi(本地)、MineruV4(云端付费)、Flash(云端免费) 三套 API 的全部功能。
>
> **核心约定**:Token 仅影响配额、并发、队列优先级与可选项,**不影响请求/响应格式**。

---

## 1. 整体设计

### 1.1 资源模型

| 资源 | 含义 |
|------|------|
| `file` | 平台中的文件,由不透明的 `file_id` 唯一标识。既包括用户上传的源文件(`purpose:parse`),也包括解析产物如 markdown/json 等(`purpose:parse_output`)。`sha256sum` 是可选元信息,用于秒传与校验 |
| `upload` | OpenAI Uploads API 风格的上传会话对象,驱动文件上传生命周期 |
| `job` | 一次解析任务(可包含多个文件) |

**ID 格式**

| 资源 | ID 示例 | 备注 |
|------|---------|------|
| `file` | `file-01HXYZ123ABCDEF` | 不透明随机 ID;全局唯一;同内容不同用户/不同上传得到不同 file_id |
| `upload` | `upload_01HXYZ123ABCDEF` | OpenAI 风格,创建 Upload 时分配 |
| `job` | `job_01HXYZ123ABCDEF` | 解析任务 |

### 1.2 端点总览

| Method | Path | 用途 |
|--------|------|------|
| `POST` | `/v1/uploads` | 创建 Upload(可选传 sha256sum 启用秒传) |
| `POST` | `/v1/uploads/{upload_id}/complete` | 完成 Upload,生成 File 对象 |
| `POST` | `/v1/uploads/{upload_id}/cancel` | 取消 Upload |
| `GET` | `/v1/uploads/{upload_id}` | 查询 Upload 状态 |
| `GET` | `/v1/files` | 列出当前租户的文件(游标分页) |
| `GET` | `/v1/files/{file_id}` | 查询文件元信息 |
| `GET` | `/v1/files/{file_id}/content` | 下载解析产物(302 重定向到 CDN,仅 `parse_output`) |
| `DELETE` | `/v1/files/{file_id}` | 从租户视图删除文件 |
| `POST` | `/v1/parse/jobs` | 创建解析任务,支持同步等待(`wait` 参数) |
| `GET` | `/v1/parse/jobs/{job_id}` | 查询任务状态与结果 |
| `GET` | `/v1/parse/jobs/{job_id}/events` | SSE 流式状态推送(可选) |
| `GET` | `/v1/parse/jobs` | 列出任务(分页) |
| `DELETE` | `/v1/parse/jobs/{job_id}` | 取消任务 |
| `GET` | `/v1/health` | 健康检查 |
| `GET` | `/v1/models` | 列出可用解析模型 |
| `GET` | `/v1/models/{model}` | 查询单个模型信息 |

### 1.3 认证

```
Authorization: Bearer <MINERU_TOKEN>
```

- **缺省 Token**:匿名访问,自动降级为 Free 档(等同当前 Flash API 限制)。
- **有效 Token**:按 Token 等级解锁更高配额、更多模型、`docx`/`html`/`latex` 等高级输出格式。
- **无效 Token**:返回 `401 Unauthorized`(注意:**不传 Token ≠ Token 无效**)。

### 1.4 通用约定

- **基址(Base URL)**
  - 云端:`https://mineru.net/api`
  - 本地:`http://localhost:8000/api`(自部署)
- **版本号**:URL 前缀 `/v1`,新增字段只增不删
- **Content-Type**:`application/json; charset=utf-8`(除上传字节流)
- **时间**:全部 ISO-8601 UTC 字符串,如 `2026-05-21T08:30:00Z`
- **ID 格式**:如 `job_01HXYZ...`、`upload_01HXYZ...`、`file-01HXYZ...`
- **空字段**:用 `null`,不省略 key

---

## 2. 通用响应结构

### 2.1 成功响应

直接返回资源对象,**无外层 envelope**:

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "queued",
  ...
}
```

### 2.2 错误响应

统一结构,HTTP 状态码 + 业务错误码:

```json
{
  "error": {
    "code": "file_too_large",
    "message": "File exceeds the 10MB limit for free tier.",
    "details": {
      "limit_bytes": 10485760,
      "actual_bytes": 23456789,
      "upgrade_url": "https://mineru.net/pricing"
    },
    "request_id": "req_01HXYZ..."
  }
}
```

| HTTP | code 示例 | 含义 |
|------|----------|------|
| 400 | `invalid_request` | 请求参数错误 |
| 400 | `invalid_sha256sum` | sha256sum 格式非法(非 64 位小写 hex) |
| 400 | `file_hash_mismatch` | 上传字节实际 SHA-256 与申请时不一致 |
| 401 | `invalid_token` | Token 无效或过期 |
| 403 | `quota_exceeded` / `feature_requires_token` | 配额耗尽或功能受限 |
| 404 | `job_not_found` / `file_not_found` / `upload_not_found` / `model_not_found` | 资源不存在 |
| 409 | `job_already_terminal` | 状态不可变更(如取消已完成任务) |
| 409 | `upload_not_ready` | 调用 Complete 时字节尚未上传 |
| 413 | `file_too_large` | 文件超出当前等级限制 |
| 429 | `rate_limit_exceeded` | 触发限流 |
| 500 | `internal_error` | 服务端错误 |
| 503 | `service_unavailable` | 模型不可用,可重试 |

---

## 3. Health — 健康检查

### GET `/v1/health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "backend_version": "3.1.14",
  "models": {
    "pipeline": "ok",
    "vlm": "ok",
    "html": "ok"
  }
}
```

---

## 4. Models — 解析模型

模型列表与查询接口对齐 **OpenAI Models API** 的请求与响应形态。

### 4.1 GET `/v1/models` — 列出全部模型

**Request**

```http
GET /v1/models  HTTP/1.1
Authorization: Bearer <token>
```

**Response (200)**

```json
{
  "object": "list",
  "data": [
    {
      "id": "auto",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru"
    },
    {
      "id": "pipeline",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru"
    },
    {
      "id": "vlm",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru"
    },
    {
      "id": "html",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru"
    }
  ]
}
```

**字段说明(对齐 OpenAI)**

| 字段 | 类型 | 说明 |
|------|------|------|
| `object`(顶层) | string | 固定为 `"list"` |
| `data[]` | array | 模型对象数组 |
| `data[].id` | string | 模型标识,用于 `POST /v1/parse/jobs` 的 `model` 字段 |
| `data[].object` | string | 固定为 `"model"` |
| `data[].created` | int | Unix 秒级时间戳(模型首次上线时间) |
| `data[].owned_by` | string | 拥有者(本平台均为 `"mineru"`,自部署可设组织名) |

> 说明:本端点仅保留 OpenAI 标准字段,**不再包含** `description` / `available_to` / `supports` 等本 API 早期字段。模型适用性(`available_to`)由 Tier(参见 §8)决定;格式适用性(`supports`)在使用模型时由服务端自动校验。

### 4.2 GET `/v1/models/{model}` — 查询单个模型

**Request**

```http
GET /v1/models/vlm  HTTP/1.1
Authorization: Bearer <token>
```

**Path Parameters**

| 参数 | 说明 |
|------|------|
| `model` | 模型 ID(`auto` / `pipeline` / `vlm` / `html`) |

**Response (200)**

```json
{
  "id": "vlm",
  "object": "model",
  "created": 1700000000,
  "owned_by": "mineru"
}
```

**Errors**

- `404 model_not_found`:模型 ID 不存在
- `403 feature_requires_token`:模型存在但当前 tier 无权使用(响应体含 `upgrade_url`)

---

## 5. Files & Uploads

文件上传采用 **OpenAI Uploads API 的三段式生命周期**:Create Upload → 客户端直传 OSS → Complete Upload。对齐 OpenAI 协议外壳,但字节传输路径适应低带宽 API 服务器(详见 §3.3 设计说明)。

```
┌──────────┐                           ┌──────────┐                    ┌──────┐
│  Client  │                           │ API 服务  │                    │ OSS  │
└────┬─────┘                           └────┬─────┘                    └──┬───┘
     │                                      │                            │
     │ 1. POST /v1/uploads                  │                            │
     │    {filename, bytes, purpose,        │                            │
     │     mime_type, sha256sum?}           │                            │
     ├─────────────────────────────────────►│                            │
     │                                      │                            │
     │ ┌────────────────────────────────────┼────────────────────────┐   │
     │ │ 若 sha256sum 命中秒传:            │                        │   │
     │ │ ◄── 200 {status:"completed",      │                        │   │
     │ │         file:{id:"file-..."}}      │                        │   │
     │ │ → 跳到步骤 4                       │                        │   │
     │ └────────────────────────────────────┼────────────────────────┘   │
     │                                      │                            │
     │ ◄── 201 {id:"upload_...",            │                            │
     │         status:"pending",            │                            │
     │         upload_url:"https://oss...", │                            │
     │         upload_method:"PUT",         │                            │
     │         upload_headers:{...}}        │                            │
     │                                      │                            │
     │ 2. PUT {upload_url} <bytes>          │                            │
     ├──────────────────────────────────────────────────────────────────►│
     │ ◄── 200 OK                           │                            │
     │                                      │                            │
     │ 3. POST /v1/uploads/{id}/complete    │                            │
     │    {sha256sum?}  ← 可选,仅校验       │                            │
     ├─────────────────────────────────────►│                            │
     │                                      │                            │
     │ ◄── 200 {status:"completed",         │                            │
     │         file:{id:"file-..."}}        │                            │
     │                                      │                            │
     │ 4. POST /v1/parse/jobs               │                            │
     │    {files:[{source:{type:"file_id",  │                            │
     │            id:"file-..."}}]}         │                            │
     ├─────────────────────────────────────►│                            │
```

### 5.1 概念

**Upload 对象**(资源类型: `upload`)

| 字段 | 说明 |
|------|------|
| `id` | `upload_<ULID>`,创建时由服务端分配 |
| `object` | 固定为 `"upload"` |
| `bytes` | 预期上传字节数 |
| `created_at` | Unix 秒级时间戳 |
| `expires_at` | Unix 秒级时间戳(默认 1 小时后;可配 `expires_after`) |
| `filename` | 文件名 |
| `purpose` | 固定为 `"parse"` |
| `mime_type` | MIME 类型 |
| `sha256sum` | 可选 64 位小写 hex,创建时提供 |
| `status` | `pending` / `completed` / `cancelled` / `expired` |
| `file` | 仅 `status=completed` 时存在,内嵌 File 对象 |

**File 对象**(资源类型: `file`)

| 字段 | 说明 |
|------|------|
| `id` | `file_<ULID>`,全局唯一的不透明标识 |
| `object` | 固定为 `"file"` |
| `bytes` | 文件字节数 |
| `created_at` | Unix 秒级时间戳(文件首次上传时间) |
| `expires_at` | Unix 秒级时间戳(到期时间;`null` = 手动删除前永不过期) |
| `filename` | 文件名 |
| `purpose` | 固定为 `"parse"` |
| `sha256sum` | 可选 64 位小写 hex |

**秒传规则**

| 提供 `sha256sum`? | 秒传(命中已有文件) | 字节完整性校验 |
|--------------------|---------------------|----------------|
| ✅ Create 时提供 | Create 直接返 `status:"completed"` + `file`,跳过 steps 2/3 | Complete 时可选二次校验 |
| ❌ 不提供 | 不支持(必上传,即使内容相同) | 无校验 |

> **秒传作用域 = 租户级**:跨租户不可见。
> **同 file_id 复用**:拿到 file_id 后,后续 `POST /v1/parse/jobs` 直接引用。

### 5.2 POST `/v1/uploads` — Create Upload

对齐 OpenAI Create Upload API。创建一个 Upload 对象,返回预签名 OSS URL 供客户端直传字节。

> ⚠️ **与 OpenAI 的唯一偏离**:响应中额外返回 `upload_url` / `upload_method` / `upload_headers`,取代 OpenAI 的 `PUT /v1/uploads/{id}/content`。原因:API 服务器带宽有限,字节不能穿越网关。
>
> 若未来 API 服务器带宽升级,可平滑迁移为代理模式(客户端代码无需改动——此处预先说明)。

**Request**

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

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | 是 | 文件名 |
| `bytes` | int | 是 | 预期字节数,服务端预校验配额 |
| `mime_type` | string | 是 | MIME 类型(如 `application/pdf`) |
| `purpose` | string | 是 | 固定为 `"parse"` |
| `sha256sum` | string | **否** | 64 位小写 hex。提供则启用秒传与字节完整性校验 |
| `expires_after` | object | 否 | 过期策略。不传则 Upload 默认 1 小时后过期 |
| `expires_after.anchor` | string | 否 | 锚点,当前仅支持 `"created_at"` |
| `expires_after.seconds` | int | 否 | 距锚点秒数,范围 `[3600, 2592000]`(1 小时 ~ 30 天) |

**Response (200) — 秒传命中(提供 sha256sum 且租户内已有相同内容)**

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "completed",
  "expires_at": 1719188511,
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

> 秒传命中时 `status` 直接为 `completed`,内嵌 `file` 对象。**无需调用 Complete**。

**Response (201) — 需要上传**

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "pending",
  "expires_at": 1719188511,
  "upload_url": "https://oss-cn-hz.aliyuncs.com/mineru/uploads/upload_01HXYZ...?Signature=...",
  "upload_method": "PUT",
  "upload_headers": {
    "Content-Type": "application/pdf",
    "x-amz-content-sha256": "a1b2c3d4e5f6..."
  }
}
```

> 若未提供 `sha256sum`,响应中 `sha256sum` 为 `null`,`upload_headers` 中不含 `x-amz-content-sha256`。

**响应字段总览**

| 字段 | 秒传(200) | 需上传(201) |
|------|-----------|------------|
| `id` | ✓ | ✓ |
| `object` | `"upload"` | `"upload"` |
| `bytes` | ✓ | ✓ |
| `created_at` | ✓ | ✓ |
| `expires_at` | ✓ | ✓ |
| `filename` | ✓ | ✓ |
| `purpose` | `"parse"` | `"parse"` |
| `mime_type` | ✓ | ✓ |
| `sha256sum` | ✓(等于输入) | ✓ 或 `null` |
| `status` | `"completed"` | `"pending"` |
| `file` | ✓(内嵌) | 不存在 |
| `upload_url` | `null` | ✓ |
| `upload_method` | 不存在 | `"PUT"` |
| `upload_headers` | 不存在 | ✓ |

**Errors**

- `400 invalid_request`:缺少必填字段
- `400 invalid_sha256sum`:sha256sum 格式非法(非 64 位小写 hex)
- `413 file_too_large`:超出当前 tier 限制(响应体含 `upgrade_url`)

### 5.3 PUT `{upload_url}` — 上传字节到 OSS

客户端凭 Create 响应中的 `upload_url` / `upload_method` / `upload_headers` 直接向 OSS 发送字节。**API 服务器完全不参与此次传输**。

```http
PUT https://oss-cn-hz.aliyuncs.com/mineru/uploads/upload_01HXYZ...?Signature=...
Content-Type: application/pdf
x-amz-content-sha256: a1b2c3d4e5f6...

<binary file bytes>
```

成功返回 `200 OK`。无需 Bearer Token,预签名 URL 自带授权。

**设计说明:为什么不提供 `PUT /v1/uploads/{id}/content`?**

OpenAI 的 Uploads API 流程为:`POST /v1/uploads` → `PUT /v1/uploads/{id}/content`(字节经 API 网关) → `POST /v1/uploads/{id}/complete`。但在本 API 中:

- **API 服务器带宽有限**,不适合代理 PDF 字节(通常 MB 级)
- **OSS 具备高外网带宽**,预签名 URL 直传是最优路径

因此我们将字节传输点从 API 网关移到了 OSS 预签名 URL,而上传生命周期(Create / Complete / Cancel)保持与 OpenAI 完全一致。这是本 API 对 OpenAI 协议的**唯一务实偏离**。

### 5.4 POST `/v1/uploads/{upload_id}/complete` — Complete Upload

字节上传完成后调用,标记 Upload 完成并生成 File 对象。对齐 OpenAI Complete Upload API。

**Request**

```http
POST /v1/uploads/upload_01HXYZ123ABCDEF/complete  HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sha256sum` | string | **否** | 64 位小写 hex。此时提供仅用于**校验**(不能用于秒传——秒传在 Create 时已完成判断)。与 Create 时提供的值必须一致,不一致则拒绝 |

**Response (200)**

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6abcd1234567890abcdef1234567890abcdef1234567890abcdef",
  "status": "completed",
  "expires_at": 1719188511,
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

> Complete 后 `upload_url` 立即失效,`file` 对象可跨 jobs 复用。

**Errors**

- `404 upload_not_found`:upload_id 不存在
- `409 upload_not_ready`:字节尚未上传完成
- `409 upload_already_terminal`:status 已是 `completed` / `cancelled` / `expired`
- `400 file_hash_mismatch`:提供的 sha256sum 与实际字节不匹配
- `400 bytes_mismatch`:实际上传字节数与创建时声明的 `bytes` 不一致

### 5.5 POST `/v1/uploads/{upload_id}/cancel` — Cancel Upload

取消进行中的 Upload,已上传字节将被清理。对齐 OpenAI Cancel Upload API。

**Request**

```http
POST /v1/uploads/upload_01HXYZ123ABCDEF/cancel  HTTP/1.1
Authorization: Bearer <token>
```

**Response (200)**

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "status": "cancelled",
  "expires_at": 1719188511
}
```

**Errors**

- `404 upload_not_found`
- `409 upload_already_terminal`:已是 `completed` / `cancelled` / `expired`

### 5.6 GET `/v1/uploads/{upload_id}` — 查询 Upload

**Request**

```http
GET /v1/uploads/upload_01HXYZ123ABCDEF  HTTP/1.1
Authorization: Bearer <token>
```

**Response (200)** — 结构与 Create / Complete 返回的 Upload 对象一致。

```json
{
  "id": "upload_01HXYZ123ABCDEF",
  "object": "upload",
  "bytes": 1048576,
  "created_at": 1719184911,
  "filename": "report.pdf",
  "purpose": "parse",
  "mime_type": "application/pdf",
  "sha256sum": "a1b2c3d4e5f6...",
  "status": "pending",
  "expires_at": 1719188511
}
```

`status` 取值:`pending`(等待上传) / `completed`(含 `file`) / `cancelled` / `expired`

### 5.7 GET `/v1/files` — 列出文件

列出当前租户的全部文件,游标分页(分页设计参考 OpenAI Files API)。

**Query Parameters**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `after` | string | `null` | 游标,上一页响应的 `last_id`。从其下一条开始返回 |
| `limit` | int | `10000` | 返回数量上限,范围 `[1, 10000]` |
| `order` | enum | `desc` | 按 `created_at` 排序,`asc` 升序 / `desc` 降序 |
| `purpose` | string | `null` | 按用途过滤:`parse`(源文件) / `parse_output`(解析产物)。不传则返回全部 |
| `sha256sum` | string | `null` | 按 SHA-256 过滤;命中租户内已有同内容文件(用于秒传探测) |

**Request**

```http
GET /v1/files?limit=100&order=desc  HTTP/1.1
Authorization: Bearer <token>
```

**Response (200)**

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
    },
    {
      "id": "file-01HXYZ456GHIJKL",
      "object": "file",
      "bytes": 204800,
      "created_at": 1719100000,
      "expires_at": null,
      "filename": "scan.jpg",
      "purpose": "parse",
      "sha256sum": null
    }
  ],
  "first_id": "file-01HXYZ123ABCDEF",
  "last_id": "file-01HXYZ456GHIJKL",
  "has_more": true
}
```

**与 OpenAI Files API 的差异**

| 维度 | OpenAI | 本 API | 原因 |
|------|--------|--------|------|
| 分页/envelope | `{object:"list",data,first_id,last_id,has_more}` | ✅ 完全对齐 | 客户端心智一致 |
| `object` | `"file"` | ✅ 对齐 | |
| ID 形式 | `file-abc123` | `file-<ULID>` | ✅ 对齐 |
| 大小字段名 | `bytes` | ✅ 对齐 | |
| 时间格式 | Unix timestamp | ✅ 对齐 | |
| `purpose` | 多种 | `parse` + `parse_output` | 源文件与解析产物 |
| `status` | 有(已 deprecated) | 无 | 跟随 OpenAI 弃用 |
| `sha256sum` | 无 | **新增** | 本 API 特有,辅助秒传 |

### 5.8 GET `/v1/files/{file_id}` — 元信息

**Response (200)**

```json
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
```

> 字段与 `GET /v1/files` 中的单条 `data[]` 元素完全一致。

### 5.9 GET `/v1/files/{file_id}/content` — 下载解析产物

下载 `parse_output` 类文件的原始内容(markdown/json/images 等)。源文件(`purpose:parse`)不提供下载——用户理应持有原始文件,平台不充当文件托管。

```http
GET /v1/files/file-01HXYZ123ABCDEF/content  HTTP/1.1
Authorization: Bearer <token>

→ 302 Found
  Location: https://cdn.mineru.net/files/file-01HXYZ.../signed?Signature=...
```

客户端跟随 302 重定向到 CDN 直取字节。所有文件类型均返回 302。

```bash
# 下载产物
curl -L "https://mineru.net/api/v1/files/file-MD.../content" \
  -H "Authorization: Bearer $TOKEN" -o report.md
```

> **设计说明**:解析产物(markdown/json/content_list/images)不再有独立的 `/v1/artifacts/` 接口,统一为 File 对象(`purpose:parse_output`),复用 Files API 的内容寻址机制。对齐 OpenAI 架构:Files 是通用内容存储层,`purpose` 区分源文件与产物。

**Errors**

- `404 file_not_found`

### 5.10 DELETE `/v1/files/{file_id}` — 删除

仅从当前租户视图删除;已关联的 jobs 不受影响。

**Request**

```http
DELETE /v1/files/file-01HXYZ123ABCDEF  HTTP/1.1
Authorization: Bearer <token>
```

**Response (200)**

```json
{
  "id": "file-01HXYZ123ABCDEF",
  "object": "file",
  "deleted": true
}
```

**Errors**

- `404 file_not_found`:文件不存在(幂等性视角下也可返回 `{deleted: true}`,实现侧可选)

---

## 6. Jobs — 解析任务

### 6.1 POST `/v1/parse/jobs` — 创建任务

**Request**

```json
{
  "files": [
    {
      "source": {
        "type": "file_id",
        "file_id": "file-01HXYZ123ABCDEF"
      },
      "options": {
        "language": "ch",
        "ocr": "auto",
        "formula": true,
        "table": true,
        "image_analysis": true,
        "page_range": "1-10"
      }
    },
    {
      "source": {
        "type": "url",
        "url": "https://example.com/scan.jpg"
      }
    },
    {
      "source": {
        "type": "inline",
        "name": "scan.jpg",
        "data": "base64..."
      }
    }
  ],
  "model": "auto",
  "output_formats": ["markdown", "json", "content_list", "images"],
  "wait": 60,
  "callback": {
    "url": "https://your.app/mineru-webhook",
    "secret": "whsec_..."
  }
}
```

#### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | array | 是 | 至少 1 个,最多由 token 等级决定 |
| `model` | enum | 否 | `auto`(默认,平台推荐模型) / `pipeline` / `vlm` / `html` |
| `output_formats` | array | 否 | 选择产物,默认 `["markdown"]`。产物以 File 对象形式存储(`purpose:parse_output`),通过 `GET /v1/files/{file_id}/content` 下载。可选值见下表 |
| `wait` | int | 否 | 同步等待秒数。不传或 `0`:异步模式,立即返回 202。传正值:阻塞等待最多 N 秒,范围 `[5, 300]`,完成返回 200(内容内联),超时返回 202(转为异步轮询) |
| `callback` | object | 否 | Webhook 通知(需 Token) |

#### `files[].source` 四种来源

```json
{ "type": "file_id",   "file_id": "file-01HXYZ123ABCDEF" }                 // 引用平台中已有文件(Create 秒传/Complete 后获得)
{ "type": "url",       "url": "https://..." }                          // 服务端拉取(内部转为 file_id)
{ "type": "inline",    "name": "report.pdf", "data": "base64..." }      // 小文件内嵌(< 1MB,内部转为 file_id)。name 必填,作为文件显示名
```

> 在 `POST /v1/parse/jobs` 中,file source 只需 `file_id`(Create 秒传命中时立即获得,或 Complete 后获得)、`url` 或 `inline` 三种。完成上传后即可复用 file_id,不再依赖 upload_id。

#### `files[].options`

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `language` | string | `"ch"` | `ch` / `en` / `ja` / `korean` / `ch_lite` / `ch_server` / `japan` / `chinese_cht` / `ta` / `te` / `ka` / `th` / `el` / `latin` / `arabic` / `east_slavic` / `cyrillic` / `devanagari`。不传则默认 `ch` |
| `ocr` | enum | `auto` | `auto` / `true` / `false` / `txt`。`txt` 强制文本提取(不 OCR) |
| `formula` | bool | `true` | 公式识别 |
| `table` | bool | `true` | 表格识别 |
| `image_analysis` | bool | `true` | 图片内容分析 |
| `page_range` | string | `null` | `"1-10"`、`"1,3,5-7"`、`"2--2"`(到倒数第二页) |

#### `output_formats` 可选值

| 值 | 类型 | 含义 | 需 Token |
|----|------|------|----------|
| `markdown` | 解析产物 | 渲染后的 markdown 文本 | 否 |
| `json` | 解析产物 | 类型化中间结构(原 `middle_json`) | 否 |
| `content_list` | 解析产物 | 扁平内容列表 | 否 |
| `content_list_v2` | 解析产物 | 新版内容列表 | 否 |
| `images` | 解析产物 | 切片图片 | 否 |
| `html` | 导出格式 | HTML 格式导出 | **是** |
| `latex` | 导出格式 | LaTeX 格式导出 | **是** |
| `docx` | 导出格式 | Word 文档导出 | **是** |
| `zip` | 打包格式 | 将 `output_formats` 中除 `zip` 外的所有产物打包为 ZIP(如 `["markdown","json","zip"]` 产出 markdown + json + 包含前两者的 zip) | 否 |

> 请求 `output_formats` 中包含 Token-only 格式但无 Token 时,返回 `403 feature_requires_token`。

**Response (202 Accepted)**

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "queued",
  "created_at": "2026-05-21T08:30:00Z",
  "model": "auto",
  "output_formats": ["markdown", "json", "content_list", "images"],
  "tier": "pro",
  "files": [
    { "file_id": "file-01HXYZ123ABCDEF", "name": "report.pdf", "status": "queued" },
    { "file_id": "file-01HXYZ456GHIJKL", "name": "scan.jpg",   "status": "queued" }
  ],
  "links": {
    "self":   "/v1/parse/jobs/job_01HXYZ123ABCDEF",
    "events": "/v1/parse/jobs/job_01HXYZ123ABCDEF/events",
    "cancel": "/v1/parse/jobs/job_01HXYZ123ABCDEF"
  }
}
```

**Response (200 OK) — `wait > 0` 且在超时内完成**

文本类产物(`markdown`)内联在 `content` 字段,二进制产物(`images`、`docx`、`zip`)仅返回 `file_id`,需通过 `GET /v1/files/{id}/content` 下载。

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "completed",
  "created_at": "2026-05-21T08:30:00Z",
  "started_at": "2026-05-21T08:30:02Z",
  "finished_at": "2026-05-21T08:30:48Z",
  "model": "vlm",
  "output_formats": ["markdown", "json", "images"],
  "tier": "pro",
  "progress": { "completed": 1, "failed": 0, "total": 1 },
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "status": "completed",
      "metadata": {
        "pages": 12,
        "model_used": "vlm",
        "language_detected": "ch",
        "processing_time_ms": 8234,
        "backend_version": "3.1.14"
      },
      "output_files": {
        "markdown": {
          "file_id": "file-01HXYZMD000001",
          "bytes": 51407,
          "content": "# Report\n\n## Section 1\n..."
        },
        "json": {
          "file_id": "file-01HXYZJSON000002",
          "bytes": 184320
        },
        "images": [
          { "path": "images/0.jpg", "file_id": "file-01HXYZIM000004", "bytes": 102400 }
        ]
      }
    }
  ]
}
```

- `markdown` — 唯一内联 `content` 的字段
- 其他文本/二进制产物(`content_list`、`content_list_v2`、`json`、`html`、`latex`、`images`、`docx`、`zip`) — 仅返回 `file_id` + `bytes`,通过 `GET /v1/files/{id}/content` 下载

**Response (202 Accepted) — `wait=0` 或 `wait` 超时**

`wait=0`(默认)或等待超时时返回 202,客户端应切换到异步轮询。超时时 `status` 为当前实际状态(`queued` 或 `running`)。

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "running",
  "created_at": "2026-05-21T08:30:00Z",
  "model": "auto",
  "output_formats": ["markdown", "json", "images"],
  "tier": "pro",
  "files": [
    { "file_id": "file-01HXYZ123ABCDEF", "name": "report.pdf", "status": "running" },
    { "file_id": "file-01HXYZ456GHIJKL", "name": "scan.jpg",   "status": "running" }
  ],
  "links": {
    "self":   "/v1/parse/jobs/job_01HXYZ123ABCDEF",
    "events": "/v1/parse/jobs/job_01HXYZ123ABCDEF/events",
    "cancel": "/v1/parse/jobs/job_01HXYZ123ABCDEF"
  }
}
```

#### 同步等待的设计考虑

`wait` 参数让一个端点同时支持同步和异步模式,但对公开部署有几点需要注意:

**LB/代理超时对齐** — 反向代理(nginx、ALB)默认 `proxy_read_timeout` 通常 60s。`wait` 值若超过 LB 超时,代理会先于服务端断开连接。部署时需全线对齐 `wait_maximum` 与代理超时(或设置代理超时 ≥ 最大 wait)。

**连接断开 ≠ 取消 job** — 客户端主动断开(如用户刷新页面)时 TCP 连接关闭,服务端通过 `ctx.Done()` 感知。**job 应继续运行直至完成**,仅停止向该连接写入。确保客户端断开连接后仍可通过 `GET /v1/parse/jobs/{job_id}` 轮询结果。

**Waiter 并发上限** — 同步等待的连接持有 goroutine + socket buffer + ctx,虽每个开销小,但大量 waiter 会放大排队的可见性。建议对同时处于同步等待状态的请求数设硬上限(如 500),超出则直接返回 202 退化为异步,避免耗尽内存或连接资源。

**同步请求独立限流** — 同步等待请求应使用独立的 rate limit 计数,防止少量用户占满 waiter 槽位。

**max_wait 配置点** — 服务端可通过配置限制 `wait` 最大值(与 LB 超时匹配);超出上限的 `wait` 值静默截断(返回 202 时已完成则返 200)。

---

### 6.2 GET `/v1/parse/jobs/{job_id}` — 查询任务

**Response (200)**

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "completed",
  "created_at": "2026-05-21T08:30:00Z",
  "started_at": "2026-05-21T08:30:02Z",
  "finished_at": "2026-05-21T08:30:48Z",
  "model": "vlm",
  "output_formats": ["markdown", "json", "images"],
  "tier": "pro",
  "progress": { "completed": 2, "failed": 0, "total": 2 },
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "status": "completed",
      "metadata": {
        "pages": 12,
        "model_used": "vlm",
        "language_detected": "ch",
        "processing_time_ms": 8234,
        "backend_version": "3.1.14"
      },
      "output_files": {
        "markdown": {
          "file_id": "file-01HXYZMD000001",
          "bytes": 51407
        },
        "json": {
          "file_id": "file-01HXYZJSON000002",
          "bytes": 184320
        },
        "content_list": {
          "file_id": "file-01HXYZCL000003",
          "bytes": 23456
        },
        "images": [
          {
            "path": "images/0.jpg",
            "file_id": "file-01HXYZIM000004",
            "bytes": 102400
          }
        ]
      }
    },
    {
      "file_id": "file-01HXYZ456GHIJKL",
      "name": "scan.jpg",
      "status": "failed",
      "error": {
        "code": "ocr_failed",
        "message": "Image too blurry to OCR."
      }
    }
  ]
}
```

#### `status` 取值

| 值 | 含义 |
|----|------|
| `queued` | 排队中 |
| `running` | 解析中 |
| `completed` | 全部成功 |
| `partial` | 部分文件失败 |
| `failed` | 全部失败 |
| `canceled` | 已取消 |

**轮询建议**:指数退避,初值 2s,上限 30s。

---

### 6.3 GET `/v1/parse/jobs/{job_id}/events` — SSE 流式状态

**Request**

```http
GET /v1/parse/jobs/job_01HXYZ.../events
Authorization: Bearer <token>
Accept: text/event-stream
```

**Response (200, stream)**

```
event: status
data: {"status":"running","progress":{"completed":0,"total":2}}

event: file_completed
data: {"file_id":"file_01HXYZA","status":"completed"}

event: status
data: {"status":"completed","progress":{"completed":2,"total":2}}

event: done
data: {"job_id":"job_01HXYZ...","status":"completed"}
```

事件类型:`status` / `file_started` / `file_completed` / `file_failed` / `done` / `error`。

---

### 6.4 GET `/v1/parse/jobs` — 列出任务

**Query Params**

| 参数 | 说明 |
|------|------|
| `status` | 过滤状态,可逗号分隔 |
| `limit` | 每页数量,默认 20,最大 100 |
| `cursor` | 游标,从上次响应的 `next_cursor` 取 |
| `created_after` | ISO 时间,过滤创建时间 |

**Response (200)**

```json
{
  "items": [
    {
      "job_id": "job_01HXYZ...",
      "status": "completed",
      "created_at": "2026-05-21T08:30:00Z",
      "files_count": 2
    }
  ],
  "next_cursor": "eyJjIjoiMjAyNi0wNS0yMVQwODozMDowMFoifQ==",
  "has_more": true
}
```

---

### 6.5 DELETE `/v1/parse/jobs/{job_id}` — 取消任务

**Response (200)**

```json
{
  "job_id": "job_01HXYZ...",
  "status": "canceled",
  "canceled_at": "2026-05-21T08:30:30Z"
}
```

**Errors**

- `409 job_already_terminal`:任务已是 `completed` / `failed` / `canceled`

---

## 7. Tier 与差异化

Tier 由 Token 决定,但**响应格式完全一致**。

| 能力 | Free(无 Token) | Pro(有 Token) |
|------|----------------|----------------|
| 单文件页数 | ≤ 20 | ≤ 1000 |
| 单文件大小 | ≤ 10 MB | ≤ 200 MB |
| 单任务文件数 | 1 | 100 |
| 并发任务 | 1 | 10+ |
| 队列优先级 | 默认 | 优先 |
| `model` 可选 | `auto` / `pipeline` / `html` | 全部含 `vlm` |
| `output_formats` 高级格式 | 拒绝 `html`/`latex`/`docx` | 全部支持 |
| `callback` | 拒绝 | 支持 |
| 产物保留期 | 24h | 7d |

超限时返回 `403 feature_requires_token` 或 `413 file_too_large`,响应体包含升级提示。

---

## 8. Webhook(异步回调)

任务结束时,服务端 POST 到 `callback.url`:

**Request to your endpoint**

```http
POST https://your.app/mineru-webhook
Content-Type: application/json
X-MinerU-Event: job.completed
X-MinerU-Signature: t=1684740000,v1=5257a869e7ecebeda32...
X-MinerU-Request-Id: req_01HXYZ...

{
  "event": "job.completed",
  "job_id": "job_01HXYZ...",
  "status": "completed",
  "occurred_at": "2026-05-21T08:30:48Z"
}
```

签名计算:`HMAC-SHA256(secret, "{timestamp}.{raw_body}")`。

事件类型:`job.completed` / `job.failed` / `job.canceled` / `job.partial`。

收到 webhook 后,客户端调用 `GET /v1/parse/jobs/{job_id}` 拉取完整结果。

---

## 9. 限流

返回头携带配额信息:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1684740060
Retry-After: 12
```

超限返回 `429 rate_limit_exceeded`。

---

## 10. 端到端示例

### 10.1 完整流程(Pro 用户,1 个 PDF,提供 sha256sum)

```bash
# 1. Create Upload(本地计算 sha256sum)
SHA=$(sha256sum report.pdf | cut -d' ' -f1)
SIZE=$(stat -f%z report.pdf)

RESP=$(curl -s -X POST https://mineru.net/api/v1/uploads \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"report.pdf\",\"bytes\":$SIZE,\"mime_type\":\"application/pdf\",\"purpose\":\"parse\",\"sha256sum\":\"$SHA\"}")
STATUS=$(echo "$RESP" | jq -r '.status')

# ─── 情况 A: 秒传 (200, status=completed) ────────────────
# → {"id":"upload_...","status":"completed","file":{"id":"file-..."}}
# 直接跳到步骤 4

# ─── 情况 B: 需上传 (201, status=pending) ────────────────
if [ "$STATUS" = "pending" ]; then
  UPLOAD_ID=$(echo "$RESP" | jq -r '.id')
  URL=$(echo "$RESP" | jq -r '.upload_url')
  METHOD=$(echo "$RESP" | jq -r '.upload_method')

  # 2. PUT 直传 OSS
  curl -X "$METHOD" "$URL" -H "Content-Type: application/pdf" --data-binary @report.pdf

  # 3. Complete Upload
  RESP=$(curl -s -X POST "https://mineru.net/api/v1/uploads/$UPLOAD_ID/complete" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"sha256sum\":\"$SHA\"}")
fi

FILE_ID=$(echo "$RESP" | jq -r '.file.id')
# → file-01HXYZ123ABCDEF

# 4. 创建任务
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"files\": [{
      \"source\": {\"type\":\"file_id\",\"file_id\":\"$FILE_ID\"},
      \"options\": {\"language\":\"ch\",\"ocr\":\"auto\"}
    }],
    \"model\": \"vlm\",
    \"output_formats\": [\"markdown\",\"json\",\"images\"]
  }"

# → 202 {"job_id":"job_...","status":"queued",...}

# 5. 轮询
curl https://mineru.net/api/v1/parse/jobs/job_... \
  -H "Authorization: Bearer $TOKEN"

# → 200 {"status":"completed","files":[{"output_files":{...}}]}

# 6. 下载产物(通过 Files API)
curl -L "https://mineru.net/api/v1/files/$MD_FILE_ID/content" \
  -H "Authorization: Bearer $TOKEN" -o report.md
```

### 10.2 最简流程(不提供 sha256sum,无秒传/无校验)

```bash
SIZE=$(stat -f%z report.pdf)

# 1. Create Upload
RESP=$(curl -s -X POST https://mineru.net/api/v1/uploads \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"report.pdf\",\"bytes\":$SIZE,\"mime_type\":\"application/pdf\",\"purpose\":\"parse\"}")

UPLOAD_ID=$(echo "$RESP" | jq -r '.id')
URL=$(echo "$RESP" | jq -r '.upload_url')

# 2. PUT 直传 OSS
curl -X PUT "$URL" -H "Content-Type: application/pdf" --data-binary @report.pdf

# 3. Complete Upload
RESP=$(curl -s -X POST "https://mineru.net/api/v1/uploads/$UPLOAD_ID/complete" \
  -H "Authorization: Bearer $TOKEN")

FILE_ID=$(echo "$RESP" | jq -r '.file.id')
# → file-01HXYZ...

# 后续同 10.1 的步骤 4-6
```

### 10.3 同步等待(已有 file_id,一步拿到 markdown)

```bash
# 上传步骤同 10.1 或 10.2,拿到 FILE_ID 后:

# 创建任务 + 同步等待最多 60s
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"files\": [{
      \"source\": {\"type\":\"file_id\",\"file_id\":\"$FILE_ID\"},
      \"options\": {\"language\":\"ch\",\"ocr\":\"auto\"}
    }],
    \"model\": \"vlm\",
    \"output_formats\": [\"markdown\",\"json\"],
    \"wait\": 60
  }"

# → 200 (在 60s 内完成):
# {
#   "job_id": "job_...",
#   "status": "completed",
#   "files": [{
#     "output_files": {
#       "markdown": {"file_id": "file-md-xxx", "content": "# Report\n..."},
#       "json":   {"file_id": "file-json-xxx", "content": {...}}
#     }
#   }]
# }

# → 202 (超时,退化为异步):
# {"job_id":"job_...","status":"running","links":{...}}
# 继续用 GET /v1/parse/jobs/job_... 轮询

# 如果返回 200, markdown 已在 .files[0].output_files.markdown.content 中
# 图片等二进制产物仍需通过 file_id 下载:
# curl -L "https://mineru.net/api/v1/files/$IMG_FILE_ID/content" \
#   -H "Authorization: Bearer $TOKEN" -o image.jpg
```

### 10.4 批量上传(100 个文件,提供 sha256sum 享受秒传)

```bash
# 1. 逐个 Create Upload,服务端自动判定秒传/需上传
> file_ids.txt
for f in *.pdf; do
  SHA=$(sha256sum "$f" | cut -d' ' -f1)
  SIZE=$(stat -f%z "$f")

  RESP=$(curl -s -X POST https://mineru.net/api/v1/uploads \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"filename\":\"$f\",\"bytes\":$SIZE,\"mime_type\":\"application/pdf\",\"purpose\":\"parse\",\"sha256sum\":\"$SHA\"}")

  STATUS=$(echo "$RESP" | jq -r '.status')

  if [ "$STATUS" = "pending" ]; then
    UPLOAD_ID=$(echo "$RESP" | jq -r '.id')
    URL=$(echo "$RESP" | jq -r '.upload_url')

    # 2. PUT 直传 OSS
    curl -s -X PUT "$URL" --data-binary @"$f"

    # 3. Complete Upload
    RESP=$(curl -s -X POST "https://mineru.net/api/v1/uploads/$UPLOAD_ID/complete" \
      -H "Authorization: Bearer $TOKEN")
  fi

  FILE_ID=$(echo "$RESP" | jq -r '.file.id')
  echo "$FILE_ID $f" >> file_ids.txt
done

# 4. 一个任务包含全部文件
FILES_JSON=$(while read fid name; do
  echo "{\"source\":{\"type\":\"file_id\",\"file_id\":\"$fid\"}}"
done < file_ids.txt | jq -s .)

curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"files\":$FILES_JSON,\"model\":\"vlm\",\"output_formats\":[\"markdown\"]}"
```

> 提示:`POST /v1/uploads` 在传入 `sha256sum` 时返回 `status:"completed"`(秒传,含 `file`)或 `status:"pending"`(含 `upload_url`);此外也可用 `GET /v1/files?sha256sum=<hex>` 作纯查询探测。

### 10.5 Free 用户最小示例(URL 来源)

```bash
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "files": [{
      "source": {"type":"url","url":"https://example.com/doc.pdf"}
    }],
    "output_formats": ["markdown"]
  }'

# → 202 {"job_id":"job_...","tier":"free","files":[{"file_id":"file-01HXYZ..."}]}
# (服务端拉取后,响应中的 file_id 已回填,可后续复用免重新上传)
```

### 10.6 与现有三套 API 的能力映射

| 现有 API | 新 API 实现路径 |
|----------|----------------|
| `POST /tasks` (本地 ParseApi,multipart) | `POST /v1/uploads` → `PUT` OSS → `POST /v1/uploads/{id}/complete` → `POST /v1/parse/jobs` |
| `GET status_url` (本地 ParseApi) | `GET /v1/parse/jobs/{id}` |
| `GET result_url` (本地,ZIP) | `GET /v1/files/{id}/content`(逐个) |
| `POST /file-urls/batch` (V4) | `POST /v1/uploads` → `PUT` OSS → `POST /v1/uploads/{id}/complete` → `POST /v1/parse/jobs` |
| `GET /extract-results/batch/{id}` (V4) | `GET /v1/parse/jobs/{id}` |
| `full_zip_url` (V4) | `output_files[]`(job response) → `GET /v1/files/{id}/content` |
| `POST /parse/file` (Flash) | `POST /v1/uploads` → `PUT` → `POST /v1/uploads/{id}/complete` → `POST /v1/parse/jobs`(无 Token) |
| `GET /parse/{task_id}` (Flash) | `GET /v1/parse/jobs/{id}` |
| `markdown_url` (Flash) | `output_files.markdown.file_id` → `GET /v1/files/{id}/content` |
| —(新增) | `POST /v1/uploads` 携带 `sha256sum` 实现秒传;或 `GET /v1/files?sha256sum=<hex>` 探测 |
| —(新增,同步) | `POST /v1/parse/jobs` 携带 `wait:60`,阻塞等待直接拿到 markdown 内容 |

---

## 11. 客户端示意

统一为单个 Parser 类,基址 + 可选 Token 决定档位:

```python
from mineru import MineruApiParser

# 本地自部署
parser = MineruApiParser("http://localhost:8000/api")

# 云端 Free 档
parser = MineruApiParser("https://mineru.net/api")

# 云端 Pro 档
parser = MineruApiParser("https://mineru.net/api", token="msk_...")

result = parser.parse("./report.pdf")
print(result.markdown())
result.save(writer)
```

调用代码完全相同,Token 仅决定能不能用 `model="vlm"` 或 `output_formats=["docx"]`。

---

## 12. 与现有三套 API 的逐步对比

新 API 与现有三套 API 在概念上有三处根本归一:

1. **上传与解析分离**:文件上传(content-addressed)与任务调度(job)是独立资源
2. **预签名 URL 统一**:本地/云端都通过 `POST /v1/uploads` → `PUT` 上传字节,不再分 multipart 直传与两步 PUT
3. **产物清单替代 ZIP 包**:每个产物独立 URL,客户端按需下载,不再统一下整包

下面分别对比每个旧 API 的每个调用步骤。

### 12.1 vs Local ParseApi(本地 FastAPI,multipart 直传)

**旧流程**:`POST /tasks` (multipart) → 轮询 `status_url` → `GET result_url` (ZIP)

#### Call 1 — 提交任务

**旧**
```http
POST /tasks  HTTP/1.1
Content-Type: multipart/form-data; boundary=...

--...
Content-Disposition: form-data; name="files"; filename="doc.pdf"
Content-Type: application/octet-stream

<binary bytes>
--...
Content-Disposition: form-data; name="lang_list"
ch
--...
Content-Disposition: form-data; name="backend"
pipeline
--...
Content-Disposition: form-data; name="return_md"
true
--...
(return_middle_json, return_content_list, return_images, ...)
--...--

→ 202
{"status_url":"http://.../status/abc","result_url":"http://.../result/abc"}
```

**新**(等价路径)
```http
POST /v1/uploads  HTTP/1.1
Content-Type: application/json

{"filename":"doc.pdf","bytes":1048576}
→ 201 {"id":"upload_...","upload_url":"http://localhost:8000/_upload/..."}

PUT {upload_url}
<binary bytes>
→ 200

POST /v1/uploads/{id}/complete
→ 200 {"status":"completed","file":{"id":"file-..."}}

POST /v1/parse/jobs  HTTP/1.1
Authorization: (省略)
Content-Type: application/json

{
  "files":[{"source":{"type":"file_id","id":"file-..."},
            "options":{"language":"ch"}}],
  "model":"pipeline",
  "output_formats":["markdown","json","content_list","images"]
}
→ 202 {"job_id":"job_...","status":"queued","links":{...}}
```

**差异**
- 旧:单次 `multipart/form-data` 把字节+配置打包;新:分 3 步(upload 申请 / PUT 字节 / 创建 job)
- 旧:配置开关散落在 form 字段(`return_md`, `return_middle_json`, ...);新:统一收纳到 `output_formats` 数组
- 旧:每次调用都重新上传;新:可秒传(若提供 sha256)
- 旧:无文件级标识;新:不透明 `file_id`(`file-xxx`)+ 可选 `sha256sum` 元信息,file_id 可跨任务复用
- 旧:`backend` 字段直接选实现;新:`model` 字段语义化(`pipeline`/`vlm`/`html`/`auto`)

#### Call 2 — 查询状态

**旧**
```http
GET http://.../status/abc
→ 200 {"status":"completed"}     # 或 running / failed
```

**新**
```http
GET /v1/parse/jobs/job_...
Authorization: Bearer <token>
→ 200 {
  "job_id":"job_...",
  "status":"completed",
  "progress":{"completed":1,"total":1},
  "files":[{"file_id":"file-01HXYZ...","output_files":{...}}]
}
```

**差异**
- 旧:status URL 由服务端在 Call 1 返回,客户端按 URL 轮询;新:RESTful 资源,客户端用 `job_id` 拼路径
- 旧:响应仅包含 `status`,结果在另一个 URL;新:同一响应里包含状态 + 进度 + 全部产物 URL(完成时)
- 新增 `progress` 字段、partial 状态

#### Call 3 — 下载结果

**旧**
```http
GET http://.../result/abc
→ 200 (Content-Type: application/zip)
<ZIP bytes>

# 解压后内嵌:
#   {name}/{method}/full.md
#   {name}/{method}/{name}_middle.json
#   {name}/{method}/{name}_content_list.json
#   {name}/{method}/{name}_model.json
#   {name}/{method}/images/*.jpg
```

**新**(逐文件下载)
```http
GET /v1/files/file-MD.../content
→ 302 Location: https://cdn/.../doc.md

GET /v1/files/file-JSON.../content
→ 302 Location: https://cdn/.../result.json

# 图片同理,可并行下载
```

**差异**
- 旧:单 ZIP 包含全部产物,客户端必须先下整包再解压;新:每个产物独立 URL,按需下载
- 旧:产物路径嵌套且依赖 `{name}/{method}/` 约定;新:扁平 ID 寻址,无目录推断
- 新:CDN 直连(302 + 预签名),不经过 API 网关

---

### 12.2 vs MinerU V4 API(云端付费,Bearer token + 两步 PUT)

**旧流程**:`POST /file-urls/batch` → `PUT` 每个 file_url → 轮询 `GET /extract-results/batch/{id}` → `GET full_zip_url`

#### Call 1 — 申请上传 URL(批量)

**旧**
```http
POST /api/v4/file-urls/batch  HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "files":[
    {"name":"doc.pdf","is_ocr":true,"page_ranges":"1-10"}
  ],
  "model_version":"vlm",
  "language":"ch",
  "enable_formula":true,
  "enable_table":true,
  "extra_formats":["docx"]
}

→ 200 {
  "code":0,
  "msg":"ok",
  "data":{
    "batch_id":"batch_...",
    "file_urls":["https://oss.../signed-url-1"]
  }
}
```

**新**(申请上传与解析配置分两步)
```http
POST /v1/uploads
{"filename":"doc.pdf","bytes":...,"sha256sum":"a1b2..."}
→ 201 {"file_id":"file-01HXYZ...","upload_id":"upload_...","upload_url":"https://oss..."}

# Call 1 此处仅含上传申请;解析参数留到 Call 3 (POST /v1/parse/jobs)
```

**差异**
- 旧:单次调用既申请上传又锁定解析参数(`batch_id` 隐式绑定);新:`POST /v1/uploads` 只管字节,解析参数留到 `POST /v1/parse/jobs`,可在不同 jobs 里复用同一 `file_id`
- 旧:响应有 `{code,msg,data}` envelope;新:直接返回资源对象,错误用 HTTP 状态码 + `{error:{code,...}}`
- 旧:`extra_formats` / 文档格式由 Token 等级隐式控制;新:`output_formats` 统一收纳,高级格式(`docx`/`html`/`latex`)需 Token,无权限返回 `403 feature_requires_token`
- 旧:每次必上传;新:`sha256sum` 命中可秒传

#### Call 2 — 上传字节

**旧**
```http
PUT https://oss.../signed-url-1
Content-Type: application/pdf
<binary bytes>
→ 200
```

**新**
```http
PUT {upload_url}
Content-Type: application/pdf
<binary bytes>
→ 200
# 若申请时带了 sha256,服务端异步校验;不一致则 file_id 失效
```

**差异**
- 调用形式几乎一致(都是预签名 PUT)
- 新增:可选 SHA-256 校验
- 旧:URL 由 `file-urls/batch` 一次性下发多个;新:每个文件一次 `POST /v1/uploads`(更细粒度的配额控制)

#### Call 3 — 创建/触发解析

**旧**:无独立调用 — 解析在 PUT 完成后由 `batch_id` 自动触发。

**新**:显式创建 job,这是新 API 的关键概念差异。
```http
POST /v1/parse/jobs
{
  "files":[{"source":{"type":"file_id","id":"file-01HXYZ..."},
            "options":{"language":"ch","ocr":"auto","page_range":"1-10"}}],
  "model":"vlm",
  "output_formats":["markdown","json","images","docx"]
}
→ 202 {"job_id":"job_...","status":"queued"}
```

**差异**
- 旧:上传 = 解析任务(`batch_id` 同时是上传组和解析任务);新:文件与任务解耦,**同一 `file_id` 可被多个 jobs 复用**(切换 model、改 page_range、跑不同 output_formats 都不必重传)
- 旧:`page_ranges` 写在 Call 1 的 file 条目里;新:写在 job 的 `options`(因为属于解析行为)
- 旧:`model_version` 在 Call 1 顶层;新:`model` 在 `POST /v1/parse/jobs` 顶层

#### Call 4 — 轮询结果

**旧**
```http
GET /api/v4/extract-results/batch/batch_...
Authorization: Bearer <token>

→ 200 {
  "code":0,
  "data":{
    "extract_result":[
      {"file_name":"doc.pdf","state":"done","full_zip_url":"https://oss.../result.zip"},
      {"state":"running"},
      {"state":"failed","err_msg":"..."}
    ]
  }
}
```

**新**
```http
GET /v1/parse/jobs/job_...
→ 200 {
  "status":"partial",
  "files":[
    {"file_id":"file-01HXYZ...","status":"completed","output_files":{
       "markdown":{"url":"..."},"json":{"url":"..."}, "images":[...]
    }},
    {"file_id":"file-01HXYZ...","status":"failed","error":{...}}
  ]
}
```

**差异**
- 旧:`{code,msg,data}` envelope;新:扁平资源对象 + HTTP 状态码
- 旧:`state` 取值 `done`/`running`/`failed`;新:`status` 含 `queued`/`running`/`completed`/`partial`/`failed`/`canceled`(多出 `partial` 表示部分成功)
- 旧:成功时只给 `full_zip_url`;新:逐 artifact URL,可只下需要的
- 新增:`progress`、`metadata.processing_time_ms` 等可观测字段

#### Call 5 — 下载结果

**旧**
```http
GET https://oss.../result.zip
→ 200 (application/zip)

# 解压后:
#   layout.json
#   {name}_content_list.json
#   full.md
#   images/*.jpg
```

**新**
```http
GET /v1/files/file-MD.../content    → 302 → markdown 文件
GET /v1/files/file-JSON.../content    → 302 → result.json
GET /v1/files/file-IM.../content    → 302 → 单张图片
```

**差异**(同 12.1 Call 3,要点一致)
- 单 ZIP → 多 URL 拆分
- 旧 ZIP 内文件名/结构需客户端解析约定;新 artifact 类型在 metadata 中明确

---

### 12.3 vs MinerU Flash API(云端免费,无 token)

**旧流程**:`POST /parse/file` → `PUT file_url` → 轮询 `GET /parse/{task_id}` → 下载 `markdown_url`

#### Call 1 — 申请上传 + 创建任务(合并)

**旧**
```http
POST /api/v1/agent/parse/file
Content-Type: application/json

{
  "file_name":"doc.pdf",
  "language":"ch",
  "page_range":"1-5",
  "is_ocr":true,
  "enable_formula":true,
  "enable_table":true
}

→ 200 {
  "code":0,
  "data":{
    "task_id":"task_...",
    "file_url":"https://oss.../signed-url"
  }
}
```

**新**(拆为两步)
```http
POST /v1/uploads
{"filename":"doc.pdf","bytes":...}
→ 201 {"upload_id":"upload_...","upload_url":"https://oss..."}

# (上传 PUT 略)

POST /v1/parse/jobs
{
  "files":[{"source":{"type":"file_id","file_id":"file-..."},
            "options":{"language":"ch","page_range":"1-5","ocr":"auto"}}],
  "output_formats":["markdown"]
}
→ 202 {"job_id":"job_...","tier":"free"}
```

**差异**
- 旧:一个 POST 同时拿到 `task_id` + `file_url`(上传与任务耦合);新:两个 POST(`/v1/uploads` 拿 `upload_id`+`upload_url`,`/v1/parse/jobs` 拿 `job_id`)
- 旧:**单文件**任务;新:任意数量文件(Free 档限 1,Pro 档限 100)
- 旧:`{code,msg,data}` envelope;新:扁平资源
- 旧:无 sha256/秒传概念;新:可选秒传
- 旧:任务参数写在 Call 1 顶层;新:文件级参数(`page_range`, `language`)在 `files[].options`,任务级参数(`model`, `output_formats`)在 job 顶层

#### Call 2 — 上传字节

```http
PUT {file_url 或 upload_url}
<binary bytes>
→ 200
```

旧/新基本一致 — 都是预签名 PUT。

#### Call 3 — 轮询状态

**旧**
```http
GET /api/v1/agent/parse/task_...

→ 200 {
  "code":0,
  "data":{
    "state":"done",            # / running / failed
    "markdown_url":"https://oss.../result.md",
    "err_msg":""
  }
}
```

**新**
```http
GET /v1/parse/jobs/job_...

→ 200 {
  "status":"completed",
  "files":[{
    "file_id":"file-01HXYZ...",
    "output_files":{"markdown":{"url":"https://cdn/.../doc.md"}}
  }]
}
```

**差异**
- 旧:扁平结构(单文件,markdown_url 直接在 data 上);新:`files[]` 数组(即使单文件),`output_files` 字典支持多种产物
- 旧:仅可获得 `markdown_url`;新:Free 档默认 markdown,但若客户端 `output_formats` 列了 json/content_list,Pro 档亦可拿到(Free 档收到 `403 feature_requires_token`)

#### Call 4 — 下载 Markdown

**旧**
```http
GET https://oss.../result.md
→ 200 (text/markdown)
# ... markdown 文本
```

**新**
```http
GET /v1/files/file-MD.../content
→ 302 Location: https://cdn/.../doc.md
# ... markdown 文本
```

**差异**
- 旧:`markdown_url` 由 Call 3 直接给出,客户端 GET 即可;新:通过 `/v1/files/{file_id}/content` 间接寻址,302 重定向到 CDN
- 间接寻址的好处:统一鉴权与计量入口,且 URL 可在产物级别独立失效

---

### 12.4 整体对比矩阵

| 维度 | Local ParseApi | V4 API | Flash API | 新 API |
|------|---------------|--------|-----------|--------|
| 鉴权 | 无 | Bearer 必填 | 无 | Bearer 可选(决定 tier) |
| 上传方式 | multipart 同请求 | 预签名 PUT(批量下发) | 预签名 PUT(单个) | 预签名 PUT(每文件) |
| 文件标识 | 无 | 无(batch 内序号) | 无(task 内单一) | 不透明 `file-xxx` + 可选 `sha256sum` |
| 秒传 | 不支持 | 不支持 | 不支持 | 支持(可选) |
| 任务粒度 | 单/多 | 多(batch) | 单 | 多(job) |
| 任务复用文件 | 否 | 否 | 否 | **是**(同 file_id 可跨 jobs) |
| 响应 envelope | 直接 JSON | `{code,msg,data}` | `{code,msg,data}` | 直接资源对象 |
| 产物交付 | 单 ZIP | 单 ZIP | 单 markdown URL | artifact 清单 + 独立 URL |
| 产物可选范围 | 多个 `return_*` 开关 | 固定包内含全部 | 仅 markdown | `output_formats` 数组 |
| 异步通知 | 无 | 无 | 无 | Webhook(Pro)+ SSE |
| 部分失败语义 | 无显式 | 列表内每条 `state` | 单任务无 | `status:partial` + 逐文件 error |

### 12.5 调用次数(单文件、不秒传、要 markdown)

| API | HTTP 调用次数 | 序列 |
|-----|--------------|------|
| Local ParseApi | 3 (+轮询) | POST /tasks → poll status → GET result |
| V4 API | 4 (+轮询) | POST file-urls/batch → PUT → poll → GET zip |
| Flash API | 4 (+轮询) | POST parse/file → PUT → poll → GET md |
| 新 API | 4 (+轮询) | POST /v1/uploads → PUT → POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(秒传命中) | 3 (+轮询) | POST /v1/uploads → POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(同 file_id 复用) | 2 (+轮询) | POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(同步 wait) | 1 | POST /v1/parse/jobs {wait:60} → 200 content 内联 |

> 同 file_id 复用是新 API 独有优势:相同文件换个 `model` / `page_range` / `output_formats` 重跑无需任何上传。

