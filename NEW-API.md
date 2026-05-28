# MinerU Unified API (v1)

> 一套 REST API,覆盖当前 ParseApi(本地)、MineruV4(云端注册用户)、Flash(云端匿名用户) 三套 API 的全部功能。
>
> **核心约定**:Token 仅影响配额、并发、队列优先级与可选项,**不影响请求/响应格式**。

---

## 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-05-28 | **Callback `seed` → `secret`**:签名字段恢复为 `secret`,`seed` 易与 Chat Completions 确定性采样参数混淆 |
| 2026-05-27 | **ID 格式统一**:File 对象 `file_<ULID>` → `file-<ULID>`(连字符),全文对齐 |
| 2026-05-27 | **Upload 状态码统一**:秒传命中与需上传均返回 200,通过 body `status` 区分 |
| 2026-05-27 | **wait 参数调整**:合法值改为 `0` / `[5, 50]`,与 nginx 默认 60s 超时安全对齐 |
| 2026-05-27 | **Tier 重命名**:`Free` → `anonymous`(无 Token),`Pro` → `registered`(有 Token),两档不变 |
| 2026-05-27 | **产物保留期修正**:统一为 30d(section 9),此前写 24h/7d 是笔误 |
| 2026-05-27 | **Health 端点精简**:弱化为仅探测 server 连通性(`status` + `version`),移除 models 检查 |
| 2026-05-27 | **Model 端点补充**:为模型对象添加 `description` 字段(对齐 OpenAI) |
| 2026-05-27 | **Files 分页默认值**:`limit` 默认 100、最大 1000(原 10000 过大) |
| 2026-05-27 | **Jobs 分页统一**:对齐 OpenAI 游标分页风格(`after`/`first_id`/`last_id`/`has_more`),取代 `cursor`/`next_cursor` |
| 2026-05-27 | **通用约定补充**:Idempotency-Key、X-Request-Id、URL source 安全约束(SSRF 防护)、inline/local 限制 |
| 2026-05-27 | **sha256sum 查询参数移除**:删除 `GET /v1/files?sha256sum=` 参数,秒传探测统一由 `POST /v1/uploads` 内置实现 |
| 2026-05-27 | **Webhook 重写**:签名方案改为 SHA256(uid+seed+content),`seed` 替代 `secret`,增加重试规则 |
| 2026-05-27 | **SSE/Webhook 可用性**:明确 mineru.net 支持两者,本地 server 均不支持 |
| 2026-05-27 | **Chat Completions 定位**:补充与 Parse Jobs 的差异说明(细粒度 vs 全文档) |
| 2026-05-27 | **Usage 端点**:新增 `GET /v1/usage` 查询当日用量与 tier 配额上限 |
| 2026-05-27 | **ID 格式确定**:采用 24 字符 base62 随机串(对齐 OpenAI),`file-` 用连字符,`upload_`/`job_` 用下划线 |
| 2026-05-27 | **Presets 端点**:新增 `GET /v1/presets`(§4.3),列出所有 preset 及其当前模型;§9 放开 anonymous 的 preset 限制 |
| 2026-05-27 | **Health 能力探测**:`features` 字段自描述 SSE/Webhook 可用性(§3),客户端按能力选择通知策略 |
| 2026-05-27 | **限流粒度确定**:per-token、per-minute、按端点类别(parse/upload/chat/read)四档分桶,独立限额(§11.1) |
| 2026-05-27 | **`input_image` purpose 新增**:Chat/Responses 图片文件使用 `purpose=input_image` 上传,区别于 `parse` 和 `parse_output`(§5.1/§6.1/§7.2) |
| 2026-05-27 | **Chat/Responses 输入限制**:每条消息最多 1 个图片或文件,图片 < 10MiB 且分辨率 < 3500×3500(§6.1) |
| 2026-05-27 | **anonymous 能力对齐**:anonymous 与 registered 在单文件页数/大小、任务文件数、并发上完全一致,差异仅限流/高级格式/callback(§9) |
| 2026-05-27 | **匿名用户身份与权限**:anonymous 按 IP 标识(限流/用量),`GET /v1/files` 和 `GET /v1/parse/jobs` 返回 `403 list_requires_token`(§1.3/§5.8/§8.4/§11) |

---

## 1. 整体设计

### 1.1 资源模型

| 资源 | 含义 |
|------|------|
| `file` | 平台中的文件,由不透明的 `file_id` 唯一标识。包括三种用途:用户上传的源文件(`purpose:parse`)、Chat/Responses 接口的图片输入(`purpose:input_image`)、解析产物如 markdown/json 等(`purpose:parse_output`)。`sha256sum` 是可选元信息,用于秒传与校验 |
| `upload` | OpenAI Uploads API 风格的上传会话对象,驱动文件上传生命周期 |
| `job` | 一次解析任务(可包含多个文件) |

**ID 格式**

所有资源 ID 采用 OpenAI 风格:前缀 + 24 字符 base62 随机串(大小写字母 + 数字)。不包含时间信息,全局唯一,不可逆推。

| 资源 | ID 示例 | 备注 |
|------|---------|------|
| `file` | `file-r9NSmHLJE6flShV5vQ0Y60Rd` | 不透明随机 ID;全局唯一;同内容不同用户/不同上传得到不同 file_id |
| `upload` | `upload_r9NSmHLJE6flShV5vQ0Y60Rd` | OpenAI 风格,创建 Upload 时分配(注意前缀用下划线,同 OpenAI `upload_` 前缀) |
| `job` | `job_r9NSmHLJE6flShV5vQ0Y60Rd` | 解析任务 |

> **文档约定**:为可读性,后文代码示例中 ID 使用缩写占位符(如 `file-01HXYZ...`、`upload_01HXYZ...`),实际 ID 长度和格式以此处定义为准。

### 1.2 端点总览

> 下表中所有路径相对于 Base URL(§1.4)。完整路径 = `{base_url}/v1/...`,如 `https://mineru.net/api/v1/health`。

| Method | Path | 用途 |
|--------|------|------|
| `GET` | `/v1/health` | 健康检查 |
| `GET` | `/v1/models` | 列出可用解析模型 |
| `GET` | `/v1/models/{model}` | 查询单个模型信息 |
| `GET` | `/v1/presets` | 列出可用解析预设 |
| `POST` | `/v1/uploads` | 创建 Upload(可选传 sha256sum 启用秒传) |
| `POST` | `/v1/uploads/{upload_id}/complete` | 完成 Upload,生成 File 对象 |
| `POST` | `/v1/uploads/{upload_id}/cancel` | 取消 Upload |
| `GET` | `/v1/uploads/{upload_id}` | 查询 Upload 状态 |
| `GET` | `/v1/files` | 列出当前租户的文件(游标分页) |
| `GET` | `/v1/files/{file_id}` | 查询文件元信息 |
| `GET` | `/v1/files/{file_id}/content` | 下载解析产物(302 重定向到 CDN,仅 `parse_output`) |
| `DELETE` | `/v1/files/{file_id}` | 从租户视图删除文件 |
| `POST` | `/v1/chat/completions` | OpenAI 兼容的文档对话/生成(Chat API) |
| `POST` | `/v1/responses` | OpenAI 兼容的文档对话/生成(Responses API) |
| `POST` | `/v1/parse/jobs` | 创建解析任务,支持同步等待(`wait` 参数) |
| `GET` | `/v1/parse/jobs/{job_id}` | 查询任务状态与结果 |
| `GET` | `/v1/parse/jobs/{job_id}/events` | SSE 流式状态推送(mineru.net 支持,本地 server 不支持) |
| `GET` | `/v1/parse/jobs` | 列出任务(分页) |
| `DELETE` | `/v1/parse/jobs/{job_id}` | 取消任务 |
| `GET` | `/v1/usage` | 查询当前用量与配额上限 |

> **SSE 与 Webhook 可用性**:mineru.net 云端同时支持 SSE 事件流(§8.3)和 Webhook 回调(§10);本地自部署 server **两者均不支持**,客户端须使用 `GET /v1/parse/jobs/{job_id}` 轮询。

### 1.3 认证

```
Authorization: Bearer <MINERU_TOKEN>
```

- **缺省 Token**:匿名访问,自动降级为 anonymous 档。anonymous 用户的身份标识为**请求 IP**,限流和用量均按 IP 统计。
- **有效 Token**:按 Token 等级解锁更高配额、`docx`/`html`/`latex` 等高级输出格式。
- **无效 Token**:返回 `401 Unauthorized`(注意:**不传 Token ≠ Token 无效**)。

### 1.4 通用约定

- **基址(Base URL)**
  - 云端:`https://mineru.net/api`
  - 本地:`http://localhost:8000/api`(自部署)
- **版本号**:URL 前缀 `/v1`,新增字段只增不删
- **Content-Type**:`application/json; charset=utf-8`(除上传字节流)
- **时间**:ISO-8601 UTC 字符串(如 `2026-05-21T08:30:00Z`)或 Unix 秒级时间戳(如 `1719184911`),各端点按场景选用。JOB/Webhook 等任务相关字段用 ISO-8601,File/Upload/Model/ChatCompletion 等资源元数据字段用 Unix 时间戳
- **ID 格式**:前缀 + 24 字符 base62 随机串,如 `file-r9NSmHLJE6flShV5vQ0Y60Rd`、`upload_r9NSmHLJE6flShV5vQ0Y60Rd`、`job_r9NSmHLJE6flShV5vQ0Y60Rd`。`file` 用连字符 `-`,`upload` 和 `job` 用下划线 `_`(对齐 OpenAI)。代码示例中使用缩写占位符(如 `file-01HXYZ...`)
- **空字段**:用 `null`,不省略 key
- **Idempotency-Key**:`POST` 端点支持 `Idempotency-Key` 请求头。传递相同 key 的重复请求返回首次请求的结果,不会重复创建资源(如重复扣费/重复创建 job)。Key 有效期 24 小时
- **X-Request-Id**:所有响应(含成功与错误)均携带 `X-Request-Id` 头,用于排查问题
- **URL source 安全约束**(`POST /v1/parse/jobs` 的 `source.type: "url"`):
  - 拒绝私有网段 IP(10.0.0.0/8、172.16.0.0/12、192.168.0.0/16)及链路本地地址
  - 拒绝 cloud metadata IP(如 `169.254.169.254`)
  - 下载超时:服务端拉取超时 30s,大小上限与当前 tier 单文件限制一致(参见 §9)
  - 拉取失败时 job 中对应 `file` item 的 `status` 为 `"failed"`
- **`inline` source 大小限制**:`data` 字段为 base64 编码后的字符串,大小不超过 **1MB**(指 base64 字符串字节数,非 decode 后原始字节)
- **`local` source 安全约束**:仅本地 server 支持;服务端须通过部署时配置的 `allowlist_root` 限制可读目录范围;拒绝路径穿越(`../`、符号链接跳转到 allowlist 外) 

---

## 2. 通用响应结构

### 2.1 成功响应

直接返回资源对象,**无外层 envelope**。所有成功响应均携带 `X-Request-Id` 头。

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
    "message": "File exceeds the 200MB limit.",
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
| 400 | `bytes_mismatch` | 实际上传字节数与创建时声明的 `bytes` 不一致 |
| 401 | `invalid_token` | Token 无效或过期 |
| 403 | `quota_exceeded` / `feature_requires_token` / `list_requires_token` | 配额耗尽、功能受限或无 scope 访问列表接口 |
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

无需鉴权。仅探测服务端连通性与能力,不检查下游模型后端状态。

```json
{
  "status": "ok",
  "version": "1.0.0",
  "features": {
    "sse": true,
    "webhook": true
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"ok"` |
| `version` | string | API 服务版本号 |
| `features.sse` | bool | 是否支持 SSE 事件流(§8.3) |
| `features.webhook` | bool | 是否支持 Webhook 回调(§10) |

> **部署差异**:mineru.net 返回 `sse: true, webhook: true`;本地 server 返回 `sse: false, webhook: false`。客户端按此自动选择轮询/SSE/webhook 策略。

---

## 4. Models — 解析模型

模型列表与查询接口对齐 **OpenAI Models API** 的请求与响应形态。**无需鉴权**,匿名访问返回完整模型列表(tier 不改变可见模型)。

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
      "id": "pipeline",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "Local pipeline-based parsing model (no GPU required)."
    },
    {
      "id": "MinerU2.5-Pro-2604-1.2B",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "VLM-based high-accuracy parsing model."
    },
    {
      "id": "MinerU2.5-Pro-2605-1.2B",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "VLM-based high-accuracy parsing model (latest)."
    },
    {
      "id": "MinerU-HTML",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "HTML page parsing model."
    }
  ]
}
```

**字段说明(对齐 OpenAI)**

| 字段 | 类型 | 说明 |
|------|------|------|
| `object`(顶层) | string | 固定为 `"list"` |
| `data[]` | array | 模型对象数组 |
| `data[].id` | string | 模型标识 |
| `data[].object` | string | 固定为 `"model"` |
| `data[].created` | int | Unix 秒级时间戳(模型首次上线时间) |
| `data[].owned_by` | string | 拥有者(本平台均为 `"mineru"`,自部署可设组织名) |
| `data[].description` | string | 模型描述(可选,对齐 OpenAI) |

> 说明:本端点仅保留 OpenAI 标准字段,**不再包含** `available_to` / `supports` 等本 API 早期字段。模型适用性(`available_to`)由 Tier(参见 §9)决定;格式适用性(`supports`)在使用模型时由服务端自动校验。

### 4.2 GET `/v1/models/{model}` — 查询单个模型

**Request**

```http
GET /v1/models/vlm  HTTP/1.1
Authorization: Bearer <token>
```

**Path Parameters**

| 参数 | 说明 |
|------|------|
| `model` | 模型 ID(`pipeline` / `MinerU2.5-Pro-2604-1.2B` / `MinerU-HTML`) |

**Response (200)**

```json
{
  "id": "MinerU2.5-Pro-2604-1.2B",
  "object": "model",
  "created": 1700000000,
  "owned_by": "mineru",
  "description": "VLM-based high-accuracy parsing model."
}
```

**Errors**

- `404 model_not_found`:模型 ID 不存在
- `403 feature_requires_token`:模型存在但当前 tier 无权使用(响应体含 `upgrade_url`)

### 4.3 GET `/v1/presets` — 列出可用预设

列出所有 parser preset 及其当前对应的模型。**无需鉴权**,匿名访问返回完整列表。

**Request**

```http
GET /v1/presets  HTTP/1.1
```

**Response (200)**

```json
{
  "object": "list",
  "data": [
    {
      "id": "pipeline",
      "description": "Local pipeline-based parsing (no GPU required).",
      "current_model": "pipeline"
    },
    {
      "id": "vlm",
      "description": "VLM-based high-accuracy parsing.",
      "current_model": "MinerU2.5-Pro-2605-1.2B"
    },
    {
      "id": "html",
      "description": "HTML page parsing.",
      "current_model": "MinerU-HTML"
    },
    {
      "id": "auto",
      "description": "Platform selects best preset per document.",
      "current_model": null
    }
  ]
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `object`(顶层) | string | 固定为 `"list"` |
| `data[]` | array | preset 对象数组 |
| `data[].id` | string | preset 标识,对应 `POST /v1/parse/jobs` 的 `preset` 参数 |
| `data[].description` | string | 用途说明 |
| `data[].current_model` | string \| null | 当前该 preset 背后实际使用的模型 ID,对应 `GET /v1/models` 中的 `id`。`auto` 不绑定单一模型,为 `null` |

> **设计说明**:`preset` 是平台精选的解析方案组合,由服务端维护对应的模型版本。客户端用 `preset` 无需关心具体模型,平台升级模型版本时无感。`current_model` 与 job 响应中的 `metadata.model_used` 形成追踪链——用户可从 job 结果反查解析时所使用的模型。

---

## 5. Files & Uploads

文件上传采用 **OpenAI Uploads API 的三段式生命周期**:Create Upload → 客户端直传 OSS → Complete Upload。对齐 OpenAI 协议外壳,但字节传输路径适应低带宽 API 服务器。

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
     │ ◄── 200 {id:"upload_...",            │                            │
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
     │            file_id:"file-..."}}]}    │                            │
     ├─────────────────────────────────────►│                            │
```

### 5.1 概念

**Upload 对象**(资源类型: `upload`)

| 字段 | 说明 |
|------|------|
| `id` | `upload_<24位base62随机>`,创建时由服务端分配 |
| `object` | 固定为 `"upload"` |
| `bytes` | 预期上传字节数 |
| `created_at` | Unix 秒级时间戳 |
| `expires_at` | Unix 秒级时间戳(默认 1 小时后;可配 `expires_after`) |
| `filename` | 文件名 |
| `purpose` | `"parse"` 或 `"input_image"` |
| `mime_type` | MIME 类型 |
| `sha256sum` | 可选 64 位小写 hex,创建时提供 |
| `status` | `pending` / `completed` / `cancelled` / `expired` |
| `file` | 仅 `status=completed` 时存在,内嵌 File 对象 |

**File 对象**(资源类型: `file`)

| 字段 | 说明 |
|------|------|
| `id` | `file-<24位base62随机>`,全局唯一的不透明标识 |
| `object` | 固定为 `"file"` |
| `bytes` | 文件字节数 |
| `created_at` | Unix 秒级时间戳(文件首次上传时间) |
| `expires_at` | Unix 秒级时间戳(到期时间;`null` = 手动删除前永不过期) |
| `filename` | 文件名 |
| `purpose` | `"parse"` / `"input_image"` / `"parse_output"` |
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
| `purpose` | string | 是 | `"parse"` 或 `"input_image"` |
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

**Response (200) — 需要上传**

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

| 字段 | 秒传(200) | 需上传(200) |
|------|-----------|------------|
| `id` | ✓ | ✓ |
| `object` | `"upload"` | `"upload"` |
| `bytes` | ✓ | ✓ |
| `created_at` | ✓ | ✓ |
| `expires_at` | ✓ | ✓ |
| `filename` | ✓ | ✓ |
| `purpose` | 等于输入 | 等于输入 |
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

### 5.4 部署场景的行为差异

API 定义完全一致,但三种部署场景下部分行为有差异:

| 场景 | 文件来源 | 上传方式 | 产物下载 |
|------|---------|---------|---------|
| **mineru.net**(云端) | `file_id` / `url` / `inline` | `upload_url` 指向 OSS,字节直传 OSS | 302 重定向到 CDN |
| **LAN server**(局域网) | `file_id` / `url` / `inline` / `local` | `upload_url` 指向 API server 自身,字节直传 API server;`local` 跳过上传 | 200 + body,不重定向 |
| **localhost**(本机) | `local` / `file_id` / ... | `local` 直接读磁盘;其他与 LAN 相同 | 200 + body,不重定向 |

**上传差异**:云端 `upload_url` 指向 OSS,LAN/本机指向 API server 自身的临时上传端点(如 `/_upload/{upload_id}`)。客户端调用流程(`POST /v1/uploads` → `PUT upload_url` → `POST /v1/uploads/{id}/complete`)完全相同,仅 URL 目标不同。

**下载差异**:云端 `GET /v1/files/{id}/content` 返回 302 重定向到 CDN,LAN/本机直接返回 200 + body。客户端使用 `-L` 跟随重定向即可通吃。

**`local` source 差异**:mineru.net 拒绝此来源返回 `400 invalid_request`,LAN/本机直接读取磁盘。

### 5.5 POST `/v1/uploads/{upload_id}/complete` — Complete Upload

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

### 5.6 POST `/v1/uploads/{upload_id}/cancel` — Cancel Upload

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

### 5.7 GET `/v1/uploads/{upload_id}` — 查询 Upload

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

### 5.8 GET `/v1/files` — 列出文件

列出当前租户的全部文件,游标分页(分页设计参考 OpenAI Files API)。

> anonymous 用户调用此端点返回 `403 list_requires_token`,需 Bearer Token 才能列出文件列表。

**Query Parameters**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `after` | string | `null` | 游标,上一页响应的 `last_id`。从其下一条开始返回 |
| `limit` | int | `100` | 返回数量上限,范围 `[1, 1000]` |
| `order` | enum | `desc` | 按 `created_at` 排序,`asc` 升序 / `desc` 降序 |
| `purpose` | string | `null` | 按用途过滤:`parse`(源文件) / `input_image`(Chat 输入图片) / `parse_output`(解析产物)。不传则返回全部 |

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
| ID 形式 | `file-abc123` | `file-<24位base62随机>` | ✅ 对齐 |
| 大小字段名 | `bytes` | ✅ 对齐 | |
| 时间格式 | Unix timestamp | ✅ 对齐 | |
| `purpose` | 多种 | `parse` + `input_image` + `parse_output` | 源文件 / Chat 输入 / 解析产物 |
| `status` | 有(已 deprecated) | 无 | 跟随 OpenAI 弃用 |

### 5.9 GET `/v1/files/{file_id}` — 元信息

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

### 5.10 GET `/v1/files/{file_id}/content` — 下载解析产物

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

### 5.11 DELETE `/v1/files/{file_id}` — 删除

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

## 6. Chat Completions — 文档对话

OpenAI 兼容的同步聊天/生成 API,为文档解析 pipeline 提供自然语言接口。
当前仅支持**单轮问答**——每次请求独立处理,不维护会话历史。

**与 Parse Jobs 的定位差异**:Chat Completions 面向**细粒度解析场景**——单张图片 OCR、单个表格提取、一段文字识别等原子操作,适合按需小范围调用。而 `POST /v1/parse/jobs` 面向**全文档解析**——PDF/DOCX/PPTX 等多页文档的结构化提取。两者互补:Chat 走单图单文本通路,Jobs 走完整 pipeline 通路。

### 6.1 消息格式(message roles)

支持以下 role,不支持 `assistant`、`tool`、`function`:

| Role | content 类型 | 说明 |
|------|-------------|------|
| `system` | `string` 或 `[{"type":"text", "text":...}]` | 系统指令 |
| `developer` | `string` 或 `[{"type":"text", "text":...}]` | OpenAI 新标准的系统指令(与 `system` 等价处理) |
| `user` | `string` 或 `[{text}, {image_url}, {file}]` | 用户消息,支持多模态 |

**user content 的 part 类型:**

| Part | 字段 | 说明 |
|------|------|------|
| `text` | `{"type":"text", "text": string}` | 文本内容 |
| `image_url` | `{"type":"image_url", "image_url": {"url": string}}` | 图片 URL 或 base64 data URI。`detail` 字段可传,会被忽略 |
| `file` | `{"type":"file", "file": {"file_id": string}}` | 仅支持已上传文件的 `file_id`(`purpose=input_image`),不支持内联 `file_data`/`filename` |

**输入限制**

| 限制项 | 值 |
|--------|-----|
| `image_url` / `file` 数量 | 每条消息最多 **1 个**图片或文件(二选一) |
| 图片文件大小 | < **10 MiB** |
| 图片分辨率 | < **3500 × 3500** 像素 |

超限返回 `413 content_too_large` 或 `400 invalid_request`。

### 6.2 POST `/v1/chat/completions` — 创建对话

**Request**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo..."
          }
        },
        {
          "type": "text",
          "text": "Text Recognition:"
        }
      ]
    }
  ],
  "model": "MinerU2.5-Pro-2604-1.2B",
  "stream": false,
  "stream_options": {
    "include_usage": true
  },
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 4096,
  "max_completion_tokens": 4096,
  "stop": null,
  "seed": 42,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "reasoning_effort": "medium",
  "n": 1
}
```

**Body Parameters**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `messages` | `array` | ✅ | 对话消息列表。仅支持 `system`/`developer`/`user` role。每个请求可有多个 system/developer/user 消息 |
| `model` | `string` | ✅ | 模型 ID。可用模型见 `GET /v1/models` |
| `stream` | `boolean` | | 是否流式输出,默认 `false`。流式格式见 [6.3 流式响应](#63-流式响应) |
| `stream_options` | `object` | | 流式选项,仅在 `stream=true` 时生效 |
| `stream_options.include_usage` | `boolean` | | 若为 `true`,最终 chunk 携带 `usage` 字段,`choices` 为空数组 |
| `temperature` | `number` | | 采样温度,0–2,默认由模型决定 |
| `top_p` | `number` | | 核采样,0–1,默认由模型决定。一般建议与 `temperature` 二选一 |
| `max_tokens` | `integer` | | 最大输出 token 数。已废弃,建议使用 `max_completion_tokens` |
| `max_completion_tokens` | `integer` | | 最大输出 token 数(含可见 token 与 reasoning tokens) |
| `stop` | `string \| array[string]` | | 最多 4 个停止序列,生成的文本不会包含停止序列 |
| `seed` | `integer` | | 若指定,系统尽量确定性采样。结合 `system_fingerprint` 监控后端变更 |
| `frequency_penalty` | `number` | | -2.0–2.0。正值降低逐字重复概率 |
| `presence_penalty` | `number` | | -2.0–2.0。正值鼓励引入新话题 |
| `reasoning_effort` | `string` | | 接受 `"none"`、`"minimal"`、`"low"`、`"medium"`、`"high"`、`"xhigh"`,当前**接受但忽略**,保留以备未来模型支持 |
| `n` | `integer` | | 候选输出数量。**当前仅允许 `1`**,传入其他值返回 `400` |

**不支持的 OpenAI 参数**

以下参数在 OpenAI Chat Completions API 中存在但 MinerU **不支持**,传入会被静默忽略或返回错误:

- `tools` / `tool_choice` / `function_call` / `functions` / `parallel_tool_calls` — 不支持 function calling
- `audio` / `modalities` — 不支持音频输出
- `web_search_options` — 不支持网页搜索
- `prediction` — 不支持 Predicted Outputs
- `response_format` — 不支持
- `logprobs` / `top_logprobs` — 不支持
- `logit_bias` — 不支持
- `service_tier` — 不支持
- `store` / `metadata` — 不支持
- `safety_identifier` / `prompt_cache_key` / `prompt_cache_retention` / `user` / `verbosity` — 不支持

**使用 image data URL 的示例**(最简)

```json
{
  "model": "MinerU2.5-Pro-2604-1.2B",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo..."
          }
        },
        {
          "type": "text",
          "text": "Text Recognition:"
        }
      ]
    }
  ]
}
```

**使用 file 的示例**(图片已上传为 file)

```json
{
  "model": "MinerU2.5-Pro-2604-1.2B",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "file",
          "file": {
            "file_id": "file-01HXYZ123ABCDEF"
          }
        },
        {
          "type": "text",
          "text": "Text Recognition:"
        }
      ]
    }
  ]
}
```

**Response (200,非流式)**

```json
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "MinerU2.5-Pro-2604-1.2B",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "# 文本识别结果\n\n这是从图片中识别出的文字内容..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 86,
    "total_tokens": 206
  }
}
```

**Response 字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | 本次 completion 的唯一标识 |
| `object` | `string` | 固定值 `"chat.completion"` |
| `created` | `integer` | Unix 时间戳(秒) |
| `model` | `string` | 实际使用的模型 |
| `system_fingerprint` | `string` | 后端配置指纹,配合 `seed` 判断确定性 |
| `choices` | `array` | 候选输出列表(`n=1` 时仅一项) |
| `choices[].index` | `integer` | 候选索引,从 0 开始 |
| `choices[].message` | `object` | 模型生成的消息 |
| `choices[].message.role` | `string` | 固定值 `"assistant"` |
| `choices[].message.content` | `string \| null` | 生成文本,可能为 null(如被内容过滤器拦截) |
| `choices[].finish_reason` | `string` | 停止原因:`"stop"`(自然结束)、`"length"`(达到 max_tokens 限制)、`"content_filter"`(内容过滤) |
| `usage` | `object` | Token 用量统计 |
| `usage.prompt_tokens` | `integer` | 输入 token 数 |
| `usage.completion_tokens` | `integer` | 输出 token 数 |
| `usage.total_tokens` | `integer` | 总计 token 数 |

### 6.3 流式响应

当 `stream=true` 时,响应为 SSE(Server-Sent Events)流,每行 `data: ` 前缀后跟一个 JSON chunk,
格式为 `chat.completion.chunk` 对象。流以 `data: [DONE]` 结束。

**Chunk 示例**

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2604-1.2B","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2604-1.2B","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":"#"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2604-1.2B","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":" 文本"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2604-1.2B","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2604-1.2B","system_fingerprint":"fp_44709d6fcb","choices":[],"usage":{"prompt_tokens":19,"completion_tokens":10,"total_tokens":29}}

data: [DONE]
```

**Chunk 字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | Completion ID(所有 chunk 相同) |
| `object` | `string` | 固定值 `"chat.completion.chunk"` |
| `created` | `integer` | 创建时间戳(所有 chunk 相同) |
| `model` | `string` | 使用的模型 |
| `system_fingerprint` | `string` | 后端配置指纹 |
| `choices[].index` | `integer` | 候选索引 |
| `choices[].delta` | `object` | 增量内容块 |
| `choices[].delta.role` | `string` | 首个 chunk 包含,值为 `"assistant"` |
| `choices[].delta.content` | `string \| null` | 增量文本 |
| `choices[].finish_reason` | `string \| null` | 最后一个有效 chunk 携带停止原因 |
| `usage` | `object \| null` | 仅在开启 `stream_options.include_usage` 时出现,携带最终 token 统计;该 chunk 的 `choices` 为空数组 |

**Errors**

- `400 unsupported_parameter`: `n` 不为 1
- `400 invalid_request`: `messages` 包含不支持的 role,或缺少必需参数
- `404 model_not_found`: 模型 ID 不存在
- `413 content_too_large`: 输入超过 token 限制
- `429 rate_limit_exceeded`: 超出速率限制

---

## 7. Responses — 文档对话(Responses API)

OpenAI 的 Responses API 是与 Chat Completions 并存的另一套对话接口。两者功能重叠但形态不同——Responses API 采用 typed item list 作为输入输出模型。选择哪一套取决于客户端的现有集成。

与 Chat Completions 一致的约束:单轮问答,system/developer/user 三种 role,无 function calling,无音频,无 web search。

### 7.1 与 Chat Completions 的关键区别

| 维度 | Chat Completions | Responses |
|------|-----------------|-----------|
| 输入 | `messages: [{role, content}]` | `input: string \| [{role, content}]` |
| 系统指令 | 在 `messages` 中作为 `system` role | 独立 `instructions` 字段 |
| 输出 | `choices[].message.content` | `output: [{type:"message", content:[{type:"output_text", text}]}]` |
| 流式 | SSE `data:` 行,每行一个 JSON chunk | SSE `event:` + `data:` 行,事件类型语义化 |
| 模型字段 | `model` | `model`(相同值) |

### 7.2 `input` 格式

支持以下形式:

| 形式 | 说明 |
|------|------|
| `string` | 纯文本,等同 user 角色的文本输入 |
| `[{role, content}]` | `EasyInputMessage` 数组 |

**`EasyInputMessage` 结构:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | `enum` | `system` / `developer` / `user` |
| `content` | `string` 或 `[content_parts]` | 文本或多模态内容 |

**content part 类型:**

| Part 类型 | 字段 | 说明 |
|-----------|------|------|
| `input_text` | `{"type":"input_text", "text": string}` | 文本内容 |
| `input_image` | `{"type":"input_image", "image_url"?: string, "file_id"?: string}` | 图片。`image_url` 支持 URL 或 base64 data URI;`file_id` 引用 `purpose=input_image` 的文件。`detail` 可传,会被忽略 |
| `input_file` | `{"type":"input_file", "file_id": string}` | 仅支持 `purpose=input_image` 的文件,不支持 `file_data` / `file_url` / `filename` |

**不支持的 input item 类型:** `Message`(含 status/type 的完整版)、`assistant` role、`FunctionCall` / `FunctionCallOutput`、`Reasoning` / `Compaction` 等 tool 和状态管理 item。

### 7.3 POST `/v1/responses` — 创建响应

**Request**

```json
{
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_image", "image_url": "https://example.com/page.png"},
        {"type": "input_text", "text": "Text Recognition:"}
      ]
    }
  ],
  "model": "MinerU2.5-Pro-2604-1.2B",
  "stream": false,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_output_tokens": 4096,
  "reasoning": {
    "effort": "medium"
  },
  "store": false
}
```

**Body Parameters**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `input` | `string \| array` | ✅ | 输入内容。字符串等同 user 文本;数组为 `EasyInputMessage` 列表。仅支持 `system`/`developer`/`user` role |
| `instructions` | `string` | | 系统指令,等同于 Chat Completions 的 `system` message |
| `model` | `string` | ✅ | 模型 ID。可用模型见 `GET /v1/models` |
| `stream` | `boolean` | | 是否流式输出,默认 `false`。流式格式见 [7.5 流式响应](#75-流式响应) |
| `temperature` | `number` | | 采样温度,0–2,默认由模型决定 |
| `top_p` | `number` | | 核采样,0–1,默认由模型决定 |
| `max_output_tokens` | `integer` | | 最大输出 token 数(含可见 token 与 reasoning tokens) |
| `reasoning` | `object` | | 推理配置,当前**接受但忽略**,保留以备未来模型支持 |
| `reasoning.effort` | `string` | | `"none"`/`"minimal"`/`"low"`/`"medium"`/`"high"`/`"xhigh"` |
| `reasoning.summary` | `string` | | `"auto"`/`"concise"`/`"detailed"` |
| `store` | `boolean` | | 是否存储本次响应。当前**仅允许 `false`**,传入 `true` 返回 `400` |

**不支持的 OpenAI 参数**

- `tools` / `tool_choice` / `parallel_tool_calls` / `max_tool_calls` — 不支持 function calling 及内置工具
- `background` — 不支持后台执行
- `conversation` / `previous_response_id` — 不支持多轮对话
- `include` / `prompt` / `top_logprobs` / `text` / `truncation` / `context_management` — 不支持
- `metadata` / `safety_identifier` / `prompt_cache_key` / `prompt_cache_retention` / `user` / `service_tier` — 不支持

**使用 input_image + data URL 的示例**(最简)

```json
{
  "model": "MinerU2.5-Pro-2604-1.2B",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo..."},
        {"type": "input_text", "text": "Text Recognition:"}
      ]
    }
  ]
}
```

**使用 input_image + file_id 的示例**(图片已上传为 file)

```json
{
  "model": "MinerU2.5-Pro-2604-1.2B",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_image", "file_id": "file-01HXYZ123ABCDEF"},
        {"type": "input_text", "text": "Text Recognition:"}
      ]
    }
  ]
}
```

**使用 input_file + file_id 的示例**(图片已上传为 file)

```json
{
  "model": "MinerU2.5-Pro-2604-1.2B",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_file", "file_id": "file-01HXYZ123ABCDEF"},
        {"type": "input_text", "text": "Text Recognition:"}
      ]
    }
  ]
}
```

**Response (200,非流式)**

```json
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "model": "MinerU2.5-Pro-2604-1.2B",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "该文档主要介绍了...",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 36,
    "output_tokens": 87,
    "total_tokens": 123
  }
}
```

**Response 字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | 本次 response 的唯一标识 |
| `object` | `string` | 固定值 `"response"` |
| `created_at` | `integer` | Unix 时间戳(秒) |
| `status` | `string` | `"completed"` / `"failed"` / `"incomplete"` |
| `model` | `string` | 实际使用的模型 |
| `output` | `array` | 模型输出项列表 |
| `output[].type` | `string` | 仅 `"message"`(无 tool call 等其他类型) |
| `output[].role` | `string` | `"assistant"` |
| `output[].content` | `array` | 输出内容列表 |
| `output[].content[].type` | `string` | `"output_text"` |
| `output[].content[].text` | `string` | 生成文本 |
| `usage` | `object` | Token 用量统计 |
| `usage.input_tokens` | `integer` | 输入 token 数 |
| `usage.output_tokens` | `integer` | 输出 token 数 |
| `usage.total_tokens` | `integer` | 总计 token 数 |
| `incomplete_details` | `object \| null` | 若 `status="incomplete"`,包含 `reason`(`"max_output_tokens"` 或 `"content_filter"`) |

### 7.4 Streaming 事件类型

当 `stream=true` 时,响应为 SSE 流。与 Chat Completions 的 `data:` 行格式不同,Responses 流式输出使用 `event:` + `data:` 双行结构,语义化的事件类型:

| 事件 | 含义 |
|------|------|
| `response.created` | Response 对象已创建 |
| `response.in_progress` | 开始处理 |
| `response.output_item.added` | 新增 output item(message) |
| `response.content_part.added` | 新增 content part(output_text) |
| `response.output_text.delta` | 文本增量(`delta` 字段) |
| `response.output_text.done` | 当前 output_text 完成 |
| `response.content_part.done` | 当前 content part 完成 |
| `response.output_item.done` | 当前 output item 完成 |
| `response.completed` | 整个 response 完成,携带完整 `response` 对象和 `usage` |

### 7.5 流式响应示例

```
event: response.created
data: {"type":"response.created","response":{"id":"resp_...","object":"response","status":"in_progress",...}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_...","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","item_id":"msg_...","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_...","output_index":0,"content_index":0,"delta":"#"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_...","output_index":0,"content_index":0,"delta":" 文本"}

...

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_...","output_index":0,"content_index":0,"text":"该文档主要介绍了..."}

event: response.content_part.done
data: {"type":"response.content_part.done","item_id":"msg_...","output_index":0,"content_index":0,"part":{"type":"output_text","text":"该文档主要介绍了...","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_...","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"该文档主要介绍了...","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_...","object":"response","status":"completed","output":[...],"usage":{"input_tokens":36,"output_tokens":87,"total_tokens":123},"model":"MinerU2.5-Pro-2604-1.2B",...}}
```

**Errors**

- `400 invalid_request`: `input` 包含不支持的 item 类型,或缺少必需参数
- `400 unsupported_parameter`: `store` 为 `true`
- `404 model_not_found`: 模型 ID 不存在
- `413 content_too_large`: 输入超过 token 限制
- `429 rate_limit_exceeded`: 超出速率限制

---

## 8. Jobs — 解析任务

### 8.1 POST `/v1/parse/jobs` — 创建任务

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
    },
    {
      "source": {
        "type": "local",
        "path": "/data/docs/report.pdf"
      }
    }
  ],
  "preset": "auto",
  "output_formats": ["markdown", "json", "content_list", "images"],
  "wait": 30,
  "callback": {
    "url": "https://your.app/mineru-webhook",
    "secret": "abc123"
  }
}
```

#### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | array | 是 | 至少 1 个,最多由 token 等级决定 |
| `preset` | enum | 否 | `auto`(默认,平台推荐组合) / `pipeline` / `vlm` / `html`。预设的解析方案组合,服务端按 preset 自动选择各阶段模型 |
| `output_formats` | array | 否 | 选择产物,默认 `["markdown"]`。产物以 File 对象形式存储(`purpose:parse_output`),通过 `GET /v1/files/{file_id}/content` 下载。可选值见下表 |
| `wait` | int | 否 | 同步等待秒数。`0`:异步模式,立即返回 202。传正值(`[5, 50]`):阻塞等待最多 N 秒,完成返回 200(内容内联),超时返回 202(转为异步轮询) |
| `callback` | object | 否 | Webhook 通知(需 Token) |

#### `files[].source`

```json
{ "type": "file_id",   "file_id": "file-01HXYZ123ABCDEF" }                 // 引用平台中已有文件(Create 秒传/Complete 后获得)
{ "type": "url",       "url": "https://..." }                          // 服务端拉取(内部转为 file_id)。文件名自动从 URL 路径推导(如 `/doc.pdf` → `doc.pdf`)
{ "type": "inline",    "name": "report.pdf", "data": "base64..." }      // 小文件内嵌(< 1MB,内部转为 file_id)。name 必填,作为文件显示名
{ "type": "local",     "path": "/data/docs/report.pdf" }                // 本地文件路径,支持三种格式: Unix 绝对路径("/data/a.pdf")、Windows 绝对路径("D:\\my-files\\a.pdf")、file URI("file:///data/a.pdf")。仅本地 server 支持,mineru.net 拒绝此类型
```

> `local` 类型的文件由服务端直接从磁盘读取,不经过上传流程,不生成 `file_id`。云端 `mineru.net` 不支持此来源,传入返回 `400 invalid_request`。

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
  "preset": "auto",
  "output_formats": ["markdown", "json", "content_list", "images"],
  "tier": "registered",
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
  "preset": "vlm",
  "output_formats": ["markdown", "json", "images"],
  "tier": "registered",
  "progress": { "completed": 1, "failed": 0, "total": 1 },
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "status": "completed",
      "metadata": {
        "pages": 12,
        "model_used": "MinerU2.5-Pro-2604-1.2B",
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
  "preset": "auto",
  "output_formats": ["markdown", "json", "images"],
  "tier": "registered",
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

**LB/代理超时对齐** — 反向代理(nginx、ALB)默认 `proxy_read_timeout` 通常 60s。`wait` 最大值为 50s,设计上确保不触发代理默认超时。若部署环境代理超时低于 50s,需上调或降低服务端 `max_wait` 配置。部署时建议全线对齐 `wait_maximum` 与代理超时。

**连接断开 ≠ 取消 job** — 客户端主动断开(如用户刷新页面)时 TCP 连接关闭,服务端通过 `ctx.Done()` 感知。**job 应继续运行直至完成**,仅停止向该连接写入。确保客户端断开连接后仍可通过 `GET /v1/parse/jobs/{job_id}` 轮询结果。

**Waiter 并发上限** — 同步等待的连接持有 goroutine + socket buffer + ctx,虽每个开销小,但大量 waiter 会放大排队的可见性。建议对同时处于同步等待状态的请求数设硬上限(如 500),超出则直接返回 202 退化为异步,避免耗尽内存或连接资源。

**同步请求独立限流** — 同步等待请求应使用独立的 rate limit 计数,防止少量用户占满 waiter 槽位。

**max_wait 配置点** — 服务端可通过配置限制 `wait` 最大值(与 LB 超时匹配);超出上限的 `wait` 值静默截断(返回 202 时已完成则返 200)。

---

### 8.2 GET `/v1/parse/jobs/{job_id}` — 查询任务

**Response (200)**

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "completed",
  "created_at": "2026-05-21T08:30:00Z",
  "started_at": "2026-05-21T08:30:02Z",
  "finished_at": "2026-05-21T08:30:48Z",
  "preset": "vlm",
  "output_formats": ["markdown", "json", "images"],
  "tier": "registered",
  "progress": { "completed": 2, "failed": 0, "total": 2 },
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "status": "completed",
      "metadata": {
        "pages": 12,
        "model_used": "MinerU2.5-Pro-2604-1.2B",
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

### 8.3 GET `/v1/parse/jobs/{job_id}/events` — SSE 流式状态

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
data: {"file_id":"file-01HXYZA","status":"completed"}

event: status
data: {"status":"completed","progress":{"completed":2,"total":2}}

event: done
data: {"job_id":"job_01HXYZ...","status":"completed"}
```

事件类型:`status` / `file_started` / `file_completed` / `file_failed` / `done` / `error`。

---

### 8.4 GET `/v1/parse/jobs` — 列出任务

> anonymous 用户调用此端点返回 `403 list_requires_token`,需 Bearer Token 才能列出任务列表。

**Query Params**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `status` | string | `null` | 过滤状态,可逗号分隔 |
| `limit` | int | `20` | 返回数量上限,范围 `[1, 100]` |
| `after` | string | `null` | 游标,上一页响应的 `last_id`。从其下一条开始返回 |
| `created_after` | ISO-8601 | `null` | 只返回此时间之后创建的任务 |
| `order` | enum | `desc` | 按 `created_at` 排序,`asc` 升序 / `desc` 降序 |

**Response (200)**

```json
{
  "object": "list",
  "data": [
    {
      "job_id": "job_01HXYZ...",
      "status": "completed",
      "created_at": "2026-05-21T08:30:00Z",
      "files_count": 2
    }
  ],
  "first_id": "job_01HXYZAAA...",
  "last_id": "job_01HXYZBBB...",
  "has_more": true
}
```

---

### 8.5 DELETE `/v1/parse/jobs/{job_id}` — 取消任务

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

## 9. Tier 与差异化

Tier 由 Token 决定,但**响应格式完全一致**。

| 能力 | anonymous(无 Token) | registered(有 Token) |
|------|---------------------|-----------------------|
| 单文件页数 | ≤ 1000 | ≤ 1000 |
| 单文件大小 | ≤ 200 MB | ≤ 200 MB |
| 单任务文件数 | 100 | 100 |
| 并发任务 | 10+ | 10+ |
| 队列优先级 | 默认 | 优先 |
| `preset` 可选 | 全部含 `vlm` | 全部含 `vlm` |
| `output_formats` 高级格式 | 拒绝 `html`/`latex`/`docx` | 全部支持 |
| `callback` | 拒绝 | 支持 |
| 产物保留期 | 30d | 30d |

> anonymous 与 registered 的**能力差异仅在于**:限流(§11)、高级输出格式、callback。其余完全一致。

超限时返回 `403 feature_requires_token` 或 `413 file_too_large`,响应体包含升级提示。

---

## 10. Webhook(异步回调)

> **可用性**:mineru.net 支持 Webhook;本地自部署 server **不支持** Webhook。

任务结束时,服务端 POST 到 `callback.url`:

**callback 创建**

创建 job 时通过 `callback` 字段传入:

```json
{
  "callback": {
    "url": "https://your.app/mineru-webhook",
    "secret": "abc123"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 回调 URL,支持 HTTP/HTTPS。接口必须支持 POST、UTF-8 编码、`Content-Type: application/json` |
| `secret` | string | 是 | 随机字符串,由英文字母、数字、下划线组成,不超过 64 字符。用于签名校验 |

**Request to your endpoint**

```http
POST https://your.app/mineru-webhook
Content-Type: application/json

{
  "checksum": "a1b2c3d4e5f6...",
  "content": "{\"event\":\"job.completed\",\"job_id\":\"job_01HXYZ...\",\"status\":\"completed\",\"occurred_at\":\"2026-05-21T08:30:48Z\"}"
}
```

| 字段 | 说明 |
|------|------|
| `checksum` | `SHA256(uid + secret + content)`,用于防篡改校验。`uid` 可在个人中心查询 |
| `content` | JSON 字符串,需自行解析。结构与 `GET /v1/parse/jobs/{job_id}` 的响应 data 部分一致 |

**回调处理规则**

- 您的服务端返回 HTTP `200` 视为接收成功,其他状态码视为失败
- 失败时最多**重试 5 次**,每次间隔递增(退避)
- 5 次均失败后不再推送,建议您检查 callback 接口状态
- 收到 webhook 后,建议调用 `GET /v1/parse/jobs/{job_id}` 拉取完整结果

**事件类型**:`job.completed` / `job.failed` / `job.canceled` / `job.partial`。

> **设计说明**:`secret` 是一次性的,与 job 生命周期一致,不需要轮换。`secret` 不会在 `GET /v1/parse/jobs/{job_id}` 的响应中 echo。

---

## 11. 限流

限流粒度为 **per-minute**，按端点类别独立计数。registered 用户按 **Token** 计数，anonymous 用户按 **请求 IP** 计数。

### 11.1 类别与限额

| 类别 | 范围 | anonymous | registered |
|------|------|-----------|------------|
| `parse` | `POST /v1/parse/jobs` | 5/min | 30/min |
| `upload` | `POST /v1/uploads`、`POST /v1/uploads/{id}/complete`、`POST /v1/uploads/{id}/cancel` | 10/min | 60/min |
| `chat` | `POST /v1/chat/completions`、`POST /v1/responses` | 10/min | 60/min |
| `read` | 其余 GET / DELETE 端点 | 60/min | 300/min |

### 11.2 响应头

每个响应携带当前端点类别的限流信息:

```http
X-RateLimit-Limit-Parse: 30
X-RateLimit-Remaining-Parse: 28
X-RateLimit-Reset-Parse: 1719184920
Retry-After: 0
```

| 头 | 说明 |
|----|------|
| `X-RateLimit-Limit-<Category>` | 当前类别每分钟上限 |
| `X-RateLimit-Remaining-<Category>` | 当前窗口剩余次数 |
| `X-RateLimit-Reset-<Category>` | 窗口重置时间(Unix 秒级时间戳) |
| `Retry-After` | 若超限,建议等待秒数;未超限为 `0` |

### 11.3 超限行为

返回 `429 rate_limit_exceeded`:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Parse rate limit (5/min) exceeded. Retry in 12s.",
    "details": {
      "category": "parse",
      "limit": 5,
      "window_seconds": 60,
      "retry_after_seconds": 12
    }
  }
}
```

> **设计说明**:按端点类别分桶。解析和上传是付费计算密集型,chat 是 VLM 推理,读取类基本免费。独立计数避免高频轮询挤占解析额度。

---

## 12. Usage — 用量查询

查询当前计费周期的用量与所属 tier 的配额上限。

### GET `/v1/usage`

**Request**

```http
GET /v1/usage  HTTP/1.1
Authorization: Bearer <token>
```

无参数,始终返回当前计费周期(当日 UTC 零点起)的快照。

**Response (200) — registered 用户**

```json
{
  "object": "usage",
  "tier": "registered",
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

**Response (200) — anonymous 用户**

```json
{
  "object": "usage",
  "tier": "anonymous",
  "billing_period": {
    "start": "2026-05-27T00:00:00Z",
    "end": "2026-05-28T00:00:00Z"
  },
  "current": {
    "pages_processed": 12,
    "files_processed": 4,
    "jobs_created": 4
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

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `object` | string | 固定 `"usage"` |
| `tier` | string | `"anonymous"` / `"registered"` |
| `billing_period.start` | ISO-8601 | 当前计费周期开始(UTC) |
| `billing_period.end` | ISO-8601 | 当前计费周期结束(UTC),与 `start` 差值 24h |
| `current.pages_processed` | int | 当日累计处理页数 |
| `current.files_processed` | int | 当日累计处理文件数 |
| `current.jobs_created` | int | 当日累计创建任务数 |
| `limits.max_pages_per_file` | int | 单文件最大页数 |
| `limits.max_file_size_bytes` | int | 单文件最大字节数 |
| `limits.max_files_per_job` | int | 单任务最大文件数 |
| `limits.max_concurrent_jobs` | int | 最大并发任务数 |
| `limits.max_file_retention_days` | int | 文件保留天数 |

**设计要点**

- registered:按 Token 统计用量;anonymous:按请求 IP 统计用量
- 不传 Token 也能查,返回 anonymous 档的 `limits`
- `limits` 字段集与 §9 Tier 表一一对应
- `billing_period` 粒度为天,始终 UTC 零点对齐,无时区歧义
- 若未来引入月度配额上限(如"每月最多 10000 页"),可在 `limits` 中增补字段,向后兼容

---

## 13. 端到端示例

### 13.1 完整流程(registered 用户,1 个 PDF,提供 sha256sum)

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

# ─── 情况 B: 需上传 (200, status=pending) ────────────────
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
    \"preset\": \"vlm\",
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

### 13.2 最简流程(不提供 sha256sum,无秒传/无校验)

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

# 后续同 13.1 的步骤 4-6
```

### 13.3 同步等待(已有 file_id,一步拿到 markdown)

```bash
# 上传步骤同 13.1 或 13.2,拿到 FILE_ID 后:

# 创建任务 + 同步等待最多 30s
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"files\": [{
      \"source\": {\"type\":\"file_id\",\"file_id\":\"$FILE_ID\"},
      \"options\": {\"language\":\"ch\",\"ocr\":\"auto\"}
    }],
    \"preset\": \"vlm\",
    \"output_formats\": [\"markdown\",\"json\"],
    \"wait\": 30
  }"

# → 200 (在 30s 内完成):
# {
#   "job_id": "job_...",
#   "status": "completed",
#   "files": [{
#     "output_files": {
#       "markdown": {"file_id": "file-md-xxx", "content": "# Report\n..."},
#       "json":   {"file_id": "file-json-xxx", "bytes": 184320}
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

### 13.4 批量上传(100 个文件,提供 sha256sum 享受秒传)

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
  -d "{\"files\":$FILES_JSON,\"preset\":\"vlm\",\"output_formats\":[\"markdown\"]}"
```

> 提示:`POST /v1/uploads` 在传入 `sha256sum` 时返回 `status:"completed"`(秒传,含 `file`)或 `status:"pending"`(含 `upload_url`)。

### 13.5 anonymous 用户最小示例(URL 来源)

```bash
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "files": [{
      "source": {"type":"url","url":"https://example.com/doc.pdf"}
    }],
    "output_formats": ["markdown"]
  }'

# → 202 {"job_id":"job_...","tier":"anonymous","files":[{"file_id":"file-01HXYZ..."}]}
# (服务端拉取后,响应中的 file_id 已回填,可后续复用免重新上传)
```

### 13.6 与现有三套 API 的能力映射

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
| —(新增) | `POST /v1/uploads` 携带 `sha256sum` 实现秒传 |
| —(新增,同步) | `POST /v1/parse/jobs` 携带 `wait:30`,阻塞等待直接拿到 markdown 内容 |

---

## 14. 客户端示意

统一为单个 Parser 类,基址 + 可选 Token 决定档位:

```python
from mineru import MineruApiParser

# 本地自部署
parser = MineruApiParser("http://localhost:8000/api")

# 云端 anonymous 档
parser = MineruApiParser("https://mineru.net/api")

# 云端 registered 档
parser = MineruApiParser("https://mineru.net/api", token="msk_...")

result = parser.parse("./report.pdf")
print(result.markdown())
result.save(writer)
```

调用代码完全相同,Token 仅决定能不能用 `preset="vlm"` 或 `output_formats=["docx"]`。

---

## 15. 与现有三套 API 的逐步对比

新 API 与现有三套 API 在概念上有三处根本归一:

1. **上传与解析分离**:文件上传(content-addressed)与任务调度(job)是独立资源
2. **预签名 URL 统一**:本地/云端都通过 `POST /v1/uploads` → `PUT` 上传字节,不再分 multipart 直传与两步 PUT
3. **产物清单替代 ZIP 包**:每个产物独立 URL,客户端按需下载,不再统一下整包

下面分别对比每个旧 API 的每个调用步骤。

### 15.1 vs Local ParseApi(本地 FastAPI,multipart 直传)

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

{"filename":"doc.pdf","bytes":1048576,"purpose":"parse"}
→ 200 {"id":"upload_...","upload_url":"http://localhost:8000/_upload/..."}

PUT {upload_url}
<binary bytes>
→ 200

POST /v1/uploads/{id}/complete
→ 200 {"status":"completed","file":{"id":"file-..."}}

POST /v1/parse/jobs  HTTP/1.1
Authorization: (省略)
Content-Type: application/json

{
  "files":[{"source":{"type":"file_id","file_id":"file-..."},
            "options":{"language":"ch"}}],
  "preset":"pipeline",
  "output_formats":["markdown","json","content_list","images"]
}
→ 202 {"job_id":"job_...","status":"queued","links":{...}}
```

**差异**
- 旧:单次 `multipart/form-data` 把字节+配置打包;新:分 3 步(upload 申请 / PUT 字节 / 创建 job)
- 旧:配置开关散落在 form 字段(`return_md`, `return_middle_json`, ...);新:统一收纳到 `output_formats` 数组
- 旧:每次调用都重新上传;新:可秒传(若提供 sha256)
- 旧:无文件级标识;新:不透明 `file_id`(`file-xxx`)+ 可选 `sha256sum` 元信息,file_id 可跨任务复用
- 旧:`backend` 字段直接选实现;新:`preset` 字段语义化(`pipeline`/`vlm`/`html`/`auto`)

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

### 15.2 vs MinerU V4 API(云端注册用户,Bearer token + 两步 PUT)

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
→ 200 {"file_id":"file-01HXYZ...","upload_id":"upload_...","upload_url":"https://oss..."}

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
  "files":[{"source":{"type":"file_id","file_id":"file-01HXYZ..."},
            "options":{"language":"ch","ocr":"auto","page_range":"1-10"}}],
  "preset":"vlm",
  "output_formats":["markdown","json","images","docx"]
}
→ 202 {"job_id":"job_...","status":"queued"}
```

**差异**
- 旧:上传 = 解析任务(`batch_id` 同时是上传组和解析任务);新:文件与任务解耦,**同一 `file_id` 可被多个 jobs 复用**(切换 model、改 page_range、跑不同 output_formats 都不必重传)
- 旧:`page_ranges` 写在 Call 1 的 file 条目里;新:写在 job 的 `options`(因为属于解析行为)
- 旧:`model_version` 在 Call 1 顶层;新:`preset` 在 `POST /v1/parse/jobs` 顶层

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

**差异**(同 13.1 Call 3,要点一致)
- 单 ZIP → 多 URL 拆分
- 旧 ZIP 内文件名/结构需客户端解析约定;新 artifact 类型在 metadata 中明确

---

### 15.3 vs MinerU Flash API(云端匿名用户,无 token)

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
→ 200 {"upload_id":"upload_...","upload_url":"https://oss..."}

# (上传 PUT 略)

POST /v1/parse/jobs
{
  "files":[{"source":{"type":"file_id","file_id":"file-..."},
            "options":{"language":"ch","page_range":"1-5","ocr":"auto"}}],
  "output_formats":["markdown"]
}
→ 202 {"job_id":"job_...","tier":"anonymous"}
```

**差异**
- 旧:一个 POST 同时拿到 `task_id` + `file_url`(上传与任务耦合);新:两个 POST(`/v1/uploads` 拿 `upload_id`+`upload_url`,`/v1/parse/jobs` 拿 `job_id`)
- 旧:**单文件**任务;新:任意数量文件
- 旧:`{code,msg,data}` envelope;新:扁平资源
- 旧:无 sha256/秒传概念;新:可选秒传
- 旧:任务参数写在 Call 1 顶层;新:文件级参数(`page_range`, `language`)在 `files[].options`,任务级参数(`preset`, `output_formats`)在 job 顶层

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
- 旧:仅可获得 `markdown_url`;新:anonymous 档默认 markdown,但若客户端 `output_formats` 列了 json/content_list,registered 档亦可拿到(anonymous 档收到 `403 feature_requires_token`)

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

### 15.4 整体对比矩阵

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
| 异步通知 | 无 | 无 | 无 | Webhook(registered)+ SSE |
| 部分失败语义 | 无显式 | 列表内每条 `state` | 单任务无 | `status:partial` + 逐文件 error |

### 15.5 调用次数(单文件、不秒传、要 markdown)

| API | HTTP 调用次数 | 序列 |
|-----|--------------|------|
| Local ParseApi | 3 (+轮询) | POST /tasks → poll status → GET result |
| V4 API | 4 (+轮询) | POST file-urls/batch → PUT → poll → GET zip |
| Flash API | 4 (+轮询) | POST parse/file → PUT → poll → GET md |
| 新 API | 4 (+轮询) | POST /v1/uploads → PUT → POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(秒传命中) | 3 (+轮询) | POST /v1/uploads → POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(同 file_id 复用) | 2 (+轮询) | POST /v1/parse/jobs → poll → GET /v1/files/{id}/content |
| 新 API(同步 wait) | 1 | POST /v1/parse/jobs {wait:30} → 200 content 内联 |

> 同 file_id 复用是新 API 独有优势:相同文件换个 `preset` / `page_range` / `output_formats` 重跑无需任何上传。



## 16. TODO

- **§8.1 `content_list` 与 `content_list_v2` 并存**:v1 何时下线?需制定迁移计划,并考虑以 `schema_version` 替代多值。
- **§8.1 计量与计费**:文档暂未涉及计费规则(按页/按 token/按文件),需补充分计费说明或独立计费文档。
- **§1.4 时间格式**:当前 JOB/Webhook 用 ISO-8601,File/Model/Chat 用 Unix 时间戳。未来评估是否需要统一为单一格式。

> 以下为原始文档结尾,end of file.

