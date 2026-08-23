# Parse Jobs

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 解析任务创建、查询和轮询
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 任务模型

Parse Job 是一次文档解析请求，可以包含一个或多个输入文件。任务创建后进入队列，服务端按 `tier`、文件类型和当前可用解析能力选择具体引擎。

Job 状态:

| 状态 | 含义 | 终态 |
|------|------|:--:|
| `queued` | 已创建，等待调度。 | 否 |
| `running` | 至少一个文件正在解析。 | 否 |
| `completed` | 全部文件解析成功。 | 是 |
| `partial` | 部分文件成功，部分文件失败。 | 是 |
| `failed` | 全部文件失败。 | 是 |
| `canceled` | 用户取消。 | 是 |

客户端应把 `completed`、`partial`、`failed`、`canceled` 都视为终态。

## POST `/v1/parse/jobs`

创建解析任务。

```json
{
  "files": [
    {
      "source": {
        "type": "file_id",
        "file_id": "file-01HXYZ123ABCDEF"
      },
      "page_range": "1~10"
    }
  ],
  "tier": null,
  "output_formats": ["markdown", "middle_json", "structured_content", "zip"],
  "callback": {
    "url": "https://your.app/mineru-webhook",
    "secret": "abc123"
  }
}
```

顶层字段:

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|:--:|------|------|
| `files` | array | 是 | 无 | 至少 1 个，最多由 access level 决定。 |
| `tier` | string 或 null | 否 | `null` | 当前服务支持的 tier 或 `null`。省略或传 `null` 表示使用默认选择策略；完整 tier 语义见 [解析 Tier](../tiers.md) 与 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。HTML 输入自动路由到 HTML 解析器。 |
| `output_formats` | array | 否 | `["markdown"]` | 请求产物格式。 |
| `callback` | object | 否 | `null` | Webhook 回调配置，官方 API registered 用户可用。 |

### `files[]`

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `source` | object | 是 | 文件来源。 |
| `page_range` | string | 否 | 针对单个文件的页码范围。 |

### Source 类型

`file_id` 引用已经存在的 File:

```json
{
  "type": "file_id",
  "file_id": "file-01HXYZ123ABCDEF"
}
```

`url` 由服务端拉取文件，并在内部转为 File:

```json
{
  "type": "url",
  "url": "https://example.com/doc.pdf"
}
```

`inline` 用于小文件内嵌，`data` 是 base64 字符串:

```json
{
  "type": "inline",
  "name": "scan.jpg",
  "data": "base64..."
}
```

`local` 仅 Local Parse Server 支持:

```json
{
  "type": "local",
  "path": "/data/docs/report.pdf"
}
```

Source 字段约束:

| `type` | 必填字段 | 约束 |
|--------|----------|------|
| `file_id` | `file_id` | File 必须存在、属于当前租户，且 `purpose` 适合解析。 |
| `url` | `url` | 官方 API 只允许公网 URL；拒绝私有网段、链路本地和 cloud metadata 地址。 |
| `inline` | `name`、`data` | base64 解码后字节数默认不超过 1MB。 |
| `local` | `path` | 仅本地 server 支持；只有 `features.sources` 包含 `local` 时可用。 |

客户端应以 `GET /v1/health` 返回的 `features.sources` 作为当前部署实际允许的 source 类型。

Local Parse Server 的 source 策略由启动参数决定:

- `--allow-local-source` 控制是否允许 `local` source；未开启时拒绝 `local` source，开启后允许读取 server 进程权限范围内的任意本地路径。
- `--max-inline-bytes` 控制 `inline` source 解码后的最大字节数，默认 1MB。
- `--allow-http-source` 控制 `url` source 是否允许 `http://`；默认只允许 `https://`。

### 文件级解析字段

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `page_range` | string | `null` | 页码范围。省略或传 `null` 均表示未指定页码范围，服务端解析整个文件。推荐 `~` 分隔，如 `1~10`、`1,3,5~7`、`-5~-1`。 |

OCR 策略和图片分析能力由 `tier` 与服务端实际引擎自动决定，客户端不能通过文件级参数单独关闭或开启。

### 输出格式

| 格式 | 产物类型 | 含义 | 需 API Key |
|------|----------|------|:--:|
| `markdown` | 文本 | Markdown 文本。 | 否 |
| `middle_json` | 文本 | 完整 Middle JSON，中间结构和高级调试。 | 否 |
| `content_list` | 文本 | 扁平内容列表。 | 否 |
| `structured_content` | 文本 | 面向 Agent 和新客户端的结构化内容 JSON。 | 否 |
| `html` | 文本 | HTML 导出。 | 是 |
| `latex` | 文本 | LaTeX 导出。 | 是 |
| `docx` | 二进制 | Word 文档导出。 | 是 |
| `zip` | 二进制 | 自包含解析包；图片 sidecar 只通过 zip 返回。 | 否 |

所有产物都以 `purpose:"parse_output"` 的 File 对象存储。Job 响应只返回产物 File 引用；所有产物内容都通过 `GET /v1/files/{file_id}/content` 获取。

`images` 不是可请求的 `output_formats` 值。需要图片 sidecar 的客户端应请求 `zip`，并从 zip 包内读取 `middle_json` 引用的图片路径。

客户端应以 `GET /v1/health` 返回的 `features.output_formats` 作为当前部署实际支持格式。Local Parse Server 可以只暴露基础格式。

`json`、`content_list_v2` 不是 NEXT 版正式格式名，不进入公开格式集合。命名决策见 [ADR-0001](../decisions/0001-json-output-formats.md)。

### 创建响应

创建成功时返回 `202`。客户端通过 `GET /v1/parse/jobs/{job_id}` 查询任务状态和结果。

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "queued",
  "created_at": "2026-05-21T08:30:00Z",
  "tier": "standard",
  "output_formats": ["markdown", "middle_json", "structured_content", "zip"],
  "access_level": "registered",
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "page_range": "1~12",
      "status": "queued"
    }
  ],
  "links": {
    "self": "/v1/parse/jobs/job_01HXYZ123ABCDEF",
    "cancel": "/v1/parse/jobs/job_01HXYZ123ABCDEF"
  }
}
```

创建响应字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `job_id` | string | Job ID。 |
| `status` | string | 当前 job 状态。 |
| `created_at` | string | ISO-8601 UTC。 |
| `tier` | string | 本次 job 的请求/解析 tier。请求省略或传 `null` 时，响应中返回解析后的 job tier；非 PDF/image 文件可能按 ADR-0024 在文件执行时归一为 `flash`，暂不暴露 file-level effective tier。 |
| `output_formats` | array | 实际接受的输出格式。 |
| `access_level` | string | `anonymous` 或 `registered`。 |
| `files[]` | array | 输入文件的队列状态。 |
| `links.self` | string | 查询任务 URL。 |
| `links.cancel` | string | 取消任务 URL。 |

### 终态响应

`GET /v1/parse/jobs/{job_id}` 查询到终态任务时返回 `200`。

```json
{
  "job_id": "job_01HXYZ123ABCDEF",
  "status": "completed",
  "created_at": "2026-05-21T08:30:00Z",
  "started_at": "2026-05-21T08:30:02Z",
  "finished_at": "2026-05-21T08:30:48Z",
  "tier": "standard",
  "output_formats": ["markdown", "middle_json", "zip"],
  "access_level": "registered",
  "progress": {
    "completed": 1,
    "failed": 0,
    "total": 1
  },
  "files": [
    {
      "file_id": "file-01HXYZ123ABCDEF",
      "name": "report.pdf",
      "page_range": "1~12",
      "status": "completed",
      "parse": {
        "model_used": "MinerU2.5-Pro-2605-1.2B",
        "duration_ms": 8234,
        "parser_version": "3.1.14"
      },
      "output_files": {
        "markdown": {
          "file_id": "file-01HXYZMD000001",
          "bytes": 51407
        },
        "middle_json": {
          "file_id": "file-01HXYZJSON000002",
          "bytes": 184320
        },
        "zip": {
          "file_id": "file-01HXYZZIP000004",
          "bytes": 286720
        }
      }
    }
  ]
}
```

文件结果字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `file_id` | string 或 null | 输入 File ID。`local` source 可以为 `null`。 |
| `name` | string | 文件显示名。 |
| `page_range` | string | 服务端规范化后的实际解析页码范围。请求未指定时为 `1~{total}`；请求指定时，服务端先按文件总页数展开负数页码，再去重并合并连续页码区间。例如 `1,2,3,-1,-1` 在 10 页文件中规范化为 `1~3,10`。 |
| `status` | string | 文件级状态。 |
| `parse.model_used` | string | 实际模型 ID。仅当该文件 `status="completed"` 时出现。 |
| `parse.duration_ms` | integer | 单文件从开始解析到结束的耗时，不包含排队时间。仅当该文件 `status="completed"` 时出现。 |
| `parse.parser_version` | string | 解析器版本。仅当该文件 `status="completed"` 时出现。 |
| `output_files` | object | 各输出格式对应的 File 引用。仅当该文件 `status="completed"` 时出现。 |
| `error` | object | 文件失败时出现，包含 `code` 和 `message`。仅当该文件 `status="failed"` 时出现。 |

`queued`、`running`、`canceled` 状态下的文件可以不返回 `parse`、`output_files` 和 `error`。`partial` job 中，成功文件按 `completed` 文件返回 `parse` 与 `output_files`，失败文件按 `failed` 文件返回 `error`。

## GET `/v1/parse/jobs/{job_id}`

查询任务状态和结果。终态 job 使用与终态响应相同的结构，但只返回产物 File 引用，不返回产物内容。未完成 job 可以只返回当前 `status`、`progress` 和 `files` 的阶段信息。

轮询建议:

- 初始间隔 2 秒。
- 使用指数退避。
- 最大间隔 30 秒。
- 进入终态后停止轮询。

错误:

| HTTP | code |
|------|------|
| 404 | `job_not_found` |

## GET `/v1/parse/jobs`

列出任务。anonymous 用户调用返回 `403 list_requires_api_key`。

查询参数:

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `status` | string | `null` | 按状态过滤，可逗号分隔。 |
| `limit` | integer | `20` | 返回数量，范围 `[1, 100]`。 |
| `after` | string | `null` | 游标，上一页响应的 `last_id`。 |
| `created_after` | string | `null` | ISO-8601 时间，只返回此时间之后创建的任务。 |
| `order` | string | `desc` | `asc` 或 `desc`。 |

响应:

```json
{
  "object": "list",
  "data": [
    {
      "job_id": "job_01HXYZ...",
      "status": "completed",
      "created_at": "2026-05-21T08:30:00Z",
      "file_count": 2
    }
  ],
  "first_id": "job_01HXYZAAA...",
  "last_id": "job_01HXYZBBB...",
  "has_more": true
}
```

错误:

| HTTP | code |
|------|------|
| 400 | `invalid_request` |
| 403 | `list_requires_api_key` |

## DELETE `/v1/parse/jobs/{job_id}`

取消未进入终态的任务。

```json
{
  "job_id": "job_01HXYZ...",
  "status": "canceled",
  "canceled_at": "2026-05-21T08:30:30Z"
}
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 404 | `job_not_found` | job 不存在或不属于当前租户。 |
| 409 | `job_already_terminal` | job 已完成、失败或已取消。 |

## 创建任务错误

`POST /v1/parse/jobs` 常见错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_request` | 请求结构非法、source 缺字段、page_range 不合法等。 |
| 400 | `unsupported_output_format` | 输出格式不支持。 |
| 400 | `unsupported_source` | 当前部署不支持该 source 类型。 |
| 403 | `feature_requires_api_key` | anonymous 使用高级输出格式或 callback。 |
| 404 | `file_not_found` | `file_id` 不存在或不可见。 |
| 413 | `file_too_large` | 文件超过当前 access level 限制。 |
| 429 | `rate_limit_exceeded` | 触发 parse 类限流。 |
| 503 | `quality_tier_unavailable` | 请求的质量 tier 不可用，或 PDF/image 默认选择策略找不到可用的非 `flash` tier。 |

## 本地 Server 差异

Local Parse Server 的任务 API 与官方 API 保持同一结构，但有以下实现差异:

- 不支持 Webhook 时，`health.features.webhook` 必须为 `false`；收到 `callback` 时应返回 `400 invalid_request` 或明确忽略。
- `health.features.output_formats` 必须反映本地 server 实际支持的输出格式；当前本地实现支持 `markdown`、`middle_json`、`content_list`、`structured_content` 和 `zip`。
- `health.features.sources` 必须反映本地 server 实际允许的 source 类型；只有启动时开启 `--allow-local-source` 才包含 `local`。
- `local` source 可以不生成输入 `file_id`；响应中应保留 `name` 和文件级状态。
- 对 PDF/image，省略 `tier` 或传 `null` 时，只能按默认选择策略选择本地可发现的非 `flash` 质量 tier，不能回退到 `flash`；如果只有 `flash` 可用，应返回 `quality_tier_unavailable`。
- 对 Office/HTML，API Server job 按批量规则处理，即使 job tier 是质量 tier，文件实际解析也按 `flash` 语义归一；text 不进入 parse job；暂不新增 file-level effective tier 字段。
