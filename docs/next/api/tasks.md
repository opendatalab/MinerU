# V1 Tasks：JSON 清单、逐文件上传与下载

`/v1/tasks` 是本地 API Server、Router 和 SaaS 的共同异步客户端契约。
客户端用 JSON 创建任务，按返回链接上传缺失文件，再轮询 `status_url`。
每个文件就绪后立即执行，完成后在 `files[].result_url` 返回独立下载链接。
客户端不拼接存储路径，也不要求 SaaS 实现本地 Files API。

快速定位：[响应字段](#taskstatusresponse) · [各接口成功响应](#各接口成功响应) · [错误响应](#各接口错误响应) · [输出格式与下载](#输出格式与下载)。

## 路由与请求

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/tasks` | 仅接受 `application/json`，返回 `202` 和稳定 `task_id`。 |
| 返回的 `upload_method` 与 `upload_url` | 每个链接上传一个文件；本地使用 `PUT /v1/tasks/{task_id}/uploads/{client_file_id}`。 |
| `GET /v1/tasks/{task_id}` | 返回 `200`，包含总状态、逐文件状态、尚有效的上传链接和已完成文件的下载链接。 |
| `DELETE /v1/tasks/{task_id}` | 返回 `200` 和取消后的完整任务对象，保留已经完成文件的下载链接。 |
| `GET /v1/tasks` | 返回 `200` 和父任务摘要分页列表。 |

创建请求不再接受表单、multipart 或文件字节，其他 Content-Type 返回 `415`。
业务参数必须放在 JSON 主体中，POST 查询参数、未知字段、非法值返回 `400 invalid_request`。
上传文件的 PUT 仍发送原始字节，使用返回的 `upload_headers`；签名 URL 必须原样使用。
Task ID 是父任务标识；内部逐文件执行 ID 不属于公共契约。

| JSON 字段 | 默认值 | 说明 |
|---|---|---|
| `files` | 必填 | 1–100 个文件组成的 JSON 数组，不接受 JSON 字符串。 |
| `tier` | 服务默认值 | `flash/basic/standard/advanced`，受部署能力约束。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 全局 PDF 页范围，支持 `1-5`、`r1`、`all` 等现有语法。 |
| `output_formats` | `["default"]` | 非空列表，值为 `default/docx/html/latex` 中一个或多个；重复值按首次出现去重。 |

`include_model_output` 默认为 `false`。为 `true` 时必须包含 `default` 输出格式，存在原始模型输出时在结果 ZIP 中增加 `model_output.json`；其他格式组合不满足条件时返回 400。创建和查询响应都携带该选项。

每个清单条目包含：

| 字段 | 必填 | 说明 |
|---|---|---|
| `client_file_id` | 否 | 任务内唯一，1–64 个字母、数字、下划线或短横线；省略或 `null` 时由服务端补全。 |
| `file_name` | 是 | 逻辑文件名，保留扩展名；不同条目允许同名。 |
| `url` | 否 | HTTP(S) 文件下载地址，受部署来源策略约束；省略或 `null` 表示客户端上传。 |
| `bytes` | 条件 | 上传项必填；URL 项可省略或为 `null`。提供时必须是严格正整数。 |
| `sha256sum` | 否 | 64 位小写十六进制 SHA-256；用于完整性校验，上传项还用于源文件缓存预检。 |
| `page_range` | 否 | 逐文件 PDF 页范围，未提供时继承全局值；非 PDF 仍只支持整本解析。 |

服务端先预留全部显式 `client_file_id`，再按缺省项的输入顺序，从 `f1` 开始分配最小未占用的 `fN`。例如显式占用了 `f2`，两项缺省 ID 分别生成 `f1`、`f3`。空字符串、非法 ID 和重复显式 ID 返回 400。
创建响应的 `files` 顺序与输入一致，`files[]` 和 `uploads[]` 始终返回非空 ID；生成后上传、查询及 Router 转发均保持不变。

先从真实文件生成 JSON 请求（本例省略 ID，由服务端补全）：

```bash
python3 - a.pdf b.pdf > task.json <<'PY'
import hashlib
import json
from pathlib import Path
import sys

files = []
for value in sys.argv[1:]:
    path = Path(value)
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    files.append({
        'file_name': path.name,
        'bytes': path.stat().st_size,
        'sha256sum': hasher.hexdigest(),
    })
print(json.dumps({'files': files, 'tier': 'flash', 'ocr_mode': 'txt',
                  'output_formats': ['default', 'latex']}, ensure_ascii=False))
PY

curl --fail-with-body "$MINERU_API_URL/v1/tasks" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @task.json
```

未配置 API Key 的本地服务可省略 Authorization，四种输出格式均可用；启用服务认证后四种格式均需认证，没有额外的格式权限限制。

## TaskStatusResponse

`POST /v1/tasks`、`GET /v1/tasks/{task_id}`、`DELETE /v1/tasks/{task_id}` 的成功响应均为完整的 `TaskStatusResponse`。
本地逐文件上传 `PUT /v1/tasks/{task_id}/uploads/{client_file_id}` 也返回同一结构。成功时直接返回对象，没有 `data` 或 `task` 外层包装。

JSON 响应使用 `Content-Type: application/json`，正文为 UTF-8。下表列出的字段在当前本地 API Server 和 Router 响应中均会出现；可空字段输出 `null`，空集合输出 `[]` 或 `{}`，不以省略字段代替。客户端应忽略未来新增的响应字段。
时间字符串使用 ISO-8601 UTC 格式，例如 `2026-10-23T04:00:00Z`；上传链接的 `expires_at` 单独使用 Unix 秒级整数。
请求中的 `ocr_mode` 和顶层 `page_range` 不在响应顶层回显；逐文件页范围通过 `files[].page_range` 返回。

### 顶层字段

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `task_id` | string | 否 | 稳定的父任务 ID；是不透明标识，不代表内部逐文件执行 ID。 |
| `status` | string | 否 | 父任务状态：`waiting_upload/queued/running/completed/partial/failed/canceled`。 |
| `status_url` | string | 否 | 查询本任务的完整 URL；使用返回地址，不自行拼接。 |
| `created_at` | string | 否 | 父任务创建时间。 |
| `started_at` | string | 是 | 已知的最早逐文件执行开始时间；尚无执行开始记录时为 `null`。 |
| `finished_at` | string | 是 | 父任务进入终态的时间；非终态时为 `null`，取消成功时也会填写。 |
| `tier` | string | 是 | 任务级档位，非空时为 `flash/basic/standard/advanced`；文件实际采用的流程还受格式约束。 |
| `output_formats` | array of string | 否 | 本次任务采用的输出格式，按首次出现去重。 |
| `include_model_output` | boolean | 否 | 是否请求在 default ZIP 中附加可用模型输出。 |
| `progress` | object | 否 | 文件计数，结构见下表。 |
| `files` | array of object | 否 | 完整逐文件快照，顺序始终与创建请求一致，结构见下表。 |
| `uploads` | array of object | 否 | 仍有效、尚未确认就绪的上传描述；无需上传或任务取消后为 `[]`。 |
| `error` | object | 是 | 父任务级错误，结构为 `ErrorDetail`；逐文件失败应检查 `files[].error`。 |

### progress：文件计数

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `completed` | integer | 已成功完成解析和全部所选格式导出的文件数。 |
| `failed` | integer | 已失败文件数，包含上传过期、下载、解析或导出失败。 |
| `total` | integer | 输入文件总数，固定为创建请求的 `files` 长度。 |

取消的文件不计入 `completed` 或 `failed`。存在排队、运行、等待上传或已取消文件时，`completed + failed` 可以小于 `total`，包括父任务已经为 `canceled` 的情况。

### files[]：逐文件对象

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `client_file_id` | string | 否 | 客户端显式提供或服务端补全的任务内唯一标识，创建后不变。 |
| `file_name` | string | 否 | 创建时提交的逻辑文件名，保留扩展名；同名文件通过 ID 区分。 |
| `bytes` | integer | 是 | 上传项的声明大小；URL 项可暂为 `null`，完整下载并校验后补齐实际字节数。 |
| `url` | string | 是 | URL 输入保留原始下载地址；客户端上传项为 `null`。 |
| `sha256sum` | string | 是 | 创建时声明的小写 SHA-256；URL 完整下载并校验后回填实际值，未声明 hash 的上传项可保持 `null`。 |
| `page_range` | string | 是 | 归一化后的逐文件页范围，初始继承请求设置，整本初始为 `""`；执行后可回填实际页号范围。 |
| `upload_status` | string | 否 | `pending/ready/not_required/expired`，与解析状态分别表达。 |
| `file_id` | string | 是 | 本地/Router 上传项的源 Files ID，准备中可能为 `null`；URL 项为 `null`。这是源文件 ID，产物使用 `result_url` 下载。 |
| `name` | string | 否 | 执行层回传的文件名；当前本地/Router 与 `file_name` 相同，仍保留此字段。 |
| `status` | string | 否 | 单文件状态：`waiting_upload/queued/running/completed/failed/canceled`；没有 `partial`。 |
| `parse` | object | 是 | 解析统计，尚无统计时为 `null`；非空结构见下表。 |
| `result_url` | string | 是 | 结果完整保存后发布的唯一下载 URL；非 `completed` 文件为 `null`。 |
| `error` | object | 是 | 该文件的错误详情；无错误时为 `null`。 |

`upload_status` 的具体含义：`pending` 表示上传尚未确认，`ready` 表示源文件已经就绪，`not_required` 表示 URL 输入不需要客户端上传，`expired` 表示尚未完成上传的条目已过上传期限。
取消尚未上传的文件后，其 `status` 为 `canceled`，`upload_status` 仍可能是 `pending`，但不再提供上传链接。

### files[].parse：解析统计

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `model_used` | string | 是 | 执行模型信息；当前本地解析实现为 `null`，不根据该字段推断 tier。 |
| `duration_ms` | integer | 是 | 毫秒级文件执行耗时；当前实现包含源读取/下载、解析和导出，不含等待上传及排队时间。 |
| `parser_version` | string | 是 | 执行 worker 的 MinerU 版本，不保证与 Router 的版本一致。 |

### uploads[]：上传描述

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `client_file_id` | string | 对应 `files[]` 中同 ID 的输入。 |
| `upload_url` | string | 上传一个文件的完整 URL；签名地址必须原样使用。 |
| `upload_method` | string | 当前为 `PUT`，客户端按返回值发送请求。 |
| `upload_headers` | object | 字符串到字符串的映射；上传时发送这些请求头，不假定总是固定头集合。 |
| `expires_at` | integer | Unix 秒级截止时间，创建任务后固定，查询不续期。 |

### ErrorDetail：父任务或逐文件错误

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `type` | string | 否 | 错误大类，例如 `invalid_request_error/engine_error/api_error`。 |
| `code` | string | 是 | 机器可读错误码。 |
| `message` | string | 否 | 人类可读错误说明，不作为客户端分支判断依据。 |
| `param` | string | 是 | 关联参数，例如 `files.1.page_range`；无具体参数时为 `null`。 |

`error=null` 不表示整个批次全部成功：父任务即使是 `partial/failed`，顶层 `error` 仍可为 `null`，错误位于具体文件的 `error` 中。

## 各接口成功响应

以下三个示例沿用同一批两个 HTML 输入：先创建任务，再上传并完成第一个文件，最后取消仍待上传的第二个文件。
示例来自当前响应结构，ID、地址、时间和耗时为示意值；JSON 展示全部字段，不使用省略号。

### POST /v1/tasks：创建任务

成功返回 **`202 Accepted` + `application/json`**，正文为 `TaskStatusResponse`。`202` 只表示任务已接单，不表示文件已经解析成功。
初始状态通常为 `waiting_upload`；URL 或缓存命中项可以直接进入 `queued`。`uploads` 只列出当前需要客户端上传的文件。

请求示例：

```json
{
  "files": [
    {"file_name": "report.html", "bytes": 64},
    {"file_name": "notes.html", "bytes": 64}
  ],
  "tier": "flash",
  "output_formats": ["default"]
}
```

完整响应：

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "waiting_upload",
  "status_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567",
  "created_at": "2026-10-23T04:00:00Z",
  "started_at": null,
  "finished_at": null,
  "tier": "flash",
  "output_formats": [
    "default"
  ],
  "include_model_output": false,
  "progress": {
    "completed": 0,
    "failed": 0,
    "total": 2
  },
  "files": [
    {
      "client_file_id": "f1",
      "file_name": "report.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "",
      "upload_status": "pending",
      "file_id": null,
      "name": "report.html",
      "status": "waiting_upload",
      "parse": null,
      "result_url": null,
      "error": null
    },
    {
      "client_file_id": "f2",
      "file_name": "notes.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "",
      "upload_status": "pending",
      "file_id": null,
      "name": "notes.html",
      "status": "waiting_upload",
      "parse": null,
      "result_url": null,
      "error": null
    }
  ],
  "uploads": [
    {
      "client_file_id": "f1",
      "upload_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/f1",
      "upload_method": "PUT",
      "upload_headers": {
        "Content-Type": "application/octet-stream"
      },
      "expires_at": 1792731600
    },
    {
      "client_file_id": "f2",
      "upload_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/f2",
      "upload_method": "PUT",
      "upload_headers": {
        "Content-Type": "application/octet-stream"
      },
      "expires_at": 1792731600
    }
  ],
  "error": null
}
```

### GET /v1/tasks/{task_id}：查询任务

成功返回 **`200 OK` + `application/json`**，正文始终为 `TaskStatusResponse`，包括任务仍在运行、等待上传、部分失败、全部失败或已取消时。
以下快照中，第一个文件已完成，可立即下载；第二个文件仍等待上传，所以父任务为 `waiting_upload`，`finished_at` 仍为 `null`。

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "waiting_upload",
  "status_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567",
  "created_at": "2026-10-23T04:00:00Z",
  "started_at": "2026-10-23T04:00:01Z",
  "finished_at": null,
  "tier": "flash",
  "output_formats": [
    "default"
  ],
  "include_model_output": false,
  "progress": {
    "completed": 1,
    "failed": 0,
    "total": 2
  },
  "files": [
    {
      "client_file_id": "f1",
      "file_name": "report.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "1",
      "upload_status": "ready",
      "file_id": "file-source0123456789abcdef",
      "name": "report.html",
      "status": "completed",
      "parse": {
        "model_used": null,
        "duration_ms": 125,
        "parser_version": "4.0.12"
      },
      "result_url": "http://127.0.0.1:8000/v1/files/file-result0123456789abcdef/content",
      "error": null
    },
    {
      "client_file_id": "f2",
      "file_name": "notes.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "",
      "upload_status": "pending",
      "file_id": null,
      "name": "notes.html",
      "status": "waiting_upload",
      "parse": null,
      "result_url": null,
      "error": null
    }
  ],
  "uploads": [
    {
      "client_file_id": "f2",
      "upload_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/f2",
      "upload_method": "PUT",
      "upload_headers": {
        "Content-Type": "application/octet-stream"
      },
      "expires_at": 1792731600
    }
  ],
  "error": null
}
```

### DELETE /v1/tasks/{task_id}：取消任务

成功返回 **`200 OK` + `application/json`**，正文为取消后的完整 `TaskStatusResponse`，`status` 为 `canceled`，`uploads` 为 `[]`。
这个接口取消未完成文件，不删除 Task 或已完成的结果；返回时后台不可中断计算或资源清理可能仍在收尾。
已经成功或失败的文件保持原状态，成功文件继续提供 `result_url`。对任意已终结的父任务再次取消，返回 `409 task_already_terminal`。

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "canceled",
  "status_url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567",
  "created_at": "2026-10-23T04:00:00Z",
  "started_at": "2026-10-23T04:00:01Z",
  "finished_at": "2026-10-23T04:00:03Z",
  "tier": "flash",
  "output_formats": [
    "default"
  ],
  "include_model_output": false,
  "progress": {
    "completed": 1,
    "failed": 0,
    "total": 2
  },
  "files": [
    {
      "client_file_id": "f1",
      "file_name": "report.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "1",
      "upload_status": "ready",
      "file_id": "file-source0123456789abcdef",
      "name": "report.html",
      "status": "completed",
      "parse": {
        "model_used": null,
        "duration_ms": 125,
        "parser_version": "4.0.12"
      },
      "result_url": "http://127.0.0.1:8000/v1/files/file-result0123456789abcdef/content",
      "error": null
    },
    {
      "client_file_id": "f2",
      "file_name": "notes.html",
      "bytes": 64,
      "url": null,
      "sha256sum": null,
      "page_range": "",
      "upload_status": "pending",
      "file_id": null,
      "name": "notes.html",
      "status": "canceled",
      "parse": null,
      "result_url": null,
      "error": null
    }
  ],
  "uploads": [],
  "error": null
}
```

### PUT /v1/tasks/{task_id}/uploads/{client_file_id}：本地上传响应

本地 API Server 和 Router 的上传成功返回 **`200 OK` + `application/json`**，正文仍为上面定义的完整 `TaskStatusResponse`。
本次上传文件的 `upload_status` 为 `ready`，通常从 `uploads` 中移除并进入 `queued/running`；其他文件和已完成结果也包含在响应中。
有效期内重复 PUT 已确认文件同样返回 `200` 和当前快照；上传链接到期后返回 `410`，不会重新读取或覆盖源字节。

SaaS 返回的上传目标可能是独立对象存储：其成功状态码和响应体由该存储服务决定，不要求返回 Tasks JSON。
通用客户端应接受目标服务的成功响应，随后查询 `status_url` 确认就绪情况，不把对象存储上传响应直接作为任务状态解析。

### GET /v1/tasks：列出父任务

成功返回 **`200 OK` + `application/json`**，正文为 `TaskListResponse`，不是完整的 `TaskStatusResponse`：

```json
{
  "object": "list",
  "data": [
    {
      "task_id": "job_0123456789abcdef01234567",
      "status": "canceled",
      "created_at": "2026-10-23T04:00:00Z",
      "file_count": 2
    }
  ],
  "first_id": "job_0123456789abcdef01234567",
  "last_id": "job_0123456789abcdef01234567",
  "has_more": false
}
```

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `object` | string | 否 | 固定为 `list`。 |
| `data` | array of object | 否 | 父任务摘要；没有结果时为 `[]`。 |
| `data[].task_id` | string | 否 | 父任务 ID。 |
| `data[].status` | string | 否 | 与完整任务对象相同的父任务状态枚举。 |
| `data[].created_at` | string | 否 | ISO-8601 UTC 创建时间。 |
| `data[].file_count` | integer | 否 | 该任务的输入文件总数。 |
| `first_id` | string | 是 | 本页第一项 task ID；空页为 `null`。 |
| `last_id` | string | 是 | 本页最后一项 task ID；空页为 `null`，作为下一页 `after`。 |
| `has_more` | boolean | 否 | 是否还有下一页。 |

列表不返回 `files/uploads/progress/result_url`；需要详情时逐任务查询。筛选与分页参数见本页“父任务列表与缓存”。

## 各接口错误响应

请求级失败返回对应非 2xx 状态码和 `Content-Type: application/json`，正文是仅包含顶层 `error` 的 JSON 对象，其值为上文的 `ErrorDetail`，不返回成功任务对象。
例如重复取消任务返回 **`409 Conflict`**：

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

| 接口 | HTTP | 常见 code | 场景 |
|---|---|---|---|
| POST | 400 | `invalid_request`、`page_range_invalid` | 字段、ID、页范围等非法，包括旧字段和新旧混传。 |
| POST | 415 | `invalid_request` | 请求不是 `application/json`。 |
| POST | 503 | `quality_tier_unavailable`、`server_shutting_down` 或模型预加载错误码 | 当前部署无法接单。 |
| GET 详情 / DELETE / 本地 PUT | 404 | `task_not_found` | task 不存在、记录已清理或不属于当前调用方。 |
| DELETE | 409 | `task_already_terminal` | 父任务已经成功、部分成功、失败或被取消；包括取消前刷新时刚进入终态。 |
| 本地 PUT | 400 | `upload_size_mismatch`、`file_hash_mismatch`、`upload_interrupted` | 上传校验失败或传输中断；未过期时可查询后重传。 |
| 本地 PUT | 404 | `upload_not_found` | task 中不存在该 `client_file_id`。 |
| 本地 PUT | 409 | `task_already_terminal`、`file_already_terminal` | 任务已取消或文件已不接受上传。 |
| 本地 PUT | 410 | `upload_expired` | 固定上传窗口已过期。 |
| GET 列表 | 400 / 404 | `invalid_request` / `task_not_found` | 筛选参数非法，或 `after` 游标不存在、已清理或越权。 |
| 所有受保护接口 | 401 | `invalid_api_key` | 服务启用认证，但调用方未提供有效凭据。 |

Router 或网关还可能返回远端依赖相关的 5xx 错误；错误格式及部署扩展见 [响应与错误](responses.md)。
本地 API 和上述任务错误示例包含 `param: null`；Router 代理层自身产生的错误也可能省略 `param`，客户端按 `null` 处理。
异步文件下载、解析或导出失败通常通过查询接口的 **`200` 快照和 `files[].error`** 表达；不要把 HTTP `200` 等同于任务成功，也不要仅凭顶层 `error` 判断全部文件是否成功。

## URL、缓存与独立上传

URL 输入同样先创建 JSON 任务：

```json
{
  "files": [
    {"client_file_id": "remote", "file_name": "document.pdf", "url": "https://example.com/download?signature=example"},
    {"client_file_id": "local", "file_name": "other.pdf", "bytes": 12345}
  ],
  "output_formats": ["html"]
}
```

- URL 项的 `upload_status` 为 `not_required`，不生成上传链接、不接受 PUT；源 `file_id` 为 `null`。创建后即进入执行队列，不等待其他上传。
- URL 保留原始签名地址，使用逻辑 `file_name` 识别类型；不要求 HEAD 或 `Content-Length`。下载后返回实际 `bytes` 和 `sha256sum`，提供的大小/hash 必须匹配。URL 每次下载，不用 hash 缓存跳过下载。
- 上传项 hash 命中时为 `ready`，不返回上传链接，立即排队；未命中或无 hash 时返回该文件的链接。上传后仍计算 hash、校验大小并对源字节去重。
- 缓存只复用源文件，不复用不匹配的解析结果；每个文件使用本次解析参数独立执行。
- 每个上传项校验完成即排队，不需要客户端发送完成确认。等待上传不占解析并发，也不预占 worker。

### 一小时上传窗口

所有上传链接的 `expires_at` 固定为创建时起 3600 秒，采用 Unix 秒。查询不会延长或换发链接。
到期后新 PUT 返回 `410 upload_expired`，包括对已经确认文件的重复 PUT。

截止前已受理的上传可以继续传完并校验，该文件仍处于 `waiting_upload`，不会被过期扫描中止。
尚未受理上传的文件到期后变为 `status=failed`、`upload_status=expired`，逐文件错误码为 `upload_expired`。
在途上传中断且已经过期时，同样进入失败终态；补交须创建新任务。
上传截止时间不限制已上传文件的排队、解析或导出时间。

有效期内，一个文件中断或校验失败可以从头重传，其他文件的状态不受影响。
重复 PUT 已确认文件不读取或覆盖源字节，并发请求也只提交一次解析。
上传响应丢失时先查询：已经 `ready` 则跳过；仍待上传且链接有效时再重试。
不支持单文件分块断点续传。

## 总状态与逐文件状态

状态响应只有元数据，不内联 Markdown、JSON 或结果文件字节。
`files` 按原清单顺序返回，使用 `client_file_id` 对应输入，不依赖文件名唯一性。

完整字段、类型和可空约束见 [TaskStatusResponse](#taskstatusresponse)，下面说明状态聚合规则。

Task 顶层没有 `result_url`；`files[]` 不暴露底层 `output_files`。
已完成文件可以在父任务仍为 `running/queued/waiting_upload` 时下载。

总状态优先级如下：

| 条件 | `status` |
|---|---|
| 存在运行中的文件，包括导出阶段 | `running` |
| 没有运行文件，存在排队文件 | `queued` |
| 剩余未完成文件全部等待上传 | `waiting_upload` |
| 所有文件成功 | `completed` |
| 所有文件终结，成功与失败并存 | `partial` |
| 所有文件终结，无成功文件 | `failed` |
| 用户取消任务 | `canceled` |

因此，第一个文件解析完成而第二个文件尚未上传时，总状态可以由 `running` 回到 `waiting_upload`，第一个文件的下载链接仍然有效。
取消只停止未完成的文件，已经成功或失败的文件保持原状态；重复取消终态任务返回 `409`。

## 输出格式与下载

一个输入文件对应一个下载链接，与该 Task 有多少输入文件无关。

| `output_formats` | 下载内容 |
|---|---|
| `["default"]` | ZIP：`markdown.md`、`middle_json.json`、`structured_content.json`、`images/`。 |
| `["docx"]` | 图片内嵌的单个 `.docx`。 |
| `["html"]` | 图片内嵌的单个 `.html`。 |
| `["latex"]` | ZIP：`document.tex`、`images/`。 |
| 多个不同值 | 一个 ZIP，平铺对应文件，不嵌套 ZIP。 |

多格式 ZIP 内 DOCX 和 HTML 分别为 `document.docx`、`document.html`。
`default` 和 LaTeX 共用同一套物化图片及 `images/` 引用；无图片时目录可为空。
DOCX 和 HTML 即使放入多格式 ZIP，也保持图片内嵌。
`default` 不自动包含 `model_output.json`；不生成跨输入文件的总 ZIP 或批次 `manifest.json`。

每个输入只解析一次，基于同一结果生成所选格式。全部所选格式完成并保存后才发布该文件的 `completed` 和链接。
任一格式失败，该文件为 `failed/result_packaging_failed` 且链接为空，不提供不完整产物；其他文件继续执行。
导出复用已物化素材，不重新打开源文件或修改原语义树。图片缺失时报错，不交付引用失效的 HTML 单文件。
HTML 的“单文件”保证图片内嵌，公式脚本仍沿用现有 renderer 的行为。

- SaaS 返回对象存储签名地址，结果字节不经过网关。
- 本地返回 `/v1/files/{file_id}/content` 的完整地址，流式输出已保存产物。
- Router 映射自身 Files 地址并流式代理 worker，不暴露内部 worker 地址。

轮询及重复下载不重新渲染或打包。下载签名过期时，可在结果保留期内再次查询任务取得有效地址；这与不可续期的上传链接不同。
已经完成的结果不会因为其他文件仍在等待而提前回收，保留期从父 Task 清理完成后开始。
认证只附加到 API 同源请求；对象存储及跨源重定向不得携带 MinerU API Key。

## 客户端示例

`scripts/http_task_example.sh` 调用同目录 Python 脚本，仅需要 Python 3.10+ 标准库。
创建请求不会自动重试；文件传输有界重试，先查询状态再处理响应丢失。
下载使用临时文件、验证对应格式后原子保存，成功文件保存在 `RESULT_DIR/<client_file_id>/<输入文件主名>.<扩展名>`。

```bash
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

OUTPUT_FORMATS=default,latex,docx RESULT_DIR=results \
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# 使用首次提交的文件列表和顺序恢复，不重新创建任务。
TASK_ID=job_... MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf
```

| 配置 | 默认值 | 说明 |
|---|---|---|
| `MINERU_API_URL` | 必填 | API 或 Router 基地址。 |
| `MINERU_API_KEY` | 空 | 服务认证。 |
| `SEND_HASH` | `1` | `0` 时不计算或提交 hash。 |
| `TASK_ID` | 空 | 恢复已有任务，跳过 ready 文件。 |
| `MINERU_TIER`、`OCR_MODE`、`PAGE_RANGE` | 服务默认 / `auto` / 整本 | 解析参数。 |
| `INCLUDE_MODEL_OUTPUT` | `0` | `1` 时附加可用模型输出，必须包含 default 格式。 |
| `OUTPUT_FORMATS` | `default` | 逗号分隔，转为 JSON 数组。 |
| `RESULT_DIR` | `results` | 逐文件结果目录。 |
| `MAX_POLLS`、`POLL_INTERVAL` | `300`、`1` | 查询预算和间隔秒数。 |
| `CONNECT_TIMEOUT`、`REQUEST_TIMEOUT` | `10`、`600` | 连接与单次传输超时秒数。 |
| `MINERU_PYTHON` | `python3` | Python 可执行文件。 |

退出码为 `0`（全部成功）、`2`（部分成功）、`1`（失败或取消）、`124`（轮询预算耗尽）。已下载的成功结果保留。
轮询超时不取消服务端任务。`UPLOAD_MODE` 与 `RESULT_FILE` 已删除。

## 部署、Router 与迁移

本地不设固定客户端上传大小或 PDF 页数上限；`usage.limits.max_file_size_bytes/max_pages_per_file` 为 `null`。
URL 下载沿用服务的来源策略：默认 HTTPS；HTTP 需启用 `--allow-http-source`，下载超时默认 60 秒、大小默认 200 MiB。
任务记录和上传索引仍为进程内状态，重启不保证恢复。

Router 按就绪文件独立选择 worker 并预占负载，同一父 Task 可使用多个 worker。
URL 交给 worker 下载，客户端上传字节按 worker 的缓存预检结果传输。
健康接口的 `features.task_output_formats` 声明 Tasks 格式，`sources` 为 `upload/url`，`task_model_output` 声明模型输出能力。
不声明新 Tasks 能力的旧 worker 不参与 Tasks 调度，更新时应先更新 worker，再更新 Router 和客户端。
Router 使用有界、可续租的内部结果租约保护 worker 产物，上传窗口不受结果租约影响；Router 丢失后租约会自行结束。已完成任务按 worker 并行续租，每轮网络等待最多 10 秒，避免失联 worker 阻塞其他 worker 的结果保留。

创建请求仅接受 `files` 和 `file_name`，不兼容 `manifest`、`filename`；新旧字段混传同样返回 400。Tasks 响应及 Files 列表、详情统一使用 `file_name`。
这是同一 `/v1` 下的破坏性变更：先停收请求并排空在途任务，再协调更新 worker、Router、SDK、file_parse 包装服务及其他客户端；SaaS 由部署方同步对齐。

本次协议不保留旧表单、直传或 `markdown/middle_json/structured_content/zip` Tasks 格式别名。
客户端应改用 JSON、`files[].result_url` 及逐文件保存方式。
API Server 和 Router 不提供 `/v1/file_parse`、`/v1/tasks/{task_id}/result`。
需要同步单文件调用时，独立启动 [file_parse 包装服务](file-parse.md)，由它调用本页的共同 Tasks 契约。
SaaS Go 服务和对象存储配置由部署方按此共同契约实现，不属于本地 Python 服务代码。

## 父任务列表与缓存

`GET /v1/tasks` 返回当前调用方的父任务摘要。支持 `status`（逗号分隔）、`limit`（默认 20，上限 100）、`after`（上一页 last_id）、`order`（默认 desc）、`created_after`。`created_after` 必须是带时区的 ISO-8601 时间，按实际时间过滤（包含边界），支持 UTC、时区偏移和小数秒；非法值返回 400。按 created_at 和 task_id 稳定排序；游标不存在、过期或属于其他调用方时返回 404。

响应为 `object=list/data/first_id/last_id/has_more`；每项包含 `task_id/status/created_at/file_count`，不展示内部子执行。

SDK 即使连接同机也计算 hash 并提交清单。缓存有效且已完成上传的同 hash 后续免传；并发首次上传允许重复传输但源字节去重。默认缓存保留 24 小时、每 5 分钟扫描，沿用 retention 配置；活动任务与传输引用保护字节。缓存过期或服务重启后允许重新上传。
