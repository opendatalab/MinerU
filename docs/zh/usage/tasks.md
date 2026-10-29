# Tasks API

本文描述 **4.1.0** 契约。`/v1/tasks` 是 API Server、Router 和兼容 SaaS 的共同异步解析入口：提交 JSON 清单、上传缺失文件、查询状态，再按文件下载结果。旧客户端请先阅读 [4.1 迁移说明](../reference/migration_4_1.md)。单文件同步调用见 [file-parse API](file_parse.md)。

## 启动与认证

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier auto
```

服务的 `/docs` 和 `/openapi.json` 提供当前部署的接口定义。配置 API Key 时，API 请求使用 `Authorization: Bearer <API_KEY>`；未配置 Key 的本地服务可省略该头。请求基地址可带反向代理路径前缀。

| 方法与路径 | 成功响应 | 用途 |
|---|---|---|
| `POST /v1/tasks` | 202，`CreateTaskResponse` | 提交 JSON 清单。 |
| `PUT files[].upload.url` | 本地/Router 为 200，`UploadFileResponse` | 上传一个文件的原始字节；对象存储接受其成功 2xx 响应。 |
| `GET /v1/tasks/{task_id}` | 200，`TaskDetailResponse` | 查询完整状态快照。 |
| `POST /v1/tasks/{task_id}/cancel` | 200，`CancelTaskResponse` | 取消未完成文件，无需请求体。 |

本地上传路径为 `/v1/tasks/{task_id}/uploads/{file_id}`，客户端应使用返回的完整 URL。没有任务/文件列表、公开 DELETE 或批次结果接口。主 API Server 和 Router 不提供 `/v1/file_parse`。

## 创建请求

`POST /v1/tasks` 只接受 `application/json`。业务参数全部放在 JSON 中；查询参数、未知字段或非法值返回 400，其他 Content-Type 返回 415。

| 字段 | 类型 | 默认值或要求 |
|---|---|---|
| `files` | object[] | 必填，1–100 项，不能传编码后的 JSON 字符串。 |
| `tier` | string 或 null | 可省略；`flash/basic/standard/advanced`，见下文。 |
| `ocr_mode` | string | `auto`；可选 `auto/txt/ocr`。 |
| `page_range` | string | 空字符串表示整本；PDF 支持 `1-5`、`r1`、`all` 等。 |
| `output_formats` | string[] | `["default"]`；非空，可选 `default/docx/html/latex`，按首次出现去重。 |

每个 `files[]` 条目：

| 字段 | 类型 | 要求 |
|---|---|---|
| `file_name` | string | 必填，1–1024 字符的逻辑文件名，保留扩展名；允许同名输入。 |
| `bytes` | integer 或 null | 上传项必填且为正整数；URL 项可省略，提供时必须与实际大小匹配。 |
| `url` | string 或 null | HTTP(S) 下载地址；省略或 null 表示客户端上传。 |
| `sha256sum` | string 或 null | 可选，64 位小写十六进制 SHA-256，用于校验；上传项还用于缓存预检。 |
| `page_range` | string 或 null | 可选，覆盖全局 PDF 页范围；非 PDF 原生文档仍整本解析。 |

请求禁止 `file_id/client_file_id`，包括值为 null。服务端按输入顺序分配公共 `file_id`；相同名称或内容仍获得不同 ID。创建后通过 ID 关联上传、查询、取消与下载。在本地/Router 中，结果 Files ID 与该 ID 一致；产物生成前按该 ID 查询 Files 返回 404。源缓存 ID 不公开。

### 档位选择

- PDF/图片省略 `tier` 或传入 null，按 **standard → basic → flash → advanced** 选择支持的默认值；仅支持 Advanced 的服务在省略档位时默认选择 Advanced。
- 显式请求不支持的 PDF/图片档位，整次创建返回 `400 invalid_request`，`error.param="tier"`，不会接单或自动升降档。
- 所有正常可用的兼容服务都支持现有原生文档格式。Office、HTML 等固定使用 Flash，不受 `/v1/tiers` 或合法请求档位限制；空字符串和未知枚举仍非法。
- 混合批次分别确定档位。例如服务支持 Standard，未指定 tier 的 PDF + DOCX 分别使用 Standard、Flash。档位在接单时确定，排队或转发时不重新选择；实际执行值见成功文件的 `parse.tier`。

### 示例请求与创建响应

以下 JSON 的文件大小、ID、URL 和时间均为示意值；实际上传应填写真实字节数。

```json
{
  "files": [{"file_name": "report.pdf", "bytes": 12345}],
  "tier": "standard",
  "ocr_mode": "auto",
  "output_formats": ["default", "docx"]
}
```

202 创建回执：

```json
{
  "task_id": "job_example",
  "status": "waiting-file",
  "files": [{
    "file_id": "file-example",
    "file_name": "report.pdf",
    "status": "waiting-file",
    "upload": {
      "url": "http://127.0.0.1:8000/v1/tasks/job_example/uploads/file-example",
      "headers": {},
      "expires_at": 1800003600
    }
  }]
}
```

202 仅表示接单。创建回执不包含文件大小、解析统计或结果链接，即使已经完成也通过 GET 获取详情。

URL 输入也使用相同创建接口，例如：

```json
{
  "files": [{"file_name": "report.pdf", "url": "https://example.com/report.pdf"}],
  "output_formats": ["html"]
}
```

URL 项由服务端下载，不需要客户端 PUT；替换为部署允许访问的真实下载地址。

## 响应字段

成功响应直接返回对象，没有 `data/task` 外层包装。以下 `?` 表示字段可省略，不是字段名的一部分。

| 响应 | 顶层字段 | files[] 字段 |
|---|---|---|
| `CreateTaskResponse` | `task_id`、`status`、`files` | `file_id`、`file_name`、`status`、`upload?` |
| `TaskDetailResponse` | `task_id`、`status`、`created_at`、`started_at?`、`finished_at?`、`output_formats`、`progress`、`files`、`error?` | `file_id`、`file_name`、`status`、`bytes?`、`url?`、`upload?`、`parse?`、`result_url?`、`error?` |
| `CancelTaskResponse` | `task_id`、`status`（固定 canceled）、`finished_at`、`files` | `file_id`、`file_name`、`status`、`parse?`、`result_url?`、`error?` |
| `UploadFileResponse` | `task_id`、`file_id`、`status`（本次文件状态） | 无 |

`task_id/file_id/file_name/status` 为字符串，`files` 为数组，`bytes` 为整数，`url/result_url` 为字符串，`output_formats` 为字符串数组。时间字段为 ISO-8601 UTC 字符串。

| 嵌套对象 | 字段与含义 |
|---|---|
| `upload` | `url: string`、`headers: object`（字符串键值，可为空）、`expires_at: integer`（Unix 秒）。仅当前需要上传时出现。 |
| `parse` | `tier: string`、`duration_ms?: integer`、`parser_version?: string`。仅成功文件返回；版本为执行 worker 的版本。耗时含源读取/下载、校验、解析、导出，不含等待上传、准备名额或解析排队。 |
| `progress` | 必带整数 `completed/failed/total`；取消文件不计入 completed 或 failed，total 为输入数量。 |
| `error` | 必填非空 `code: string`、`message: string`，以及 `param?: string`。可出现在 GET 顶层/文件内和取消回执的失败文件内；不包含 type。 |

Tasks 成功响应递归省略 null，保留 `0/false/[]/{}`。每次响应都是完整快照，字段缺失表示当前无值，不能保留旧上传链接。客户端应容忍未来新增字段。公共 HTTP 错误同样省略空 param；Files 和 file-parse 的空值规则独立。

GET 成功完成示例（200）：

```json
{
  "task_id": "job_example",
  "status": "completed",
  "created_at": "2026-10-28T00:00:00Z",
  "started_at": "2026-10-28T00:00:02Z",
  "finished_at": "2026-10-28T00:00:10Z",
  "output_formats": ["default", "docx"],
  "progress": {"completed": 1, "failed": 0, "total": 1},
  "files": [{
    "file_id": "file-example",
    "file_name": "report.pdf",
    "status": "completed",
    "bytes": 12345,
    "parse": {"tier": "standard", "duration_ms": 8000, "parser_version": "4.1.0"},
    "result_url": "http://127.0.0.1:8000/v1/files/file-example/content"
  }]
}
```

本地 PUT 确认示例：`{"task_id":"job_example","file_id":"file-example","status":"pending"}`。对象存储的成功 PUT 不要求返回这个 JSON，随后查询 Tasks 确认即可。

## 上传、URL 与缓存

上传使用原始字节和返回的 `upload.url/headers`，固定 PUT，无需 complete。客户端只在文件为 `waiting-file` 且存在 `upload` 时发送文件；没有 upload 时只轮询。校验完成的文件独立排队，不等待其他文件，也不在等待上传期间预占解析名额。

- 上传项的 hash 命中有效缓存时免传；缓存仅复用源字节，每次任务仍按本次参数解析。并发首次上传允许重复传输，但源存储去重。
- URL 项由服务端下载，不返回上传动作，不要求 HEAD 或 Content-Length；使用 `file_name` 识别格式，下载后补齐 bytes，校验提供的大小/hash。URL 每次下载，不按 hash 跳过下载。
- 本地 URL 来源默认仅 HTTPS；HTTP 需启用 `--allow-http-source`，默认下载超时 60 秒、大小上限 200 MiB。客户端直接上传没有固定大小或 PDF 页数上限，部署可另加网关限制。
- 上传链接从创建起固定有效 3600 秒，查询不续期。截止前已受理的上传可完成；尚未受理的文件到期变为 `failed/upload_expired`。有效上传后的解析不受上传截止时间限制。
- 成功 PUT 后不重复上传；响应丢失先查询。`pending/running/completed` 跳过上传；`waiting-file` 无 upload 继续等待；失败且仍有有效 upload 时才有界重试。过期须创建新任务，不支持单文件分块断点续传。

本地源缓存默认保留 24 小时，每 5 分钟扫描；活动任务和传输引用保护源字节。API/Router 的任务记录和索引为进程内状态，重启不保证恢复。

## 状态与取消

| 文件状态 | 含义 |
|---|---|
| `waiting-file` | 等待/接收上传、URL 下载、校验或对象存储确认。 |
| `pending` | 输入就绪，等待解析资源。 |
| `running` | 解析、导出或保存结果。 |
| `completed` | 所选格式全部成功，结果可下载。 |
| `failed` | 输入、解析或导出失败，读取 error。 |
| `canceled` | 未完成文件被取消。 |

父任务优先汇总为 running，其次 pending，再次 waiting-file；全部成功为 completed，全部终结且成功/失败并存为 partial，无成功文件为 failed。取消操作使父任务为 canceled。父任务可能从 running 回到 waiting-file，已完成文件的链接仍可用。

取消保留已成功/失败的文件，将其他文件置为 canceled；返回的文件状态仅为 completed/failed/canceled。底层不可中断计算或清理可能仍在收尾，但迟到结果不覆盖取消终态。已终结任务再次取消返回 `409 task_already_terminal`。

取消一个仍在处理的单文件任务，200 回执示例：

```json
{
  "task_id": "job_example",
  "status": "canceled",
  "finished_at": "2026-10-28T00:00:05Z",
  "files": [{"file_id": "file-example", "file_name": "report.pdf", "status": "canceled"}]
}
```

## 输出与下载 {#outputs-and-downloads}

| output_formats | 每个输入的产物 |
|---|---|
| `["default"]` | ZIP：`markdown.md`、`middle_json.json`、`structured_content.json`、`images/`。 |
| `["docx"]` | 图片内嵌的 DOCX。 |
| `["html"]` | 图片内嵌的 HTML；公式脚本仍遵循 renderer 行为。 |
| `["latex"]` | ZIP：`document.tex`、`images/`。 |
| 多种 | 一个平铺 ZIP，不嵌套 ZIP；DOCX/HTML 文件名为 `document.docx/document.html`。 |

每个输入解析一次；全部所选格式完成后发布一个 `files[].result_url`。某格式导出失败时该文件为 `failed/result_packaging_failed`，不返回部分产物；其他文件继续。没有批次总 ZIP 或顶层 result_url。已完成文件无需等待父任务终结即可下载。

ModelJson 仅用于内部处理。`include_model_output` 已删除，提交该字段（包括 false）返回 400，产物不含 `model_output.json`。

本地与 Router 返回 Files 下载地址，SaaS 可返回对象存储签名地址。使用原始结果 URL，不拼接存储路径；仅对 API 同源请求附加 API Key，每次重定向重新判断，不能向对象存储泄露 Key。签名过期可在结果保留期内重新查询任务获取链接。轮询/下载不重新渲染；未完成的其他文件不会导致已完成结果提前回收。

## 错误

请求级错误为非 2xx 和 `{"error": ...}`，例如：

```json
{
  "error": {
    "code": "task_already_terminal",
    "message": "Task is canceled"
  }
}
```

| HTTP | 常见 code | 场景 |
|---|---|---|
| 400 | `invalid_request`、`page_range_invalid` | 非法字段、档位、页范围或旧参数。 |
| 400 | `upload_size_mismatch`、`file_hash_mismatch`、`upload_interrupted` | 上传校验或传输失败。 |
| 401 | `invalid_api_key` | 服务启用认证而凭据无效。 |
| 404 | `task_not_found`、`upload_not_found` | 任务不可见/已清理，或上传文件 ID 不存在。 |
| 409 | `task_already_terminal`、`file_already_terminal` | 取消终态任务或向不接受上传的文件发送数据。 |
| 410 | `upload_expired` | 上传窗口到期。 |
| 415 | `invalid_request` | 创建请求不是 JSON。 |
| 503 | `quality_tier_unavailable`、`server_shutting_down` 或模型预加载错误码 | 当前服务不能接单。 |

异步下载、解析和导出失败通常通过 **200 查询快照内的 files[].error** 返回；HTTP 200 不等于解析成功。公共错误只包含必填的非空 code、message 和可选 param，不再输出或接受 type。param 无值时省略，接收端也允许 null；根据 HTTP 状态与 code 分支处理，不解析 message 判断类别。请检查所有文件状态，缺少顶层 error 不代表全部成功。

## 可运行客户端示例

以下 Bash 示例从仓库根目录运行，复用标准库 Python 客户端，自动计算真实文件大小和 hash、处理上传与逐文件下载：

```bash
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf notes.docx

OUTPUT_FORMATS=default,docx RESULT_DIR=results \
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf

# 使用原来的文件列表和顺序继续，不重新创建任务。
TASK_ID=job_example MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf
```

`MINERU_API_KEY` 设置认证，`MINERU_TIER/OCR_MODE/PAGE_RANGE` 设置解析选项，`SEND_HASH=0` 关闭预检 hash。`MAX_POLLS/POLL_INTERVAL` 默认 300/1，限制轮询；`CONNECT_TIMEOUT/REQUEST_TIMEOUT` 限制单次传输。创建请求不自动重试。退出码：0 全部成功、2 部分成功、1 失败/取消、124 轮询耗尽，已下载文件保留。

直接 HTTP 调用时，通过配置基址加 `/v1/tasks/{URL 编码后的 task_id}` 查询，通过同一路径加 `/cancel` 发送 POST 取消。创建回执中的 ID 是恢复入口，Tasks 不返回 status_url。
