# 单文件同步解析：file_parse

`mineru-kit file-parse-server` 提供独立的同步 HTTP 包装服务：接收一个文件，创建 Tasks、上传、等待，完成后直接返回 ZIP、DOCX 或 HTML 附件。它连接本地 API、Router 或 SaaS；切换部署只修改上游地址和认证。

## 启动

```bash
mineru-kit file-parse-server --api-url http://127.0.0.1:8000

# 上游基地址可以带路径前缀；不要附加 /v1/tasks。
MINERU_API_URL=https://mineru.net/api MINERU_API_KEY=your-key \
  mineru-kit file-parse-server
```

| 启动参数 | 默认值 | 含义 |
|---|---|---|
| `--api-url` | `MINERU_API_URL`，否则必填 | 上游统一 Tasks API 基地址。 |
| `--api-key` | `MINERU_API_KEY`，否则为空 | 上游 Bearer Token。 |
| `--host`、`--port` | `127.0.0.1`、`8003` | 包装服务监听地址。 |
| `--state-dir` | `${MINERU_HOME:-~/.mineru}/file-parse` | 恢复记录及传输临时目录。 |
| `--retention-seconds` | `86400` | 首次观察终态后保留恢复记录的秒数；`0` 禁用回收。 |

包装入口不提供认证，面向本机或可信网络；部署方可以在反向代理设置访问认证。`--api-key` 只用于连接上游。请求不能覆盖上游地址或认证，入站 Authorization 不会转发到上游。

服务只执行 HTTP 上传、查询、下载，不初始化解析模型。OpenAPI 位于 `/openapi.json`，交互文档位于 `/docs`。这些路由属于独立包装服务，不注册到原 API Server 或 Router。

## 上传与等待

`POST /v1/file_parse` 仅接受 `multipart/form-data`，业务参数放在表单字段中；不接受查询参数、未知字段或重复标量字段。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `file` | 必填 | 一个非空文件，保留扩展名；不接受多文件。 |
| `tier` | 上游默认 | 可选，仅对 PDF/图片生效；`flash/basic/standard/advanced`。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 与 Tasks 一致，PDF 支持 `1-5`、`r1`、`all`；格式支持范围由上游校验。 |
| `output_formats` | `default` | `default/docx/html/latex`；重复同名字段表达列表，重复值按首次出现去重。 |
| `wait_timeout` | `3600` | 整数秒，范围 `0–3600`。 |
| `request_id` | 自动生成 | 1–64 个字母、数字、下划线或短横线。提前指定并保存它，可在连接中断后查询。 |

未提供 tier 表单字段时，包装层在上游 Tasks 请求中省略该字段，由上游按 **standard → basic → flash → advanced** 选择默认档位；仅开放 advanced 时自动选择 advanced。
显式值原样传递，PDF/图片请求的档位不受上游支持时返回错误，不自动升降档。
所有正常可用服务都支持 Office、HTML 等现有原生格式，固定使用 flash，即使传入上游未开放的合法档位也不改变原生解析路径。
表单空字符串或字符串 `"null"` 不是省略字段，属于非法值；OpenAPI 将 tier 声明为可省略的字符串枚举，不声明可空值。完整规则见 [Tasks tier 语义](tasks.md#tier-selection)。

```bash
# 成功时 response.body 是附件；超时或错误时是 JSON，请同时检查 response.headers。
curl --max-time 7200 -D response.headers -o response.body \
  http://127.0.0.1:8003/v1/file_parse \
  -F 'file=@report.pdf' \
  -F 'request_id=report-20261023-001' \
  -F 'tier=flash' -F 'ocr_mode=txt' \
  -F 'output_formats=default' -F 'output_formats=docx'
```

`output_formats` 的列表语义与 [Tasks](tasks.md) 一致；multipart 中不使用 JSON 字符串或逗号分隔字符串。

已删除 `include_model_output`；表单携带该字段时返回 400，不接受旧参数。

## 响应结构与格式 {#responses}

以下三个接口均属于独立 file_parse 包装服务。它们的 JSON 状态对象是下文的 `StatusResponse`，独立于 Tasks 的创建、详情和取消响应模型。

| 接口 | HTTP | Content-Type | 正文格式 |
|---|---|---|---|
| `POST /v1/file_parse`，同步等待成功 | 200 | 随输出格式 | ZIP、DOCX 或 HTML 附件字节。 |
| `POST /v1/file_parse`，等待到期或 `wait_timeout=0` | 202 | `application/json` | 完整 `StatusResponse`，保存 `status_url` 后继续查询。 |
| `GET /v1/file_parse/{request_id}`，查询成功 | 200 | `application/json` | 完整 `StatusResponse`，即使已知任务状态为失败、取消或提交结果未知。 |
| `GET /v1/file_parse/{request_id}/result`，下载成功 | 200 | 随输出格式 | 与同步成功时相同的附件字节。 |
| 上述接口的已知业务、存储或上游调用错误 | 4xx / 5xx | `application/json` | 包装服务错误对象；已有恢复记录时同时携带状态字段。 |

JSON 正文使用 UTF-8；可空字段输出 `null`，不以省略字段代替。下列示例的 ID、地址为示意值。

### POST /v1/file_parse：200 附件响应

成功时直接返回文件内容，不包含 JSON 外层包装，也不把文件编码为 Base64。
文件名为 `<输入文件主名>.<产物扩展名>`，例如输入 `report.pdf`：

| `output_formats` | 下载文件名 | Content-Type | 文件内容 |
|---|---|---|---|
| `["default"]` | `report.zip` | `application/zip` | `markdown.md`、`middle_json.json`、`structured_content.json`、`images/`；不包含模型输出。 |
| `["docx"]` | `report.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | 图片内嵌的 DOCX。 |
| `["html"]` | `report.html` | `text/html; charset=utf-8` | 图片内嵌的 UTF-8 HTML。 |
| `["latex"]` | `report.zip` | `application/zip` | `document.tex`、`images/`。 |
| 多个不同格式 | `report.zip` | `application/zip` | 各格式平铺在一个 ZIP 中，不嵌套 ZIP；DOCX/HTML 成员为 `document.docx`、`document.html`。 |

包装服务原样返回验证过的上游产物，不重新渲染或打包。完整成员规则、图片及公式行为沿用 [Tasks 输出格式](tasks.md#outputs-and-downloads)。

```http
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Disposition: attachment; filename="report.zip"
X-Request-ID: report-20261023-001
X-Task-ID: job_0123456789abcdef01234567

<ZIP 文件二进制内容>
```

| 响应头 | 类型 | 返回条件及含义 |
|---|---|---|
| `Content-Type` | string | 成功附件的 MIME 类型，按上表确定。 |
| `Content-Disposition` | string | `attachment` 及建议文件名；非 ASCII 文件名可能使用标准的 `filename*` 编码。 |
| `X-Request-ID` | string | 成功附件必带，值为包装服务的请求 ID。 |
| `X-Task-ID` | string | 成功附件必带，值为已确认的上游 Tasks ID。 |

### StatusResponse：查询与 202 的完整 JSON 对象

以下七个顶层字段均会出现。状态对象不返回 Tasks 的 `files/progress`，不暴露上游上传链接或签名下载地址。

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `request_id` | string | 否 | 包装服务的恢复标识，来自请求或自动生成。 |
| `task_id` | string | 是 | 已确认的上游父任务 ID；尚未获得有效创建响应时为 `null`。 |
| `status` | string | 否 | 包装服务保存的单文件状态，具体取值见下表。 |
| `status_url` | string | 否 | 包装服务自身的完整状态查询 URL。 |
| `result_url` | string | 是 | `completed` 时指向包装服务自身的 `/v1/file_parse/{request_id}/result`，其他状态为 `null`。 |
| `output_formats` | array of string | 否 | 按首次出现去重后的输出格式。 |
| `error` | object | 是 | 保存的最近错误或上游逐文件错误；无错误时为 `null`，具体字段见“错误响应”。 |

| `status` | 含义 |
|---|---|
| `prepared` | 本地源文件和恢复记录已持久化，尚未开始提交。 |
| `submitting` | 正在向上游创建任务，尚无确认后的 `task_id`。 |
| `waiting-file` | 上游仍在接收、下载、校验或确认输入；上游省略 upload 不表示已经就绪。 |
| `pending` | 已进入上游执行队列。 |
| `running` | 上游正在解析或导出。 |
| `completed` | 上游完整产物已生成，可通过包装服务下载。 |
| `failed` | 已知失败，读取 `error`。 |
| `canceled` | 上游单文件任务已被取消。 |
| `submission_unknown` | 创建请求可能已被上游接受，但包装服务没有获得有效任务 ID，不自动重放创建请求。 |

包装服务仅处理一个输入，不使用父 Tasks 的 `partial` 聚合状态。`completed` 表示上游产物已就绪；实际下载仍可能因结果过期或传输失败返回错误。

### POST /v1/file_parse：202 JSON 响应

等待预算耗尽时，返回 `202 Accepted` 和完整 `StatusResponse`。这表示上游继续执行，不是解析失败。
`wait_timeout=0` 在源文件确认就绪后直接返回 `202`；即使任务已经完成，也返回 JSON，此时 `status` 可以是 `completed`，`result_url` 已非空。

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
Location: http://127.0.0.1:8003/v1/file_parse/report-20261023-001
Retry-After: 2
X-Request-ID: report-20261023-001
X-Task-ID: job_0123456789abcdef01234567
```

| 响应头 | 返回条件及含义 |
|---|---|
| `Location` | 202 必带，与正文 `status_url` 相同。 |
| `Retry-After` | 202 必带，当前值为字符串 `2`，单位为秒，提示查询间隔。 |
| `X-Request-ID` | 202 必带，与正文 `request_id` 相同。 |
| `X-Task-ID` | 已确认上游任务 ID 时携带，与正文 `task_id` 相同。 |

完整 JSON 正文示例：

```json
{
  "request_id": "report-20261023-001",
  "task_id": "job_0123456789abcdef01234567",
  "status": "running",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-20261023-001",
  "result_url": null,
  "output_formats": [
    "default"
  ],
  "error": null
}
```

### GET /v1/file_parse/{request_id}：200 状态查询

查询已有记录且上游状态刷新成功时，返回 `200 OK`、`Content-Type: application/json` 和完整 `StatusResponse`。
返回值可以是 `running`，也可以是 `completed/failed/canceled/submission_unknown`；HTTP `200` 不表示解析成功。
当前状态查询不统一附加 `X-Request-ID/X-Task-ID`，从 JSON 字段读取这两个标识。

```json
{
  "request_id": "report-20261023-001",
  "task_id": "job_0123456789abcdef01234567",
  "status": "completed",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-20261023-001",
  "result_url": "http://127.0.0.1:8003/v1/file_parse/report-20261023-001/result",
  "output_formats": [
    "default"
  ],
  "error": null
}
```

若刷新上游时本次调用失败，则返回非 2xx 错误 JSON；原任务仍可能存在并继续执行，使用相同 `request_id` 继续查询。

### GET /v1/file_parse/{request_id}/result：200 附件下载

成功响应的二进制格式、MIME、命名及 `X-Request-ID/X-Task-ID` 与同步 POST 的 200 附件相同。
调用者不需要上游 API Key。每次下载先查询原任务并获取当前下载链接；结果未就绪返回 `409 result_not_ready`，没有已确认上游任务时也可能返回 `409 task_not_created`。
下载中断可以再次 GET。签名失效时重新查询原任务，最多尝试 3 次；上游结果已过期且无法恢复时返回错误，不自动重新解析。

## 错误响应 {#error-responses}

file_parse 使用自己的 `code/message/upstream_status` 错误结构；主 Tasks 接口的 `ErrorDetail` 为必填 `code/message` 和可选 `param`，不包含 `type`。
包装层直接返回的 `error` 包含以下三个字段，JSON 值中的 `upstream_status` 不省略：

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `error.code` | string | 否 | 包装层错误码或保留的上游错误码。 |
| `error.message` | string | 否 | 错误说明。 |
| `error.upstream_status` | integer | 是 | 已知上游 HTTP 错误状态码；本地错误、网络异常或通过任务状态得知的解析失败可以为 `null`。 |

### 没有恢复记录时

只返回顶层 `error`，没有 `request_id/task_id/status_url` 等状态字段。例如提交 JSON 而不是 multipart：

```http
HTTP/1.1 415 Unsupported Media Type
Content-Type: application/json
```

```json
{
  "error": {
    "code": "unsupported_media_type",
    "message": "Expected multipart/form-data",
    "upstream_status": null
  }
}
```

### 已有恢复记录时

返回完整 `StatusResponse` 的七个顶层字段，但用本次操作的错误覆盖正文 `error`。`status` 仍表示该记录当前状态，不会因为本次 HTTP 失败就统一改成 `failed`。
例如查询尚未完成的结果，HTTP 为 `409 Conflict`，任务仍为 `running`：

```json
{
  "request_id": "report-20261023-001",
  "task_id": "job_0123456789abcdef01234567",
  "status": "running",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-20261023-001",
  "result_url": null,
  "output_formats": [
    "default"
  ],
  "error": {
    "code": "result_not_ready",
    "message": "The result is not ready",
    "upstream_status": null
  }
}
```

当前错误响应不保证附加 `X-Request-ID/X-Task-ID` 或 `Location`，恢复信息从 JSON 正文读取。只取得了客户端预先指定的 ID、但没有对应恢复记录时，仍使用前一种仅含 `error` 的结构。

状态查询成功响应中的 `error` 是持久化的错误快照：它可能是上述三字段结构并额外含 `http_status`，也可能保留上游逐文件错误的 `code/message` 和可选 `param`。历史恢复记录保持原样，可能仍含旧的 `type` 或其他字段；这不代表新 Tasks 协议接受旧字段。
其中 `http_status` 是包装层保存的错误响应状态，不是 `upstream_status`；客户端应容忍错误对象的扩展字段。
例如创建结果未知后，状态查询仍可返回 `200 OK` 和以下完整状态，不自动再次创建任务：

```json
{
  "request_id": "report-20261023-001",
  "task_id": null,
  "status": "submission_unknown",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-20261023-001",
  "result_url": null,
  "output_formats": [
    "default"
  ],
  "error": {
    "code": "submission_unknown",
    "message": "Task creation outcome is unknown; not resubmitting",
    "upstream_status": null,
    "http_status": 502
  }
}
```

### 常见错误状态码

| HTTP | 常见 code | 含义 |
|---|---|---|
| 400 | `invalid_request`、`upload_interrupted` | 参数、多文件、空文件等校验失败，或完整接受前上传中断。 |
| 404 | `request_not_found` | 包装服务中没有记录，或恢复记录已过保留期。 |
| 408 | `upload_receive_timeout` | 接收入站请求体时发生传输空闲超时。 |
| 409 | `request_id_conflict`、`result_not_ready`、`task_not_created` 或取消错误码 | ID 对应不同请求、结果未就绪、尚无确认任务或上游已取消。 |
| 410 | `upload_expired` | 包装层确认上游上传窗口已经过期。 |
| 415 | `unsupported_media_type` | POST 不是 `multipart/form-data`。 |
| 500 | `storage_error` | 包装服务保存源文件、结果或访问本地存储失败。 |
| 502 | `submission_unknown`、`upstream_unreachable`、`upstream_protocol_error`、`invalid_result`、下载错误码或上游错误码 | 上游创建、查询、传输或解析失败，或结果校验失败。 |

上游认证失败以包装层的 `502` 返回，例如 `error.code="invalid_api_key"`、`error.upstream_status=401`；包装入口本身没有 API Key 认证，不把上游配置错误伪报为入口鉴权失败。
上游明确拒绝档位请求时也沿用包装层 `502`：显式 tier 不受支持保留 `error.code="invalid_request"`、`error.upstream_status=400`；没有可用默认档位的标准 Tasks 错误响应保留 `error.code="quality_tier_unavailable"`、`error.upstream_status=503`。两者记录为 `failed`，`task_id=null`，不标记为提交结果未知。
创建阶段的 5xx 响应中，仅符合共享错误结构的 `503 quality_tier_unavailable` 可确认为档位拒单；其余 5xx、408、畸形响应或网络异常仍按 `submission_unknown` 处理。明确拒单和提交结果未知都不会自动重放创建，保留期内相同 request_id 的重复请求和重启恢复沿用原记录。
`submission_unknown` 首次创建失败通常返回 `502`；后续请求仍可能得到 `409/502`，应查询原恢复记录，不自动重放创建。

## 等待、超时与断连后查询

每次调用在确认上游文件就绪后开始计时，等待包含排队、解析与导出。接收文件、上游上传、文件下载与校验准备及结果下载不占此预算，独立使用传输空闲超时。缓存命中跳过上传；只有进入 pending/running/completed 或失败/取消终态后，才清理本地保留的源文件。

- 正常轮询间隔为 2 秒。
- 连接、元数据请求、传输空闲超时分别为 10、30、600 秒。
- HTTP 客户端和反向代理的读取超时应覆盖上传、同步等待和结果传输；例如客户端配置 7200 秒，代理按业务文件大小配置相应超时。

```bash
curl http://127.0.0.1:8003/v1/file_parse/report-20261023-001
curl --fail-with-body -o report.zip \
  http://127.0.0.1:8003/v1/file_parse/report-20261023-001/result
```

完整接收后的客户端断连不取消已托管的上传或解析。上传失败先查询状态，确认已经就绪便不重传；若保留的源文件仍可用，使用相同请求 ID 和相同输入可继续原任务。

包装服务当前不提供独立的取消接口。等待超时、客户端断连或包装服务关闭都不会自动向上游发送取消请求。
已取得非空 `task_id` 且有权使用对应上游身份时，可向配置的上游 API 调用 `POST /v1/tasks/{task_id}/cancel`；不要使用包装服务的基址，也不要把 `request_id` 当作上游任务 ID。
上游取消成功后，再查询 `/v1/file_parse/{request_id}` 可观察 `canceled` 状态。取消能力沿用上游 Tasks 契约；file_parse 自身的 `status_url`、响应字段和 null 规则保持不变。

## 去重与持久化边界

完整接收并持久化源文件后，包装服务才接受请求并托管上游上传。此后连接中断不取消上传或解析；文件尚未完整接收时中断不会创建任务。

同一 `request_id`、相同文件内容、逻辑文件名及归一化解析参数复用原任务；不同输入返回 `409 request_id_conflict`。`wait_timeout` 不参与身份比较。没有预先提供 ID 且尚未收到响应的调用者，可能无法自行找回任务。

请求映射使用独立 SQLite 数据库，按上游地址和认证隔离。数据库不保存上游 API Key。单个状态目录只允许一个包装服务进程；服务重启后恢复已知任务，尚待上传且链接有效时继续上传。上游任务记录如果因重启或过期而消失，不会自动重建。

源文件在上游确认就绪后删除，终结失败也会清理源文件。结果只临时下载，发送完成或中断即删除；重启清理遗留传输文件。恢复记录的保留期限不延长上游结果有效期，查询也不会为上游上传链接续期。结果及请求过期后应使用新的请求 ID 发起新的解析。

## Python HTTP 示例

仓库提供 `scripts/http_file_parse_example.py`，依赖 `httpx`，提前打印并保存请求 ID；成功时原子保存附件，`202` 时打印查询信息并以 `124` 退出。

```bash
python scripts/http_file_parse_example.py report.pdf --output results/report.docx \
  --output-format docx --request-id report-20261023-002

python scripts/http_file_parse_example.py report.pdf --wait-timeout 0 \
  --request-id report-20261023-003
```

后续查询和下载仍使用上面的包装接口，不需要安装 MinerU SDK 或重建 `ParseResult`。

## 升级与恢复记录

本次协议升级要求 API Server、Router、SDK、Gradio 与包装服务同步更新，先停收请求并排空旧任务。
恢复记录目录使用 `tasks-v5-` 加既有上游地址与认证摘要，不加载、不迁移也不自动删除旧 `tasks-v4-`、`tasks-v3-`、`tasks-v2-` 或无前缀目录。包装服务自身的 `status_url` 与 Location 头保持不变，上游 Tasks 查询改用固定路径。旧 request_id 不再恢复；新空间内继续支持幂等和进程重启恢复。

上游 POST 只返回创建回执。包装服务先持久化服务端生成的 task_id/file_id 和原始请求上下文，再 GET 查询详情；查询失败或重启仍恢复同一任务，不重放创建。只有 Tasks 成功响应省略 null，本文的 StatusResponse、status_url 和错误响应保持原规则。
