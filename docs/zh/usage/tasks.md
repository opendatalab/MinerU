# V1 Tasks：JSON 清单、逐文件上传与下载

`/v1/tasks` 是本地 API Server、Router 和 SaaS 的共同异步客户端契约。

`/v1/tiers` 只约束 PDF/图片。所有正常可用的兼容服务均支持现有原生文档格式并固定使用 flash，无需额外能力字段。Router 转发原生文件时省略 tier；PDF/图片转发已确定的档位。
客户端用 JSON 创建任务，按 `files[].upload` 上传缺失文件，再通过固定路径 `GET /v1/tasks/{task_id}` 轮询。
每个文件就绪后立即执行，完成后在 `files[].result_url` 返回独立下载链接。
客户端不拼接存储路径，也不要求 SaaS 实现本地 Files API。

快速定位：[创建请求](#create-request) · [响应字段](#response-models) · [共用响应对象](#shared-response-objects) · [各接口成功响应](#success-responses) · [错误响应](#error-responses) · [输出格式与下载](#outputs-and-downloads)。

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

## 路由概览

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/tasks` | 仅接受 `application/json`，返回 `202` 和稳定 `task_id`。 |
| `PUT` 返回的 `files[].upload.url` | 每个链接上传一个文件；本地使用 `PUT /v1/tasks/{task_id}/uploads/{file_id}`。 |
| `GET /v1/tasks/{task_id}` | 返回 `200`，包含总状态、逐文件状态、尚有效的上传链接和已完成文件的下载链接。 |
| `POST /v1/tasks/{task_id}/cancel` | 返回 `200 CancelTaskResponse`，包含所有文件的终态及已经完成文件的下载链接。 |

Task ID 是父任务标识；内部逐文件执行 ID 不属于公共契约。
不提供任务记录或产物的公开 DELETE 接口；旧 DELETE 取消入口不保留别名，认证通过后返回 405。

未配置 API Key 的本地服务可省略 Authorization，四种输出格式均可用；启用服务认证后四种格式均需认证，没有额外的格式权限限制。

## 创建任务请求 {#create-request}

以下字段仅用于 `POST /v1/tasks` 的 JSON 请求体。创建请求不再接受表单、multipart 或文件字节，其他 Content-Type 返回 `415`。
业务参数必须放在 JSON 主体中，创建接口的查询参数、未知字段、非法值返回 `400 invalid_request`。

### 顶层请求字段

| JSON 字段 | 默认值 | 说明 |
|---|---|---|
| `files` | 必填 | 1–100 个文件组成的 JSON 数组，不接受 JSON 字符串。 |
| `tier` | 服务默认值 | 可选，仅对 PDF/图片生效；`flash/basic/standard/advanced`，规则见下文。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 全局 PDF 页范围，支持 `1-5`、`r1`、`all` 等现有语法。 |
| `output_formats` | `["default"]` | 非空列表，值为 `default/docx/html/latex` 中一个或多个；重复值按首次出现去重。 |

ModelJson 仅用于内部解析与后处理，不属于 HTTP 或 SDK 结果。请求携带已删除的 `include_model_output` 字段（包括值为 `false`）返回 400。

### tier：默认选择与文件作用范围 {#tier-selection}

省略 `tier` 或传入 JSON `null` 时，服务端从支持的档位中按 **standard → basic → flash → advanced** 选择第一个；仅服务只开放 advanced 时默认选择 advanced。
仅开放 flash 的服务可以直接接收未指定 tier 的 PDF/图片；仅开放 advanced 时，未指定 tier 默认使用 `advanced`。

请求包含 PDF/图片且显式提供 tier 时，服务支持该档位就原样使用；不支持则在创建阶段返回 `400 invalid_request`，`error.param="tier"`，整个请求不接单、不分配上传链接，也不自动升档或降档。
空字符串和未知枚举始终是参数错误，包括纯原生文档请求。

Office、HTML 等已支持的原生文档固定使用 flash，不受请求 tier 影响。纯原生文档批次忽略合法的显式 tier，不要求服务开放该 PDF/图片档位；原生解析是所有正常可用服务的必备能力，不从 `/v1/tiers` 判断。
例如，部署只对 PDF/图片开放 standard 时：

| 输入 | 请求 tier | 执行结果 |
|---|---|---|
| PDF + DOCX | 省略或 `null` | PDF 使用 standard，DOCX 使用 flash。 |
| PDF + DOCX | `standard` | PDF 使用 standard，DOCX 使用 flash。 |
| PDF + DOCX | `basic` 或 `advanced` | 创建时返回 400。 |
| 只有 DOCX | `basic` 或 `advanced` | DOCX 使用 flash。 |

可在一个请求中混合提交文件，客户端无需预先拆分档位：

```json
{
  "files": [
    {"file_name": "report.pdf", "bytes": 12000},
    {"file_name": "contract.docx", "bytes": 8000}
  ]
}
```

服务在接单时确定并保存选档，等待上传、排队和后续转发不重新选择默认值。Router 按每个文件的有效档位选择 worker，并显式传给下游；原档位后来不可用时按文件报告失败，不自动改档。
实际执行档位始终读取成功文件的 `files[].parse.tier`，同一批次允许不同档位。
该规则也适用于 HTTP SDK 和 file_parse 提交的单文件 Tasks；直接本地解析和 Doclib 保持各自的选档策略。

### files[]：输入清单

请求体的 `files` 数组中，每个条目包含以下字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file_name` | 是 | 逻辑文件名，保留扩展名；不同条目允许同名。 |
| `url` | 否 | HTTP(S) 文件下载地址，受部署来源策略约束；省略或 `null` 表示客户端上传。 |
| `bytes` | 条件 | 上传项必填；URL 项可省略或为 `null`。提供时必须是严格正整数。 |
| `sha256sum` | 否 | 64 位小写十六进制 SHA-256；用于完整性校验，上传项还用于源文件缓存预检。 |
| `page_range` | 否 | 逐文件 PDF 页范围，未提供时继承全局值；非 PDF 仍只支持整本解析。 |

### 文件 ID 与输入关联

服务端为每个输入分配唯一的 `file_id`，请求不得包含 `client_file_id` 或 `file_id`，即使值为 `null` 也返回 `400 invalid_request`。
创建响应的 `files` 顺序与输入一致，客户端先按原始顺序绑定 ID，随后使用 ID 关联上传、查询、取消和结果。同名输入或相同内容仍获得不同 ID。
该 ID 从创建到结果下载保持不变；在本地和 Router 中，产物注册后的 Files ID 与它一致。产物生成前只预留 ID，对 Files 元数据和内容接口的访问返回 `404 file_not_found`。
源缓存引用不属于公共 ID；跨任务可共享源字节，但每次任务独立生成产物。Router 内部映射各层生成的 ID，对客户端始终保留本层 ID。

### 创建请求示例

先从真实文件生成 JSON 请求（文件 ID 统一由服务端生成）：

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

对应的完整 JSON 请求和创建回执见 [创建任务成功响应](#create-task)。

## 响应模型 {#response-models}

创建、详情、取消和上传确认使用独立模型。成功响应直接返回对象，没有 `data` 或 `task` 包装。GET 新增详情字段不会自动进入其他响应。

| 接口 | 模型 | 顶层字段 |
|---|---|---|
| [POST 创建](#create-response) | `CreateTaskResponse` | `task_id/status/files`。 |
| [GET 查询](#get-response) | `TaskDetailResponse` | `task_id/status/created_at/output_formats/progress/files`，以及有值的 `started_at/finished_at/error`。 |
| [POST 取消](#cancel-response) | `CancelTaskResponse` | `task_id/status/finished_at/files`。 |
| [本地 PUT 上传确认](#upload-response) | `UploadFileResponse` | `task_id/file_id/status`，其中 status 是本次上传文件的状态。 |

这些成功响应递归省略值为 `null` 的可选字段，保留 `0/false/[]/{}`，例如 `progress.failed=0` 和 `upload.headers={}`。OpenAPI 将可省略字段标为非必填。
每次响应是该接口当前的完整快照；缺少可选字段表示当前没有该值，不能沿用上一次响应的上传地址或结果信息。客户端应忽略未来新增的字段。
共享的非成功错误响应，以及 Files、file_parse 自身的响应，保持各自现有空值规则。

JSON 正文使用 UTF-8。时间字符串使用 ISO-8601 UTC；上传链接 `expires_at` 使用 Unix 秒级整数。
请求中的 `tier/ocr_mode/page_range` 和逐文件 `sha256sum/page_range` 不作响应回显；实际档位位于成功文件的 `parse.tier`。

### POST 创建响应 {#create-response}

`POST /v1/tasks` 返回 **202 Accepted** 和 `CreateTaskResponse`。202 表示接单，具体状态以响应为准。完整示例见 [创建任务成功响应](#create-task)。

#### 创建回执的顶层字段

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `task_id` | string | 是 | 稳定的父任务 ID，用于后续查询、上传和取消。 |
| `status` | string | 是 | 当前任务总状态：`waiting-file/pending/running/completed/partial/failed/canceled`。 |
| `files` | array of object | 是 | 每个输入的文件 ID、当前状态和上传动作，顺序与创建请求一致。 |

#### 创建回执的 files[]

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `file_id` | string | 是 | 服务端分配的公共文件 ID，创建后不变。 |
| `file_name` | string | 是 | 输入逻辑文件名，同名文件通过 ID 区分。 |
| `status` | string | 是 | 当前文件状态：`waiting-file/pending/running/completed/failed/canceled`，没有 `partial`。 |
| `upload` | object | 否 | 仅在当前需要客户端上传时返回，见 [待上传描述](#upload-link)。 |

创建回执不返回文件大小、解析统计或结果链接，即使文件已经完成也通过 GET 获取详情。

### GET 查询响应 {#get-response}

`GET /v1/tasks/{task_id}` 返回 **200 OK** 和 `TaskDetailResponse`，包括任务失败、取消等终态。完整示例见 [查询任务成功响应](#get-task)。

客户端保留配置 API 基址的代理前缀，并追加 `/v1/tasks/{URL 编码后的 task_id}` 查询；Tasks 不返回 `status_url`。

#### 查询详情的顶层字段

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `task_id` | string | 是 | 稳定的父任务 ID，不代表内部逐文件执行 ID。 |
| `status` | string | 是 | `waiting-file/pending/running/completed/partial/failed/canceled`。 |
| `created_at` | string | 是 | 父任务创建时间。 |
| `started_at` | string | 否 | 已知最早的逐文件执行开始时间，无记录时省略。 |
| `finished_at` | string | 否 | 父任务进入终态的时间，非终态时省略。 |
| `output_formats` | array of string | 是 | 本次输出格式，按首次出现去重。 |
| `progress` | object | 是 | 逐文件进度计数，字段见本节下表。 |
| `files` | array of object | 是 | 全部文件的当前详情，始终保持创建请求顺序。 |
| `error` | object | 否 | 有父任务级错误时返回 [ErrorDetail](#error-detail)，逐文件失败见 `files[].error`。 |

#### 查询详情的 progress

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `completed` | integer | 是 | 解析及全部所选格式均成功的文件数。 |
| `failed` | integer | 是 | 失败文件数，包含上传过期、下载、解析和导出失败。 |
| `total` | integer | 是 | 固定为创建请求中的输入文件数。 |

三个计数均保留零值。取消文件不计入 `completed` 或 `failed`。

#### 查询详情的 files[]

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `file_id` | string | 是 | 服务端分配的公共文件 ID，创建后不变。 |
| `file_name` | string | 是 | 输入逻辑文件名，同名文件通过 ID 区分。 |
| `status` | string | 是 | `waiting-file/pending/running/completed/failed/canceled`，没有 partial。 |
| `bytes` | integer | 否 | 输入大小，URL 下载前未知时省略，校验后补齐实际字节数。 |
| `url` | string | 否 | URL 输入的原始下载地址，上传输入省略。 |
| `upload` | object | 否 | 当前可执行的 [上传动作](#upload-link)，服务端准备、输入就绪或终结时省略。 |
| `parse` | object | 否 | 文件成功完成后返回的实际档位和 [解析统计](#parse-info)。 |
| `result_url` | string | 否 | 文件 `completed` 时返回其完整产物的下载地址。 |
| `error` | object | 否 | 文件错误详情，结构为 [ErrorDetail](#error-detail)，无错误时省略。 |

`waiting-file` 包括等待上传、接收、对象存储确认、URL 下载及完整性校验；`pending` 表示输入已就绪并等待解析资源；`running` 包含解析、导出和结果保存。
只在 `waiting-file` 且存在 `upload` 时上传；没有 upload 时只轮询，不能据此认为输入已经就绪。成功 PUT 后即使仍为 waiting-file 也不重复传输。
只有上传失败、查询确认仍需上传且链接有效时才有界重试。上传过期使用 `failed` 和 `error.code=upload_expired`。

### POST 取消响应 {#cancel-response}

`POST /v1/tasks/{task_id}/cancel` 无需请求体，返回 **200 OK** 和 `CancelTaskResponse`。已成功或失败的文件保留终态，其余文件转为 `canceled`；已终结的父任务再次取消返回 `409 task_already_terminal`。完整示例见 [取消任务成功响应](#cancel-task)。

#### 取消回执的顶层字段

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `task_id` | string | 是 | 被取消的父任务 ID。 |
| `status` | string | 是 | 固定为 `canceled`。 |
| `finished_at` | string | 是 | 父任务进入取消终态的时间。 |
| `files` | array of object | 是 | 全部文件的终态，顺序与创建请求一致。 |

#### 取消回执的 files[]

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `file_id` | string | 是 | 服务端分配的公共文件 ID，与创建回执一致。 |
| `file_name` | string | 是 | 输入逻辑文件名，同名文件通过 ID 区分。 |
| `status` | string | 是 | 文件终态，只会是 `completed/failed/canceled`。 |
| `parse` | object | 否 | 成功文件附带实际档位和 [解析统计](#parse-info)。 |
| `result_url` | string | 否 | 成功文件附带完整产物的下载地址。 |
| `error` | object | 否 | 失败文件附带 [ErrorDetail](#error-detail)，无错误时省略。 |

取消回执不包含上传动作、输入大小或来源 URL。
取消操作保留任务和已完成产物，返回后可直接使用成功文件的下载链接。底层不可中断计算和资源清理可能仍在收尾，迟到结果不会覆盖已公布的取消终态。

### PUT 上传确认响应 {#upload-response}

向返回的 `files[].upload.url` 发送原始字节，使用 `files[].upload.headers`，签名 URL 必须原样使用。
本地 API Server 和 Router 的 `PUT /v1/tasks/{task_id}/uploads/{file_id}` 返回 **200 OK** 和 `UploadFileResponse`。完整示例见 [本地上传确认成功响应](#confirm-upload)。

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `task_id` | string | 是 | 本次上传所属的父任务 ID。 |
| `file_id` | string | 是 | 本次上传文件的公共 ID，与创建回执一致。 |
| `status` | string | 是 | 本次文件的当前状态：`waiting-file/pending/running/completed/failed/canceled`，不是父任务总状态。 |

本次文件完成保存及校验后进入 `pending/running`；响应不包含 `files` 或其他任务详情，最新详情通过 GET 获取。
有效期内重复 PUT 已确认文件仍返回当前文件回执，且不读取或覆盖源字节；上传窗口到期后返回 `410 upload_expired`。
SaaS 的上传目标可能是独立对象存储：通用客户端接受成功的 2xx 响应，不要求其返回上述本地 PUT 回执，随后查询 Tasks 确认就绪。

## 共用响应对象 {#shared-response-objects}

以下对象只在各接口字段表标注的位置出现，不代表每种响应都包含这些字段。

### files[].upload：待上传描述 {#upload-link}

适用于 **POST 创建、GET 查询** 的 `files[].upload`。
固定使用 PUT 上传原始字节；对象不重复携带 file_id，也不返回 upload_method。查询仍提供当前有效的上传入口。

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `url` | string | 是 | 完整上传地址，签名 URL 原样使用。 |
| `headers` | object | 是 | 字符串请求头映射，允许为空对象。 |
| `expires_at` | integer | 是 | 固定的 Unix 秒级截止时间，查询不续期。 |

### files[].parse：解析统计 {#parse-info}

适用于 **GET 查询、POST 取消** 的 `files[].parse`。
成功后的实际 tier 来自结果元数据，不以请求 tier 代替。PDF 和图片沿用多档位规则，HTML、Office 等原生文档使用 flash，同一批次可出现不同档位。

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `tier` | string | 是 | 实际执行档位：flash/basic/standard/advanced。 |
| `duration_ms` | integer | 否 | 已知的毫秒级耗时，包含源读取/下载、校验、解析和导出，不含等待上传、准备名额及解析队列时间；未知时省略。 |
| `parser_version` | string | 否 | 已知的执行 worker 版本，不保证与 Router 版本一致；未知时省略。 |

### ErrorDetail：父任务或逐文件错误 {#error-detail}

成功快照中适用于 **GET 查询** 的顶层 `error` 和 `files[].error`，以及 **POST 取消** 的 `files[].error`。

| 字段 | JSON 类型 | 必带 | 含义 |
|---|---|---|---|
| `message` | string | 是 | 错误说明。 |
| `code` | string | 是 | 稳定的非空错误码，供程序判断错误原因。 |
| `param` | string 或 null | 否 | 关联参数路径，例如 `files.0.page_range`；有值时返回，无值时省略。接收端允许缺省或 `null`。 |

缺少顶层 `error` 不代表所有文件成功，`partial/failed` 的原因可能只在逐文件 `error` 中。
HTTP 非成功响应也使用相同的 `ErrorDetail`，外层为 `{"error": ...}`。所有错误均不包含 `type`，`code/message` 必带，`param` 无值时省略；完整示例及错误码见 [各接口错误响应](#error-responses)。

## 各接口成功响应 {#success-responses}

以下完整示例使用同一批两个 HTML 输入，ID、地址、时间和耗时均为示意值。

### POST /v1/tasks：创建任务 {#create-task}

返回 **202 Accepted** 和 `CreateTaskResponse`。202 表示接单，具体状态以响应为准。请求：

```json
{
  "files": [
    {
      "file_name": "report.html",
      "bytes": 64
    },
    {
      "file_name": "notes.html",
      "bytes": 64
    }
  ],
  "tier": "flash",
  "output_formats": [
    "default"
  ]
}
```

创建回执：

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "waiting-file",
  "files": [
    {
      "file_id": "file-111111111111111111111111",
      "file_name": "report.html",
      "status": "waiting-file",
      "upload": {
        "url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/file-111111111111111111111111",
        "headers": {
          "Content-Type": "application/octet-stream"
        },
        "expires_at": 1792731600
      }
    },
    {
      "file_id": "file-222222222222222222222222",
      "file_name": "notes.html",
      "status": "waiting-file",
      "upload": {
        "url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/file-222222222222222222222222",
        "headers": {
          "Content-Type": "application/octet-stream"
        },
        "expires_at": 1792731600
      }
    }
  ]
}
```

### GET /v1/tasks/{task_id}：查询任务 {#get-task}

返回 **200 OK** 和 `TaskDetailResponse`，包括任务失败、取消等终态。以下第一个文件完成、第二个仍待上传，父任务仍为 waiting-file，finished_at 省略：

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "waiting-file",
  "created_at": "2026-10-23T04:00:00Z",
  "started_at": "2026-10-23T04:00:01Z",
  "output_formats": [
    "default"
  ],
  "progress": {
    "completed": 1,
    "failed": 0,
    "total": 2
  },
  "files": [
    {
      "file_id": "file-111111111111111111111111",
      "file_name": "report.html",
      "bytes": 64,
      "status": "completed",
      "parse": {
        "tier": "flash",
        "duration_ms": 125,
        "parser_version": "4.0.12"
      },
      "result_url": "http://127.0.0.1:8000/v1/files/file-111111111111111111111111/content"
    },
    {
      "file_id": "file-222222222222222222222222",
      "file_name": "notes.html",
      "status": "waiting-file",
      "upload": {
        "url": "http://127.0.0.1:8000/v1/tasks/job_0123456789abcdef01234567/uploads/file-222222222222222222222222",
        "headers": {
          "Content-Type": "application/octet-stream"
        },
        "expires_at": 1792731600
      },
      "bytes": 64
    }
  ]
}
```

### POST /v1/tasks/{task_id}/cancel：取消任务 {#cancel-task}

无需请求体。例如：

```bash
curl --fail-with-body -X POST "$MINERU_API_URL/v1/tasks/$TASK_ID/cancel" \
  -H "Authorization: Bearer $MINERU_API_KEY"
```

返回 **200 OK** 和 `CancelTaskResponse`。已成功或失败的文件保留终态，其余文件转为 canceled；已终结的父任务再次取消返回 `409 task_already_terminal`。

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "status": "canceled",
  "finished_at": "2026-10-23T04:00:03Z",
  "files": [
    {
      "file_id": "file-111111111111111111111111",
      "file_name": "report.html",
      "status": "completed",
      "parse": {
        "tier": "flash",
        "duration_ms": 125,
        "parser_version": "4.0.12"
      },
      "result_url": "http://127.0.0.1:8000/v1/files/file-111111111111111111111111/content"
    },
    {
      "file_id": "file-222222222222222222222222",
      "file_name": "notes.html",
      "status": "canceled"
    }
  ]
}
```

### PUT /v1/tasks/{task_id}/uploads/{file_id}：本地上传确认 {#confirm-upload}

本地 API Server 和 Router 返回 **200 OK** 和 `UploadFileResponse`，不返回整份任务详情。本次文件完成保存及校验后进入 pending/running，最新详情通过 GET 获取。

```json
{
  "task_id": "job_0123456789abcdef01234567",
  "file_id": "file-111111111111111111111111",
  "status": "pending"
}
```

有效期内重复 PUT 已确认文件仍返回当前文件回执，且不读取或覆盖源字节；上传窗口到期后返回 410。
SaaS 的上传目标可能是独立对象存储：通用客户端接受成功的 2xx 响应，不要求其返回本地 PUT 回执，随后查询 Tasks 确认就绪。

## 各接口错误响应 {#error-responses}

请求级失败返回对应非 2xx 状态码和 `Content-Type: application/json`，正文是仅包含顶层 `error` 的 JSON 对象，其值为上文的 `ErrorDetail`，不返回成功任务对象。
例如重复取消任务返回 **`409 Conflict`**：

```json
{
  "error": {
    "code": "task_already_terminal",
    "message": "Task is canceled"
  }
}
```

| 接口 | HTTP | 常见 code | 场景 |
|---|---|---|---|
| POST 创建 | 400 | `invalid_request`、`page_range_invalid` | 字段、ID、页范围等非法，包括旧字段和新旧混传。 |
| POST 创建 | 415 | `invalid_request` | 请求不是 `application/json`。 |
| POST 创建 | 503 | `quality_tier_unavailable`、`server_shutting_down` 或模型预加载错误码 | 当前部署无法接单。 |
| GET 详情 / POST cancel / 本地 PUT | 404 | `task_not_found` | task 不存在、记录已清理或不属于当前调用方。 |
| POST cancel | 409 | `task_already_terminal` | 父任务已经成功、部分成功、失败或被取消；包括取消前刷新时刚进入终态。 |
| 本地 PUT | 400 | `upload_size_mismatch`、`file_hash_mismatch`、`upload_interrupted` | 上传校验失败或传输中断；未过期时可查询后重传。 |
| 本地 PUT | 404 | `upload_not_found` | task 中不存在该 `file_id`。 |
| 本地 PUT | 409 | `task_already_terminal`、`file_already_terminal` | 任务已取消或文件已不接受上传。 |
| 本地 PUT | 410 | `upload_expired` | 固定上传窗口已过期。 |
| 所有受保护接口 | 401 | `invalid_api_key` | 服务启用认证，但调用方未提供有效凭据。 |

Router 或网关还可能返回远端依赖相关的 5xx 错误；错误格式及部署扩展见 [响应与错误](../../next/api/responses.md)。
本地 API、Router 和 SaaS 使用同一错误结构。`param` 只用于定位具体参数；没有对应参数时省略，客户端也接受 `null`。
异步文件下载、解析或导出失败通常通过查询接口的 **`200` 快照和 `files[].error`** 表达；不要把 HTTP `200` 等同于任务成功，也不要仅凭顶层 `error` 判断全部文件是否成功。

## URL、缓存与独立上传

URL 输入同样先创建 JSON 任务：

```json
{
  "files": [
    {"file_name": "document.pdf", "url": "https://example.com/download?signature=example"},
    {"file_name": "other.pdf", "bytes": 12345}
  ],
  "output_formats": ["html"]
}
```

- URL 项保持 `waiting-file` 和 `upload 省略`，由服务端下载和校验；确认有效后进入 `pending`。不生成客户端上传链接，不接受 PUT，不等待其他文件上传。
- URL 保留原始签名地址，使用逻辑 `file_name` 识别类型；不要求 HEAD 或 `Content-Length`。下载后回填实际 `bytes`；请求提供的大小/hash 必须匹配，但响应不回显 hash。URL 每次下载，不用 hash 缓存跳过下载。
- 上传项 hash 命中时直接进入 `pending`，不返回上传链接；未命中或无 hash 时返回该文件的链接。上传后仍计算 hash、校验大小并对源字节去重。
- 缓存只复用源文件，不复用不匹配的解析结果；每个文件使用本次解析参数独立执行。
- 每个上传项校验完成即排队，不需要客户端发送完成确认。等待客户端上传不预占 worker；worker 上 URL 下载与校验使用受控准备阶段，不占解析并发。

### 一小时上传窗口

所有上传链接的 `expires_at` 固定为创建时起 3600 秒，采用 Unix 秒。查询不会延长或换发链接。
到期后新 PUT 返回 `410 upload_expired`，包括对已经确认文件的重复 PUT。

截止前已受理的上传可以继续传完并校验，该文件仍处于 `waiting-file`，不会被过期扫描中止。
尚未受理上传的文件到期后变为 `status=failed`，逐文件错误码为 `upload_expired`。URL 下载不受客户端上传截止时间影响，沿用来源下载超时。
在途上传中断且已经过期时，同样进入失败终态；补交须创建新任务。
上传截止时间不限制已上传文件的排队、解析或导出时间。

有效期内，一个文件中断或校验失败可以从头重传，其他文件的状态不受影响。
重复 PUT 已确认文件不读取或覆盖源字节，并发请求也只提交一次解析。
上传响应丢失时先查询：进入 `pending/running/completed` 后跳过上传；仍为 `waiting-file` 且 upload 省略 时只轮询；有有效上传目标时才有限重试。
不支持单文件分块断点续传。

## 总状态与逐文件状态

状态响应只有元数据，不内联 Markdown、JSON 或结果文件字节。
`files` 按原清单顺序返回，使用 `file_id` 对应输入，不依赖文件名唯一性。

各响应字段、类型及省略规则见 [响应模型](#response-models)，下面说明状态聚合规则。

Task 顶层没有 `result_url`；`files[]` 不暴露底层 `output_files`。
已完成文件可以在父任务仍为 `running/pending/waiting-file` 时下载。

总状态优先级如下：

| 条件 | `status` |
|---|---|
| 存在运行中的文件，包括导出阶段 | `running` |
| 没有运行文件，存在排队文件 | `pending` |
| 剩余未完成文件全部处于文件准备阶段 | `waiting-file` |
| 所有文件成功 | `completed` |
| 所有文件终结，成功与失败并存 | `partial` |
| 所有文件终结，无成功文件 | `failed` |
| 用户取消任务 | `canceled` |

因此，第一个文件解析完成而第二个文件尚未上传时，总状态可以由 `running` 回到 `waiting-file`，第一个文件的下载链接仍然有效。
取消只停止未完成的文件，已经成功或失败的文件保持原状态；重复取消终态任务返回 `409`。

## 输出格式与下载 {#outputs-and-downloads}

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
`default` 不包含 `model_output.json`；不生成跨输入文件的总 ZIP 或批次 `manifest.json`。

每个输入只解析一次，基于同一结果生成所选格式。全部所选格式完成并保存后才发布该文件的 `completed` 和链接。
任一格式失败，该文件为 `failed/result_packaging_failed` 且不返回结果链接，不提供不完整产物；其他文件继续执行。
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
下载使用临时文件、验证对应格式后原子保存，成功文件保存在 `RESULT_DIR/<file_id>/<输入文件主名>.<扩展名>`。

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
| `TASK_ID` | 空 | 恢复已有任务，只处理仍有上传动作的文件。 |
| `MINERU_TIER`、`OCR_MODE`、`PAGE_RANGE` | 服务默认 / `auto` / 整本 | 解析参数。 |
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
Router 的源缓存使用按调用方隔离的私有引用，不等待或转发 worker 的源 Files ID；结果仍通过 Files 映射下载。私有源引用在父任务完成清理后开始保留期，过期独立回收，活动传输继续保护字节。
健康接口的 `features.task_output_formats` 声明 Tasks 格式，`sources` 为 `upload/url`。
Router 创建任务前使用候选 worker 的受保护 `/v1/usage` 校验认证，不调用列表接口，不预占解析名额。
不声明新 Tasks 能力的旧 worker 不参与 Tasks 调度，更新时应先更新 worker，再更新 Router 和客户端。
Router 使用有界、可续租的内部结果租约保护 worker 产物，上传窗口不受结果租约影响；Router 丢失后租约会自行结束。已完成任务按 worker 并行续租，每轮网络等待最多 10 秒，避免失联 worker 阻塞其他 worker 的结果保留。

### 动态 upstream 管理与级联

Router 可以在运行期间注册、查询和安全注销远程 upstream。upstream 可以是独立的 V1 API Server，也可以是另一层 Router；操作只影响接收请求的这一层，不会注销下层的 worker 或关闭远程进程。本地托管 worker 不在管理接口范围内。

配置独立管理 Key 后启用接口，未配置时这些路径返回 404。`--admin-api-key` 优先于 `MINERU_ROUTER_ADMIN_API_KEY`，也支持 `--reload`。管理 Key 不等于解析 API Key，不会被转发给下层；解析请求继续使用原有认证。

```bash
# 空节点池也可启动；注册健康节点前 /v1/health 返回 503，管理接口仍可访问。
export MINERU_ROUTER_ADMIN_API_KEY='replace-with-a-management-secret'
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus none

# 在另一个终端设置相同管理 Key 后操作。
curl -X POST http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"url":"http://worker-a:8000"}'
curl http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
# 使用注册或列表响应中的 worker_id。
curl -i -X DELETE http://127.0.0.1:8002/v1/router/workers/remote-1 \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
```

| 接口 | 响应与行为 |
|---|---|
| `GET /v1/router/workers` | `200`，返回 `{"object":"list","data":[...]}`，包含启动参数和运行期注册的远程节点。 |
| `POST /v1/router/workers` | 请求只接受 `url`。新增返回 `201`，同一规范化 URL 重复注册返回 `200`；保留原 ID 和已有引用，draining 节点恢复接单。 |
| `DELETE /v1/router/workers/{worker_id}` | 无引用时删除并返回 `204`；仍有引用时返回 `202` 与 draining 快照；未知或本地托管节点返回 `404 worker_not_found`。 |

节点快照包含 `worker_id`、`url`、`state`（`active/draining`）、`healthy`、`tiers`、`active_jobs` 和 `retained_routes`。注册 URL 须为 HTTP/HTTPS base URL，允许路径前缀，禁止内嵌凭证、查询参数和 fragment。非法输入返回 `400 invalid_request`，缺失或错误管理凭证返回 `401 invalid_api_key`，错误均使用 `error.code/message` 和可选 `param`。

注册后执行一次健康与能力探测，暂时离线或返回非法 tier 数据的节点仍保留并返回 `healthy=false`，后续由周期检查重试。非法能力不替换上一份合法快照，也不影响管理列表查询。只有健康的 active 节点参与新任务调度及 health/models/tiers 能力聚合；健康恢复不会撤销 draining。

注销立即停止新预占，已有预占、上传交接、查询、取消、结果续租与下载继续执行。节点必须等执行预占、请求/流式传输、执行与文件路由及租约全部释放后才会移除；`active_jobs=0` 不能单独作为关闭远程服务的依据。结果默认保留 24 小时，节点可能在这段时间内保持 draining；后台复用五分钟扫描完成回收，也可重复 DELETE 检查。`--retention-seconds 0` 下长期结果引用可以一直阻止移除，无强制删除接口。

尚未分派的文件使用剩余可用节点，没有匹配节点时按原契约返回 `quality_tier_unavailable`，不自动降档或新增等待队列。允许注销最后一个节点。列表中节点消失只表示本层已释放引用；下层可能仍在为其他调用方服务。

两层部署时，顶层直接接入下层 Router 的地址：

```bash
mineru-kit router --local-gpus none \
  --upstream-url http://router-a:8002 \
  --upstream-url http://router-b:8002
```

也可通过上述 POST 动态加入 Router C。建议各下层管理独立 worker 池并保持无环拓扑；本次不提供全局负载统计或跨 Router 任务接管。

**动态清单仅当前进程有效。** 注册和注销不会修改启动配置；重启后恢复 `--upstream-url` 中的节点，动态节点须重新注册，worker ID 不能跨重启作为稳定标识。顶层仍有单点，任一 Router 重启后的旧任务恢复不在保证范围内。

### 协议迁移

创建请求仅接受 `files` 和 `file_name`，不兼容 `manifest`、`filename`；新旧字段混传同样返回 400。Tasks 响应及 Files 详情统一使用 `file_name`。
任务和文件列表接口已移除；使用创建响应中的 `task_id` 和固定查询路由恢复任务，按 `files[].result_url` 下载。合法认证下 `GET /v1/tasks` 返回 405，`GET /v1/files` 返回 404。
这是同一 `/v1` 下的破坏性变更：先停收请求并排空在途任务，再协调更新 worker、Router、SDK、file_parse 包装服务及其他客户端；SaaS 由部署方同步对齐。

本次协议不保留旧表单、直传或 `markdown/middle_json/structured_content/zip` Tasks 格式别名。
客户端应改用 JSON、`files[].result_url` 及逐文件保存方式。
API Server 和 Router 不提供 `/v1/file_parse`、`/v1/tasks/{task_id}/result`。
需要同步单文件调用时，独立启动 [file_parse 包装服务](file_parse.md)，由它调用本页的共同 Tasks 契约。
SaaS Go 服务和对象存储配置由部署方按此共同契约实现，不属于本地 Python 服务代码。

## 源文件缓存

SDK 即使连接同机也计算 hash 并提交清单。缓存有效且已完成上传的同 hash 后续免传；并发首次上传允许重复传输但源字节去重。默认缓存保留 24 小时、每 5 分钟扫描，沿用 retention 配置；活动任务与传输引用保护字节。缓存过期或服务重启后允许重新上传。
