# file-parse API：单文件同步解析

本文描述 **4.1.0** 接口。`mineru-kit file-parse-server` 是独立的 HTTP 包装服务：接收一个文件，通过 [Tasks](tasks.md) 创建、上传、等待，成功后直接返回 ZIP、DOCX 或 HTML 附件。它可连接本地 API、Router 或兼容 SaaS，不加载解析模型。

## 启动

先启动上游解析服务，再在另一个终端启动包装服务：

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier auto
```

```bash
mineru-kit file-parse-server --api-url http://127.0.0.1:8000
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--api-url` | `MINERU_API_URL`，否则必填 | 上游 API 基址，可含代理前缀，不要追加 `/v1/tasks`。 |
| `--api-key` | `MINERU_API_KEY`，否则为空 | 仅用于上游的 Bearer Token。 |
| `--host`、`--port` | `127.0.0.1`、`8003` | 包装服务监听地址。 |
| `--state-dir` | `${MINERU_HOME:-~/.mineru}/file-parse` | 恢复数据库和传输临时目录。 |
| `--retention-seconds` | `86400` | 首次观察终态后的记录保留秒数；0 禁用回收。 |

包装入口本身没有 API Key 认证，面向本机或可信网络；需要对外提供访问时在反向代理设置认证。入站 Authorization 不会转发给上游，请求也不能覆盖上游地址或凭据。

包装服务有独立的 `/docs`、`/openapi.json`。以下路由只在该服务上提供，不注册到主 API Server 或 Router。

## 请求字段

`POST /v1/file_parse` 仅接受 `multipart/form-data`。不接受查询参数、未知字段、多个文件或重复标量字段；`output_formats` 可以重复。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `file` | 必填 | 一个非空文件，保留文件扩展名。 |
| `tier` | 上游默认 | 可省略；`flash/basic/standard/advanced`，仅对 PDF/图片生效。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | PDF 支持 `1-5`、`r1`、`all` 等；格式支持范围由上游校验。 |
| `output_formats` | `default` | `default/docx/html/latex`，重复字段表达列表，按首次出现去重。 |
| `wait_timeout` | `3600` | 整数秒，范围 0–3600。 |
| `request_id` | 自动生成 | 1–64 位字母、数字、下划线或短横线；预先指定并保存以便断连后恢复。 |

省略 tier 时由上游按 Standard → Basic → Flash → Advanced 选择 PDF/图片档位；仅开放 Advanced 时默认选择 Advanced。显式不支持的 PDF/图片档位返回错误，不自动改档。现有原生文档固定使用 Flash，即使传入上游未开放的合法档位。表单空字符串或字符串 `"null"` 不等于省略，均为非法 tier。

`output_formats` 不接受 JSON 字符串或逗号分隔值；多格式使用多个同名表单字段。`include_model_output` 已移除，传入即返回 400。

## 上传与附件响应

以下为 Bash/curl 示例；Windows PowerShell 可使用 `curl.exe` 并调整续行语法。服务收到真实文件后自动计算大小和 hash。

```bash
# response.body 可能是附件或 JSON，必须同时检查状态码和响应头。
curl --max-time 7200 -D response.headers -o response.body \
  http://127.0.0.1:8003/v1/file_parse \
  -F 'file=@report.pdf' \
  -F 'request_id=report-example' \
  -F 'tier=standard' \
  -F 'output_formats=default' -F 'output_formats=docx'
```

| 接口 | HTTP | 正文 |
|---|---|---|
| POST，同步等待成功 | 200 | 附件原始字节，无 JSON/Base64 外层。 |
| POST，等待到期或 `wait_timeout=0` | 202 | 完整 StatusResponse。 |
| `GET /v1/file_parse/{request_id}`，查询成功 | 200 | 完整 StatusResponse，包括已知失败、取消或提交结果未知状态。 |
| `GET /v1/file_parse/{request_id}/result`，下载成功 | 200 | 与同步 POST 相同的附件。 |

以输入 `report.pdf` 为例：

| output_formats | 文件名 | Content-Type / 内容 |
|---|---|---|
| `default` | `report.zip` | `application/zip`；Markdown、Middle JSON、Structured Content、images/。 |
| `docx` | `report.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document`；内嵌图片。 |
| `html` | `report.html` | `text/html; charset=utf-8`；内嵌图片。 |
| `latex` | `report.zip` | `application/zip`；document.tex、images/。 |
| 多格式 | `report.zip` | 各格式平铺的 ZIP，内容与 [Tasks 输出](tasks.md#outputs-and-downloads)一致。 |

成功附件包含 `Content-Disposition`、`X-Request-ID` 和 `X-Task-ID`。202 包含 `Location`（状态查询 URL）、`Retry-After: 2`、`X-Request-ID`；已确认上游任务时还包含 `X-Task-ID`。状态查询与错误响应不保证附加这些头，恢复信息应从 JSON 正文读取。

## 状态查询

StatusResponse 的七个字段始终出现，可空字段使用 null，不采用 Tasks 的省略规则：

| 字段 | 类型 | 含义 |
|---|---|---|
| `request_id` | string | 包装服务的请求 ID。 |
| `task_id` | string 或 null | 已确认的上游任务 ID，尚未确认时为 null。 |
| `status` | string | 单文件状态，见下表。 |
| `status_url` | string | 包装服务自身的完整查询地址。 |
| `result_url` | string 或 null | completed 时指向包装服务自身的结果接口，否则为 null。 |
| `output_formats` | string[] | 归一化输出格式列表。 |
| `error` | object 或 null | 最近保存的错误或上游文件错误，没有时为 null。 |

| 状态 | 含义 |
|---|---|
| `prepared` | 源文件和恢复记录已保存，尚未提交。 |
| `submitting` | 正在创建上游任务，尚无确认的 task_id。 |
| `waiting-file` | 上游输入准备中，不能仅因 upload 缺失而认定就绪。 |
| `pending`、`running` | 上游排队、解析或导出中。 |
| `completed` | 上游产物就绪；下载仍可能因过期或传输问题失败。 |
| `failed`、`canceled` | 已知失败或被上游取消。 |
| `submission_unknown` | 创建可能成功，但未获得有效任务 ID；不会自动重放创建。 |

包装服务只处理一个输入，没有 partial 状态，也不返回 Tasks 的 files/progress 或上游签名链接。示例 202 或 GET 状态正文：

```json
{
  "request_id": "report-example",
  "task_id": "job_example",
  "status": "running",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-example",
  "result_url": null,
  "output_formats": ["default", "docx"],
  "error": null
}
```

```bash
curl http://127.0.0.1:8003/v1/file_parse/report-example
curl --fail-with-body -D result.headers -o report.zip \
  http://127.0.0.1:8003/v1/file_parse/report-example/result
```

下载未就绪返回 `409 result_not_ready`；没有确认的上游任务时可能返回 `409 task_not_created`。下载中断可重试 GET。签名失效时包装服务查询原任务，最多尝试三次；无法恢复的过期结果不会自动重新解析。

## 等待、幂等与恢复

`wait_timeout` 在确认上游文件就绪后计时，覆盖排队、解析和导出。接收源文件、上游上传/下载/校验和结果下载不占这段预算，因此它不是整个 HTTP 请求的总超时。正常轮询间隔为 2 秒；连接、元数据请求、传输空闲超时分别为 10、30、600 秒。零等待可用于先获得状态查询入口。

完整接收并持久化源文件后，服务才接管请求；此前断连不会创建任务。接管后客户端断连、等待超时或包装服务关闭不会自动取消上游任务。

- 同一 request_id、文件内容、逻辑文件名及归一化解析参数复用原任务；输入不同返回 `409 request_id_conflict`。wait_timeout 不参与身份比较。
- SQLite 恢复记录按上游地址和认证隔离，不保存 API Key。一个状态目录仅允许一个包装服务进程。重启继续已知任务及仍有效的待上传操作。
- 创建回执中的 task_id/file_id 先持久化，再查询详情；首次查询失败或重启不会再次创建任务。创建结果未知时保留 submission_unknown，不盲目重试。
- 源文件在上游确认就绪或终结失败后清理；附件仅临时下载，发送完成或中断即删除。恢复记录不延长上游上传链接或结果有效期；上游任务消失后不会重建。
- 协议升级的新空间为 `tasks-v5-`；旧空间不加载、不迁移，也不自动删除。详见 [4.1 迁移说明](../reference/migration_4_1.md)。

包装服务没有独立取消接口。持有已确认的 task_id 且有上游权限时，向**配置的上游 API 基址**发送 `POST /v1/tasks/{task_id}/cancel`，然后查询包装服务观察 canceled；不要使用包装基址或把 request_id 当作 task_id。未预先保存 ID 且未收到响应时，调用者可能无法找回任务。

## 错误响应

包装层直接产生的 error 包含 `code: string`、`message: string`、`upstream_status: integer|null`。没有恢复记录时仅返回 error：

```json
{
  "error": {
    "code": "unsupported_media_type",
    "message": "Expected multipart/form-data",
    "upstream_status": null
  }
}
```

有记录时返回完整七字段 StatusResponse，用本次操作的错误覆盖 error；例如下载未就绪可返回 HTTP 409，同时 status 仍为 running。查询成功时的持久化 error 可能额外包含 http_status，或保留上游的 code/message 和可选 param。历史恢复记录可能仍含旧 type 或其他字段；客户端应容忍扩展字段，这不代表新 Tasks 协议接受旧字段。

| HTTP | 常见 code | 含义 |
|---|---|---|
| 400 | `invalid_request`、`upload_interrupted` | 参数、多文件、空文件或接受前断连。 |
| 404 | `request_not_found` | 记录不存在或已清理。 |
| 408 | `upload_receive_timeout` | 接收入站请求体时传输空闲超时。 |
| 409 | `request_id_conflict`、`result_not_ready`、`task_not_created` 或取消错误码 | 请求冲突、结果未就绪、尚无任务或上游取消。 |
| 410 | `upload_expired` | 上游上传窗口已过期。 |
| 415 | `unsupported_media_type` | 不是 multipart/form-data。 |
| 500 | `storage_error` | 本地存储失败。 |
| 502 | `submission_unknown`、`upstream_unreachable`、`upstream_protocol_error`、`invalid_result` 或上游错误码 | 上游请求、传输、解析或结果校验失败。 |

上游认证失败通过包装层 502 返回，`upstream_status=401`；明确的档位拒单也返回 502，并保留上游 400/503。它们是 failed，不是 submission_unknown。创建阶段其他 5xx、408、网络异常或畸形响应可能无法判断是否接单，按 submission_unknown 处理；符合共享错误结构的 `503 quality_tier_unavailable` 是明确拒单。两种情况都不会自动重放创建，相同 ID 沿用已有记录。

## Python 示例客户端

仓库脚本依赖 httpx，提前打印请求 ID，成功时原子保存附件；202 时打印恢复信息并以 124 退出。从仓库根目录执行：

```bash
python scripts/http_file_parse_example.py report.pdf --output report.docx \
  --base-url http://127.0.0.1:8003 --output-format docx --request-id report-docx

python scripts/http_file_parse_example.py report.pdf --wait-timeout 0 \
  --base-url http://127.0.0.1:8003 --request-id report-background
```
