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
| `tier` | 上游默认 | `flash/basic/standard/advanced`。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 与 Tasks 一致，PDF 支持 `1-5`、`r1`、`all`；格式支持范围由上游校验。 |
| `output_formats` | `default` | `default/docx/html/latex`；重复同名字段表达列表，重复值按首次出现去重。 |
| `include_model_output` | `false` | 附加可用原始模型输出；开启时 output_formats 必须包含 default，参与 request_id 的解析身份。 |
| `wait_timeout` | `3600` | 整数秒，范围 `0–3600`。 |
| `request_id` | 自动生成 | 1–64 个字母、数字、下划线或短横线。提前指定并保存它，可在连接中断后查询。 |

```bash
# 成功时 response.body 是附件；超时或错误时是 JSON，请同时检查 response.headers。
curl --max-time 7200 -D response.headers -o response.body \
  http://127.0.0.1:8003/v1/file_parse \
  -F 'file=@report.pdf' \
  -F 'request_id=report-20261022-001' \
  -F 'tier=flash' -F 'ocr_mode=txt' \
  -F 'output_formats=default' -F 'output_formats=docx'
```

`output_formats` 的列表语义与 [Tasks](tasks.md) 一致；multipart 中不使用 JSON 字符串或逗号分隔字符串。

| 选择 | HTTP 成功附件 |
|---|---|
| 单选 `default` | ZIP，包含 Markdown、中间协议、Structured Content、外置图片。 |
| 单选 `docx` | 图片内嵌的 DOCX。 |
| 单选 `html` | 图片内嵌的 HTML。 |
| 单选 `latex` | ZIP，包含 TeX 和外置图片。 |
| 多个不同格式 | 上游生成的单个 ZIP。 |

成功返回 `200`、正确的 Content-Type 和 `Content-Disposition: attachment`，文件名为 `<输入文件主名>.<扩展名>`，并携带 `X-Request-ID`、`X-Task-ID`。包装服务原样返回验证过的上游产物，不重新渲染或打包。

每次调用在确认上游文件就绪后开始计时，等待包含排队、解析与导出。接收文件、上游上传和结果下载不占此预算，独立使用传输空闲超时。缓存命中跳过上传。

- 正常轮询间隔为 2 秒。
- 连接、元数据请求、传输空闲超时分别为 10、30、600 秒。
- `wait_timeout=0` 在文件就绪后直接返回 `202` 查询信息；即使已经完成也不下载附件。
- 等待到期返回 `202` JSON 和 `Location: <status_url>`，上游继续执行。
- HTTP 客户端和反向代理的读取超时应覆盖上传、同步等待和结果传输；例如客户端配置 7200 秒，代理按业务文件大小配置相应超时。

## 超时与断连后查询

```json
{
  "request_id": "report-20261022-001",
  "task_id": "task_example",
  "status": "running",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-20261022-001",
  "result_url": null,
  "output_formats": ["default", "docx"],
  "error": null
}
```

`GET /v1/file_parse/{request_id}` 查询状态，完成后 `result_url` 指向包装服务自己的 `/v1/file_parse/{request_id}/result`。调用者不需要上游密钥。

```bash
curl http://127.0.0.1:8003/v1/file_parse/report-20261022-001
curl --fail-with-body -o report.zip \
  http://127.0.0.1:8003/v1/file_parse/report-20261022-001/result
```

结果未就绪时下载接口返回 `409 result_not_ready`。下载中断可以再次 GET；每次会查询最新下载链接。签名失效时重新查询原任务，最多尝试 3 次；上传失败同样先查询状态，确认已上传成功便不重传。

状态包括接收完成后的 `prepared/submitting`，以及单文件 Tasks 的 `waiting_upload/queued/running/completed/failed/canceled`。`submission_unknown` 表示创建请求可能已被上游接受，但包装服务没有获得有效任务 ID：不会自动重放创建请求，不能承诺找回这个不确定窗口内的任务。

错误响应使用 `error.code/message/upstream_status`。已接受请求的错误还包含请求 ID 和查询地址。参数错误返回 `400`，内容类型错误返回 `415`，ID 冲突或结果未就绪返回 `409`，记录不存在或已过保留期返回 `404`；上游调用或解析失败通常返回 `502` 并保留上游错误码。等待预算耗尽返回 `202`，不是解析失败。

## 去重与持久化边界

完整接收并持久化源文件后，包装服务才接受请求并托管上游上传。此后连接中断不取消上传或解析；文件尚未完整接收时中断不会创建任务。

同一 `request_id`、相同文件内容、逻辑文件名及归一化解析参数复用原任务；不同输入返回 `409 request_id_conflict`。`wait_timeout` 不参与身份比较。没有预先提供 ID 且尚未收到响应的调用者，可能无法自行找回任务。

请求映射使用独立 SQLite 数据库，按上游地址和认证隔离。数据库不保存上游 API Key。单个状态目录只允许一个包装服务进程；服务重启后恢复已知任务，尚待上传且链接有效时继续上传。上游任务记录如果因重启或过期而消失，不会自动重建。

源文件在上游确认就绪后删除，终结失败也会清理源文件。结果只临时下载，发送完成或中断即删除；重启清理遗留传输文件。恢复记录的保留期限不延长上游结果有效期，查询也不会为上游上传链接续期。结果及请求过期后应使用新的请求 ID 发起新的解析。

## Python HTTP 示例

仓库提供 `scripts/http_file_parse_example.py`，依赖 `httpx`，提前打印并保存请求 ID；成功时原子保存附件，`202` 时打印查询信息并以 `124` 退出。

```bash
python scripts/http_file_parse_example.py report.pdf --output results/report.docx \
  --output-format docx --request-id report-20261022-002

python scripts/http_file_parse_example.py report.pdf --wait-timeout 0 \
  --request-id report-20261022-003
```

后续查询和下载仍使用上面的包装接口，不需要安装 MinerU SDK 或重建 `ParseResult`。
