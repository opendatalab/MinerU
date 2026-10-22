# V1 Tasks：JSON 清单、逐文件上传与下载

`/v1/tasks` 是本地 API Server、Router 和 SaaS 的共同异步客户端契约。
客户端用 JSON 创建任务，按返回链接上传缺失文件，再轮询 `status_url`。
每个文件就绪后立即执行，完成后在 `files[].result_url` 返回独立下载链接。
客户端不拼接存储路径，也不要求 SaaS 实现本地 Files API。

## 路由与请求

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/tasks` | 仅接受 `application/json`，返回 `202` 和稳定 `task_id`。 |
| 返回的 `upload_method` 与 `upload_url` | 每个链接上传一个文件；本地使用 `PUT /v1/tasks/{task_id}/uploads/{client_file_id}`。 |
| `GET /v1/tasks/{task_id}` | 返回 `200`，包含总状态、逐文件状态、尚有效的上传链接和已完成文件的下载链接。 |
| `DELETE /v1/tasks/{task_id}` | 取消未完成文件，保留已经完成文件的下载链接。 |

创建请求不再接受表单、multipart 或文件字节，其他 Content-Type 返回 `415`。
业务参数必须放在 JSON 主体中，POST 查询参数、未知字段、非法值返回 `400 invalid_request`。
上传文件的 PUT 仍发送原始字节，使用返回的 `upload_headers`；签名 URL 必须原样使用。
Task ID 是父任务标识，不承诺同时是 `/v1/parse/jobs` 的 Job ID。

| JSON 字段 | 默认值 | 说明 |
|---|---|---|
| `manifest` | 必填 | 1–100 个文件组成的 JSON 数组，不接受 JSON 字符串。 |
| `tier` | 服务默认值 | `flash/basic/standard/advanced`，受部署能力约束。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 全局 PDF 页范围，支持 `1-5`、`r1`、`all` 等现有语法。 |
| `output_formats` | `["default"]` | 非空列表，值为 `default/docx/html/latex` 中一个或多个；重复值按首次出现去重。 |

每个清单条目包含：

| 字段 | 必填 | 说明 |
|---|---|---|
| `client_file_id` | 是 | 任务内唯一，1–64 个字母、数字、下划线或短横线。 |
| `filename` | 是 | 逻辑文件名，保留扩展名；不同条目允许同名。 |
| `url` | 否 | HTTP(S) 文件下载地址，受部署来源策略约束；省略或 `null` 表示客户端上传。 |
| `bytes` | 条件 | 上传项必填；URL 项可省略或为 `null`。提供时必须是严格正整数。 |
| `sha256sum` | 否 | 64 位小写十六进制 SHA-256；用于完整性校验，上传项还用于源文件缓存预检。 |
| `page_range` | 否 | 逐文件 PDF 页范围，未提供时继承全局值；非 PDF 仍只支持整本解析。 |

先从真实文件生成 JSON 请求：

```bash
python3 - a.pdf b.pdf > task.json <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest = []
for index, value in enumerate(sys.argv[1:], start=1):
    path = Path(value)
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    manifest.append({
        'client_file_id': f'f{index}',
        'filename': path.name,
        'bytes': path.stat().st_size,
        'sha256sum': hasher.hexdigest(),
    })
print(json.dumps({'manifest': manifest, 'tier': 'flash', 'ocr_mode': 'txt',
                  'output_formats': ['default', 'latex']}, ensure_ascii=False))
PY

curl --fail-with-body "$MINERU_API_URL/v1/tasks" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @task.json
```

未配置 API Key 的本地服务可省略 Authorization，四种输出格式均可用；启用服务认证后四种格式均需认证，没有额外的格式权限限制。

## URL、缓存与独立上传

URL 输入同样先创建 JSON 任务：

```json
{
  "manifest": [
    {"client_file_id": "remote", "filename": "document.pdf", "url": "https://example.com/download?signature=example"},
    {"client_file_id": "local", "filename": "other.pdf", "bytes": 12345}
  ],
  "output_formats": ["html"]
}
```

- URL 项的 `upload_status` 为 `not_required`，不生成上传链接、不接受 PUT；源 `file_id` 为 `null`。创建后即进入执行队列，不等待其他上传。
- URL 保留原始签名地址，使用逻辑 `filename` 识别类型；不要求 HEAD 或 `Content-Length`。下载后返回实际 `bytes` 和 `sha256sum`，提供的大小/hash 必须匹配。URL 每次下载，不用 hash 缓存跳过下载。
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

| 字段 | 说明 |
|---|---|
| `task_id`、`status`、`status_url` | 稳定父任务标识、聚合状态、查询地址。 |
| `files[].upload_status` | `pending/ready/not_required/expired`。 |
| `files[].status` | `waiting_upload/queued/running/completed/failed/canceled`，包含导出过程。 |
| `files[].result_url` | 该文件产物完整保存前为 `null`；`completed` 时为唯一下载链接。 |
| `files[].error` | 该文件的上传过期、下载、解析或导出错误。 |
| `uploads` | 仍有效的待上传项：`client_file_id/upload_url/upload_method/upload_headers/expires_at`。 |
| `progress` | 文件计数：`completed/failed/total`，`total` 固定为清单长度；取消不计为解析失败。 |
| `error` | 父任务级错误；逐文件错误应读取 `files[].error`。 |

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
| `OUTPUT_FORMATS` | `default` | 逗号分隔，转为 JSON 数组。 |
| `RESULT_DIR` | `results` | 逐文件结果目录。 |
| `MAX_POLLS`、`POLL_INTERVAL` | `300`、`1` | 查询预算和间隔秒数。 |
| `CONNECT_TIMEOUT`、`REQUEST_TIMEOUT` | `10`、`600` | 连接与单次传输超时秒数。 |
| `MINERU_PYTHON` | `python3` | Python 可执行文件。 |

退出码为 `0`（全部成功）、`2`（部分成功）、`1`（失败或取消）、`124`（轮询预算耗尽）。已下载的成功结果保留。
轮询超时不取消服务端任务。`UPLOAD_MODE` 与 `RESULT_FILE` 已删除。

## 部署、Router 与迁移

本地不设固定客户端上传大小或 PDF 页数上限；`usage.limits.max_file_size_bytes/max_pages_per_file` 为 `null`。
URL 下载沿用 Jobs 的策略：默认 HTTPS；HTTP 需启用 `--allow-http-source`，下载超时默认 60 秒、大小默认 200 MiB。
任务记录和上传索引仍为进程内状态，重启不保证恢复。

Router 按就绪文件独立选择 worker 并预占负载，同一父 Task 可使用多个 worker。
URL 交给 worker 下载，客户端上传字节按 worker 的缓存预检结果传输。
健康接口的 `features.task_output_formats` 单独声明 Tasks 格式；原 `features.output_formats` 继续描述 Jobs。
不声明新 Tasks 能力的旧 worker 不参与 Tasks 调度，更新时应先更新 worker，再更新 Router 和客户端。
Router 使用有界、可续租的内部结果租约保护 worker 产物，上传窗口不受结果租约影响；Router 丢失后租约会自行结束。

本次协议不保留旧表单、直传或 `markdown/middle_json/structured_content/zip` Tasks 格式别名。
客户端应改用 JSON、`files[].result_url` 及逐文件保存方式。
独立 `/v1/uploads`、`/v1/parse/jobs` 与 SDK 的现有请求及输出协议保持不变。
API Server 和 Router 不提供 `/v1/file_parse`、`/v1/tasks/{task_id}/result`。
需要同步单文件调用时，独立启动 [file_parse 包装服务](file-parse.md)，由它调用本页的共同 Tasks 契约。
SaaS Go 服务和对象存储配置由部署方按此共同契约实现，不属于本地 Python 服务代码。
