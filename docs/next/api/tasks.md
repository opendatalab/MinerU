# V1 tasks：逐文件上传与结果下载链接

状态: Implemented（自部署 API server 与 Router）
读者: HTTP 客户端、SaaS 接口与自部署服务开发者

共同流程是：**提交文件清单 → 上传缺失文件 → 轮询任务 → 使用返回链接下载 ZIP**。
SaaS 可以把上传、下载地址指向 OSS，本地服务可以指向自身的文件接口。客户端只使用服务返回的链接，不拼接存储路径，也不要求 SaaS 实现本地 Files API。
SaaS 的 Go 服务、OSS 配置及部署由其开发方实现；这里定义客户端共同契约。

## 路由与状态

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/tasks` | 表单提交文件清单，或本地单文件 multipart 直传；返回 `202`。 |
| 返回的 `upload_method` 与 `upload_url` | 每个链接上传一个文件，本地使用 `PUT /v1/tasks/{task_id}/uploads/{client_file_id}`。 |
| `GET /v1/tasks/{task_id}` | 返回 `200` 和任务元数据、缺失文件上传项、可用的结果链接。 |
| `DELETE /v1/tasks/{task_id}` | 取消等待上传或正在执行的任务。 |

任务首次创建就获得稳定的 `task_id`；执行提交后，它也是当前部署的公开 `job_id`。等待上传阶段使用 tasks 查询，既有 `/v1/parse/jobs` 执行资源尚未创建。

状态为 `waiting_upload`、`queued`、`running`、`completed`、`partial`、`failed`、`canceled`。
全部文件校验通过后自动启动，不要求客户端发送完成确认。等待上传不占解析并发，也不预占 Router worker。
`completed/partial` 只在最终 ZIP 已保存后发布；打包时仍为 `running`。`partial` 保留成功文件产物及失败文件的错误。

## 表单请求

`POST /v1/tasks` 只接受 `application/x-www-form-urlencoded` 或 `multipart/form-data`。
所有业务参数放在表单主体；URL 查询参数返回 `400 invalid_request`，JSON 和原始字节创建请求返回 `415`。
文件 PUT 使用原始字节。服务返回的 OSS 签名 URL 可以包含查询串，客户端必须原样使用。

| 表单字段 | 默认值 | 说明 |
|---|---|---|
| `manifest` | 无 | JSON 数组形式的**文本字段**；批量任务必填，最多 100 项。 |
| `tier` | 服务默认值 | `flash/basic/standard/advanced`，受部署能力约束。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | 全局 PDF 页范围；仍支持 `1-5`、`r1`、`all` 等现有语法。 |
| `output_formats` | `markdown` | 可重复字段，支持 `markdown/middle_json/structured_content/zip`；最终 ZIP 始终生成。 |
| `files` | 无 | 仅本地单文件 multipart 直传使用，最多一个文件。 |

每个清单条目包含：

| 字段 | 必填 | 说明 |
|---|---|---|
| `client_file_id` | 是 | 一次任务内唯一，1–64 个字母、数字、下划线或短横线。 |
| `filename` | 是 | 逻辑文件名，保留扩展名；不同条目允许同名。 |
| `bytes` | 是 | 文件实际字节数，正整数。 |
| `sha256sum` | 否 | 64 位小写十六进制 SHA-256；省略时视为需要上传。 |
| `page_range` | 否 | 逐文件页范围，未提供时继承全局值；非 PDF 仍只支持整本解析。 |

用真实文件生成清单，再发送表单：

```bash
python3 - a.pdf b.pdf > manifest.json <<'PY'
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
print(json.dumps(manifest, ensure_ascii=False))
PY

curl --fail-with-body "$MINERU_API_URL/v1/tasks" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  --data-urlencode 'tier=flash' \
  --data-urlencode 'ocr_mode=txt' \
  --data-urlencode 'manifest@manifest.json'
```

匿名本地服务可省略 Authorization。无需 hash 时从清单中省略 `sha256sum`。
`client_file_id` 用于关联文件和上传项，不能按文件名判断对应关系。

## 缓存、上传与恢复

- 全部 hash 命中：返回空 `uploads`，直接提交解析。
- 部分命中：仅返回未命中文件的上传项，命中文件已经是 `ready`。
- 无 hash 或未命中：上传完整文件；服务端计算实际 hash、校验大小与可选 hash，并去重保存。

缓存是**源文件缓存**，本次解析仍使用本次 tier、页范围和其他参数。
新请求不会凭文件 hash 复用不匹配的解析结果。

响应中的共同字段：

| 字段 | 说明 |
|---|---|
| `task_id`、`status`、`status_url` | 稳定任务 ID、当前状态和查询地址。 |
| `files` | 按输入顺序返回清单、`upload_status: pending/ready`、逐文件解析状态及错误。 |
| `uploads` | 尚需上传的文件，每项包含 `client_file_id`、`upload_url`、`upload_method: PUT`、`upload_headers`、`expires_at`。 |
| `result_url` | 结果未就绪时为 `null`；`completed/partial` 时为最终 ZIP 下载地址。 |
| `error` | 任务级错误，例如上传过期或结果打包失败。 |

本地上传准备默认有效期为 3600 秒，链接的 `expires_at` 为 Unix 秒。
一个文件完整校验后才变为 `ready`；中断内容不会进入解析或缓存。
已成功的文件独立保留，后续文件中断只需重传失败文件，继续使用原 `task_id`。
重复 PUT 已确认文件不会覆盖源字节；同时完成多个文件只会触发一次解析。

本轮提供文件级重试，单个文件失败后从头重传，不支持分块断点续传。
上传响应丢失时先查询任务：文件已 `ready` 就跳过，否则重试该项。
链接过期而任务仍有效时，重新查询同一任务获取当前链接；本地准备任务整体过期后返回 `failed/upload_expired`。

## 本地单文件快捷直传

```bash
curl --fail-with-body "$MINERU_API_URL/v1/tasks" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -F 'tier=flash' \
  -F 'ocr_mode=txt' \
  -F 'files=@document.pdf'
```

返回异步任务，不保持连接等待解析。需要声明 hash 时，同时提交仅含一个条目的 manifest 文本字段，例如 `-F 'manifest=<single-manifest.json'`。

多个文件字段，或者多文件清单混入文件字节，返回 `400`，应改用逐文件上传链接。
直传时实际发送的字节全部接受完整性校验；它不承诺在 hash 命中时跳过网络传输。稳定的省流量由清单预检提供。

## 下载结果

轮询只访问响应的 `status_url`。状态接口不内联 Markdown、Middle JSON 或 ZIP 内容。

- SaaS 返回 OSS 签名下载地址，客户端直接从 OSS 下载，结果字节不经过 API 网关。
- 本地返回 `/v1/files/{file_id}/content` 的完整地址，服务端从已保存文件流式输出。
- Router 返回自身的 Files 地址并流式代理 worker 产物，不泄露内部 worker 地址。

所有 tasks 自动生成 ZIP；单文件保留现有自包含产物，批量使用 `0001/`、`0002/` 等目录并包含 `manifest.json`。
manifest 使用 `client_file_id` 关联输入，`directory` 指向包内目录，不写入 worker 的私有 task/file ID 或短期链接。任务 ID 以 HTTP 响应为准。
ZIP 在后台只生成一次；重复查询或下载不重新打包。打包失败返回 `failed/result_packaging_failed`，不会发布虚假的下载链接。
签名链接过期但产物仍在保留期内时，重新查询任务获得有效地址，不重新提交解析。

认证只附加到与 API 同源的请求。同源由协议、主机、有效端口共同决定；OSS、外部下载地址及跨源重定向不附加 MinerU API Key。
上传额外请求头使用 `upload_headers`，不硬编码云存储 SDK 或 OSS 域名。

## 完整客户端示例

`scripts/http_task_example.sh` 调用同目录的 `http_task_example.py`，只需要 Python 3.10+ 标准库，无需安装 MinerU 或云存储 SDK。
默认顺序上传，每个文件最多尝试三次，仅重试网络及临时服务错误；创建请求不自动重试。
脚本校验任务身份、状态和文件对应关系，轮询有预算，下载先写临时文件并验证 ZIP 后再保存最终文件。

```bash
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

SEND_HASH=0 MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# 恢复时使用首次提交的文件列表和顺序，不重新创建任务。
TASK_ID=job_... MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

UPLOAD_MODE=direct MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf
```

| 配置 | 默认值 | 说明 |
|---|---|---|
| `MINERU_API_URL` | 必填 | API 或 Router 基地址。 |
| `MINERU_API_KEY` | 空 | API 认证，匿名本地服务可省略。 |
| `UPLOAD_MODE` | `preflight` | `preflight/direct`，direct 只支持一个文件。 |
| `SEND_HASH` | `1` | `0` 时不计算或提交 hash。 |
| `TASK_ID` | 空 | 继续已有任务，按原文件列表跳过 ready 文件。 |
| `MINERU_TIER`、`OCR_MODE`、`PAGE_RANGE` | 服务默认 / `auto` / 整本 | 解析参数。 |
| `OUTPUT_FORMATS` | `markdown` | 逗号分隔，自动加入 ZIP。 |
| `RESULT_FILE` | `result.zip` | 最终文件位置。 |
| `MAX_POLLS`、`POLL_INTERVAL` | `300`、`1` | 最多查询次数与间隔秒数。 |
| `CONNECT_TIMEOUT`、`REQUEST_TIMEOUT` | `10`、`600` | 连接和单次传输超时秒数。 |
| `MINERU_PYTHON` | `python3` | 可指定 Python 可执行文件。 |

退出码为 `0`（全部成功且保存 ZIP）、`2`（部分成功且保存 ZIP）、`1`（失败或取消）、`124`（轮询预算耗尽）。
取得任务 ID 后发生错误会打印恢复方式；轮询超时不会取消任务。

## 本地限制与 Router 调度

本地不设固定上传字节数或 PDF 页数上限；`usage.limits.max_file_size_bytes` 与 `max_pages_per_file` 为 `null`。
每任务最多 100 文件。既有 Jobs 的 URL/inline 来源预算仍按其专用配置生效。
源文件、上传索引和任务记录仍是进程内状态，重启不保证恢复；完成后的任务及产物沿用部署的保留策略。

Router 的 tasks 与 V1 Uploads 复用按调用方隔离的源字节缓存。
全部客户端文件齐备后才选择并预占 worker，再向该 worker 提交 tasks 和传输其缺失文件。
worker 负责最终 ZIP，Router 只查询任务与 File 元数据并映射公开下载地址。
同负载轮询，缓存命中不把任务固定到繁忙 worker；一个批量任务仍交给同一个 worker。
失败、取消、终态和后台回收通过同一机制释放预占，等待上传不计入执行负载。

## 迁移

旧 `/v1/file_parse` 和 `/v1/tasks/{task_id}/result` 已移除；不提供同步等待 HTTP 路由或结果正文路由。
客户端改为轮询 `status_url` 并下载 `result_url`，删除旧 `MODE=sync`、`RESPONSE_FORMAT`、`wait_timeout` 配置。
旧 tasks JSON 来源、原始字节创建请求和 POST 查询参数不再接受；URL、inline、local、file_id 等资源来源继续使用既有 `/v1/parse/jobs` 协议。
独立 `/v1/uploads` 与 `/v1/parse/jobs` 的 JSON 请求协议保持不变。
