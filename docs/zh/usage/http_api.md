# V1 HTTP API

推荐 HTTP 客户端使用 tasks 的表单清单与下载链接协议。本页先介绍该流程，再给出 Python SDK、WebUI 使用的既有 Uploads/Jobs 资源工作流。Python 客户端见 [Python SDK](sdk_api.md)。

先启动本地服务；使用 SaaS 时，将客户端指向实现同一 tasks 契约的服务地址：

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

## tasks：清单、逐文件上传与 ZIP 下载

API server、Router 和采用同一契约的 SaaS 使用相同客户端流程：提交清单、上传缺失文件、轮询状态、下载返回的 `result_url`。
SaaS 可以返回 OSS 直链，本地返回 Files 下载地址；客户端原样使用链接，不拼接下载路径，也不要求云端实现本地 Files API。
完整契约见 [V1 tasks](../../next/api/tasks.md)。

```bash
# 单文件或多个文件：默认先预检，只上传未命中缓存的项。
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# hash 可选。
SEND_HASH=0 MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# 使用首次提交的文件列表和顺序继续原任务。
TASK_ID=job_... MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# 本地仅保留单文件 multipart 快捷提交，仍返回异步任务。
curl --fail-with-body 'http://127.0.0.1:8000/v1/tasks' \
  -F 'tier=flash' -F 'ocr_mode=txt' -F 'files=@document.pdf'
```

`POST /v1/tasks` 的参数全部放在表单中。`manifest` 是 JSON 数组形式的文本字段，每项包含唯一 `client_file_id`、`filename`、实际 `bytes`，可选 `sha256sum` 与逐文件 `page_range`。
多文件任务只能提交清单，不能在同一个请求中附带文件字节。单文件直传需要 hash 时，可附带单条 manifest。
解析选项仍使用 `tier`、`ocr_mode`、`page_range` 和重复的 `output_formats`，ZIP 始终生成。
POST 查询参数返回 `400`，JSON 和原始字节创建请求返回 `415`。独立 Uploads/Jobs 的现有 JSON 协议不受影响。

创建立即返回稳定的 `task_id` 和 `status_url`；`uploads` 只包含需要上传的项，每项给出上传方法、地址、请求头与有效期。
所有文件校验通过后自动开始，客户端不再发送完成确认。未提供 hash 时正常上传，服务端仍计算 hash 并去重保存。
文件完整上传后独立标记为 `ready`；后续文件中断只需重传失败文件，已经成功的文件保留。当前支持文件级重试，不支持单文件分块续传。

客户端只轮询 `status_url`。状态为 `waiting_upload/queued/running/completed/partial/failed/canceled`。
最终 ZIP 已保存后才返回 `completed/partial` 和 `result_url`；状态接口不内联解析正文，下载不经过 SaaS API 网关。
批量 ZIP 含编号目录及逐文件 manifest。打包失败是任务失败，不能返回不可用的下载链接。

完整客户端由 [`scripts/http_task_example.sh`](https://github.com/opendatalab/MinerU/blob/master/scripts/http_task_example.sh) 与同目录 Python 文件维护，只需 Python 3.10+ 标准库。
`UPLOAD_MODE=direct` 只支持一个文件；`RESULT_FILE` 默认 `result.zip`。`MAX_POLLS/POLL_INTERVAL` 控制有界轮询，`CONNECT_TIMEOUT/REQUEST_TIMEOUT` 控制传输预算。
每个文件最多尝试三次，创建任务不自动重试；响应丢失先查询文件状态，链接过期则查询原任务刷新。
API Key 只附加到同源请求，OSS、外部下载和跨源重定向不携带该 Key。
退出码为 `0`（成功）、`2`（部分成功且已保存 ZIP）、`1`（失败或取消）、`124`（等待预算耗尽，保留任务）。

本地不设固定的上传字节数或 PDF 页数上限，每任务最多 100 文件。等待上传不占解析名额，也不预占 Router worker。
Router 按负载选择 worker，并在向上游提交前预占；缓存命中不固定到繁忙 worker。批量文件仍交给同一个 worker，最终 ZIP 在 worker 生成，Router 只映射资源并流式下载。

## 既有 Uploads / Jobs 资源示例

需要操作独立 Upload、File 与 Job 资源时，继续使用以下脚本；其协议和退出码保持不变，与上面的 tasks 脚本分别维护：

> [`scripts/http_api_example.sh`](https://github.com/opendatalab/MinerU/blob/master/scripts/http_api_example.sh)

```bash
export MINERU_API_URL=http://127.0.0.1:8000
export MINERU_API_KEY=secret-key        # 匿名本地访问时省略
./scripts/http_api_example.sh document.pdf
```

脚本行为由 `tests/unittest/test_http_api_example_script.py` 在真实 Flash 服务和脚本化 mock 服务上验证，无需下载模型。退出码面向自动化保持稳定：

| 码 | 含义 |
| --- | --- |
| `0` | 任务 completed，全部请求产物已保存 |
| `1` | 脚本、传输、协议、下载或本地写入失败 |
| `2` | 任务 partial：已先保存成功文件的产物 |
| `3` | 任务 failed |
| `4` | 任务 canceled |
| `124` | 轮询预算耗尽；打印 `job_id`，可继续轮询 |

## 资源级接口的鉴权与字节上传

服务以 `--api-key` 启动时，`/v1/*` 请求携带 `Authorization: Bearer $MINERU_API_KEY`；匿名本地访问可省略该头。

字节上传原样使用 `POST /v1/uploads` 返回的方法、URL 和请求头：

- 自部署服务返回同源 URL（如 `/v1/uploads/{id}/content`）。同源指 scheme + host + 有效端口，由客户端判断而非假设；同源上传在服务要求鉴权时必须携带 API 鉴权。
- 官方 API 返回自带授权的预签名对象存储 URL。**不要**向不同源的 URL 附加 MinerU API Key——那会泄露凭据。服务返回的上传头（`upload_headers`）始终原样保留。

示例脚本实现了与 Python SDK（`api_client._same_origin_upload_headers`）相同的规则：相对 URL 先按 API 基地址解析，再比较 scheme + host + 有效端口，仅对同源上传附加 MinerU Key。相对 URL 解析后即同源；同一主机的不同端口按不同源处理。

下载走 `GET /v1/files/{file_id}/content`，可能返回 `302` 重定向。示例跟随重定向但不使用 `--location-trusted`，凭据不会被重新发送到不同源（包括同主机不同端口；要求 curl >= 7.83.0）。

## 资源级脚本的请求周期细节

- **上传生命周期。** 创建上传时传 `sha256sum` 可启用完整性校验和秒传；命中已存在文件时响应直接为 `completed` 并内嵌 `file` 对象，跳过字节上传和完成两步。
- **HTTP 200 不等于合法响应。** 脚本在每一步校验 JSON 结构：ID 必须是非空字符串、状态必须属于支持集合；否则立即失败，而不是带着 null ID 进入轮询。
- **有界轮询。** `MAX_POLLS` 限制轮询请求次数（每次还包括请求超时和轮询间隔）；预算耗尽时脚本退出 `124` 并打印 `job_id`——任务未被取消，可直接继续轮询。
- **终态。** `completed`、`partial`、`failed`、`canceled` 都是终态。`partial` 表示部分文件成功：脚本先保存成功文件的产物、打印含错误的逐文件结果，然后才退出 `2`；`failed` 退出 `3`，`canceled` 退出 `4`。
- **原子下载。** 产物先写 `*.part`，成功后才移到最终文件名；`completed` 任务若有产物下载失败或缺失，不会退出 `0`。
- **产物。** 所有产物都是 `purpose:"parse_output"` 的文件；内容一律通过 `GET /v1/files/{file_id}/content`、用 `files[].output_files` 下的 id 获取。以 `GET /v1/health`（`features.sources`、`features.output_formats`）为当前部署实际能力的依据；渲染器支持的格式不代表 API 暴露。

## 生产部署注意事项

- 谨慎绑定 `--host`。默认 `127.0.0.1` 只监听回环；使用 `0.0.0.0` 暴露到网络时，应配合 `--api-key` 或带鉴权的反向代理。
- 存储与重启边界：文件字节保存在 `--upload-dir` 下（未配置时为临时目录，正常关闭即删除），但上传/文件/任务的资源索引是进程内状态。即使保留了目录，服务重启后旧的 `upload_id`/`file_id`/任务 ID 也不保证可用。该 API 不是可恢复的持久化任务服务；持久化阅读、索引和缓存需求使用文档库——但文档库不是这套上传/任务接口的透明持久化替代品。
- `--concurrency` 限制并发解析任务数；超出后表现为 `queued` 时间变长，而不是失败。
- `GET /v1/usage` 报告用量。用 `/v1/health` 和 `/docs`（启用时）的 OpenAPI 文档区分服务端故障与客户端错误；重试 `failed` 任务前先查看服务端日志。

新的便捷工作流统一使用 `/v1/tasks` 和返回链接，不提供同步等待或结果正文路由。根路径旧接口不提供，旧客户端迁移见[迁移指南](../reference/migration_4.md)。
