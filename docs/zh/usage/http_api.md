# V1 HTTP API

4.1 使用 Tasks 作为唯一异步解析入口。API Server、Router、SDK 和 WebUI 统一提交 JSON 清单、按需上传、查询状态并下载逐文件结果。Python 调用见 [Python SDK](sdk_api.md)，完整字段见 [Tasks 契约](tasks.md)。

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
OUTPUT_FORMATS=default,docx MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_api_example.sh a.pdf
```

## 文件清单与上传

Tasks 与 Files JSON 统一使用 `file_name`；旧字段 `manifest`、`filename` 及新旧混传返回 400。

向 `POST /v1/tasks` 提交 `application/json`：files 每项包含逻辑 file_name、上传文件的 bytes 和可选 sha256sum/page_range，也可提供 URL（bytes 可省略）。禁止传入 client_file_id 或 file_id；服务端生成唯一 file_id，按输入顺序返回，并在上传、查询、取消和产物下载中保持不变。解析选项为 tier/ocr_mode/page_range/output_formats。

创建返回稳定的 `task_id`；仅需要客户端上传的文件携带 `upload`，包含 `url/headers/expires_at`，其他文件省略该字段。客户端固定使用 PUT 上传缺失文件，使用响应指定的 PUT 地址及请求头，无需 complete。每个文件就绪即独立排队，其他文件等待上传不会阻塞它。
同机也使用相同流程。SDK 总是计算 hash；已完成上传的有效缓存可以免传。并发首次请求允许重复传输，但源字节去重。缓存默认保留 24 小时、每 5 分钟扫描，正在执行及传输的引用受保护。

上传链接固定有效一小时，查询不会续期。截止前已受理上传可继续完成。中断后先查询，进入 pending/running/completed 后跳过上传；waiting-file 且 upload 省略 时只等待，有有效上传目标时才有限重试。过期文件进入 failed，错误码为 upload_expired。

## 状态与结果

使用配置的 API 基址（保留代理前缀）和编码后的 task_id 构造 `GET /v1/tasks/{task_id}` 轮询。父任务状态为 `waiting-file/pending/running/completed/partial/failed/canceled`；通过创建回执中的 `file_id` 对应输入。逐文件使用同一组状态但没有 partial，不再返回 upload_status。waiting-file 包括上传、服务端下载和校验，pending 表示输入已就绪并排队等待解析。
每个成功文件在 `files[].result_url` 提供唯一下载地址，即使父任务尚未完成也可下载。顶层没有 result_url，不生成跨输入文件的总 ZIP。

| output_formats | 内容 |
|---|---|
| default | ZIP：Markdown、Middle JSON、Structured Content、images/ |
| docx | 图片内嵌的 DOCX |
| html | 图片内嵌的 HTML |
| latex | TEX 与 images/ 的 ZIP |
| 多种 | 对应文件平铺在一个 ZIP 中 |

ModelJson 仅在解析内部使用，API、SDK、CLI 和 Gradio 均不输出模型文件。`include_model_output` 已删除，提交该字段返回 400。
`POST /v1/tasks/{task_id}/cancel` 取消未完成文件。任务和文件列表已移除；Files 保留按 ID 查询元数据和下载结果；不提供任务记录或产物删除接口。逐文件 `parse` 在完成前省略，成功后包含实际 `tier`、耗时和解析器版本；响应顶层不再有 tier。逐文件响应不回显 hash、页范围、底层源缓存 ID 或重复的 name；请求参数仍保留。

客户端原样使用结果链接：本地/Router 返回 Files 地址，SaaS 可返回对象存储签名地址。只为 API 同源请求附加 API Key，每次重定向重新判断；签名失效可重新查询结果链接。

## 示例客户端与迁移

两个 shell 示例入口都复用标准库 Python Tasks 客户端。`RESULT_DIR` 指定逐文件目录；`OUTPUT_FORMATS` 默认为 default，`SEND_HASH=0` 可关闭预检 hash。
使用相同文件顺序设置 `TASK_ID` 可继续原任务。创建请求不自动重试；`MAX_POLLS/POLL_INTERVAL` 限制轮询，`CONNECT_TIMEOUT/REQUEST_TIMEOUT` 限制单次传输。
退出码为 0（全部成功）、2（部分成功）、1（失败或取消）、124（轮询耗尽）。已下载产物保留。

旧独立上传和 Jobs 接口已移除，客户端与服务端需协调升级，详见 [4.1 迁移说明](../reference/migration_4_1.md)。需要同步单文件返回时使用独立的 [file-parse 包装服务](file_parse.md)。

POST 创建、GET 详情、POST cancel 和本地 PUT 分别返回创建回执、查询详情、逐文件取消结果和上传确认；Tasks 成功响应省略 null，但保留零值和空集合。POST cancel 取消未完成文件，成功文件的产物仍可直接下载。
