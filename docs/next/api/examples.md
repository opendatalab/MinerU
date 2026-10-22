# Tasks 使用示例

完整的流式上传、缓存预检、有限重试、恢复和逐文件下载由 `scripts/http_task_example.py` 实现；两个 shell 示例入口都调用它。

```bash
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh report.pdf
OUTPUT_FORMATS=default,docx MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_api_example.sh a.pdf b.pdf
TASK_ID=job_... MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
```

直接提供 URL 时只需提交 JSON：

```json
{"manifest":[{"client_file_id":"report","filename":"report.pdf","url":"https://example.com/report.pdf"}],"output_formats":["default"],"include_model_output":false}
```

对 `/v1/tasks` 执行 POST，保存返回的 `task_id/status_url`。本地文件清单包含 `bytes` 和可选 `sha256sum`；仅对响应中的待上传项使用返回的 PUT 地址。
完成后下载对应 `files[].result_url`。列表使用 `GET /v1/tasks?status=running,queued&limit=20`，取消使用 `DELETE /v1/tasks/{task_id}`。

详见 [Tasks 契约](tasks.md)、[SDK](../sdk/api-parser.md) 和 [同步包装服务](file-parse.md)。
