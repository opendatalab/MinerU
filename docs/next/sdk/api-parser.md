# API-backed Parser: `MinerUApiParser`

`MinerUApiParser` 实现 `DocumentParser`，通过 [Tasks](../api/tasks.md) 调用 API Server、Router 或 SaaS，返回严格 Middle JSON 2.0 的 `ParseResult`。它不使用 Doclib 结果缓存，也不自行选择是否允许远端上传。

```python
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(api_url="http://127.0.0.1:8000", tier="flash", ocr_mode="auto", include_images=True)
result = parser.parse("report.pdf", page_range="1-5")
print(result.markdown())
```

构造参数保留 `api_url/api_key/tier/ocr_mode/include_images/include_model_output`。
地址默认读取 `MINERU_API_URL`，否则使用 `https://mineru.net/api`；密钥默认读取 `MINERU_API_KEY`。
`tier=None` 和 `ocr_mode=None` 省略对应请求字段；服务默认 OCR 模式是 `auto`。
`parse()` 和 `parse_async()` 均接收 `page_range/status_callback/duration_callback`。

## 请求与结果

1. 校验文件及页范围，计算大小和 SHA-256，包括同机 HTTP 请求。
2. 向 `/v1/tasks` 提交一个 `client_file_id="file"` 的 files 数组，固定请求 `output_formats=["default"]`。
3. 缓存命中不上传；未命中按服务返回的 PUT 链接和 headers 流式上传，并按清单大小设置 `Content-Length`。无需完成确认。
4. 按返回的同源 `status_url` 每秒查询，轮询预算 3600 次；用 `client_file_id` 关联结果。
5. 从该文件的 `result_url` 流式下载并校验 ZIP，读取 `middle_json.json`。
6. `include_images=True` 恢复图片素材；`include_model_output=True` 请求并读取可用的 `model_output.json`。没有原始模型输出时不生成占位数据。

HTTP 输入只有上传和 URL；SDK 的路径参数始终代表客户端文件，不再传递服务端路径。上传 HTML 不自动附带同目录资源，应使用自包含文件或 Tasks URL 输入；直接本地 Python parser 仍可读取本地路径。

## 状态和错误

`ApiTaskStatus` 包含 `waiting_upload/queued/running/completed/partial/failed/canceled`；`ApiJobStatus` 是保留的 Python 类型别名。
状态回调在创建和查询后触发，回调异常只记录日志。服务端完成表示开始下载，不等于 SDK 已经返回。
`duration_callback` 在开始下载前接收服务端单文件 `parse.duration_ms`，包含解析和导出；缺失或非法值不触发回调。

创建请求没有幂等保证，不自动重试。上传响应丢失后先查状态：已 ready 则继续，仍 pending 才重传。查询、传输和下载有界重试；任务查询和用量查询对 408/429/500/502/503/504 最多请求 3 次，支持 `Retry-After` 秒数和 HTTP 日期。单次建议等待超过 60 秒时返回原错误，不提前重试；无有效建议时使用指数退避。下载签名失效后重新查询任务。密钥仅附加到 API 同源请求，每次重定向重新判断来源。

结构化业务错误保留 code/message/param。传输失败保留 `_APITransportError` 阶段及超时信息，Doclib 继续区分 `remote_timeout`、`remote_unreachable` 与业务 `parse_failed`。

旧协议客户端不能连接移除旧接口后的服务，更新 worker、Router 和 SDK 后再切换调用方；参见 [迁移说明](../api/changes.md)。
