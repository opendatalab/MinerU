# API-backed Parser: `MinerUApiParser`

`MinerUApiParser` 实现 `DocumentParser`，通过 [Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/) 调用 API Server、Router 或 SaaS，返回严格 Middle JSON 2.0 的 `ParseResult`。它不使用 Doclib 结果缓存，也不自行选择是否允许远端上传。

```python
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(api_url="http://127.0.0.1:8000", tier="flash", ocr_mode="auto", include_images=True)
result = parser.parse("report.pdf", page_range="1-5")
print(result.markdown())
```

构造参数保留 `api_url/api_key/tier/ocr_mode/include_images`。
地址默认读取 `MINERU_API_URL`，否则使用 `https://mineru.net/api`；密钥默认读取 `MINERU_API_KEY`。
`tier=None` 和 `ocr_mode=None` 省略对应请求字段；服务默认 OCR 模式是 `auto`。
`/v1/tiers` 仅描述 PDF/图片档位，原生文档始终可提交给所连接服务并使用 flash。SDK 不自行选择默认 tier：PDF/图片由服务端按 **standard → basic → flash → advanced** 选择支持的档位，仅开放 advanced 时自动选择 advanced。
显式 tier 原样发送，服务不支持该 PDF/图片档位时返回错误，不自动升降档。
Office、HTML 等已支持的原生文档始终使用 flash，忽略合法的显式 tier；单文件 HTTP 调用也遵循 [Tasks 规则](https://opendatalab.github.io/MinerU/zh/usage/tasks/#tier-selection)，与直接本地解析及 Doclib 的选档策略独立。
`parse()` 和 `parse_async()` 均接收 `page_range/status_callback/duration_callback`。

## 请求与结果

1. 校验文件及页范围，计算大小和 SHA-256，包括同机 HTTP 请求。
2. 向 `/v1/tasks` 提交单文件 files 数组，不指定任何文件 ID，固定请求 `output_formats=["default"]`，从创建回执记录服务端生成的 file_id，再通过首次 GET 校验详情。
3. 缓存命中不上传；未命中遍历 `files[]` 中非空的 `upload`，固定使用 PUT 向返回的链接按 headers 流式上传，并按清单大小设置 `Content-Length`。无需完成确认。
4. 以配置的 API 基址追加 `/v1/tasks/{编码后的 task_id}`，后续每秒轮询，保留代理路径前缀，轮询预算 3600 次；用服务端生成的 `file_id` 关联结果。
5. 从该文件的 `result_url` 流式下载并校验 ZIP，读取 `middle_json.json`。
6. `include_images=True` 恢复图片素材。`ParseResult` 只持有 MiddleJson，不携带或导出 ModelJson。

HTTP 输入只有上传和 URL；SDK 的路径参数始终代表客户端文件，不再传递服务端路径。上传 HTML 不自动附带同目录资源，应使用自包含文件或 Tasks URL 输入；直接本地 Python parser 仍可读取本地路径。

## 状态和错误

`ApiTaskStatus` 包含 `waiting-file/pending/running/completed/partial/failed/canceled`；`ApiJobStatus` 是保留的 Python 类型别名。
状态回调在创建和查询后触发，回调异常只记录日志。服务端完成表示开始下载，不等于 SDK 已经返回。
单文件成功后的 `parse` 包含实际 `tier`、`duration_ms` 和 `parser_version`。`duration_callback` 在开始下载前接收服务端单文件 `parse.duration_ms`，包含解析和导出；缺失或非法值不触发回调。

`parse_async(..., cancel_on_cancel=True)` 将协程取消联动到 `POST /v1/tasks/{task_id}/cancel`，默认关闭，普通 SDK、Doclib 和 file-parse 的既有行为不变。Gradio 显式开启此选项。取消发生在创建期间时，会等待同一次创建回执以取得服务端 ID，创建整体最多等待 120 秒，不重复提交；拿到未终结任务后，最多用 5 秒发送一次取消。404 和 `409 task_already_terminal` 视为已无可取消任务，其他失败记录日志，最终仍传播原始 `CancelledError`。网络失败且没有取得 ID 时无法确认远端取消。已知任务终结后的下载或本地输出取消不会调用远端取消，也不删除产物。

创建请求没有幂等保证，不自动重试。上传响应丢失后先查状态：进入 pending/running/completed 后不重传；waiting-file 且 未返回 upload 时仅轮询；出现上传目标后才在原重试预算内发送。PUT 成功后等待确认，不因状态仍是 waiting-file 而重复上传。查询、传输和下载有界重试；任务查询和用量查询对 408/429/500/502/503/504 最多请求 3 次，支持 `Retry-After` 秒数和 HTTP 日期。单次建议等待超过 60 秒时返回原错误，不提前重试；无有效建议时使用指数退避。下载签名失效后重新查询任务。密钥仅附加到 API 同源请求，每次重定向重新判断来源。

结构化业务错误保留 code/message/param。传输失败保留 `_APITransportError` 阶段及超时信息，Doclib 继续区分 `remote_timeout`、`remote_unreachable` 与业务 `parse_failed`。

旧协议客户端不能连接移除旧接口后的服务，更新 worker、Router 和 SDK 后再切换调用方；参见 [迁移说明](https://opendatalab.github.io/MinerU/zh/reference/migration_4_1/)。
