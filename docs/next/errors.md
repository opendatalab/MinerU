# 错误与恢复边界

不同入口拥有不同的错误结构。核对实现时沿实际消费者追踪，不把内部异常字段直接复制到公共 HTTP。

## 公开 HTTP

| 入口 | 错误对象 | 规范与实现 |
|---|---|---|
| Tasks、Files、API 能力接口、Router | `error.code/message`，以及有值时的 `param`；无 `type` | [HTTP 错误](api/responses.md)、`mineru/tasks/contracts.py` |
| file-parse 包装服务 | `code/message/upstream_status`；有记录时还返回完整状态对象 | [file-parse](https://opendatalab.github.io/MinerU/zh/usage/file_parse/)、`mineru/kit/file_parse/` |

HTTP 200 的任务快照可能包含失败文件，必须检查 status 和逐文件 error。
响应正文、状态码、字段省略规则以各接口规范为准，不能套用下面 CLI/Doclib 的默认映射。

## CLI、Doclib 与本地 Python

`mineru/errors.py` 定义 `MineruError(code, message, param)`，实例另有 `type` 分类。
`InvalidRequestError`、`NotFoundError`、`PermissionError_`、`ConflictError`、`EngineError` 和
`ServerNotRunningError` 等子类用于内部产品路径。
HTTP 映射由 `http_status_for()` 按具体 code 优先、type 默认值其次处理。
`error_response()` 产生内部错误 envelope；这些字段不构成 Tasks 的 HTTP 契约。

直接 `parse()` 还可能抛出文件系统、参数校验或底层引擎异常。
API-backed parser 具有独立的业务错误与传输错误路径；不要只捕获 `MineruError` 就认为覆盖所有失败。

## 常见恢复原则

| 场景 | 处理 |
|---|---|
| `page_range_invalid`、格式或参数错误 | 修正请求，不原样重试 |
| `invalid_api_key` | 检查目标服务及凭据；不可把推理服务 Key 用于 Tasks |
| `quality_tier_unavailable` | 核对目标服务能力及显式档位，不能静默改档 |
| CLI `parse_wait_timeout` | 解析继续执行；使用同一文档／任务引用查询 |
| Tasks 创建结果未知 | 不盲目重放；已有 ID 时查询原任务 |
| 上传响应丢失 | 先查状态；只有仍需上传且链接有效时有界重试 |
| file-parse `submission_unknown` | 保留 request_id，包装服务不自动再次创建 |
| 历史缓存无法转换、缺少素材 | 从源文件重新解析，不能仅改 schema 版本 |

本地 Doclib 的文件、文档和 parse record 错误分别归所属记录管理。
持久化和索引失败的后续改进在 [当前待办](backlog.md)，不能以历史计划替代当前行为。
`retryable`、`user_action`、`docs_url` 尚未成为通用错误协议。
