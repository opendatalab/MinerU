# Tasks 唯一异步接口迁移

这是破坏性 HTTP 接口升级：独立 `/v1/uploads` 全部路由及 `/v1/parse/jobs` 创建、列表、查询和取消路由已移除，返回 404；不保留别名或客户端自动回退。

| 旧方式 | 新方式 |
|---|---|
| Tasks 创建请求 `manifest` | `files`，仍为 1–100 项数组；旧字段或新旧混传返回 400。 |
| Tasks / Files JSON `filename` | `file_name`；下载附件命名规则保持不变。 |
| 创建请求必填 `client_file_id` | 可省略或为 `null`；服务端按输入顺序补全最小可用的 `fN`，避开所有显式 ID，响应仍必填。 |
| 创建 Upload、PUT、complete | POST `/v1/tasks` 提交清单，只 PUT 缺失文件；上传成功自动排队。 |
| 创建、查询、取消 Job | POST `/v1/tasks`；按 task_id GET/DELETE，不提供任务列表。 |
| `files[].output_files` | `files[].result_url`，每个输入一个完整产物链接。 |
| `markdown/middle_json/structured_content/zip` 输出选项 | `default` 默认 ZIP；另可组合 `docx/html/latex`。 |
| `local/inline/file_id` 输入 | 文件上传或 URL；同机先计算 hash，缓存未命中也需上传。 |
| `include_model_output` 与模型输出文件 | 参数已删除，HTTP 提交返回 400；ModelJson 仅在内部解析使用，不进入 API、SDK、CLI 或 Gradio 结果。 |
| Health `output_formats` | `task_output_formats`；sources 为 upload/url；删除 `task_model_output`。 |
| GET 任务列表 / 文件列表 | 已移除；合法认证下 `/v1/tasks` 返回 405，`/v1/files` 返回 404。 |
| 响应 `uploads/pending_uploads` | 改为各个 `files[].upload`，子字段为 `url/headers/expires_at`，不重复携带文件 ID；上传路由路径仍含 `/uploads/`。 |
| Tasks `status_url` | 删除，客户端使用配置的 API 基址和编码后的 task_id 构造固定查询路由。 |
| 响应顶层 `tier` | 改为成功文件的 `files[].parse.tier`，取实际结果元数据；请求 tier 仍保留。 |
| 响应 `files[].sha256sum/page_range/file_id/name` | 删除；请求 hash 和页范围不变，源引用只在内部管理。 |
| `upload_status` | 删除；仅保留 files[].status。waiting-file 统一输入准备，pending 表示已就绪后的解析排队。 |
| 状态 `waiting_upload/queued` | 改为 `waiting-file/pending`，不提供旧值别名；等待文件且 upload=null 时只轮询。 |
| `upload_method` | 字段删除，客户端固定使用 PUT，并原样使用 URL 和请求头。 |
| `files[].parse.model_used` | 删除；成功后的 parse 包含 `tier/duration_ms/parser_version`，完成前为 null。 |

`--allow-local-source` 和 `--max-inline-bytes` 同时删除。Files 按 ID 查询、下载与删除、Health、Tiers、Models、Usage 保留；同步 `file_parse` 继续作为独立包装服务运行。
SDK 保留 `MinerUApiParser` 方法和返回类型，状态类型改为 `ApiTaskStatus`，保留 `ApiJobStatus` Python 别名。直接本地 parser 与 HF CPU/ZeroGPU 内嵌解析不经过 HTTP。

停收旧请求、等待在途任务完成，再协调升级 worker、Router、SDK、file_parse 包装服务和应用。SaaS Go 服务需部署方按同一契约配套更新；进程内任务和缓存索引不保证跨重启恢复。

Router 通过候选 worker 的 `/v1/usage` 校验认证，不再依赖 Files 列表。
file_parse 恢复状态 namespace 增加 `tasks-v4-` 前缀，旧 `tasks-v3-`、`tasks-v2-` 及无前缀记录目录保留但不加载；新空间内继续支持 request_id 幂等和重启恢复。不提供旧客户端或旧 worker 的协议回退。

Router 源缓存与下游源 Files ID 解耦，仍保留结果 Files 映射和结果租约。file_parse 自身基于 request_id 的 status_url 不属于 Tasks 响应，本次保持不变。
