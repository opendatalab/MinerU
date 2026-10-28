# Tasks 唯一异步接口迁移

## 单 tier 上限与原生文档统一契约

API 启动改为 `--tier auto|flash|basic|standard|advanced`，移除 `--tiers`，不支持 `--no-flash` 或 `--no-advanced`。上限包含自身和较低档位；standard 不再隐含 advanced，WebUI、Router、Doclib 托管配置同样遵循该规则。

`/v1/tiers` 的响应结构不变，作用范围收敛为 PDF/图片。所有兼容服务均支持现有原生文档格式，固定使用 flash；无需原生能力声明。SaaS 可以只发布 standard，原生文件仍由同一服务处理。Gradio 不再因远端缺少 flash 而启动本地补充服务。

Python `create_app(tiers=...)` 保留精确 PDF/图片集合，模型依赖及预加载仍按实际需求合并。HTTP 请求默认、逐文件实际档位和产物协议保持原规则。

`--tier auto`（包括省略参数）按有效小模型后端分流。CPU 可用逻辑核至少 4 时默认开放到 Basic；GPU 经既有选型确认可用且对应显存／预算至少 5.5 GiB 时，才开放 Standard/Advanced，且不额外要求 4 核。Torch、Apple Metal 及符合既有绑定规则的独显免测速，其他 ONNX 候选仍要求有效校准胜出。Windows/Linux 在 CPU 达标或存在内存达标的兼容 GPU 候选时同步完成适用 ONNX 选型，最终 CPU 或失败结果不能凭静态库存重新升级。无需启用 `--preload-models`，首次校准可能准备或下载 OCR rec 模型。显式启动档位以及已保存的设备选择不强制测速；模型预加载与请求默认选档顺序保持原规则。

新增 `mineru-kit hardware detect [--json]` 和 `hardware info [--json]`，分别用于同源推荐与只读问题反馈诊断。软件版本只展示指定清单内已安装的包；没有 `--quiet` 或 `auto-tier` 别名。

这是一轮能力语义和启动参数升级：先停收并排空在途任务，再协调更新 worker、Router 和客户端。SaaS Go 服务由部署方同步对齐，避免新客户端把原生文件交给仍限制 Flash 的旧服务。模型故障恢复、Doclib 原生执行路径及缓存语义不属于本轮改动。

## 既有 Tasks 接口迁移


这是破坏性 HTTP 接口升级：独立 `/v1/uploads` 全部路由及 `/v1/parse/jobs` 创建、列表、查询和取消路由已移除，返回 404；不保留别名或客户端自动回退。

| 旧方式 | 新方式 |
|---|---|
| Tasks 创建请求 `manifest` | `files`，仍为 1–100 项数组；旧字段或新旧混传返回 400。 |
| Tasks / Files JSON `filename` | `file_name`；下载附件命名规则保持不变。 |
| 创建请求 `client_file_id` | 字段移除；请求也禁止指定 file_id，包括 null。服务端统一生成公共 file_id，响应按输入顺序返回。 |
| 创建 Upload、PUT、complete | POST `/v1/tasks` 提交清单，只 PUT 缺失文件；上传成功自动排队。 |
| 创建、查询、取消 Job | POST `/v1/tasks` 创建，GET `/v1/tasks/{task_id}` 查询，POST `/v1/tasks/{task_id}/cancel` 取消；不提供任务列表。 |
| `files[].output_files` | `files[].result_url`，每个输入一个完整产物链接。 |
| `markdown/middle_json/structured_content/zip` 输出选项 | `default` 默认 ZIP；另可组合 `docx/html/latex`。 |
| `local/inline/file_id` 输入 | 文件上传或 URL；同机先计算 hash，缓存未命中也需上传。 |
| `include_model_output` 与模型输出文件 | 参数已删除，HTTP 提交返回 400；ModelJson 仅在内部解析使用，不进入 API、SDK、CLI 或 Gradio 结果。 |
| Health `output_formats` | `task_output_formats`；sources 为 upload/url；删除 `task_model_output`。 |
| GET 任务列表 / 文件列表 | 已移除；合法认证下 `/v1/tasks` 返回 405，`/v1/files` 返回 404。 |
| 响应 `uploads/pending_uploads` | 改为各个 `files[].upload`，子字段为 `url/headers/expires_at`，不重复携带文件 ID；上传路由路径仍含 `/uploads/`。 |
| Tasks `status_url` | 删除，客户端使用配置的 API 基址和编码后的 task_id 构造固定查询路由。 |
| 响应顶层 `tier` | 改为成功文件的 `files[].parse.tier`，取实际结果元数据；请求 tier 仍保留。 |
| 响应源文件 `file_id` 与 `client_file_id` | 原源资源 ID 不再公开；新 files[].file_id 从创建到产物保持不变，并与结果 Files ID 一致。源缓存仍为私有引用。 |
| 响应 `files[].sha256sum/page_range/name` | 删除；请求 hash 和页范围不变。 |
| 共用 `TaskStatusResponse` | 分为 CreateTaskResponse、TaskDetailResponse、CancelTaskResponse、UploadFileResponse；GET 扩展不进入其他响应。 |
| Tasks 成功响应的 null 占位字段 | 省略无值可选字段，保留零值和空集合；file_parse 自身及共享错误 envelope 保持现状。 |
| `upload_status` | 删除；仅保留 files[].status。waiting-file 统一输入准备，pending 表示已就绪后的解析排队。 |
| 状态 `waiting_upload/queued` | 改为 `waiting-file/pending`，不提供旧值别名；等待文件且没有 upload 时只轮询。 |
| `upload_method` | 字段删除，客户端固定使用 PUT，并原样使用 URL 和请求头。 |
| `files[].parse.model_used` | 删除；成功后的 parse 包含 `tier/duration_ms/parser_version`，完成前省略。 |

`--allow-local-source` 和 `--max-inline-bytes` 同时删除。Files 按 ID 查询和下载、Health、Tiers、Models、Usage 保留；同步 `file_parse` 继续作为独立包装服务运行。
SDK 保留 `MinerUApiParser` 方法和返回类型，状态类型改为 `ApiTaskStatus`，保留 `ApiJobStatus` Python 别名。直接本地 parser 与 HF CPU/ZeroGPU 内嵌解析不经过 HTTP。

停收旧请求、等待在途任务完成，再协调升级 worker、Router、SDK、file_parse 包装服务和应用。SaaS Go 服务需部署方按同一契约配套更新；进程内任务和缓存索引不保证跨重启恢复。

Router 通过候选 worker 的 `/v1/usage` 校验认证，不再依赖 Files 列表。
file_parse 恢复状态 namespace 增加 `tasks-v5-` 前缀，旧 `tasks-v4-`、`tasks-v3-`、`tasks-v2-` 及无前缀记录目录保留但不加载；新空间内继续支持 request_id 幂等和重启恢复。不提供旧客户端或旧 worker 的协议回退。

Router 源缓存与下游源 Files ID 解耦，仍保留结果 Files 映射和结果租约。file_parse 自身基于 request_id 的 status_url 不属于 Tasks 响应，本次保持不变。

公共 file_id 在创建时预留；产物生成前访问对应 Files 元数据或内容接口均返回 404。产物完成后，下载地址使用相同 ID。Router 按层保存公共 ID 与下游 ID 的映射，不要求不同服务层使用相同字符串。

file_parse 获得创建回执后先持久化 task_id/file_id，再查询详情。首次 GET 失败或随后重启不会重复创建；仅创建请求的结果未知时进入 submission_unknown。

取消入口改为 `POST /v1/tasks/{task_id}/cancel`，继续返回 200 和逐文件终态；不保留 DELETE 取消别名。任务记录和产物的公开 DELETE 接口均不提供，在认证通过后返回 405。内部保留期回收、缓存引用释放及活动下载保护保持不变。
