# 从 4.0.x 迁移到 4.1

本页说明相对 4.0.12 的接口与运行环境变化。HTTP 路径仍使用 `/v1`，但这是破坏性协议升级，旧客户端与新服务端不能直接混用。

## Python 与安装

Python 支持范围为 `>=3.11,<3.15`，不再支持 Python 3.10。建议使用 Python 3.12 新建环境，重新安装所需依赖，并更新服务启动配置中的解释器路径。

停止旧服务并排空任务后执行：

```bash
uv venv --python 3.12 .mineru-4.1
# Linux/macOS
source .mineru-4.1/bin/activate
# Windows PowerShell 改用：.\.mineru-4.1\Scripts\Activate.ps1
uv pip install -U "mineru>=4.1,<5"
mineru version --json
```

需要 Torch 或高吞吐 VLM 时，沿用 [torch/full 安装组合](../quick_start/extension_modules.md)。可选引擎仍受各自 Python wheel、驱动和平台限制。

## 服务档位与原生文档

API 启动使用单个上限参数 `--tier auto|flash|basic|standard|advanced`，移除 `--tiers`，不支持 `--no-flash`、`--no-advanced`。上限包含自身及较低档位；`--tier standard` 不再同时开放 Advanced。

```bash
mineru-kit api-server --tier auto
mineru-kit hardware detect --json
mineru-kit hardware info --json
```

- `auto`（或省略启动参数）按实际小模型后端和可用设备推荐上限。ONNX 在 Windows/Linux 按需选型、校准；首次可能准备或下载 OCR rec 模型，不要求开启 `--preload-models`。Torch 路径检查设备可用性，不执行 ONNX 校准。
- `hardware detect` 使用同一推荐逻辑；`hardware info` 只读报告硬件、运行环境与已有校准记录，不测速、不下载模型、不写配置。
- `/v1/tiers` 只描述 PDF/图片能力。所有正常可用的兼容服务都支持现有原生文档格式，Office、HTML 等固定使用 Flash，不要求服务在 `/v1/tiers` 中列出 Flash。
- HTTP 请求省略 `tier` 或传入 `null`，PDF/图片按 **Standard → Basic → Flash → Advanced** 选择支持的默认值；仅开放 Advanced 时才自动选择 Advanced。显式指定不支持的 PDF/图片档位返回 400，不自动改档。纯原生文档忽略合法的请求档位；混合批次分别执行，实际档位见 `files[].parse.tier`。
- Python `create_app(tiers=...)` 保留精确 PDF/图片档位集合。WebUI 和 Doclib 托管服务遵循新的启动上限；Gradio 不再因远端未声明 Flash 而额外启动本地原生解析服务。

设备配置与支持边界见[档位与运行环境](../usage/tiers.md)。服务开放的最高档位和单次请求的默认档位是两个不同设置。

## HTTP 接口替换

独立 `/v1/uploads` 全部路由和 `/v1/parse/jobs` 创建、列表、查询、取消路由已移除，返回 404，不保留别名或自动回退。

| 旧方式 | 4.1 方式 |
|---|---|
| 创建 Upload → PUT → complete → 创建 Job | `POST /v1/tasks` 提交 JSON 清单，只 PUT 缺失文件，校验成功后自动排队。 |
| 查询、取消 Job | `GET /v1/tasks/{task_id}`、`POST /v1/tasks/{task_id}/cancel`。 |
| Tasks 表单或 multipart 创建 | `application/json`；单文件 multipart 使用独立的 [file-parse 服务](../usage/file_parse.md)。 |
| `manifest`、`filename` | `files` 数组（1–100 项）、`file_name`；旧字段或新旧混传返回 400。 |
| `client_file_id`、客户端指定 `file_id` | 服务端生成每个输入的公共 `file_id`，创建响应顺序与输入一致；请求禁止这些字段，包括 `null`。 |
| `local/inline/file_id` 来源 | 上传或 HTTP(S) URL；同机也计算 hash 做缓存预检，未命中时上传。 |
| `uploads/pending_uploads` | 每个文件的 `files[].upload`，字段为 `url/headers/expires_at`。上传路径仍可包含 `/uploads/`，这不是旧独立上传 API。 |
| `upload_method`、上传 complete | 固定 PUT 原始字节，使用响应给出的 URL 和请求头，无需 complete。 |
| `upload_status`、`waiting_upload/queued` | 统一 `files[].status`，使用 `waiting-file/pending`；等待文件但没有 `upload` 时只轮询。 |
| Tasks `status_url` | 用配置的 API 基址和 URL 编码后的 `task_id` 构造查询路径，保留代理前缀。 |
| 共用任务响应 | 创建、详情、取消、本地上传分别使用 `CreateTaskResponse`、`TaskDetailResponse`、`CancelTaskResponse`、`UploadFileResponse`。 |
| 成功响应的 `null` 占位字段 | Tasks 递归省略无值字段，保留零值和空集合。file-parse 状态与 HTTP 错误保持各自规则。 |
| 公共错误 `type`、可空 `code` | 统一为必填非空 `code`、`message` 和可选 `param`；不输出或接受 type，空 param 省略。HTTP 失败与任务内错误使用同一结构。 |
| 顶层 `tier`、`files[].parse.model_used` | 成功文件的 `files[].parse` 提供实际 `tier` 及有值的 `duration_ms/parser_version`。 |
| 响应 `sha256sum/page_range/name`、源缓存 ID | 不再回显；请求 hash 和页范围仍可用。公共 `file_id` 从创建到结果保持不变。 |
| `files[].output_files`、批次总 ZIP | 每个成功输入一个 `files[].result_url`，按文件下载；父任务没有结果链接。 |
| `markdown/middle_json/structured_content/zip` 输出选项 | `default` 返回包含这些通用结果和图片的 ZIP；另外支持 `docx/html/latex`，可组合。 |
| `include_model_output`、`model_output.json` | 参数移除，提交即返回 400；ModelJson 仅用于内部处理，不进入 API、SDK、CLI 或 Gradio 结果。 |
| Health `features.output_formats`、`task_model_output` | 使用 `features.task_output_formats`，来源为 `upload/url`；移除 `task_model_output`。 |
| 任务/文件列表、DELETE 取消 | 无列表或公开删除接口；合法认证下 `GET /v1/tasks` 为 405、`GET /v1/files` 为 404，取消使用 POST。 |

`--allow-local-source` 和 `--max-inline-bytes` 同时移除。Files 按 ID 查询与下载、Health、Tiers、Models、Usage 保留。产物生成前，对预留 `file_id` 的 Files 查询或下载返回 404。

完整字段、状态机、上传有效期及错误说明见 [Tasks 接口](../usage/tasks.md)。SDK 保留 `MinerUApiParser` 方法和返回类型，状态类型使用 `ApiTaskStatus`，仍保留 `ApiJobStatus` Python 别名；这个别名不恢复旧 HTTP 协议。

公共错误升级同样要求 worker、Router、SDK、file-parse 与 SaaS 协调切换；客户端根据 HTTP 状态和 code 处理失败。内部 CLI/Doclib 错误分类以及 file-parse 历史恢复记录不随此调整改写。

## file-parse 与恢复边界

`mineru-kit file-parse-server` 是独立的单文件同步包装服务，通过 Tasks 连接本地 API、Router 或兼容 SaaS。主 API Server 和 Router 不提供 `/v1/file_parse`。

等待结束返回 202 不代表取消；保存 `request_id/status_url` 后继续查询或下载。包装服务恢复已知的上游任务，不会在创建结果未知时自动重提，也不能恢复已因上游重启或过期而消失的任务。

新恢复空间使用 `tasks-v5-` 前缀。旧 `tasks-v4-`、`tasks-v3-`、`tasks-v2-` 及无前缀目录保留，但不加载、不迁移。旧空间的请求 ID 不会恢复；新空间内继续支持幂等与包装服务重启恢复。状态和附件示例见 [file-parse 接口](../usage/file_parse.md)。

## 升级顺序与验证

1. 停收旧请求，等待在途任务结束并保存需要的产物。API/Router 的任务与缓存索引是进程内状态，不保证跨重启恢复。
2. 准备 Python 3.11–3.14 环境，先升级 worker，再升级 Router、SDK、WebUI、file-parse 包装服务及应用。SaaS 部署需同步对齐共同契约，不依赖旧 worker 回退。
3. 更新请求字段、查询/取消路径和逐文件下载逻辑，保存创建返回的任务与文件 ID；不要盲目重试创建请求。
4. 用 PDF 与原生文档混合批次验证档位、缓存免传、逐文件结果，再检查失败、取消、上传过期及同步等待超时后的查询。确认 Markdown、结构化 JSON 和素材完整后恢复接单。

更早版本用户还需参考 [3.x → 4.0 迁移](migration_4.md)。
