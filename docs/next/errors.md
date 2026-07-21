# 错误码体系

状态: Draft
读者: API/CLI/SDK 开发者、服务端开发者
范围: 解析 API 错误码、本地 CLI 错误码、错误响应格式和兼容性说明
非目标: 每个 endpoint 的完整业务规则
来源: 由旧错误码底稿迁移整理而来；旧底稿已归档删除

## 1. 定位

错误码体系负责统一 API、CLI 和 SDK 对错误的表达。调用方应该能稳定地区分请求校验、文件上传、认证配额、解析执行、任务生命周期、服务端异常和本地运行错误。

解析 API 层的错误码由本地 parse-server 和 `mineru.net/api` 共用。CLI 透传 server 返回的错误；只有在连接建立前或通信层失败时，才产生 CLI 本地错误。CLI 本地错误仍使用同一套 `error` envelope，不另起一套错误码体系。

错误码也是 Agent 的控制协议。Agent 看到错误后，应能判断下一步是重试、启动 server、显式加 `--remote`、选择 `--tier flash`，还是停止并把问题交给用户。

## 2. 设计原则

- API、CLI 和 SDK 共享同一套 `type + code`。
- CLI 透传 server 返回的错误；只在调用 server 前或通信层产生本地错误。
- 错误格式兼容 OpenAI API。
- `message` 面向人类，`code`、`param`、`retryable` 和 `user_action` 面向程序。
- 不显式 `--remote` 时，不用错误恢复逻辑静默上传文档。
- PDF/image 的默认选择策略不能降级为 `flash`；找不到非 `flash` 质量 tier 时必须报错。Office/HTML 这类仅支持 flash tier 的输入归一规则见 [ADR-0024](decisions/0024-file-type-tier-normalization.md)。文本文件无需解析，显式 parse 请求返回 `parse_not_required`。
- 未来新增 `code` 只能追加，不改变既有语义。

## 3. 错误响应格式

基础格式与 OpenAI 兼容：

```json
{
  "error": {
    "type": "engine_error",
    "code": "quality_tier_unavailable",
    "message": "Default tier selection requires basic, standard, or advanced, but only flash is available. Start a parse-server, use --remote, or explicitly pass --tier flash.",
    "param": "tier",
    "retryable": false,
    "user_action": "start_parse_server_or_use_remote_or_explicit_flash",
    "docs_url": null
  }
}
```

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `type` | string | 是 | 错误大类 |
| `code` | string \| null | 建议 | 机器可读错误码 |
| `message` | string | 是 | 人类可读描述，包含关键上下文和修复建议 |
| `param` | string \| null | 否 | 出错参数 |
| `retryable` | bool | 建议 | 调用方是否可以在不改变请求的情况下重试 |
| `user_action` | string \| null | 建议 | Agent/CLI 可执行的下一步动作 |
| `docs_url` | string \| null | 否 | 指向错误说明或修复文档 |

`retryable`、`user_action` 和 `docs_url` 是 MinerU 扩展字段。OpenAI-compatible 客户端可以忽略它们。

## 4. Type 枚举

| type | 含义 | HTTP |
|------|------|------|
| `invalid_request_error` | 参数、文件、状态或资源不存在等客户端错误 | 400 / 404 / 409 / 413 |
| `authentication_error` | API Key 无效或过期 | 401 |
| `permission_error` | 权限、配额或隐私策略不允许 | 403 |
| `rate_limit_error` | 限流 | 429 |
| `engine_error` | 解析引擎、tier、parse-server 或解析执行错误 | 500 / 503 / 504 |
| `timeout_error` | 客户端或请求等待窗口到期，但后台任务不一定失败 | 408 |
| `api_error` | 服务端内部错误或通信不可用 | 500 / 503 |

## 5. Tier 与引擎错误

Tier 语义见 [解析 Tier](tiers.md)。本节定义 `flash`、`basic`、`standard` 和默认选择策略相关错误。

| type | code | HTTP | retryable | param | 触发场景 | user_action |
|------|------|------|:--:|-------|----------|-------------|
| `engine_error` | `quality_tier_unavailable` | 503 | 否 | `tier` | PDF/image 主动阅读场景需要非 `flash` 质量 tier，但只有 `flash` 可用 | `start_parse_server_or_use_remote_or_explicit_flash` |
| `engine_error` | `no_engine` | 503 | 否 | `tier` | 本地无匹配实体 tier 的引擎 | `enable_parse_server_or_change_tier` |
| `engine_error` | `engine_unavailable` | 503 | 是 | null | 引擎进程未启动、崩溃或暂不可用 | `wait_or_restart_parse_server` |
| `engine_error` | `parse_server_unavailable` | 503 | 是 | null | local 或 remote parse-server 不可达 | `wait_or_check_parse_server` |
| `engine_error` | `tier_mismatch` | 400 | 否 | `tier` | parse-server 不支持请求 tier | `choose_supported_tier` |
| `engine_error` | `parse_failed` | 500 | 否 | null | 引擎明确返回解析失败，如损坏、加密、模型无法处理 | `inspect_file_or_try_different_tier` |
| `engine_error` | `parse_timeout` | 504 | 是 | null | 解析超时 | `retry_or_use_lower_tier` |
| `timeout_error` | `parse_wait_timeout` | 408 | 是 | `wait` | CLI `parse --wait` 等待窗口到期，解析任务仍在运行 | `poll_parse_or_rerun_with_longer_wait` |
| `engine_error` | `parse_oom` | 500 | 是 | null | 本地显存或内存不足 | `use_lower_tier_or_remote` |
| `invalid_request_error` | `parse_server_model_not_ready` | 400 | 否 | `parse_server.local.mode` / `parse_server.local.managed_tier` | 配置 managed local parse-server 时，目标 tier 的本地模型文件未准备好 | `run_mineru_kit_models_download_for_tier` |
| `invalid_request_error` | `remote_unsupported_for_file_type` | 400 | 否 | `remote` | Office/HTML 单文件主动解析请求 remote | `use_local_flash_or_choose_pdf_image` |
| `invalid_request_error` | `tier_unsupported_for_file_type` | 400 | 否 | `tier` | Office/HTML 单文件主动解析显式请求质量 tier | `use_tier_flash_or_choose_pdf_image` |
| `invalid_request_error` | `tier_unsupported_for_remote` | 400 | 否 | `tier` | PDF/image remote 解析显式请求 `flash` | `choose_remote_quality_tier_or_local_flash` |
| `invalid_request_error` | `parse_not_required` | 400 | 否 | `path` / `doc_ref` / `locator` | 对文本文件请求解析、读取解析结果或作废解析结果 | `read_source_file_directly` |

关键约束：

- `quality_tier_unavailable` 不能自动 fallback 到 `flash`。
- `tier_mismatch` 不自动降级。例如用户请求 `standard`，server 只支持 `basic`，必须报错。
- `parse_failed` 不做 remote/local fallback；文件级解析失败应直接暴露。
- `engine_unavailable` 和 `parse_server_unavailable` 可重试，但重试不得改变 `privacy`。

## 6. 隐私与远端错误

| type | code | HTTP | retryable | param | 触发场景 | user_action |
|------|------|------|:--:|-------|----------|-------------|
| `permission_error` | `remote_not_allowed` | 403 | 否 | `remote` | 请求需要远端才能满足，但用户未显式允许 remote | `rerun_with_remote_if_acceptable` |
| `authentication_error` | `invalid_api_key` | 401 | 否 | null | API Key 无效或过期 | `set_valid_api_key` |
| `permission_error` | `feature_requires_api_key` | 403 | 否 | null | 匿名用户请求需认证功能 | `login_or_set_api_key` |
| `permission_error` | `list_requires_api_key` | 403 | 否 | null | 匿名用户请求列表类接口，例如文件列表或任务列表 | `login_or_set_api_key` |
| `permission_error` | `quota_exceeded` | 403 | 否 | null | 配额耗尽 | `wait_or_use_local` |
| `rate_limit_error` | `rate_limit_exceeded` | 429 | 是 | null | 触发限流 | `retry_later` |
| `api_error` | `remote_timeout` | 503 | 是 | null | 远端 API 超时 | `retry_or_use_local` |
| `api_error` | `remote_unreachable` | 503 | 是 | null | 远端网络不可达 | `check_network_or_use_local` |

`remote_not_allowed` 表示系统知道远端可能解决问题，但不能替用户上传文档。Agent 可以建议用户加 `--remote`，但不能自行添加。

## 7. 请求、文件和上传错误

| type | code | HTTP | retryable | param | 触发场景 |
|------|------|------|:--:|-------|----------|
| `invalid_request_error` | `invalid_request` | 400 | 否 | 出错参数 | 参数格式或组合非法 |
| `invalid_request_error` | `unsupported_output_format` | 400 | 否 | `output_formats` | 输出格式不支持 |
| `invalid_request_error` | `unsupported_source` | 400 | 否 | `source` | 当前部署不支持该 source 类型或 source 策略 |
| `invalid_request_error` | `page_range_invalid` | 400 | 否 | `page_range` | 页码范围格式非法或超出文档页数 |
| `invalid_request_error` | `file_type_unsupported` | 400 | 否 | `file` | 文件类型不支持 |
| `invalid_request_error` | `file_encrypted` | 400 | 否 | `file` | 文件加密或受密码保护 |
| `invalid_request_error` | `file_corrupted` | 400 | 否 | `file` | 文件损坏无法读取 |
| `invalid_request_error` | `file_too_large` | 413 | 否 | `file` | 超出大小限制 |
| `invalid_request_error` | `file_not_found` | 404 | 否 | `file` / `file_id` | 本地路径或远端 file_id 不存在 |
| `invalid_request_error` | `file_permission_denied` | 403 | 否 | `file` | 本地文件无读取权限 |
| `invalid_request_error` | `file_hash_mismatch` | 400 | 否 | `sha256sum` | 上传字节 SHA256 与声明不一致 |
| `invalid_request_error` | `bytes_mismatch` | 400 | 否 | `bytes` | 上传字节数与声明不一致 |
| `invalid_request_error` | `upload_not_found` | 404 | 否 | `upload_id` | upload_id 不存在 |
| `invalid_request_error` | `upload_not_ready` | 409 | 是 | null | 字节未上传即 complete |
| `invalid_request_error` | `upload_expired` | 409 | 否 | null | 上传会话过期 |

### 7.1 本地 doclib 错误归属

本地 doclib 中的错误按对象生命周期归属，避免把路径实例错误、内容身份错误和解析批次错误混在一起。

| 归属 | 存储字段 | 语义 | 典型错误 |
|------|----------|------|----------|
| File | `files.error_code` / `files.error_msg` | 某个路径实例不可用 | 路径不存在、权限不足、文件被锁、stat / SHA256 读取失败 |
| Doc | `docs.error_code` / `docs.error_msg` | SHA256 对应的文档内容本身存在问题 | metadata 读取失败、文件加密、文件损坏、内容不可识别 |
| Parse | `parses.error_code` / `parses.error_msg` | 某个 tier 和页码范围的解析批次失败 | parse-server 不可用、模型失败、超时、OOM、没有可访问文件 |

关键规则：

- 已经计算出 SHA256 后，metadata、加密、损坏等内容级错误写入 `docs.error_*`。
- ParseWorker 根据 `sha256` 找不到任何 active file 时，当前批次写入 `parses.error_code=no_accessible_file`。
- 如果 parse 阶段定位到具体 file row 且路径不可读，可以同时更新 file 的可达性或 file 级错误；parse batch 仍应保留自己的失败原因。

## 8. Job 与缓存错误

| type | code | HTTP | retryable | param | 触发场景 | user_action |
|------|------|------|:--:|-------|----------|-------------|
| `invalid_request_error` | `job_not_found` | 404 | 否 | `job_id` | job_id 不存在 | `check_job_id` |
| `invalid_request_error` | `job_already_terminal` | 409 | 否 | null | 取消已完成或已失败 job | `read_existing_result` |
| `invalid_request_error` | `not_cached` | 409 | 否 | null | `--no-wait` 且请求内容不在缓存 | `rerun_with_wait_or_submit_job` |
| `invalid_request_error` | `cache_miss` | 409 | 否 | null | 请求强依赖缓存但缓存不存在 | `parse_document` |

## 9. 通用 API 与服务错误

| type | code | HTTP | retryable | param | 触发场景 | user_action |
|------|------|------|:--:|-------|----------|-------------|
| `invalid_request_error` | `model_not_found` | 404 | 否 | `model` | 请求的模型不存在 | `check_model_id` |
| `api_error` | `internal_error` | 500 | 否 | null | 服务端未预期错误 | `report_with_request_id` |
| `api_error` | `service_unavailable` | 503 | 是 | null | 服务暂不可用或依赖暂不可用 | `retry_later` |
| `api_error` | `server_busy` | 503 | 是 | null | server 暂时无法接收请求，或 SQLite 锁竞争在有限重试后仍未恢复 | `retry_later` |

## 10. CLI 本地错误

CLI 在调用 server 前或通信层面产生本地错误。它们使用同一 `error` 结构，但没有 HTTP 状态码。

| type | code | retryable | 触发场景 | user_action |
|------|------|:--:|----------|-------------|
| `invalid_request_error` | `file_not_found` | 否 | 本地文件路径不存在 | `check_path` |
| `invalid_request_error` | `file_permission_denied` | 否 | 本地文件无读取权限 | `fix_file_permission` |
| `api_error` | `server_not_running` | 是 | CLI 无法连接 doclib UDS | `run_mineru_server_start` |
| `api_error` | `server_instance_mismatch` | 是 | endpoint 指向的进程不是写入该 endpoint 的 server 实例 | `restart_server` |
| `api_error` | `server_protocol_error` | 是 | CLI 与 server 协议不兼容或响应损坏 | `upgrade_or_restart_server` |

CLI 在 TTY 中可以用表格或 rich 文本展示，但非 TTY、`--json` 或 Agent 调用场景应输出结构化错误。

## 11. 示例

### 默认选择无可用质量 tier

```json
{
  "error": {
    "type": "engine_error",
    "code": "quality_tier_unavailable",
    "message": "Default tier selection requires basic, standard, or advanced, but only flash is available. Start a local parse-server, use --remote, or explicitly pass --tier flash.",
    "param": "tier",
    "retryable": false,
    "user_action": "start_parse_server_or_use_remote_or_explicit_flash",
    "docs_url": null
  }
}
```

### API Key 无效

```json
{
  "error": {
    "type": "authentication_error",
    "code": "invalid_api_key",
    "message": "Invalid API key provided. Check that your API key is correct.",
    "param": null,
    "retryable": false,
    "user_action": "set_valid_api_key",
    "docs_url": null
  }
}
```

### 远端未授权

```json
{
  "error": {
    "type": "permission_error",
    "code": "remote_not_allowed",
    "message": "This request requires remote parsing, but remote upload was not explicitly allowed. Re-run with --remote if uploading this document is acceptable.",
    "param": "remote",
    "retryable": false,
    "user_action": "rerun_with_remote_if_acceptable",
    "docs_url": null
  }
}
```

### 本地无匹配 tier 的引擎

```json
{
  "error": {
    "type": "engine_error",
    "code": "no_engine",
    "message": "No local engine available for tier 'basic'. Available tiers: flash. Use --remote or --tier flash.",
    "param": "tier",
    "retryable": false,
    "user_action": "enable_parse_server_or_change_tier",
    "docs_url": null
  }
}
```

### 本地 server 未启动

```json
{
  "error": {
    "type": "api_error",
    "code": "server_not_running",
    "message": "Cannot connect to mineru server at the configured UDS socket path. Run 'mineru server start' to start the server.",
    "param": null,
    "retryable": true,
    "user_action": "run_mineru_server_start",
    "docs_url": null
  }
}
```

## 12. SDK 映射

SDK 应暴露结构化异常，而不是只抛出字符串。

建议最小字段：

```python
class MinerUError(Exception):
    type: str
    code: str | None
    message: str
    param: str | None
    retryable: bool | None
    user_action: str | None
```

SDK 可以按 `type` 提供子类，例如 `MinerUEngineError`、`MinerUInvalidRequestError`，但必须保留原始 `code`。

## 13. OpenAI 兼容性

| 维度 | OpenAI | MinerU |
|------|--------|--------|
| 顶层结构 | `{"error": {...}}` | 相同 |
| `type` | 固定错误大类 | 兼容基础类型，新增 `permission_error` 和 `engine_error` |
| `code` | 可选机器码 | 相同 |
| `message` | 必带 | 相同，包含修复建议 |
| `param` | 可选 | 相同 |
| 扩展字段 | 不保证 | MinerU 可增加 `retryable`、`user_action`、`docs_url` |

新增的 MinerU `type`:

- `permission_error`: OpenAI 通常把 403 归入 `invalid_request_error`；MinerU 单独区分权限、配额和 access level。
- `engine_error`: OpenAI 没有解析引擎、tier 和 parse-server 的对应概念；MinerU 用它表达解析执行层错误。

## 与其他文档的关系

- API 响应见 [Unified API](api.md)。
- CLI 输出行为见 [CLI 规格](cli.md) 和 [CLI 文档](cli/README.md)。
- SDK 异常封装见 [SDK 设计](sdk.md)。
- Tier 行为见 [解析 Tier](tiers.md)。
- ParseWorker 路由见 [系统架构](architecture.md)。

## 未决问题

错误码版本策略、trace/request id、`parse_failed` 是否继续拆分为 `parse_oom`、`parse_gpu_error` 等更细粒度错误、CLI exit code 映射和 `user_action` 枚举，集中维护在 [开放问题清单](open-questions.md)。
