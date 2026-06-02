# MinerU 错误码体系（初稿）

> **状态**：初稿，待团队评审完善。
>
> 本文档定义 MinerU 解析 API（本地 Server / mineru.net）和 CLI 的统一错误码体系。

---

## 设计原则

- 与 OpenAI API 错误格式兼容
- 解析 API 层（远端 / 本地 Server）和 CLI 层共享同一套 `type` + `code`，CLI 不发明额外错误码
- CLI 透传 Server 返回的错误，仅在连接建立前（文件系统、网络层）产生 CLI 本地错误
- 未来新增 `code` 只需追加，不破坏现有消费者

## 错误响应格式

与 OpenAI 兼容的错误响应结构：

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_too_large",
    "message": "File exceeds the 200MB limit. Limit: 209715200 bytes, actual: 234567890 bytes.",
    "param": "file"
  }
}
```

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `type` | string | ✅ | 错误大类，枚举值见下表 |
| `code` | string \| null | 可选 | 具体错误码，机器可读 |
| `message` | string | ✅ | 人类可读的错误描述，包含上下文信息和修复建议 |
| `param` | string \| null | 可选 | 导致错误的参数名 |

### `type` 枚举

| type | 含义 | 对应 HTTP |
|------|------|----------|
| `invalid_request_error` | 请求参数非法、文件格式不支持等客户端错误 | 400 / 404 / 409 / 413 |
| `authentication_error` | API Key 无效或过期 | 401 |
| `permission_error` | 权限不足（配额耗尽、功能需认证等） | 403 |
| `rate_limit_error` | 限流 | 429 |
| `engine_error` | 解析引擎相关错误（无引擎、引擎崩溃、解析失败） | 500 / 503 / 504 |
| `api_error` | 服务端内部错误 | 500 / 503 |

---

## 一、解析 API 错误码

Server（本地 / 远端）在处理请求时返回的错误。本地 Server 和 mineru.net/api 共用同一套错误码。CLI 透传这些错误。

### 1.1 请求校验

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `invalid_request_error` | `invalid_request` | 400 | 参数格式 / 组合非法 | 出错的参数名 |
| `invalid_request_error` | `page_range_invalid` | 400 | 页码范围超出文档实际页数或格式非法 | `pages` |
| `invalid_request_error` | `file_type_unsupported` | 400 | 文件类型不在支持列表中 | `file` |
| `invalid_request_error` | `file_encrypted` | 400 | 加密 / 受密码保护 | `file` |
| `invalid_request_error` | `file_corrupted` | 400 | 文件损坏无法读取 | `file` |
| `invalid_request_error` | `file_too_large` | 413 | 超出大小限制 | `file` |

### 1.2 上传 / 文件

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `invalid_request_error` | `file_not_found` | 404 | file_id 无效 | `file_id` |
| `invalid_request_error` | `file_hash_mismatch` | 400 | 上传字节 SHA256 与声明不一致 | `sha256sum` |
| `invalid_request_error` | `bytes_mismatch` | 400 | 上传字节数与声明不一致 | `bytes` |
| `invalid_request_error` | `upload_not_found` | 404 | upload_id 不存在 | `upload_id` |
| `invalid_request_error` | `upload_not_ready` | 409 | 字节未上传即调用 complete | — |
| `invalid_request_error` | `upload_expired` | 409 | 上传会话超时 | — |

### 1.3 认证与配额

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `authentication_error` | `invalid_api_key` | 401 | API Key 无效或过期 | — |
| `permission_error` | `feature_requires_api_key` | 403 | 匿名用户请求需认证功能 | — |
| `permission_error` | `list_requires_api_key` | 403 | 匿名用户无权访问列表接口 | — |
| `permission_error` | `quota_exceeded` | 403 | 配额耗尽 | — |
| `rate_limit_error` | `rate_limit_exceeded` | 429 | 限流 | — |

### 1.4 解析执行

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `engine_error` | `no_engine` | 503 | 本地无匹配 tier 的引擎 | `tier` |
| `engine_error` | `engine_unavailable` | 503 | 引擎进程未启动或崩溃 | — |
| `engine_error` | `parse_failed` | 500 | 引擎返回的解析错误 | — |
| `engine_error` | `parse_timeout` | 504 | 解析超时 | — |

### 1.5 Job 生命周期

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `invalid_request_error` | `job_not_found` | 404 | job_id 不存在 | `job_id` |
| `invalid_request_error` | `job_already_terminal` | 409 | 取消已完成 / 已失败的 job | — |

### 1.6 通用服务端

| type | code | HTTP | 触发场景 | param |
|------|------|------|---------|-------|
| `api_error` | `internal_error` | 500 | 未预期错误 | — |
| `api_error` | `service_unavailable` | 503 | 服务暂不可用，可重试 | — |
| `invalid_request_error` | `model_not_found` | 404 | 请求的模型不存在 | `model` |

### 响应示例

```json
// 400 — 文件类型不支持
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_type_unsupported",
    "message": "File type '.xyz' is not supported. Supported types: pdf, docx, pptx, xlsx, jpg, png.",
    "param": "file"
  }
}

// 401 — API Key 无效
{
  "error": {
    "type": "authentication_error",
    "code": "invalid_api_key",
    "message": "Invalid API key provided. Check that your API key is correct.",
    "param": null
  }
}

// 503 — 本地无引擎
{
  "error": {
    "type": "engine_error",
    "code": "no_engine",
    "message": "No local engine available for tier 'standard'. Available tiers: flash. Use --remote or --tier flash.",
    "param": "tier"
  }
}
```

---

## 二、CLI 本地错误码

CLI 在调用 Server 之前或通信层面产生的错误，不来自 API 响应。这些错误没有 HTTP 状态码，但使用相同的 `error` 结构。

### 2.1 文件系统

| type | code | 触发场景 |
|------|------|---------|
| `invalid_request_error` | `file_not_found` | 本地文件路径不存在 |
| `invalid_request_error` | `file_permission_denied` | 无读取权限 |

### 2.2 Server 通信

| type | code | 触发场景 |
|------|------|---------|
| `api_error` | `server_not_running` | CLI 无法连接 UDS |
| `api_error` | `server_busy` | Server 队列满 |

### 2.3 远端通信

| type | code | 触发场景 |
|------|------|---------|
| `api_error` | `remote_timeout` | 远端 API 等待超时 |
| `api_error` | `remote_unreachable` | 网络不可达 |

### 2.4 缓存

| type | code | 触发场景 |
|------|------|---------|
| `invalid_request_error` | `not_cached` | `--no-wait` 时请求内容不在缓存中 |

### 响应示例

```json
// CLI — Server 未启动
{
  "error": {
    "type": "api_error",
    "code": "server_not_running",
    "message": "Cannot connect to mineru server at /tmp/mineru.sock. Run 'mineru server start' to start the server.",
    "param": null
  }
}

// CLI — 本地文件不存在
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_not_found",
    "message": "File not found: /Users/foo/doc.pdf",
    "param": null
  }
}
```

---

## 与 OpenAI 的兼容性说明

| 维度 | OpenAI | MinerU |
|------|--------|--------|
| 顶层结构 | `{"error": {...}}` | 相同 |
| `type` 枚举 | `invalid_request_error`、`authentication_error`、`rate_limit_error`、`api_error` | 相同 + 新增 `permission_error`、`engine_error` |
| `code` | 可选，具体错误码 | 相同 |
| `message` | 必带，人类可读 | 相同，包含修复建议 |
| `param` | 可选，出错参数 | 相同 |

新增的 `type`：
- `permission_error`：OpenAI 无此分类（OpenAI 的 403 归入 `invalid_request_error`），MinerU 单独区分配额和权限类错误
- `engine_error`：解析引擎相关错误，OpenAI 无对应概念

---

## 待讨论

- CLI 本地错误是否也用同样的 JSON 结构输出（当 `--json` 或非 tty 时）
- 错误码粒度是否需要进一步拆分（如 `parse_failed` 是否细分为 `parse_oom`、`parse_gpu_error` 等）
- 退出码（exit code）与 error code 的映射关系
