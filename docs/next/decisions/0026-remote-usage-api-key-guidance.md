# ADR-0026: Remote API Usage 与 API Key 引导

状态: Accepted
日期: 2026-07-14
相关文档: ../api/usage-limits.md, ../cli/mineru-parse.md, 0005-doclib-interface-client-server-contract.md, 0015-cli-output-json-composition.md, 0023-cli-runtime-contract.md

## 背景

MinerU Unified API 提供 `GET /v1/usage`，用于查询 Remote API 当前身份的计费周期、累计用量和限制。Official API 支持 anonymous 和 registered 两种 access level；Local Parse Server 通常没有账号配额或计费限制，因此本地进程统计不应与 Remote API usage 混在同一个用户命令中。

当前 `mineru` 没有查询 Remote API usage 的命令。Remote API Key 可以通过 doclib runtime config 的 `parse_server.remote.api_key` 配置，但 CLI 也没有在合适的错误场景中提供注册、创建 API Key 和设置 API Key 的可操作引导。

匿名 Remote API 是合法能力。引导出现得过于频繁，会让用户误以为注册和 API Key 是使用 MinerU 或 Remote API 的前置条件。因此引导必须由明确的 access level 或错误码驱动，并且只针对 Official API 展示 Official API 平台链接。

本文中的用户可见术语统一使用 **API Key**。`https://mineru.net/apiManage/token` 是既有兼容 URL，只有 URL path 保留 `token`，其他命令、字段说明和文案不使用 Token 指代 API Key。

## 决策

### 1. 增加 `mineru usage`

新增顶层命令:

```bash
mineru usage
mineru usage --json
```

命令只表示当前配置的 Remote API usage，不查询或合并 Local Parse Server usage。顶层 help 使用明确说明:

```text
usage    Show Remote API usage and limits.
```

普通文本输出标题使用 `Remote API Usage`，避免把 `usage` 误解为 CLI 使用手册。CLI 使用手册继续通过 `mineru --help` 和 `mineru <command> --help` 获取。

### 2. 通过 doclib 查询 Remote API

`mineru usage` 不直接读取 SQLite 配置，也不在 CLI 中自行实现 Remote HTTP client。调用链统一为:

```text
mineru usage
  -> DoclibClient.get_remote_usage()
  -> GET /api/v1/remote-usage
  -> DoclibServer.get_remote_usage()
  -> configured Remote GET /v1/usage
```

doclib public method 命名为 `get_remote_usage()`，本地 doclib HTTP route 为 `GET /api/v1/remote-usage`，返回模型为
`RemoteUsageResponse`。本地 route 使用 `remote-usage` 明确该资源来自 Remote API，并避免与未来可能增加的本地运行统计混淆。

doclib 负责:

- 读取 `parse_server.remote.url`。
- 解析 Remote API Key。
- 构造 Bearer authentication header；未配置 API Key 时不发送该 header。
- 请求 Remote `/v1/usage`。
- 保留 Remote API 的结构化错误语义。

这与现有 Remote parse 路径的配置所有权和进程边界一致，不为 CLI 建立第二套配置、认证或错误处理逻辑。

Remote API Key 由 doclib 中的统一 resolver 按以下优先级解析:

```text
parse_server.remote.api_key
  > MINERU_API_KEY
  > anonymous
```

- SQLite runtime config 中的空字符串视为未配置。
- `MINERU_API_KEY` 是进程环境变量 fallback，不加入 `config.py` 的 startup config model，也不持久化。
- Remote parse、Remote health probe 和 Remote usage 必须使用同一个 resolver，不能分别实现凭据回退。
- `rate_limit_exceeded` 和 `quota_exceeded` 是否属于 anonymous 请求，根据 resolver 返回的有效 API Key 是否为空判断。
- guidance 中的配置命令写入 `parse_server.remote.api_key`，因此会覆盖环境变量 fallback。

### 3. Usage JSON 契约

`mineru usage --json` 使用 CLI 自己的响应结构，并将 Remote `/v1/usage` 响应放在 `usage` 字段中:

```json
{
  "remote_url": "https://mineru.net/api",
  "usage": {
    "object": "usage",
    "access_level": "anonymous",
    "billing_period": {
      "start": "2026-07-14T00:00:00Z",
      "end": "2026-07-15T00:00:00Z"
    },
    "current": {
      "pages_processed": 26,
      "files_processed": 0,
      "jobs_created": 8
    },
    "limits": {
      "max_pages_per_file": 200,
      "max_file_size_bytes": 209715200,
      "max_files_per_job": 200,
      "max_concurrent_jobs": 500,
      "max_file_retention_days": 30
    }
  },
  "guidance": null
}
```

服务端返回的限制值是运行时能力，CLI 不硬编码文档示例中的数值。

doclib server 返回纯 `RemoteUsageResponse`，不生成 CLI guidance。CLI 使用一个输出模型同时服务 JSON 和文本模式；该模型添加
`remote_url` 和可空的 `guidance`，不要求 CLI JSON 与 doclib server 响应结构一致。通用 CLI runtime 不增加专用 JSON renderer。

### 4. API Key 引导只在五种时机出现

只有满足本节条件时，CLI 才主动展示 API Key 引导。除此之外不主动提示注册或配置 API Key。

| 时机 | 附加条件 | 是否必须配置 API Key |
|------|----------|----------------------|
| `mineru usage` 返回 `access_level=anonymous` | Official API | 否 |
| `invalid_api_key` | Official API | 是，必须替换无效 API Key |
| `feature_requires_api_key` | Official API | 是，必须配置 API Key 才能使用该能力 |
| `rate_limit_exceeded` | Official API 且当前未配置 API Key，即 anonymous 请求 | 否；首先遵循 `Retry-After`，API Key 是提高 access level 的可选方案 |
| `quota_exceeded` | Official API 且当前未配置 API Key，即 anonymous 请求 | 否；API Key 是切换 registered access level 的可选方案 |

以下场景明确不展示 API Key 引导:

- 普通 anonymous Remote parse 成功。
- 每次 `mineru parse --remote` 开始前。
- Local Parse Server 启动、状态或错误。
- 与认证、特性、限流和配额无关的解析错误。
- CLI help。
- 自定义或 self-hosted Remote URL 返回同名错误。

### 5. 只为 Official API 展示平台注册链接

Official API 的 canonical base URL 是:

```text
https://mineru.net/api
```

Official API Key 管理 URL 是:

```text
https://mineru.net/apiManage/token
```

只有当前配置的 Remote URL 被严格识别为 Official API 时，才可以展示该管理 URL 和 Official API 注册文案。判断应规范化无意义的尾部 `/`，但不能把任意自定义域名、代理、self-hosted server 或仅仅返回相同错误码的服务当成 Official API。

非官方 Remote Server 的注册和 API Key 签发机制未知。CLI 保留其原始成功或错误响应，但不提供 Official API 平台引导。

### 6. 普通文本输出

`mineru usage` 的普通文本输出完整展示:

- 当前配置的 Remote URL。
- `access_level`。
- UTC billing period。
- `current` 中的全部用量字段。
- `limits` 中的全部限制字段。

文件大小和保留期分别使用适合人类阅读的 MiB 和 day 单位；JSON 输出仍保留服务端原始单位和值。示例:

```text
Remote API Usage

Remote URL: https://mineru.net/api
Access level: anonymous
Billing period: 2026-07-14 00:00 UTC - 2026-07-15 00:00 UTC

Current
  Pages processed: 26
  Files processed: 0
  Jobs created: 8

Limits
  Max pages per file: 200
  Max file size: 200 MiB
  Max files per job: 200
  Max concurrent jobs: 500
  File retention: 30 days
```

普通文本模式在原有成功或错误信息之后展示可操作信息，但不回显当前 API Key:

```text
Manage or create an API Key:
https://mineru.net/apiManage/token

Set the API Key:
mineru config set parse_server.remote.api_key '<API_KEY>'
```

可选引导必须使用不会暗示注册为必需条件的文案。`invalid_api_key` 和 `feature_requires_api_key` 可以明确说明，继续当前受限操作需要有效 API Key。

### 7. JSON guidance 契约

错误驱动的 JSON 输出不得覆盖或向 `error.message` 拼接教程。引导使用顶层 `guidance` sibling field:

```json
{
  "error": {
    "type": "authentication_error",
    "code": "invalid_api_key",
    "message": "Remote authentication failed: user authenticate failed",
    "param": "parse_server.remote.api_key"
  },
  "guidance": {
    "type": "configure_official_api_key",
    "required": true,
    "message": "Configure a valid Official API Key to continue.",
    "url": "https://mineru.net/apiManage/token",
    "command": "mineru config set parse_server.remote.api_key '<API_KEY>'"
  }
}
```

Anonymous usage 的成功 JSON 在 `usage` 字段之外使用相同 guidance 结构:

```json
{
  "remote_url": "https://mineru.net/api",
  "usage": {
    "object": "usage",
    "access_level": "anonymous",
    "billing_period": {},
    "current": {},
    "limits": {}
  },
  "guidance": {
    "type": "configure_official_api_key",
    "required": false,
    "message": "An Official API Key is optional and enables registered access.",
    "url": "https://mineru.net/apiManage/token",
    "command": "mineru config set parse_server.remote.api_key '<API_KEY>'"
  }
}
```

规则:

- 成功的 usage JSON 使用固定结构，没有适用引导时返回 `"guidance": null`；错误 JSON 只在存在引导时增加该字段。
- `guidance.type` 是稳定的机器判断字段，调用方不依赖 `message` 分支。
- `required` 明确区分必须修复和可选增强，防止 Agent 把匿名访问误判为错误。
- `url` 和 `command` 分开，便于人类界面和 Agent 分别使用。
- `guidance` 是 CLI JSON composition 字段，不修改 Unified API 的 `error` envelope。
- 不复用 `ParseResponse.tip`、`ForgetPathResponse.warnings` 或 `DocContentResponse.next_request`。这些字段分别表示 parse 补充文本、操作风险和渐进读取续请求，语义更窄。

### 8. Remote 错误信号与客户端归一化

截至 2026-07-14，Official Remote API 的实测和现有记录如下:

| 信号 | Remote 行为 | 客户端状态 |
|------|-------------|------------|
| `access_level=anonymous` | `GET /v1/usage` 返回标准字段 | 可直接判断 |
| `invalid_api_key` | HTTP 401，body 仍可能使用 legacy `msgCode=A0202` / `msg=user authenticate failed` | API client 和 health probe 已归一化为 `invalid_api_key` |
| `feature_requires_api_key` | HTTP 403，标准 `error` envelope | API client 保留 `error.code` |
| `rate_limit_exceeded` | 已有真实 Remote parse 记录，包含标准 code 和 retry message | API client 和 doclib parse row 保留 code |
| `quota_exceeded` | 本 ADR 按 Remote API 返回标准 `error` envelope 处理 | API client 保留 `error.code` |

结构化错误必须沿以下路径保留:

```text
Remote HTTP response
  -> MinerU API client error
  -> doclib operation / parse row
  -> mineru CLI error JSON
```

HTTP 401 和 legacy Official API authentication body 继续归一化为:

```json
{
  "type": "authentication_error",
  "code": "invalid_api_key",
  "message": "Remote authentication failed: user authenticate failed",
  "param": "parse_server.remote.api_key"
}
```

`feature_requires_api_key`、`rate_limit_exceeded` 和 `quota_exceeded` 使用服务端标准 `error.code`，不通过英文 message 猜测错误类型。

## 替代方案

### 方案 A: CLI 直接请求 Remote `/v1/usage`

未采用。

这会让 CLI 直接读取 doclib runtime config 或复制配置解析逻辑，并形成第二套 HTTP、authentication 和错误归一化路径。

### 方案 B: 把 Local 和 Remote usage 合并到 `mineru usage`

未采用。

Local Parse Server 的启动期计数不是账号配额或计费数据。混合展示会让用户误以为本地解析也受 Remote 限制。

### 方案 C: 每次 anonymous Remote parse 都提示注册

未采用。

Anonymous 是合法使用方式。高频提示会制造注册为必要条件的错误印象，并污染 Agent 和脚本输出。

### 方案 D: 把配置教程拼入 `error.message`

未采用。

`error` 是稳定的机器可读错误 envelope。教程、URL 和命令需要独立演进，不应改变服务端错误语义或迫使 Agent 解析自由文本。

### 方案 E: 复用 `tip`、`warnings` 或 `next_request`

未采用。

- `tip` 是 parse response 中的自由文本字符串。
- `warnings` 表示操作风险。
- `next_request` 表示内容分页或续读请求。

API Key 引导跨 usage success 和 error response，需要独立、结构化且可表达 required/optional 的字段。

## 影响

### 对 doclib

- 增加查询 configured Remote `/v1/usage` 的 interface/client/server 能力。
- Remote URL 和 API Key 继续由 doclib config service 管理。
- Remote 标准错误码必须完整保留到调用方。

### 对 CLI

- 增加顶层 `mineru usage [--json]`。
- 普通输出使用 `Remote API Usage` 标题。
- 五种明确时机可以附加 Official API Key 引导；其他场景不主动提示。
- JSON error envelope 保持不变，`guidance` 作为 sibling field。

### 对安全和隐私

- 不回显已配置 API Key。
- `config show/get/set/unset` 继续使用现有敏感配置脱敏逻辑。
- 不把 API Key 放入 telemetry。
- 不向自定义 Remote Server 用户展示 Official API 注册机制。
- 引导命令只包含 `<API_KEY>` placeholder，不包含实际凭据。
- 不为 API Key 增加专用隐藏输入流程；现有 `config set <key> <value>` 契约保持不变。用户执行含真实 API Key 的命令时，
  shell 是否记录命令由其 shell 配置决定，这是保持通用配置接口简单性的已知取舍。

### 对兼容性

- Unified API `/v1/usage` 和 error envelope 不改变。
- `guidance` 是 MinerU CLI JSON composition 的新增可选字段。
- JSON 调用方应继续忽略未知 sibling fields。

## 后续动作

1. 实现统一的 doclib Remote API Key resolver，并迁移 parse 和 health probe。
2. 实现 doclib Remote usage contract 和 Remote HTTP 调用。
3. 实现 `mineru usage` 普通文本与 JSON 输出。
4. 实现 Official URL 判定和五种 API Key guidance 触发规则。
5. 增加 Remote error normalization、doclib contract、CLI JSON 和普通文本测试。
6. 更新 CLI、API、README、SKILL 和 E2E 文档。
