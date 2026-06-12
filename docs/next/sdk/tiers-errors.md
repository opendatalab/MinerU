# Tier 与错误

状态: Draft
读者: SDK 开发者、CLI 开发者、核心开发者
范围: SDK 层 tier 语义、隐私策略、错误映射和重试建议
底稿: `../../../NEXT-SDK.md`

## Tier 语义

SDK 层必须使用与产品一致的 tier 语义:

| Tier | SDK 含义 |
|------|----------|
| `flash` | 用户显式请求快速 CPU-only 解析。不能作为默认选择的回退。 |
| `standard` | 本地或自部署 parse-server 可用的标准质量解析能力。 |
| `pro` | 当前最高质量解析能力；`mineru.net/api` 长期只提供该 tier。 |

完整产品语义见 [解析 Tier](../tiers.md)。

## 默认选择规则

SDK 中未指定 tier，或 Python 调用传 `tier=None` 时，必须遵循:

| 可用能力 | 默认选择结果 |
|----------|-------------|
| 只有 `standard` | `standard` |
| 只有 `pro` | `pro` |
| 同时发现 `standard` 和 `pro` | `pro` |
| 都没有 | 报错 |

默认选择不能变成 `flash`。如果用户想使用 `flash`，必须显式传入 `tier="flash"` 或 `backend="flash"`。

## Tool SDK 的 tier 处理

`mineru.parser.parse()` 的目标行为:

- 用户传 `tier`，SDK 解析为 backend。
- 用户传 `backend`，视为高级覆盖。
- `backend` 覆盖 `tier` 时应在文档中明确。
- 如果需要能力发现但当前环境无法发现，应返回明确错误。

目标映射:

| Tier | Backend |
|------|---------|
| `flash` | `flash` |
| `standard` | `pipeline` |
| `pro` | hybrid 默认高质量 backend |

`tier=None` 使用默认选择策略，结果为 `pro` 或 `standard`。

## Doclib SDK 的 tier 处理

`MineruClient.parse(tier=None)` 表示由 doclib 规则决定:

1. 先匹配 parsing rules。
2. 再使用默认 tier。
3. 如果用户显式 remote，则可使用 remote parse-server。
4. 如果用户未允许 remote，则只能使用本地能力。

doclib 的质量优先规则:

- 用户读取文档时默认最低接受 `standard`。
- 找不到 `standard` / `pro` 且不允许 remote 时，应报错。
- 不应擅自返回 `flash` 结果作为读取结果。

## 隐私策略

SDK 默认隐私优先:

| SDK | 默认行为 |
|-----|----------|
| `mineru.parser` | 只解析本地文件，不上传远端。 |
| `MinerUApiParser` | 调用方显式传入 `api_url`，由调用方承担远端许可语义。 |
| `MineruClient` | `remote=False`，不得上传到 remote parse-server 或 mineru.net。 |

只有显式 remote 许可才可以上传用户文件到远端。

## 错误层级

目标 SDK exception 基类是 `MineruError`，字段对齐 API error envelope:

```python
class MineruError(Exception):
    code: str
    message: str
    param: str | None
    type: str
```

推荐子类:

| 子类 | 场景 |
|------|------|
| `InvalidRequestError` | 参数、路径、页码、文件格式错误。 |
| `NotFoundError` | 文件、job、upload、model 不存在。 |
| `PermissionError_` | 权限或隐私策略不允许。 |
| `ConflictError` | 状态冲突。 |
| `EngineError` | 解析引擎不可用或解析失败。 |
| `ServerNotRunningError` | 本地 doclib 未运行。 |

## 错误映射

| code | SDK 分类 | 是否建议重试 |
|------|----------|--------------|
| `invalid_request` | InvalidRequestError | 否 |
| `page_range_invalid` | InvalidRequestError | 否 |
| `file_not_found` | NotFoundError | 否 |
| `file_too_large` | InvalidRequestError | 否 |
| `invalid_api_key` | Permission/Auth | 否 |
| `feature_requires_api_key` | PermissionError_ | 否 |
| `rate_limit_exceeded` | MineruError | 是，按 `Retry-After` |
| `engine_unavailable` | EngineError | 可稍后重试 |
| `parse_failed` | EngineError | 通常否，除非错误可恢复 |
| `parse_timeout` | EngineError | 是 |
| `server_not_running` | ServerNotRunningError | 启动 server 后重试 |
| `parse_server_unavailable` | MineruError | 是，或切换 local/remote |
| `quality_tier_unavailable` | EngineError | 启动 parse-server 或允许 remote |

## 重试策略

SDK 可以内建有限重试，但必须可配置:

- 网络连接临时失败: 可重试。
- `rate_limit_exceeded`: 按 `Retry-After`。
- `service_unavailable`: 指数退避。
- `parse_timeout`: 查询状态或重新提交由调用方决定。
- 参数错误、权限错误、文件不存在: 不重试。

## 未决问题

是否为 retryable 错误增加稳定字段、是否在 SDK exception 上暴露 `user_action`，集中维护在 [开放问题清单](../open-questions.md)。`quality_tier_unavailable` 应进入稳定错误映射。
