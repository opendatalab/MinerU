# `mineru usage`

状态: Accepted
范围: 查询当前配置的 Remote API 用量和限制

## 命令

```bash
mineru usage
mineru usage --json
```

该命令通过本地 doclib 调用当前配置 Remote API 的 `GET /v1/usage`，不统计 Local Parse Server 活动。

普通输出展示 Remote URL、access level、billing period、全部 current 字段和全部 limits 字段。JSON 使用 CLI 自己的响应结构，
包含 `remote_url`、嵌套的 `usage` 和可空的 `guidance`。

## API Key

Remote API Key 的解析优先级为：

```text
parse_server.remote.api_key > MINERU_API_KEY > anonymous
```

Anonymous 是合法 access level。只有当前 Remote URL 是 `https://mineru.net/api` 时，以下场景才展示 Official API Key 引导：

- anonymous usage，可选引导；
- `invalid_api_key`，必须修复；
- `feature_requires_api_key`，必须配置；
- anonymous `rate_limit_exceeded`，可选引导；
- anonymous `quota_exceeded`，可选引导。

JSON 错误中的 `guidance` 是 `error` 的 sibling；成功 usage 响应中的 `guidance` 与 `usage` 同级，没有适用引导时为 `null`。
自定义 Remote Server 不展示 Official API 注册链接或配置文案。

完整设计见 [ADR-0026](../decisions/0026-remote-usage-api-key-guidance.md)。
