# Webhooks

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: Parse Job 异步回调协议
底稿: `../../../NEXT-API.md`

## 可用性

Webhook 是官方远程 API 能力。本地 Local Parse Server 默认不支持 Webhook。客户端应先通过 `GET /v1/health` 的 `features.webhook` 判断是否可用。

Webhook 只面向 registered 请求。anonymous 请求传入 `callback` 时，官方 API 返回 `403 feature_requires_api_key`。

## 创建回调

创建 parse job 时传入 `callback`:

```json
{
  "callback": {
    "url": "https://your.app/mineru-webhook",
    "secret": "abc123"
  }
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `url` | string | 是 | 接收回调的 HTTP 或 HTTPS URL，必须支持 POST。 |
| `secret` | string | 是 | 当前 job 的签名密钥，由英文字母、数字、下划线组成，不超过 64 字符。 |

`secret` 与 job 生命周期绑定，不会在 `GET /v1/parse/jobs/{job_id}` 响应中返回。

## 回调请求

任务进入终态后，官方 API 向 `callback.url` 发送 POST:

```http
POST https://your.app/mineru-webhook
Content-Type: application/json
```

```json
{
  "checksum": "a1b2c3d4e5f6...",
  "content": "{\"event\":\"job.completed\",\"job_id\":\"job_01HXYZ...\",\"status\":\"completed\",\"occurred_at\":\"2026-05-21T08:30:48Z\"}"
}
```

字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `checksum` | string | `SHA256(uid + secret + content)`，用于防篡改校验。 |
| `content` | string | JSON 字符串，接收方需要自行解析。 |

`content` 解析后至少包含:

| 字段 | 说明 |
|------|------|
| `event` | `job.completed`、`job.failed`、`job.canceled` 或 `job.partial`。 |
| `job_id` | Job ID。 |
| `status` | Job 终态。 |
| `occurred_at` | 事件发生时间，ISO-8601 UTC。 |

接收端应先校验 `checksum`，再解析 `content`。收到 webhook 后，仍建议调用 `GET /v1/parse/jobs/{job_id}` 拉取完整结果。

## 重试

接收端返回 HTTP 200 视为成功。其他状态码、连接失败或超时都视为失败。

官方 API 最多重试 5 次，使用递增退避间隔。5 次均失败后不再推送。Webhook 失败不改变 job 的终态。

## 本地 Server 差异

Local Parse Server 不需要实现 Webhook。它应在 `health.features.webhook` 中返回 `false`。

如果客户端仍传入 `callback`，本地实现应选择以下一种明确行为:

- 返回 `400 invalid_request`，提示当前部署不支持 webhook。
- 在兼容模式下忽略 `callback`，但必须在文档和日志中明确说明不会发生回调。

本地 server 不能让用户误以为回调会发生。
