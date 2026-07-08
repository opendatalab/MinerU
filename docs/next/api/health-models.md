# Health、Models 与 Tiers

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: 服务健康检查、模型发现、解析档位发现
来源: 由根目录旧 Unified API 底稿迁移整理而来

## GET `/v1/health`

健康检查无需鉴权。它只表示 API 服务是否可达，以及当前部署公开哪些外围能力；它不要求对每个下游模型执行深度推理探测。

请求:

```http
GET /v1/health HTTP/1.1
```

官方 API 响应:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "features": {
    "sse": true,
    "webhook": true
  }
}
```

响应字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `status` | string | 是 | 固定为 `"ok"`。如果服务不可用，应由 HTTP 状态码表达，而不是返回其他 status。 |
| `version` | string | 是 | API 服务版本。 |
| `features` | object | 是 | 当前部署支持的外围能力。 |
| `features.sse` | bool | 是 | 是否支持 Parse Job SSE 事件流。 |
| `features.webhook` | bool | 是 | 是否支持 Webhook 回调。 |

错误:

| HTTP | code | 场景 |
|------|------|------|
| 503 | `service_unavailable` | 服务启动中或无法处理请求。 |

## GET `/v1/models`

列出当前服务可发现的模型。该 endpoint 对齐 OpenAI Models API 的基础形态，无需鉴权。anonymous 与 registered 看到同一份模型列表；是否可使用某个能力由具体 endpoint 和 access level 校验。

请求:

```http
GET /v1/models HTTP/1.1
```

响应:

```json
{
  "object": "list",
  "data": [
    {
      "id": "pipeline",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "Pipeline-based parsing model."
    },
    {
      "id": "MinerU2.5-Pro-2605-1.2B",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "VLM-based high-accuracy parsing model."
    }
  ]
}
```

模型对象字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 模型 ID。 |
| `object` | string | 是 | 固定为 `"model"`。 |
| `created` | integer | 是 | Unix 秒级时间戳，表示模型首次上线或被当前部署注册的时间。 |
| `owned_by` | string | 是 | 官方 API 固定为 `"mineru"`；自部署可使用组织名。 |
| `description` | string | 否 | 模型说明。 |

## GET `/v1/models/{model}`

查询单个模型。

请求:

```http
GET /v1/models/MinerU2.5-Pro-2605-1.2B HTTP/1.1
```

响应:

```json
{
  "id": "MinerU2.5-Pro-2605-1.2B",
  "object": "model",
  "created": 1700000000,
  "owned_by": "mineru",
  "description": "VLM-based high-accuracy parsing model."
}
```

错误:

| HTTP | code | 场景 |
|------|------|------|
| 404 | `model_not_found` | 模型不存在。 |

## GET `/v1/tiers`

列出当前服务提供的解析档位。客户端应优先使用 tier，而不是绑定具体模型 ID。服务端可以在不破坏客户端的情况下升级某个 tier 背后的模型版本。

请求:

```http
GET /v1/tiers HTTP/1.1
```

官方 API 响应:

```json
{
  "object": "list",
  "data": [
    {
      "id": "high",
      "description": "High-accuracy VLM-based parsing.",
      "current_model": "MinerU2.5-Pro-2605-1.2B"
    }
  ]
}
```

Tier 对象字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 当前服务提供的真实质量 tier，例如 `medium` 或 `high`。 |
| `description` | string | 是 | 档位说明。 |
| `current_model` | string 或 null | 是 | 当前 tier 背后的模型 ID。 |

## Tier 选择语义

`medium` 和 `high` 是 v1 协议中面向用户读取文档的真实质量档位。具体服务当前支持哪些 tier，以 `/v1/tiers` 返回为准。

请求创建 parse job 时，客户端可以省略 `tier` 或传 `null`，表示使用默认选择策略:

| 可发现能力 | 默认选择结果 |
|------------|---------------|
| 只有 `medium` | `medium` |
| 只有 `high` | `high` |
| 同时有 `medium` 和 `high` | `high` |
| 没有 `medium` 和 `high` | 报错 |

默认选择策略在任何语境下都不能等价于 `flash`。`flash` 是 watch、发现、索引阶段使用的快速 CPU 引擎，不是用户读取文档时的默认质量档位。

更完整产品语义见 [解析 Tier](../tiers.md)。

## 本地 Server 差异

Local Parse Server 的 `health` 通常返回:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "features": {
    "sse": false,
    "webhook": false
  }
}
```

本地实现可以额外返回运维字段，例如:

| 字段 | 类型 | 说明 |
|------|------|------|
| `parser_version` | string | 本地解析器版本。 |
| `models` | object | 各后端健康状态，例如 `{"medium":"ok","high":"missing"}`。 |

这些额外字段只供运维和调试使用，通用客户端不能依赖。

本地 `GET /v1/models` 和 `GET /v1/tiers` 必须反映当前 server 实际可用能力。常见组合:

| 本地 api-server 进程能力 | `/v1/tiers` 应返回 |
|-------------------------|-------------------|
| Medium | `medium` |
| High | `high` |
| 只有 Flash | 不应把 `flash` 作为用户质量解析 tier；默认选择请求应失败。 |

本地或兼容服务应以 `/v1/tiers` 返回值作为能力发现事实。它可以只暴露一个 tier，也可以暴露多个 tier；客户端不应假设具体进程模型。

如果本地 server 支持 `flash`，它可以在内部 watch 或索引机制中使用，但不应出现在面向用户质量解析的默认选择候选中。

`mineru.net/api` 在相当长时间内只提供 `high`。远端 `/v1/tiers` 因此应返回 `high`；省略 `tier` 或传 `null` 等价于使用 `high`。
