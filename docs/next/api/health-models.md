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
    "webhook": true,
    "output_formats": ["markdown", "middle_json", "content_list", "structured_content", "html", "latex", "docx", "zip"],
    "sources": ["file_id", "url", "inline"]
  }
}
```

响应字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `status` | string | 是 | 固定为 `"ok"`。如果服务不可用，应由 HTTP 状态码表达，而不是返回其他 status。 |
| `version` | string | 是 | API 服务版本。 |
| `features` | object | 是 | 当前部署支持的外围能力。 |
| `features.webhook` | bool | 是 | 是否支持 Webhook 回调。 |
| `features.output_formats` | array | 是 | 当前部署支持的 parse job `output_formats` 值。 |
| `features.sources` | array | 是 | 当前部署允许的 parse job source 类型。 |

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
      "id": "standard",
      "description": "Standard parsing for most documents.",
      "current_model": "MinerU2.5-Pro-2605-1.2B"
    }
  ]
}
```

Tier 对象字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 当前服务提供的真实质量 tier，例如 `basic`、`standard` 或 `advanced`。 |
| `description` | string | 是 | 档位说明。 |
| `current_model` | string 或 null | 是 | 当前 tier 背后的模型 ID。 |

## Tier 选择语义

v1 协议中面向用户读取文档的质量档位以 [解析 Tier](../tiers.md) 为准。具体服务当前支持哪些 tier，以 `/v1/tiers` 返回为准。

请求创建 parse job 时，客户端可以省略 `tier` 或传 `null`，表示使用 [解析 Tier](../tiers.md) 中定义的默认选择策略。对 PDF/image 这类支持多 tier 的输入，默认选择不能等价于 `flash`；Office/HTML 按 [ADR-0024](../decisions/0024-file-type-tier-normalization.md) 的批量规则归一为 `flash`，text 不作为解析输入。

更完整产品语义见 [解析 Tier](../tiers.md)。

## 本地 Server 差异

Local Parse Server 的 `health` 通常返回:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "features": {
    "webhook": false,
    "output_formats": ["markdown", "middle_json", "content_list", "structured_content", "zip"],
    "sources": ["file_id", "url", "inline"]
  }
}
```

如果 Local Parse Server 启动时开启 `--allow-local-source`，`features.sources` 额外包含 `local`。

本地实现可以额外返回运维字段，例如:

| 字段 | 类型 | 说明 |
|------|------|------|
| `parser_version` | string | 本地解析器版本。 |
| `models` | object | 各解析档位健康状态，例如 `{"basic":"ok","standard":"ok","advanced":"missing"}`。 |

这些额外字段只供运维和调试使用，通用客户端不能依赖。

本地 `GET /v1/models` 和 `GET /v1/tiers` 必须反映当前 server 实际可用能力。常见组合:

| 本地 api-server 进程能力 | `/v1/tiers` 应返回 |
|-------------------------|-------------------|
| `--tier flash` | `flash`；默认选择请求应失败，除非显式请求 Flash。 |
| `--tier basic` | `flash`、`basic` |
| `--tier basic --no-flash` | `basic` |
| `--tier standard` | `flash`、`basic`、`standard`、`advanced` |
| `--tier standard --no-flash` | `basic`、`standard`、`advanced` |

本地或兼容服务应以 `/v1/tiers` 返回值作为能力发现事实。客户端不应根据启动参数猜测具体能力。

如果本地 server 支持 `flash`，它可以用于显式 `flash`、仅支持 flash tier 的输入归一化、watch 或索引机制，但不应出现在 PDF/image 面向用户质量解析的默认选择候选中。

`mineru.net/api` 在相当长时间内只提供 `standard`。远端 `/v1/tiers` 因此应返回 `standard`；省略 `tier` 或传 `null` 等价于使用 `standard`。
