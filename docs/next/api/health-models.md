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
    "task_output_formats": ["default", "docx", "html", "latex"],
    "sources": ["upload", "url"]
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
| `features.task_output_formats` | array | 是 | 当前部署支持的 Tasks `output_formats` 值。 |
| `features.sources` | array | 是 | 当前部署允许的 Tasks 输入类型（upload/url）。 |

错误:

| HTTP | code | 场景 |
|------|------|------|
| 503 | 模型预加载错误码 | 模型预加载未成功，服务尚不能处理请求。本地实现返回 `model_preload_dependency_missing`、`model_preload_files_missing`、`model_preload_device_unavailable` 或 `model_preload_failed`，使用统一的 `error.code/message` 结构。 |

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
      "id": "mineru-4.0.11",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mineru",
      "description": "MinerU virtual model. PDF/image quality is selected per request via tier; native documents always use flash. this id identifies the server version, not a specific engine."
    }
  ]
}
```

- `/v1/models` 只广告一个**虚拟模型**：id 为 `mineru-<服务版本>`，标识服务端版本，不对应任何可单独调用的引擎。解析质量始终通过 Task 的 `tier` 字段选择，模型 id 不参与路由。
- `created` 为该部署的注册时间（本地 server 取进程启动时刻），同一部署内多次请求返回相同值。
- 模型列表与启动 `--tier` 上限无关：档位增减只反映在 `/v1/tiers`，虚拟模型恒在列表中。
- 服务端升级后 id 随版本变化（如 `mineru-4.0.11` → `mineru-4.0.12`）。客户端应重新发现，不要缓存 id 硬编码进请求。
- 历史上广告过的 `MinerU-Flash`、`Hybrid-Basic`、`MinerU-HTML`、`MinerU2.5-Pro-2605-1.2B` 已从列表移除；这些名字从来不是可传入的参数。其中 `MinerU2.5-Pro-2605-1.2B` 仍是 standard/advanced 档背后的真实 VLM 权重仓库名，只是不再出现在本 endpoint。

模型对象字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 虚拟模型 ID，`mineru-<服务版本>`。 |
| `object` | string | 是 | 固定为 `"model"`。 |
| `created` | integer | 是 | Unix 秒级时间戳，表示被当前部署注册的时间。 |
| `owned_by` | string | 是 | 官方 API 固定为 `"mineru"`；自部署可使用组织名。 |
| `description` | string | 否 | 模型说明，注明虚拟模型语义与 tier 路由方式。 |

## GET `/v1/models/{model}`

查询单个模型。

请求:

```http
GET /v1/models/mineru-4.0.11 HTTP/1.1
```

响应:

```json
{
  "id": "mineru-4.0.11",
  "object": "model",
  "created": 1700000000,
  "owned_by": "mineru",
  "description": "MinerU virtual model. PDF/image quality is selected per request via tier; native documents always use flash. this id identifies the server version, not a specific engine."
}
```

仅接受当前部署广告的虚拟模型 id（精确匹配），其余一律 404；服务端升级后旧版本 id 同样返回 404，客户端应重新发现。

错误:

| HTTP | code | 场景 |
|------|------|------|
| 404 | `model_not_found` | 模型不存在。 |

## GET `/v1/tiers`

列出当前服务允许 PDF/图片选择的解析档位；所有服务均支持现有原生文档格式并固定使用 flash。客户端应优先使用 tier，而不是绑定具体模型 ID。服务端可以在不破坏客户端的情况下升级某个 tier 背后的模型版本。

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
      "description": null,
      "current_model": "mineru-4.0.11"
    }
  ]
}
```

Tier 对象字段:

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 当前服务对 PDF/图片开放的质量 tier，例如 `basic`、`standard` 或 `advanced`。 |
| `description` | string | 是 | 档位说明。 |
| `current_model` | string 或 null | 是 | 当前 tier 背后的模型 ID，与 `/v1/models` 广告的虚拟模型一致：各档共用同一 id，差异由 `id`（档位）表达。 |

## Tier 选择语义

v1 协议中面向用户读取文档的质量档位以 [解析 Tier](../tiers.md) 为准。具体服务当前支持哪些 tier，以 `/v1/tiers` 返回为准。

请求创建 Task 时，客户端可以省略 `tier` 或传 `null`，表示使用 [解析 Tier](../tiers.md) 中定义的默认选择策略。对 PDF/image 按 `standard -> basic -> flash -> advanced` 选择默认值，仅开放 advanced 时自动选择 advanced；OFD/EPUB/Office/HTML/MHTML/CSV 按 [ADR-0024](../decisions/0024-file-type-tier-normalization.md) 与 [ADR-0028](../decisions/0028-csv-structured-flash-parsing.md) 的批量规则归一为 `flash`，其它 text 不作为解析输入。

更完整产品语义见 [解析 Tier](../tiers.md)。

## 本地 Server 差异

Local Parse Server 的 `health` 通常返回:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "features": {
    "webhook": false,
    "task_output_formats": ["default", "docx", "html", "latex"],
    "sources": ["upload", "url"]
  }
}
```


本地 `GET /v1/models` 和 `GET /v1/tiers` 必须反映当前 server 实际可用能力。常见组合:

| 本地 api-server 配置 | `/v1/tiers` 应返回 |
|----------------------|-------------------|
| `--tier flash` | `flash` |
| `--tier basic` | `flash`、`basic` |
| `--tier standard` | `flash`、`basic`、`standard` |
| `--tier advanced` | 全部四档；Tasks 默认 standard |
| `--tier auto` 或省略 | 根据硬件推荐的最高档及全部较低档位 |
| Python `create_app(tiers={"standard"})` / SaaS 固定策略 | 仅 `standard`；原生文件仍可解析并返回 flash |

本地或兼容服务应以 `/v1/tiers` 返回值作为能力发现事实。客户端不应根据启动参数猜测具体能力。

本地 `GET /v1/models` 恒返回当前版本的单一虚拟模型。经 Router 聚合多 worker 时，不同版本的 worker 会各自贡献 `mineru-<version>` 条目：Router 的 `/v1/models` 出现多个 `mineru-*` id 表示集群内存在版本漂移（滚动升级未完成），属预期诊断信号而非错误。

公开列表中的 flash 只表示 PDF/图片允许显式选择 Flash。原生文档是所有正常可用服务的必备解析能力，不依赖该列表，也不增加独立能力字段。

`mineru.net/api` 在相当长时间内只提供 `standard`。远端 `/v1/tiers` 因此只返回 `standard`；PDF/图片省略 `tier` 或传 `null` 使用 standard，原生文档固定 flash。
