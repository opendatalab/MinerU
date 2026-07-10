# API-backed Parser: `MinerUApiParser`

状态: Draft
读者: SDK 开发者、parse-server 开发者、集成方
范围: 通过 v1 API 委托解析的 `DocumentParser` 实现
来源: 由旧 SDK 底稿迁移整理而来；旧底稿已归档删除

## 定位

`MinerUApiParser` 是 Tool SDK 中的 API-backed parser。它实现 `DocumentParser` 接口，但不直接调用本地解析后端，而是通过 v1 Unified API 创建 upload 和 parse job。

它的价值是让调用方使用同一个 parser 心智:

```python
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(
    api_url="https://mineru.net/api",
    api_key="...",
    tier="high",
)

result = parser.parse("report.pdf")
print(result.markdown())
```

适用场景:

- Python 调用方希望使用远端 `mineru.net` 解析能力。
- doclib worker 需要调用 local parse-server 或 remote parse-server。
- 用户希望复用 `DocumentParser` / `ParseResult` 接口，而不是直接操作 HTTP API。

非目标:

- 不提供 doclib 搜索、watch、配置能力。
- 不维护本地缓存。
- 不隐式决定是否可上传远端；调用方传入远端 `api_url`，或通过环境变量配置远端 API，即表示调用方已经允许上传到该 API。

## 构造参数

目标公开签名:

```python
class MinerUApiParser(DocumentParser):
    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        tier: str | None = None,
        include_images: bool = False,
        include_model_output: bool = False,
    ) -> None: ...
```

字段:

| 参数 | 说明 |
|------|------|
| `api_url` | v1 API base URL，如 `https://mineru.net/api` 或 `http://127.0.0.1:16580`。省略时读取 `MINERU_API_URL`，再退回默认官方 API。 |
| `api_key` | Bearer token。省略时读取 `MINERU_API_KEY`；Local Parse Server 未启用鉴权时可为空。 |
| `tier` | `flash`、`medium`、`high`、`xhigh` 或 `None`。`None` 表示 SDK 在 HTTP 请求中省略 `tier`，让 v1 API 使用默认选择策略；非 PDF/image 文件按 API Server 批量规则归一，见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。 |
| `include_images` | 是否从 zip 产物读取图片 sidecar，并挂载到 `ParseResult` 图片缓存。 |
| `include_model_output` | 是否请求并保留模型原始输出；开启时通过 zip 产物读取。 |

## 行为流程

`parse(path)` 的流程:

1. 检查文件存在。
2. 判断 `api_url` 是否是本地地址。
3. 本地地址先读取 `GET /v1/health`；只有 `features.sources` 包含 `local` 时才使用 `local` source。
4. 非本地地址，或本地 server 未允许 `local` source 时，使用 Uploads API: create upload -> PUT upload_url -> complete upload，再以 `file_id` source 创建解析任务。
5. 调用 `POST /v1/parse/jobs`；默认请求 `output_formats:["middle_json"]`，需要模型输出或图片缓存时请求 `["zip"]`。
6. 如果 job 未完成，则轮询 `GET /v1/parse/jobs/{job_id}`。
7. 普通模式下载 `middle_json` output file；zip 模式下载 `zip` output file。
8. zip 模式从 zip 中读取 middle JSON，并在 `include_images=True` 时读取 middle JSON 引用的图片 sidecar。
9. 将 middle JSON 转为 `ParseResult`。

### 传输重试

客户端对可安全重放的传输操作执行有限重试:

- health、upload 状态、job 轮询和产物下载使用 `GET`，可以重试。
- 向 `upload_url` 写入相同文件字节使用 `PUT`，可以重试。
- upload complete 的响应丢失后，客户端先查询 upload 状态；已经完成则读取其中的 File，仍为 pending 才再次 complete。

`POST /v1/parse/jobs` 在 API 提供幂等键契约前不会自动重试。该请求的响应丢失时，服务端可能已经创建任务；直接重放会产生重复任务和费用。

重试耗尽后，doclib 远端解析分别使用 `remote_timeout` 或 `remote_unreachable`，不把客户端传输错误归类为 `parse_failed`。远端 job 自身返回的结构化 `parse_failed` 保持原语义，客户端不根据错误消息文本猜测错误来源。

本地 source:

Local Parse Server 必须通过 `--allow-local-source` 开启，并在 `features.sources` 返回 `local`，SDK 才会使用该形式。

```json
{
  "source": {
    "type": "local",
    "path": "/absolute/path/report.pdf"
  }
}
```

远端 source:

```json
{
  "source": {
    "type": "file_id",
    "file_id": "file-..."
  }
}
```

## Tier 映射

`MinerUApiParser` 面向 v1 API 时只使用 tier 术语:

| SDK tier | API tier |
|----------|----------|
| `None` | 省略 |
| `flash` | `flash` |
| `medium` | `medium` |
| `high` | `high` |
| `xhigh` | `xhigh` |

它不接受也不保存 backend 参数。backend 是本地 parser 层的高级实现概念，不应出现在 API-backed parser 的公开构造参数、实例属性或请求 payload 中。

## 输出格式

默认只请求 `middle_json`。如果 `include_images=True` 或 `include_model_output=True`，则请求 `zip`，并从 zip 中恢复 `ParseResult` 所需的 middle JSON。API 不支持把 `images` 作为独立 `output_formats` 请求值。

目标可选参数:

| 参数 | 说明 |
|------|------|
| `include_images` | 请求 `zip` 并从其中读取图片 sidecar 到 `ParseResult` 图片缓存。 |
| `include_model_output` | 请求 `zip` 并尝试保留模型输出。 |

## 错误映射

API error envelope 应映射为 SDK exception:

| API code | SDK 行为 |
|----------|----------|
| `invalid_api_key` | 抛出认证错误。 |
| `file_too_large` | 抛出用户可修复错误。 |
| `quality_tier_unavailable` | 抛出引擎不可用错误。 |
| `rate_limit_exceeded` | 抛出可重试错误，并保留 `Retry-After`。 |
| `service_unavailable` | 抛出可重试错误。 |
| `remote_timeout` | 远端传输超时，有限重试后仍未恢复。 |
| `remote_unreachable` | 远端连接或协议错误，有限重试后仍未恢复。 |

当前实现有内部 `_V1APIError`。目标公开契约应复用 `mineru.errors.MineruError` 或统一 SDK exception hierarchy。

## 隐私约束

`MinerUApiParser` 是显式远端/本地 API client。它不会自己决定“是否可以 remote”。调用方只要传入 `https://mineru.net/api`，就表示允许上传到该 API。

doclib SDK 或 CLI 在调用它之前必须完成隐私策略判断:

- 用户未显式 remote 时，不应构造指向 `mineru.net` 的 `MinerUApiParser`。
- 用户指定 local parse-server 时，优先使用本地 URL。
- API-backed parser 只负责执行请求，不负责产品策略。

## 与 `mineru.parser.api_server`

`mineru.parser.api_server` 是 `MinerUApiParser` 可以连接的服务端实现之一。它提供 v1 Unified API 的 runtime 和 Pydantic contract。

文档关系:

- HTTP 规格见 [Unified API](../api.md)。
- `MinerUApiParser` 是该 HTTP 规格的 Python parser 封装。
- `api_server` 是该 HTTP 规格的本地无状态 server 实现。

## 公开入口

`MinerUApiParser` 保持现有命名和位置，公开入口为:

```python
from mineru.parser import MinerUApiParser
```

不新增 `mineru.parser.remote` 迁移路径。低层 `V1ApiClient` 是否公开仍集中维护在 [开放问题清单](../open-questions.md)。API-backed parser 的异常应收敛到统一 `MineruError` 体系。
