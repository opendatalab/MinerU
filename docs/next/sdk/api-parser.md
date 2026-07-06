# API-backed Parser: `MinerUApiParser`

状态: Draft
读者: SDK 开发者、parse-server 开发者、集成方
范围: 通过 v1 API 委托解析的 `DocumentParser` 实现
底稿: `../../../NEXT-SDK.md`

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
- 不隐式决定是否可上传远端；调用方必须显式传入 `api_url`。

## 构造参数

目标公开签名:

```python
class MinerUApiParser(DocumentParser):
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str | None = None,
        tier: str | None = None,
        wait: int = 600,
        timeout: int = 660,
    ) -> None: ...
```

字段:

| 参数 | 说明 |
|------|------|
| `api_url` | v1 API base URL，如 `https://mineru.net/api` 或 `http://127.0.0.1:16580/api`。 |
| `api_key` | Bearer token。Local Parse Server 未启用鉴权时可为空。 |
| `tier` | `medium`、`high` 或 `None`。`None` 表示 SDK 在 HTTP 请求中省略 `tier` 或发送 JSON `null`，让 v1 API 使用默认选择策略。不得传 `flash`，因为 v1 parse API 的质量解析不接受 flash 作为默认结果。 |
| `wait` | 创建 job 时的同步等待秒数。 |
| `timeout` | HTTP 请求超时。 |

## 行为流程

`parse(path)` 的流程:

1. 检查文件存在。
2. 判断 `api_url` 是否是本地地址。
3. 本地地址优先使用 `local` source。
4. 非本地地址使用 Uploads API: create upload -> PUT upload_url -> complete upload。
5. 调用 `POST /v1/parse/jobs`，请求 `output_formats:["middle_json"]`。
6. 如果 job 未完成，则轮询 `GET /v1/parse/jobs/{job_id}`。
7. 下载 `middle_json` output file。
8. 将返回的 middle JSON 转为 `ParseResult`。

本地 source:

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
| `None` | 省略或 `null` |
| `medium` | `medium` |
| `high` | `high` |

它不接受也不保存 backend 参数。backend 是本地 parser 层的高级实现概念，不应出现在 API-backed parser 的公开构造参数、实例属性或请求 payload 中。

## 输出格式

默认只请求 `middle_json`，因为 `ParseResult` 可以基于 pages 渲染 markdown、content list 和 images。

目标可选参数:

| 参数 | 说明 |
|------|------|
| `output_formats` | 高级用户可指定 API 产物。默认 `["middle_json"]`。 |
| `download_outputs` | 是否把 API 产物引用下载进本地对象。默认只下载构造 `ParseResult` 需要的 `middle_json`。 |

## 错误映射

API error envelope 应映射为 SDK exception:

| API code | SDK 行为 |
|----------|----------|
| `invalid_api_key` | 抛出认证错误。 |
| `file_too_large` | 抛出用户可修复错误。 |
| `quality_tier_unavailable` | 抛出引擎不可用错误。 |
| `rate_limit_exceeded` | 抛出可重试错误，并保留 `Retry-After`。 |
| `service_unavailable` | 抛出可重试错误。 |

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
