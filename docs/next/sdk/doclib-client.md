# Doclib SDK: `DoclibClient`

状态: Draft
读者: SDK 开发者、`mineru` CLI 开发者、MCP/桌面端集成方
范围: 本地 doclib client 的公开能力、方法和错误模型
来源: 由旧 SDK 底稿迁移整理而来；旧底稿已归档删除

## 定位

`DoclibClient` 是本地 doclib 的 Product SDK。它通过 Unix Domain Socket 或 TCP loopback 连接 doclib server，使用本地文档库能力:

- 文件入库。
- 解析请求与缓存。
- 全文搜索和文件名搜索。
- 文件信息查询。
- watch、exclude、parsing rule 配置。
- server 状态和关闭。

它不是 v1 Unified API client。v1 API 面向 parse-server / mineru.net；doclib client 面向本机长期运行的文档库。

Doclib SDK 是 doclib 本地 API 的推荐承载方式。MinerU 项目内部除 doclib client 外，不直接通过 HTTP 协议访问 doclib API；外部客户端未来可以绕过 Python SDK，直接使用 doclib HTTP API 与 doclib server 交互。

当前实现类名是 `DoclibClient`，位于 `mineru.doclib.client`。是否额外提供 `MineruClient` 或顶层 `mineru.client` 别名仍是开放问题。

## 构造参数

目标公开签名:

```python
from pathlib import Path

from mineru.doclib.client import DoclibClient

class DoclibClient:
    def __init__(
        self,
        endpoint_path: str | Path | None = None,
        socket_path: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        api_prefix: str = "/api/v1",
    ) -> None: ...
```

约束:

- 构造 client 不应启动 doclib server。
- 默认通过 `$MINERU_HOME/doclib.endpoint.json` 发现 doclib endpoint。
- `socket_path` 表示显式 UDS endpoint；`base_url` 只表示显式 TCP endpoint，例如 `http://127.0.0.1:15980`。
- `socket_path` 和 `base_url` 不能同时传入。
- 无法连接任何候选 endpoint 时，方法调用抛出 `ServerNotRunningError`。
- client 当前支持 `close()`；是否增加 context manager 仍是迁移任务。

示例:

```python
from mineru.doclib.client import DoclibClient

client = DoclibClient()
status = client.get_server_status()
```

## Parse 方法

当前稳定入口使用 typed request / response model:

```python
from mineru.doclib.types import ParseRequest, ParseResponse

def ensure_parse(self, request: ParseRequest) -> ParseResponse: ...
```

`ParseRequest` 语义:

| 参数 | 说明 |
|------|------|
| `path` | 本地文件路径。 |
| `tier` | `flash`、`medium`、`high`、`xhigh` 或 `None`。`None` 表示使用 doclib 规则和默认策略；非 PDF/image 单文件未指定 tier 时归一为 `flash`，显式质量 tier 报错。 |
| `page_range` | 页码范围。 |
| `force` | 跳过已有 done 缓存并重新解析；可复用 active parse，只为未覆盖页创建新 parse。 |
| `remote` | 是否允许调用远端 parse-server。默认 `False`。远端 URL 来自 doclib config；API Key 优先来自 doclib config，未配置时使用环境变量。 |

返回值是 typed `ParseResponse`:

```python
class ParseResponse:
    sha256: str
    short_id: str | None
    tier: str
    page_range: str
    status: Literal["pending", "done"]
    cache_hit: bool
    wait_parse_ids: list[int]
    created_parse_ids: list[int]
    reused_parse_ids: list[int]
    tip: str | None
```

状态:

| 状态 | 含义 |
|------|------|
| `pending` | 已入队。 |
| `parsing` | 正在解析。 |
| `done` | 已完成或命中缓存。 |
| `failed` | 解析失败。 |

当前没有稳定的 `DoclibClient.parse(path, ...)` convenience wrapper。CLI 会自行构造 `ParseRequest`，然后调用 `ensure_parse()`，再按需要调用内容读取接口。

## Parse 与 Doc 辅助方法

| 方法 | 说明 |
|------|------|
| `list_parses(...)` | 对应 `GET /parses`，按 ids、sha256、tier、status 或 page_range 查询 parse records 和覆盖状态。 |
| `get_parse(parse_id)` | 对应 `GET /parses/{id}`，查询单条 parse record。 |
| `get_doc_content(doc_ref, tier=..., page_range=None, after=None, limit=..., format=...)` | 对应 `GET /docs/{doc_ref}/content`，从保存的 JSON 结果读取时转换。 |
| `read_content(locator, context=..., limit=..., format=...)` | 对应 `GET /content`，按 Agent locator 读取内容或图片。 |
| `export_doc_content(doc_ref, request)` | 对应 `POST /docs/{doc_ref}/exports`，导出结构化内容。 |
| `invalidate(request)` | 对应 `POST /invalidate`，将已有解析结果标记为失效；不自动触发重新解析。 |

`parse_status(sha256, tier)` 不再作为 NEXT 稳定方法；调用方应使用 `list_parses(doc_ref=..., tier=...)` 或 `get_parse(parse_id)`。`get_doc_content()` 是 doclib 层能力，不等同于 v1 Files API 的 `GET /v1/files/{id}/content`。

`force=True` 与 `invalidate()` 不等价。`force=True` 跳过 done cache，并通过 `wait_parse_ids` 等待复用或新建的 active parse；旧结果仍然有效。`invalidate()` 会让旧结果退出缓存命中、读取合并、搜索刷新和 compaction 选择。

## Search 与 Info

```python
def search(
    self,
    query: str,
    file_type: str | None = None,
    tier: Tier | None = None,
    min_tier: Tier | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict: ...

def find(self, query: str, ext: str | None = None, limit: int = 50) -> dict: ...

def info(self, file_path: str) -> dict: ...
```

目标响应类型:

| 方法 | 响应 |
|------|------|
| `search()` | `SearchResponse`，包含全文结果、snippet、files、tier；支持 `file_type`、`tier`、`min_tier` 过滤。 |
| `find()` | `FindResponse`，包含文件名搜索结果；支持 `ext` 过滤。 |
| `get_file_by_path()` | `FileInfoResponse`，包含文件元信息、doc metadata 和 parse tiers。 |

`search()` 的 `files` 返回与文档 SHA 关联的全部 file aliases，按 file id 降序排列；每项包含 path、filename、ext 和 status。已索引的 orphan 文档使用空列表。active 优先只属于非 JSON CLI 展示策略，不改变 SDK 响应。搜索结果可信度只通过 `tier` 表达。

## Remote API 方法

| 方法 | 响应 |
|------|------|
| `get_remote_usage()` | `RemoteUsageResponse`，返回当前配置 Remote API 的 access level、billing period、current usage 和 limits。 |

该方法通过 doclib 读取 Remote URL 和有效 API Key，并调用本地 doclib route `GET /api/v1/remote-usage`。它不返回 Local Parse
Server 运行统计。

## Config 方法

Doclib client 暴露运行时配置能力:

| 方法 | 说明 |
|------|------|
| `get_config()` / `get_config_key()` | 返回运行时 KV 配置。 |
| `set_config()` / `unset_config()` | 更新或删除运行时 KV 配置。 |
| `add_watch()` / `list_watches()` / `remove_watch()` | 管理 watch target。 |
| `add_exclude_rule()` / `list_exclude_rules()` / `remove_exclude_rule()` | 管理排除规则。 |
| `add_parsing_rule()` / `list_parsing_rules()` / `remove_parsing_rule()` | 管理解析规则。 |

这些方法对应 [配置体系](../config.md) 中的运行时配置，不负责启动前配置。

## Server 方法

| 方法 | 说明 |
|------|------|
| `get_server_status()` | 返回 doclib、队列、watch 和 parse-server 状态。 |
| `shutdown_server()` | 请求 doclib server 关闭。 |

`get_server_status()` 是 CLI、MCP 和桌面端判断本地能力的主入口。

## 错误模型

doclib client 当前从服务端 error envelope 还原 `MineruError`。目标行为:

| 场景 | 错误 |
|------|------|
| UDS / TCP 连接失败 | `ServerNotRunningError`。 |
| 服务端返回 error envelope | `MineruError`，保留 `code`、`message`、`param`。 |
| 服务端返回非 JSON | `MineruError("internal_error", ...)`。 |

SDK 不应把 doclib 内部异常原样泄漏给调用者。

## 隐私策略

`ParseRequest.remote=False` 是默认行为，表示不得上传到远端。此时 doclib 只能使用本地 flash 或 local parse-server。

只有调用方显式设置 `ParseRequest.remote=True`，doclib 才可以连接 remote parse-server 或 mineru.net。即便远端可用，SDK 也不能自动越过该开关。

## 当前与目标差异

| 主题 | 当前 | 目标 |
|------|------|------|
| 包路径 | `mineru.doclib.client.DoclibClient` | 保持；是否增加 `MineruClient` / `mineru.client` 别名仍未决。 |
| 返回类型 | 已返回 typed Pydantic response | 保持；是否增加 `raw=True` 兼容层未决。 |
| 高层 parse helper | 当前稳定入口是 `ensure_parse(ParseRequest)` | 是否增加 `parse(path, ...)` convenience wrapper 未决。 |
| context manager | 仅 `close()` | 是否支持 `with DoclibClient() as client` 未决。 |

## 未决问题

Product SDK 入口、async client 和高层 parse helper 命名，集中维护在 [开放问题清单](../open-questions.md)。
