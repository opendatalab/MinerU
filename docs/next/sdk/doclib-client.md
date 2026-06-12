# Doclib SDK: `MineruClient`

状态: Draft
读者: SDK 开发者、`mineru` CLI 开发者、MCP/桌面端集成方
范围: 本地 doclib client 的公开能力、方法和错误模型
底稿: `../../../NEXT-SDK.md`

## 定位

`MineruClient` 是本地 doclib 的 Product SDK。它通过 Unix Domain Socket 连接 doclib server，使用本地文档库能力:

- 文件入库。
- 解析请求与缓存。
- 全文搜索和文件名搜索。
- 文件信息查询。
- watch、exclude、parsing rule 配置。
- server 状态和关闭。

它不是 v1 Unified API client。v1 API 面向 parse-server / mineru.net；doclib client 面向本机长期运行的文档库。

Doclib SDK 是 doclib 本地 API 的推荐承载方式。MinerU 项目内部除 doclib client 外，不直接通过 HTTP 协议访问 doclib API；外部客户端未来可以绕过 Python SDK，直接使用 doclib HTTP API 与 doclib server 交互。

## 构造参数

目标公开签名:

```python
from mineru.doclib.client import MineruClient

class MineruClient:
    def __init__(
        self,
        socket_path: str = "/tmp/mineru.sock",
        timeout: int = 60,
    ) -> None: ...
```

约束:

- 构造 client 不应启动 doclib server。
- 无法连接 UDS 时，方法调用抛出 `ServerNotRunningError`。
- client 应支持 `close()`，并最终支持 context manager。

示例:

```python
from mineru.doclib.client import MineruClient

client = MineruClient()
status = client.server_status()
```

## Parse 方法

```python
def parse(
    self,
    path: str,
    *,
    tier: str | None = None,
    pages: str | None = None,
    format: str = "markdown",
    force: bool = False,
    remote: bool = False,
) -> dict: ...
```

语义:

| 参数 | 说明 |
|------|------|
| `path` | 本地文件路径。 |
| `tier` | `flash`、`standard`、`pro` 或 `None`。`None` 表示使用 doclib 规则和默认策略。 |
| `pages` | 页码范围。 |
| `format` | 返回格式，当前主要是 `markdown`。 |
| `force` | 跳过已有 done 缓存并重新解析；可复用 active parse，只为未覆盖页创建新 parse。 |
| `remote` | 是否允许调用远端 parse-server。默认 `False`。远端 URL 和 API Key 来自 doclib config 或环境变量。 |

返回值当前是 dict，目标应稳定为 `ParseResponse`:

```python
class ParseResponse:
    sha256: str
    tier: str
    pages: str
    status: str
    cache_hit: bool
    wait_parse_ids: list[int]
    created_parse_ids: list[int]
    reused_parse_ids: list[int]
    markdown: str | None
    tip: str | None
```

状态:

| 状态 | 含义 |
|------|------|
| `pending` | 已入队。 |
| `parsing` | 正在解析。 |
| `done` | 已完成或命中缓存。 |
| `failed` | 解析失败。 |

## Parse 与 Doc 辅助方法

| 方法 | 说明 |
|------|------|
| `list_parses(...)` | 对应 `GET /parses`，按 ids、sha256、tier、status 或 pages 查询 parse records 和覆盖状态。 |
| `get_parse(parse_id)` | 对应 `GET /parses/{id}`，查询单条 parse record。 |
| `parse_content(sha256, tier, pages=None, format="markdown", output=None)` | 对应 `GET /docs/{sha256}/content`，从保存的 JSON 结果读取时转换。 |
| `invalidate(target="parses", path=None, sha256=None, tier=None)` | 对应 `POST /invalidate`，将已有解析结果标记为失效；不自动触发重新解析。 |

`parse_status(sha256, tier)` 不再作为 NEXT 稳定方法；调用方应使用 `list_parses(sha256=..., tier=...)`。`parse_content()` 是 doclib 层能力，不等同于 v1 Files API 的 `GET /v1/files/{id}/content`。

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
| `search()` | `SearchResponse`，包含全文结果、snippet、paths、tier；支持 `file_type`、`tier`、`min_tier` 过滤。 |
| `find()` | `FindResponse`，包含文件名搜索结果；支持 `ext` 过滤。 |
| `info()` | `FileInfo`，包含文件元信息、doc metadata 和 parse tiers。 |

`search()` 的 `paths` 优先返回 active files；如果某个已索引 doc 没有任何 active file，则 fallback 返回非 active file paths。搜索结果可信度只通过 `tier` 表达。

## Config 方法

doclib client 暴露运行时配置能力:

| 方法 | 说明 |
|------|------|
| `config_show()` | 返回运行时 KV 配置。 |
| `config_watch_add(path, removable=False, label=None)` | 添加 watch target。 |
| `config_watch_list()` | 列出 watch targets。 |
| `config_watch_rm(path)` | 删除 watch target。 |
| `config_exclude_add(pattern, priority=0)` | 添加排除规则。 |
| `config_exclude_list()` | 列出排除规则。 |
| `config_exclude_rm(rule_id)` | 删除排除规则。 |
| `config_parsing_rules_add(...)` | 添加解析规则。 |
| `config_parsing_rules_list()` | 列出解析规则。 |
| `config_parsing_rules_rm(rule_id)` | 删除解析规则。 |

这些方法对应 [配置体系](../config.md) 中的运行时配置，不负责启动前配置。

## Server 方法

| 方法 | 说明 |
|------|------|
| `server_status()` | 返回 doclib、队列、watch 和 parse-server 状态。 |
| `shutdown()` | 请求 doclib server 关闭。 |

`server_status()` 是 CLI、MCP 和桌面端判断本地能力的主入口。

## 错误模型

doclib client 当前从服务端 error envelope 还原 `MineruError`。目标行为:

| 场景 | 错误 |
|------|------|
| UDS 连接失败 | `ServerNotRunningError`。 |
| 服务端返回 error envelope | `MineruError`，保留 `code`、`message`、`param`。 |
| 服务端返回非 JSON | `MineruError("internal_error", ...)`。 |

SDK 不应把 doclib 内部异常原样泄漏给调用者。

## 隐私策略

`MineruClient.parse(remote=False)` 是默认行为，表示不得上传到远端。此时 doclib 只能使用本地 flash 或 local parse-server。

只有调用方显式设置 `remote=True`，doclib 才可以连接 remote parse-server 或 mineru.net。即便远端可用，SDK 也不能自动越过该开关。

## 当前与目标差异

| 主题 | 当前 | 目标 |
|------|------|------|
| 包路径 | `mineru.doclib.client.MineruClient` | 保持；可选增加 `mineru.client` 别名。 |
| 返回类型 | 多数方法返回 `dict` | 稳定为 typed response，保留 dict 兼容。 |
| context manager | 仅 `close()` | 支持 `with MineruClient() as client`。 |
| API key 参数 | `parse()` 接收但请求模型未显式声明 | 补齐 doclib request 类型或从配置读取。 |

## 未决问题

Product SDK 入口、async client 和 `parse()` 输出参数命名，集中维护在 [开放问题清单](../open-questions.md)。
