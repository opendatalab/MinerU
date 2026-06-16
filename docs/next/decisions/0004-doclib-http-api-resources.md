# ADR-0004: Doclib HTTP API 资源模型

状态: Accepted
日期: 2026-06-09
相关文档: ../architecture.md, ../workflows.md, ../sdk/doclib-client.md, 0003-parse-request-wait-batches.md

## 背景

doclib server 的本地 HTTP API 需要同时服务 CLI、SDK、桌面端和编程 Agent。当前代码中的 `/parse`、`/parse/status`、`/parse/content`、`/parse/invalidate` 更接近 RPC 风格，且 parse 相关能力容易分散:

- 发起解析请求。
- 查询解析记录状态。
- 查询某个文档在某个 tier 下的页码覆盖情况。
- 读取解析内容。
- 失效已有解析缓存。

下一版需要使用一组更稳定、更成体系的资源路径。

## 决策

doclib HTTP API 使用以下核心资源:

```http
POST /parses
GET  /parses
GET  /parses/{id}

GET  /docs
GET  /docs/by-path?path=...
GET  /docs/{sha256}
GET  /docs/{sha256}/content

GET  /search
POST /invalidate
```

### `parses`

`parse` 在 doclib API 中表示一条解析记录。一个 parse record 对应一个 `sha256 + tier + page_range` 的解析批次，也对应 `parses` 表中的一行。

`POST /parses` 的语义是 ensure parse records:

- 普通请求命中 done cache 时，可以不创建新 parse record。
- 请求页码已被 `pending` / `parsing` parse 覆盖时，复用 active parse 并提升 priority。
- 未覆盖页码创建新的 parse record。
- `--force` 跳过 done cache，但仍可复用 active parse。

`POST /parses` 返回本次请求需要等待的解析记录集合:

```json
{
  "sha256": "...",
  "tier": "standard",
  "page_range": "1~10",
  "status": "pending",
  "cache_hit": false,
  "wait_parse_ids": [12, 13],
  "created_parse_ids": [13],
  "reused_parse_ids": [12]
}
```

`GET /parses` 是唯一的 parse 状态与覆盖查询入口:

```http
GET /parses?ids=12,13
GET /parses?sha256=...&tier=standard
GET /parses?sha256=...&tier=standard&status=pending
GET /parses?sha256=...&tier=standard&page_range=1~10
```

`ids` 查询用于 CLI wait 等精确请求状态判断。`sha256 + tier + page_range` 查询用于计算页覆盖状态。
所有 list 响应统一包含 `total`、`limit`、`offset` 分页元数据。

`GET /parses/{id}` 返回单条 parse record。

### `docs`

`docs` 表示按 `sha256` 去重后的文档内容实体。

```http
GET /docs
GET /docs/by-path?path=/a/b/report.pdf
GET /docs/{sha256}
```

`GET /docs` 返回 active docs 集合，不接受 path 参数。active docs 指当前至少被一个 `status=active` 的 file row 引用的 doc；它不是 `docs` 表所有记录的完整导出。orphan docs 以及只被 `deleted` / `unreachable` file row 引用的 docs 不进入普通列表，后续由 cleanup / maintenance 视图处理。

按路径查当前文档使用 `GET /docs/by-path?path=...`，返回单个 doc；因为 `files.path` 唯一，且任一时刻一个 file row 只绑定一个 `sha256`。

`GET /docs/{sha256}` 默认只返回文档级信息。调用方需要路径实例时，可以传 `expand_files=true`，此时响应中的 `files` 必须是列表；默认不展开时 `files` 可以为 `null` 或省略。

`GET /docs/{sha256}/content` 从有效 done parse record 的 Middle JSON 中读取并转换内容:

```http
GET /docs/{sha256}/content?tier=standard&page_range=1~10&format=markdown
```

读取内容不触发解析。缺页时应返回明确错误或 pending 状态，由调用方再决定是否 `POST /parses`。

### `invalidate`

`POST /invalidate` 是操作型 endpoint，与 `GET /search` 一样作为本地 doclib 的高层能力入口。

请求体必须包含 `target`，第一版仅支持 `target="parses"`:

```json
{
  "target": "parses",
  "path": "/a/b/report.pdf",
  "tier": "standard"
}
```

返回:

```json
{
  "target": "parses",
  "sha256": "...",
  "tier": "standard",
  "invalidated_count": 2
}
```

`POST /invalidate` 不自动创建新的 parse record。

## 替代方案

### 方案 A: 保留 `/parse/status`、`/parse/content`、`/parse/invalidate`

拒绝。这会让 parse 既是动作又是资源前缀，状态查询、内容读取和缓存生命周期混在同一 RPC 风格命名下。

### 方案 B: 使用 `/parse-batches`

拒绝。`parse batch` 更利于解释语义，但 API 路径较长，且数据库表已经叫 `parses`。统一使用 `/parses` 更短，也能和实现模型对齐。

### 方案 C: 使用 `/parse-requests`

拒绝。`POST /parse-requests` 更严格表达“发起一次请求”，但会额外引入 request 资源。当前系统真正落库和可查询的实体是 parse record，因此使用 `POST /parses` 并明确 ensure 语义。

## 影响

API 影响:

- 删除 `/parse/status`，统一使用 `GET /parses` 和 `GET /parses/{id}`。
- `/parse/content` 替换为 `GET /docs/{sha256}/content`。
- `/parse/invalidate` 替换为 `POST /invalidate`。
- `/parse` 替换为 `POST /parses`。

SDK 影响:

- `MineruClient.parse()` 对应 `POST /parses`。
- `MineruClient.parse_status()` 不再作为稳定方法，改为 `list_parses()` / `get_parse()`。
- `MineruClient.parse_content()` 对应 `GET /docs/{sha256}/content`。
- `MineruClient.invalidate(..., target="parses")` 对应 `POST /invalidate`。

CLI 影响:

- `mineru parse --wait` 使用 `POST /parses` 返回的 `wait_parse_ids`，再调用 `GET /parses?ids=...`。
- `mineru library invalidate` 或等价命令调用 `POST /invalidate`。

## 后续动作

1. 更新 doclib routes、client 和 CLI 调用路径。
2. 实现 `GET /parses` 的 id 查询、状态过滤和 coverage 计算。
3. 实现 `GET /docs`、`GET /docs/{sha256}` 和 `GET /docs/{sha256}/content`。
4. 将旧 `/parse/*` routes 从 NEXT 版移除。
