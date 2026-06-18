# 端到端工作流

状态: Draft
读者: 核心开发者、SDK/API/CLI 设计参与者、Agent 能力开发者
范围: 从文件发现、入库、解析、缓存、搜索到 Agent/用户读取的端到端行为
非目标: 重复 API 字段级定义；替代数据库 schema；替代 CLI 参数手册

## 1. 定位

本文回答“一个文件从出现到被用户或 Agent 读取，中间经过哪些系统组件、状态和产物”的问题。

它串联以下专题:

- 术语以 [术语表](glossary.md) 为准。
- 默认选择 / `flash` / `standard` / `pro` 的语义以 [解析 Tier](tiers.md) 为准。
- doclib、worker、SQLite 和 parse-server 的内部结构以 [系统架构](architecture.md) 为准。
- CLI 行为以 [CLI 规格](cli.md) 为准。
- v1 API 行为以 [Unified API](api.md) 为准。
- SDK 层次以 [SDK 设计](sdk.md) 为准。
- Middle JSON 产物以 [Middle JSON](middle-json.md) 为准。

核心目标:

1. 明确 watch 和主动 parse 的差异。
2. 明确 `flash` 何时可以自动使用，何时不可以作为最终阅读质量。
3. 明确 doclib server、parse-server、Tool SDK、Doclib SDK 的调用边界。
4. 明确缓存、搜索索引和解析产物如何落到实际使用的 `tier`。
5. 给后续实现提供可拆任务和验收检查点。

## 2. 参与组件

| 组件 | 中文名 | 职责 |
|------|--------|------|
| `mineru` CLI | 用户工具 | 面向普通用户和 Agent 的本地入口。 |
| `mineru-kit` CLI | 专家工具 | 暴露 backend、批处理、api-server 等高级能力。 |
| Doclib SDK | 本地文档库 SDK | 连接 doclib server，使用 parse/search/watch/config 能力。 |
| Tool SDK | 工具 SDK | 进程内直接解析文件，返回 `ParseResult`。 |
| doclib server | 本地文档库服务 | 入库、缓存、搜索、watch、配置、任务调度。 |
| parse-server | 解析服务 | 无状态解析服务，提供 v1 Unified API，执行 `standard` / `pro`。 |
| Local Parse Server | 本地解析服务 | 用户可信环境内的 parse-server。 |
| Remote Parse Server | 远端解析服务 | `mineru.net/api` 或显式配置的远端兼容服务。 |
| Backend | 解析后端 | 实际解析实现，例如 `flash`、`pipeline`、`vlm`、`hybrid`。 |
| Render | 输出渲染 | 将 Middle JSON 转为 Markdown / Content List / HTML 等。 |

## 3. 全局不变量

这些规则在所有流程中都成立:

1. **隐私优先**: 未显式允许 remote 时，不上传用户文档。
2. **质量优先**: 用户或 Agent 主动读取文档时，默认使用默认选择策略，且不会解析为 `flash`。
3. **发现与阅读分离**: watch 可以自动使用 `flash`；主动阅读不能静默使用 `flash`。
4. **结果记录实际 tier**: 默认选择是请求时选择逻辑，任务、缓存、产物和 metadata 记录实际使用的实体 tier。
5. **backend 只在专家层和 Tool SDK parser 层暴露**: Tool SDK 的直接 parser 可以接受专家 `backend` 参数；API-backed parser、Doclib SDK、doclib server API 和 v1 API 只面向 `tier`，不暴露 `backend`。
6. **缓存按内容和 tier 隔离**: 同一 `sha256 + tier` 的解析结果可以复用；不同 tier 的结果不能互相覆盖。
7. **fallback 不扩大隐私边界**: local 失败不能自动改成 remote；remote 失败可以 fallback 到 local。
8. **Flash 可长期作为 backend 名称**: `flash` 同时是解析档位，也是快速 CPU PDF 解析 backend。
9. **doclib 只持久化 JSON**: doclib 的 `parsed/` 目录只保存按页组织的 Middle JSON 批次文件；Markdown、Content List、HTML 等格式读取时从 JSON 快速转换。

## 4. 总览流程

```text
文件来源
  ├─ watch 自动发现
  │   -> doclib files/docs
  │   -> flash parse
  │   -> 搜索索引 / 基础预览
  │
  ├─ mineru parse / Agent 主动读取
  │   -> doclib ParseService
  │   -> 查缓存
  │   -> 默认选择/standard/pro/flash 路由
  │   -> ParseWorker
  │   -> Tool SDK 或 parse-server
  │   -> Middle JSON batch JSON
  │   -> read-time render: Markdown / Content List / HTML
  │   -> 返回用户或 Agent
  │
  ├─ SDK 调用
  │   ├─ Tool SDK: 直接 parser -> backend
  │   ├─ API-backed Parser: v1 API -> parse-server
  │   └─ Doclib SDK: doclib server -> 缓存/任务/搜索
  │
  └─ v1 API 调用
      -> Official API 或 Local Parse Server
      -> upload/file/job
      -> parse-server
      -> job result / requested formats
```

## 5. 流程 A: Watch 自动发现与索引

### 5.1 触发条件

文件位于 watch 目录下，或全量扫描发现新文件、修改文件、删除文件、可插拔设备恢复可达。

### 5.2 默认目标

watch 的目标是发现文件、建立基础索引和支持后续搜索，不是提供最终阅读质量。

默认 tier:

| 场景 | 默认 tier |
|------|-----------|
| watch 自动发现 | `flash` |
| parsing-rule 明确指定 | rule 中的 tier |

### 5.3 步骤

1. `WatchLoop` 接收文件系统事件。
2. 使用扩展名白名单和 exclude rule 过滤。
3. 执行统一文件发现/刷新: 新路径写入 `files` 并设置 `sha256=NULL`；已知路径先比较 `mtime_ms` 和 `size_bytes`。
4. 如果已知路径未变化，跳过重新入库。
5. 如果已知路径变化，更新文件 stat，清空 `files.sha256`，让 ingest worker 重新处理。
6. `IngestWorker` 抢占 `status=active AND sha256 IS NULL` 的文件。
7. 计算 SHA256，提取基础 metadata。
8. 写入或更新 `docs`。
9. 写入文件名索引。
10. 根据 watch 默认策略或 parsing-rule 创建 parse 任务。
11. 如果使用默认策略，则创建 `flash` 任务。
12. `ParseWorker` 调用本地 `flash` backend。
13. 将本批次的 Middle JSON pages 写入 `parsed/<sha-prefix>/<sha>/<tier>/<page_range>_<done_at>.json`。
14. 读取本批次结果并在内存中生成 Markdown 文本，用于更新内容索引。

同一文件变化检查规则也适用于用户显式输入 source file path 的操作:

| 操作 | 入口 |
|------|------|
| 主动解析 | `POST /parses` / SDK `ensure_parse()` / `mineru parse <path>` |
| 文件信息 | `GET /files/by-path?path=...` / SDK `get_file_by_path()` / `mineru show file <path>` |
| 按路径取文档 | `GET /docs/by-path?path=...` / SDK `get_doc_by_path(path)` |
| 按路径失效解析 | `POST /invalidate { path }` / SDK `invalidate(path=...)` / `mineru invalidate <path>` |

### 5.4 状态变化

| 阶段 | 主要状态 |
|------|----------|
| 发现路径 | `files.status=active` |
| 检测到变化 | 更新 `files.size_bytes/mtime_ms`，并设置 `files.sha256=NULL` |
| 文件缺失且设备可达 | `files.status=deleted`，保留 `sha256`，写入 `deleted_at` |
| removable 设备不可达 | 后台 `DeviceMonitor` 可将 `watches.status=unreachable`，并批量把相关 active files 变为 `unreachable`；同步 path 调用只允许更新当前文件和 watch 状态 |
| removable 设备恢复 | watch 和 files 恢复为 `active`，并立即对该 watch 执行 scan |
| 入库完成 | `files.sha256` 指向 `docs.sha256` |
| 创建任务 | `parses.status=pending`，`tier=flash` 或 rule tier |
| 执行解析 | `parses.status=parsing` |
| 解析成功 | `parses.status=done`，JSON 批次写入 `data_dir/parsed/{sha256}/{tier}` 下的批次文件 |
| 解析失败 | `parses.status=failed`，写入 `error_code` / `error_msg` |

### 5.5 文件可达性规则

watch event、所有 scan 操作和所有同步 source file path 操作都必须检查 path 当前可达性。同步 path 操作包括:

- 主动解析: `POST /parses` / SDK `ensure_parse()` / `mineru parse <path>`。
- 文件信息: `GET /files/by-path?path=...` / SDK `get_file_by_path()` / `mineru show file <path>`。
- 按路径取文档: `GET /docs/by-path?path=...` / SDK `get_doc_by_path(path)`。
- 按路径失效解析: `POST /invalidate { path }` / SDK `invalidate(path=...)` / `mineru invalidate <path>`。

如果 path 不存在:

1. 如果它属于 removable watch，且 watch root 当前不可达，则标记为 `unreachable`；同步 path 调用只处理当前文件。
2. 否则标记为 `deleted`，保留 `sha256`，写入 `deleted_at`。
3. 已经是 `deleted` 的 file 不因 watch root 不可达改回 `unreachable`。
4. `PermissionError` / stat `OSError` 不表示文件删除，应写入 `files.error_code` / `files.error_msg`。
5. 标记 `deleted` 或 `unreachable` 时不清理 FTS。

`active`、`unreachable` 和 `deleted` file row 都保护对应 doc。只有完全没有任何 file row 关联的 doc 才算 orphan。
`GET /docs` / `mineru list docs` 默认只展示 active docs，即至少被一个 `status=active` file row 引用的 doc；被 `deleted` / `unreachable` file row 保护的 doc 由 cleanup / maintenance 视图处理。

watch scan 的 P0 删除检测采用“两阶段刷新”:

1. scan 开始时先检查 watch root 是否可达；不可达时只更新 `watches.status=unreachable` 并返回。
2. root 可达时，先读取该 watch 下 `status=active` 的已知 file paths，并逐个调用统一文件刷新逻辑。
3. 再执行 `os.walk()` 扫描文件系统，发现当前存在的新文件或变化文件。
4. 因此同一个已知且仍存在的文件可能在一轮 scan 中被 refresh 两次；P0 接受这部分后台 IO 成本，换取不增加 `files.last_seen_at` 字段和不引入独立删除判断逻辑。

### 5.6 输出

watch 默认输出面向系统内部:

- `files` / `docs` 记录。
- 文件名索引。
- `flash` tier 的每批次 Middle JSON 文件。
- 内容全文索引。
- 可搜索 snippet。

### 5.7 约束

- watch 自动产出的 `flash` 结果可以用于搜索和预览；P0 不基于启发式自动提示或自动排队升级。
- watch 自动产出的 `flash` 结果不能在用户主动阅读时被静默当作最终解析内容返回。
- 后台自动升级只由用户显式配置的 parsing-rules 触发；如果 parsing-rule 指定 `standard` 或 `pro`，必须经过能力检查和隐私检查。
- parsing-rule 只有显式允许 remote，才可以上传远端。

### 5.8 Cleanup 边界

deleted file cleanup 与 orphan doc cleanup 分开执行。

deleted file cleanup:

- 手动命令立即删除所有 `status=deleted` 的 file row。
- 后台任务固定保留 7 天后删除 deleted file row。
- 删除 file row 时删除对应 `fts_filenames`。
- 不自动删除 docs、parses、parsed JSON 或 `fts_contents`。

orphan doc cleanup:

- 只处理没有任何 file row 关联的 docs。
- 执行时删除 docs、parses、`fts_contents` 和 parsed JSON。

### 5.9 Forget Path

`forget` 用于让 doclib 忘记一个 path 的记录。

流程:

1. 判断 path 是否是已配置 watch root；如果是，拒绝并提示先执行 watch remove。
2. 匹配 DB 中的 file rows:
   - 精确匹配 `files.path = path` 时，按文件处理。
   - 存在 `files.path` 以 `path/` 为前缀时，按目录处理。
   - 两者都没有时，返回 0。
3. 如果 path 位于 active watch root 下，返回 warning，说明后续 scan 可能重新发现。
4. dry-run 时只返回匹配数量、匹配类型和 warnings。
5. 执行时删除匹配 file rows 对应的 `fts_filenames`。
6. 删除匹配的 `files` rows。
7. 不删除 `docs`、`parses`、parsed JSON 或 `fts_contents`。

目录 path 在 P0 永远递归匹配；不提供 `recursive` 参数。`forget` 不是 ignore rule，也不是 invalidate。

### 5.10 Scan Task

`scan` 是统一后台任务，用于扫描文件或目录并更新 doclib 的文件状态。

所有完整 scan 都必须走 `ScanService + ScanWorker`:

- 手动 `mineru scan <path>`。
- SDK / HTTP API scan。
- watch initial scan。
- watch rescan。
- 设备恢复后的 watch scan。

scan 与 ingest 的边界:

- scan 负责 stat、发现、变化检测、deleted / unreachable / error 状态刷新。
- ingest 负责 SHA256、metadata、docs、FTS 内容和初始 parse batch。
- scan done 不表示 ingest 或 parse 完成。

scan kind:

- `manual`: 显式 scan 请求，可以针对文件、目录或历史 path。
- `watch`: watch 产生的完整 scan，必须绑定 `watch_id`。

scan source 是 best-effort 记录字段，P0 不强求 `cli` / `sdk` / `api` 准确区分。

exclude 规则:

| 场景 | 是否应用 exclude |
|------|------------------|
| manual scan 文件 | 否 |
| manual scan 目录 | 是 |
| watch scan | 是 |
| watch event 单文件 | 是 |
| parse 单文件 | 否 |

完整规则见 [ADR-0009](decisions/0009-doclib-scan-task.md)。

## 6. 流程 B: 用户或 Agent 主动读取本地文件

### 6.1 触发条件

用户或 Agent 执行:

```bash
mineru parse report.pdf
```

或通过 Doclib SDK 调用:

```python
client.parse("report.pdf")
```

### 6.2 默认目标

主动读取的目标是尽可能准确地理解文档。未指定 tier 时，使用默认选择策略，并至少解析到 `standard`。

### 6.3 步骤

1. CLI 或 SDK 将请求发送给 doclib server。
2. `ParseService` 查找文件路径。
3. 如果文件尚未入库，先同步执行 ingest，得到 `sha256`。
4. 确定页码范围、输出格式、等待策略和隐私偏好。
5. 如果用户未指定 tier，使用默认选择策略。
6. 默认选择策略通过当前可用 parse-server 能力发现选择最高非 `flash` tier。
7. 将任务和缓存键落到实际使用的实体 tier。
8. 查询 `(sha256, tier, page_range)` 是否已有可复用结果。
9. 缓存命中则直接 render 并返回。
10. 缓存未命中则创建 `parses` 任务。
11. `ParseWorker` 抢占任务。
12. 按 tier 和 privacy 路由到本地 backend、Local Parse Server 或 Remote Parse Server。
13. 生成本批次 Middle JSON pages。
14. 将本批次 pages 写入 JSON 文件。
15. 根据请求输出格式，从已保存的 JSON 转换得到 Markdown / Content List / HTML 等输出。
16. 读取本批次结果并在内存中生成 Markdown 文本，用于更新内容索引。
17. 返回结果或任务状态。

### 6.4 Tier 决策

| 请求 | 实际行为 |
|------|----------|
| 未指定 tier | 选择最高可用非 `flash` tier |
| `tier=flash` | 显式使用本地 `flash` backend |
| `tier=standard` | 使用本地或自部署 parse-server 的 `standard` 能力 |
| `tier=pro` | 使用本地 `pro` 或 `mineru.net/api` 的 `pro` 能力 |

结果只需要记录实际使用的 `tier`。如果默认选择最终选择 `pro`，产物、缓存和 metadata 使用 `pro`。

### 6.5 隐私决策

| 请求条件 | 允许行为 |
|----------|----------|
| 未传 `--remote` / `remote=False` | 只能使用本地 backend 或 Local Parse Server |
| 显式 `--remote` / `remote=True` | 可以上传到 Remote Parse Server |
| 远端失败 | 可以 fallback 到 local |
| 本地失败 | 不能自动 fallback 到 remote，除非用户已经显式允许 remote |

### 6.6 输出策略

| 输出目标 | 行为 |
|----------|------|
| STDOUT | 默认适合 Agent 消费，长文档可返回部分页和继续 marker |
| 文件输出 | 写入指定路径 |
| JSON | 输出已持久化的 Middle JSON pages 或更高层 response envelope，具体格式由 CLI/API 文档定 |
| Markdown | 默认阅读输出；读取时从 Middle JSON 转换 |
| Content List / HTML | 读取时从 Middle JSON 转换 |

doclib 不持久化 Markdown、Content List 或 HTML。它们都是 CPU-only 的派生格式，转换成本低，按需生成即可。

### 6.7 约束

- 如果只有 `flash` 可用，默认选择应失败，不静默返回 `flash`。
- 如果本地没有 `standard` / `pro` 且未允许 remote，应返回可操作错误。
- 如果缓存中只有 `flash`，主动读取未指定 tier 不能直接命中该缓存。
- 如果用户显式 `tier=flash`，可以返回 `flash` 缓存或重新解析。

## 7. 流程 C: Agent 从搜索结果继续阅读

### 7.1 触发条件

Agent 先通过搜索找到候选文档，然后决定读取其中一个文档。

典型路径:

```text
watch -> flash index -> search result -> Agent chooses document -> mineru parse
```

### 7.2 步骤

1. Agent 调用 search。
2. doclib 查询文件名索引和内容索引。
3. search result 返回路径、sha256、snippet、当前索引来源 tier。
4. Agent 选择具体文档。
5. Agent 发起主动 parse 请求。
6. 请求未指定 tier 时使用默认选择策略。
7. doclib 检查是否已有 `standard` 或 `pro` 缓存。
8. 如果没有，则创建高优先级 parse 任务。
9. 解析完成后返回可阅读输出，并写入更高质量索引。

### 7.3 关键区别

搜索结果可以来自 `flash`，但读取动作不能默认停留在 `flash`。

| 阶段 | 可接受最低 tier |
|------|-----------------|
| 自动发现 | `flash` |
| 搜索召回 | `flash` |
| 搜索 snippet | `flash`，但应可标记来源 tier |
| Agent 主动阅读 | `standard`，除非显式选择 `flash` |
| Agent 引用/citation | 应优先来自 `standard` 或 `pro` |

### 7.4 Agent marker

当输出被截断或只解析部分页时，应输出可机器理解的 marker:

```text
<!-- mineru:next page_range=6~10 tier=pro -->
```

marker 不应依赖自然语言提示。具体 marker 格式可在 CLI 或 Agent 文档中继续细化。

## 8. 流程 D: parse-server API 解析

### 8.1 触发条件

调用方直接使用 Official API、Local Parse Server，或通过 `MinerUApiParser` 使用 v1 API。

### 8.2 Official API 主线

1. 客户端创建 upload 或提供已有 `file_id`。
2. 客户端创建 parse job。
3. 请求可以携带 `tier`；省略或传 JSON `null` 时使用默认选择策略。
4. API service 做鉴权、配额和输入校验。
5. 调度到可用解析能力。
6. 解析服务生成 Middle JSON 和请求的输出格式。
7. job 进入终态。
8. 客户端轮询、等待、SSE 或 webhook 获取结果。

### 8.3 Local Parse Server 差异

Local Parse Server 复用同一套客户端协议，但可以简化:

- 鉴权默认可关闭。
- Webhook 可以不实现。
- 上传可以使用本地临时存储。
- `local` source 可以引用 allowlist 内路径。
- 文件下载可以直接返回 body。

### 8.4 与 doclib 的关系

doclib 可以调用 parse-server，但 parse-server 不依赖 doclib。

| 方向 | 说明 |
|------|------|
| doclib -> Local Parse Server | 本机高质量解析，隐私仍在本地可信环境内 |
| doclib -> Remote Parse Server | 用户显式允许 remote 后才可上传 |
| API client -> parse-server | 无状态解析请求，不使用 doclib 搜索和缓存 |
| MinerUApiParser -> parse-server | 用 Python parser 接口封装 v1 API |

## 9. 流程 E: Tool SDK 直接解析

### 9.1 触发条件

开发者在进程内直接调用:

```python
from mineru.parser import parse

result = parse("report.pdf", tier="standard")
```

或使用具体 parser:

```python
from mineru.parser import PdfPipelineParser

with PdfPipelineParser() as parser:
    result = parser.parse("report.pdf")
```

### 9.2 行为边界

Tool SDK 是无状态解析工具层:

- 不启动 doclib server。
- 不管理 watch。
- 不写 doclib 搜索索引。
- 不自动上传远端。
- 不应在 import 阶段加载重依赖。

### 9.3 适用场景

| 场景 | 是否适合 Tool SDK |
|------|------------------|
| `mineru-kit parse` | 是 |
| parse-server worker 内部执行 | 是 |
| 单文件嵌入式解析 | 是 |
| 本地文档库搜索 | 否，使用 doclib |
| watch 自动发现 | 否，由 doclib 管理 |
| 长期缓存和增量页码任务 | 否，由 doclib 管理 |

## 10. JSON 产物、页合并与缓存

### 10.1 产物目录

doclib 解析产物按内容和实际使用的 tier 隔离。`parsed/` 目录下只保存 JSON:

```text
~/.mineru/
  parsed/
    ab/
      ab3f...7e2d/
        flash/
          1~5_1710000000000.json
          98~102_1710000123000.json
        standard/
          1~10_1710001234000.json
          11~20_1710002234000.json
        pro/
          1~20_1710003234000.json
```

单个 JSON 文件表示一次解析批次，基本形态:

```json
{
  "pages": []
}
```

其中 `pages` 按页组织。文件名由本批次页码范围和完成时间组成，便于同一文档、同一 tier 下存在多个不同页码范围的解析批次。

doclib 不在 `parsed/` 中保存:

- `output.md`
- `content_list.json`
- `content_list_v2.json`
- `output.html`

这些格式都由 Middle JSON 读取时转换得到。

### 10.2 缓存键与覆盖判断

基础缓存键:

```text
(sha256, tier)
```

页码范围由 `parses.page_range` 和对应 JSON 文件共同表达。请求某个页码范围时，doclib 会:

1. 查询同一 `sha256 + tier` 下已完成批次。
2. 忽略已经 invalidate 的批次。
3. 忽略 JSON 文件已经丢失的批次。
4. 用有效已完成批次的 `page_range` 覆盖集合判断请求页码是否已经满足。
5. 对未覆盖的页创建新的 parse 任务。
6. 如果页码已经被 pending / parsing 批次覆盖，则提升优先级，而不是重复创建任务。

这意味着同一个文档可以多轮解析不同页，最终由多个 JSON 批次共同覆盖用户需要的页码集合。

`--force` 会跳过第 4 步的 done 缓存命中判断，但仍可复用已经覆盖请求页码的 active parse。它不会删除或作废旧 done 批次。若 force 关联的 wait parse 失败，旧批次仍可继续用于读取和搜索，但本次 force 请求应显示为失败。

### 10.3 页合并与 compaction

Middle JSON 按页组织，因此不同解析批次可以按 `page_idx` 合并。

compaction 负责:

1. 扫描同一 `sha256 + tier` 下多个有效 done 批次。
2. 合并相邻或重叠的页码范围。
3. 读取旧批次 JSON。
4. 按 `page_idx` 收集页面内容；如果同一页出现多次，`done_at` 较新的有效批次覆盖较旧批次。
5. 删除旧 JSON 文件。
6. 为合并后的页码范围写出新的 JSON 文件。
7. 用较少的 done parse row 替代旧 row。

已 invalidate 的批次不参与 compaction 的页面选择；其 JSON 文件可以由 cleanup 或 compaction 的清理阶段删除。

compaction 不生成 Markdown 或 Content List。

### 10.4 读取时转换

所有非 JSON 格式都是读取时转换:

| 请求格式 | 来源 | 是否持久化 |
|----------|------|:--:|
| JSON | 已保存的 Middle JSON 批次 | 是 |
| Markdown | Middle JSON -> render | 否 |
| Content List | Middle JSON -> render | 否 |
| Content List v2 | Middle JSON -> render | 否 |
| HTML | Middle JSON 或 Markdown -> render | 否 |

这个设计有两个好处:

1. 派生格式转换只需要 CPU，通常足够快，不值得持久化多份。
2. JSON 按页组织，方便增量解析、批次合并和局部重算。

### 10.5 Metadata 更新

文档 metadata 可由更高质量 tier 覆盖低质量 tier。

建议优先级:

```text
flash < standard < pro
```

约束:

- `flash` 不应覆盖来自 `standard` 或 `pro` 的 metadata。
- `standard` 可以覆盖 `flash`。
- `pro` 可以覆盖 `standard` 和 `flash`。

### 10.6 搜索索引

搜索索引应尽量保留最高可用 tier 的文本。

如果文档先由 `flash` 建索引，后续 `standard` 或 `pro` 完成后，应刷新内容索引。

FTS 更新可以在解析完成后临时从 ParseResult 或 Middle JSON 渲染 Markdown 文本；这份 Markdown 只进入搜索索引，不作为 doclib 产物文件保存。

## 11. 错误与恢复

| 错误 | 典型原因 | 恢复建议 |
|------|----------|----------|
| `quality_tier_unavailable` | 主动阅读需要高质量 tier，但只有 `flash` | 启动本地解析服务或允许远端解析 |
| `no_engine` | 请求 tier 没有可用解析服务或解析后端 | 配置 local parse-server、调整 tier |
| `engine_unavailable` | 解析服务或解析后端暂不可用 | 稍后重试或重启 parse-server |
| `parse_server_unavailable` | Local/Remote Parse Server 不可达 | 检查 URL、进程、网络和 API Key |
| `tier_mismatch` | parse-server 不支持请求 tier | 改用可发现 tier |
| `parse_failed` | 文档损坏、加密或 backend 处理失败 | 检查文件或换 tier |
| `parse_timeout` | 解析超过等待时间 | 轮询任务状态或缩小页码范围 |
| `parse_oom` | 本地内存或显存不足 | 使用较低 tier、减少页数、或使用 remote |

恢复约束:

- 重试不得改变 `privacy`。
- 自动恢复不得把主动阅读降级到 `flash`。
- 崩溃恢复时，doclib 启动应释放 stale lock 并处理 parsing 状态任务。
- managed Local Parse Server 连续失败后，可以降级 disabled 并要求用户修复。

## 12. 实现切分

### P0: 打通主链路

1. doclib server 能启动、停止、恢复 stale lock。
2. watch 能发现文件并创建 `flash` 任务。
3. ingest 能写入 `files` / `docs` / 文件名索引。
4. `flash` backend 能产生基础 Middle JSON pages。
5. `mineru parse` 能同步入库文件并查缓存。
6. 默认选择能通过 parse-server 能力发现选择 `standard` 或 `pro`。
7. ParseWorker 能按 tier 和 privacy 路由。
8. 解析产物按 `sha256 + tier + page_range + done_at` 写入 JSON。
9. 主动阅读不会静默命中 `flash`。
10. 错误返回可操作 code 和 suggestion。

### P0: Agent-native 阅读

1. 搜索结果返回来源 tier。
2. Agent 主动读取时提升到 `standard` 或 `pro`。
3. 输出支持截断 marker 和下一步 page hint。
4. Middle JSON 支持稳定 page/block locator。
5. Markdown 输出可携带可选 locator marker。
6. 高质量解析完成后刷新搜索索引。

### P2: 体验与优化

1. parse-server managed 模式自动拉起和退避。
2. 增量页码解析与 compaction。
3. 多文件批处理。
4. 解析队列优先级和取消。
5. Web UI 状态展示。
6. 跨平台 watch 稳定性。

## 13. 端到端验收

### 13.1 Watch 到搜索

输入:

- watch 目录中新增 `report.pdf`。

期望:

1. `files` 有路径记录。
2. `docs` 有 SHA256 记录。
3. `parses` 有 `tier=flash` 的任务。
4. 任务完成后写入 `parsed/<sha-prefix>/<sha>/flash/<page_range>_<done_at>.json`。
5. 搜索可以命中文档。
6. 搜索结果以机器可读字段标记来源 tier，例如 `tier=flash`。

### 13.2 主动 parse 本地成功

输入:

```bash
mineru parse report.pdf --tier standard --wait 60
```

期望:

1. 不上传远端。
2. 使用 Local Parse Server 或本地可用 standard 能力。
3. JSON 产物写入 `parsed/<sha-prefix>/<sha>/standard/<page_range>_<done_at>.json`。
4. 返回 Markdown 或指定格式；非 JSON 格式从 JSON 转换得到。
5. `tier` 记录为 `standard`。

### 13.3 主动 parse 默认选择成功

输入:

```bash
mineru parse report.pdf --wait 60
```

前提:

- 可发现 Local Parse Server 支持 `standard` 和 `pro`。

期望:

1. 请求未指定 tier，进入默认选择策略。
2. 实际选择 `pro`。
3. 任务和 JSON 产物记录 `tier=pro`。
4. 不记录 `requested_tier` 字段。
5. 不创建默认选择专用缓存目录。

### 13.4 主动 parse 默认选择失败

输入:

```bash
mineru parse report.pdf
```

前提:

- 只有 `flash` 可用。
- 未显式 `--remote`。

期望:

1. 返回 `quality_tier_unavailable`。
2. 不返回 `flash` 作为最终阅读结果。
3. suggestion 包含启动本地解析服务、允许 remote、或显式 `--tier flash`。

### 13.5 显式 flash

输入:

```bash
mineru parse report.pdf --tier flash
```

期望:

1. 使用本地 `flash` backend。
2. 可命中 watch 产生的 `flash` 缓存。
3. 返回结果 metadata 中 `tier=flash`。
4. 不调用 Remote Parse Server。

### 13.6 增量页码解析与合并

输入:

```bash
mineru parse report.pdf --tier standard --pages 1~5
mineru parse report.pdf --tier standard --pages 6~10
```

期望:

1. 两次请求可以生成两个 done parse batch。
2. `parsed/<sha-prefix>/<sha>/standard/` 下保存两个 JSON 文件。
3. 每个 JSON 文件只包含对应页码范围的 `pages`。
4. 后续请求 `--pages 1~10` 可以由已完成批次覆盖，不需要重新解析。
5. compaction 可以将两个批次合并成更少的 parse row 和 JSON 文件。
6. 合并时按 `page_idx` 组织页面；同一页如果被重复解析，较新批次覆盖较旧批次。

### 13.7 Force 重新解析

输入:

```bash
mineru parse report.pdf --tier standard --pages 1~5
mineru parse report.pdf --tier standard --pages 1~5 --force
```

期望:

1. 第二次请求调用 `POST /parses`，返回 `wait_parse_ids`。
2. 旧 done batch 不被删除，也不被标记失效。
3. 如果已有 active parse 覆盖 `1~5`，第二次请求可以复用该 parse，并把 id 放入 `reused_parse_ids`。
4. 如果没有 active parse 覆盖，第二次请求创建新 parse，并把 id 放入 `created_parse_ids`。
5. CLI wait 查询 `GET /parses?ids=...`，等待所有 `wait_parse_ids` 完成。
6. 如果第二次解析成功，读取 `1~5` 时使用 `done_at` 更新的有效 batch。
7. 如果第二次解析失败，第一次解析结果仍然可读、可搜索，但本次 force 请求显示为失败。

### 13.8 Invalidate 缓存

输入:

```bash
mineru library invalidate report.pdf --tier standard
```

期望:

1. CLI 调用 `POST /invalidate`，请求体包含 `target="parses"`。
2. 命中的已有 parse 被标记为失效。
3. 失效 parse 不参与后续缓存命中。
4. 失效 parse 不参与读取时页合并。
5. 失效 parse 不参与搜索索引刷新和 compaction 最新页选择。
6. invalidate 不自动创建新的 parse。

### 13.9 Remote 隐私边界

输入:

```bash
mineru parse report.pdf --tier pro
```

前提:

- 本地没有 `pro`。
- 未显式 `--remote`。

期望:

1. 不上传。
2. 返回本地能力不足错误。
3. suggestion 提示 `--remote`，但不替用户执行。

输入:

```bash
mineru parse report.pdf --tier pro --remote
```

期望:

1. 允许上传到 Remote Parse Server。
2. 远端失败时可以 fallback 到 local，如果 local 有对应能力。
3. 结果 `via` 记录实际执行路径。

## 14. 未决问题

未决问题集中维护在 [开放问题清单](open-questions.md)。本文只保留已经确定的端到端流程和验收口径。
