# 系统架构

状态: Draft
读者: 项目核心开发者、负责本地文档库和服务端实现的开发者
范围: 本地文档库、server、worker、数据库、文件处理、配置和生命周期
非目标: API 字段级定义；CLI 参数完整手册
底稿: `../../NEXT-DESIGN.md`

## 1. 系统上下文

Next MinerU 的本地能力中心由 `mineru doclib` 和可选的 `local parse-server` 组成。

```text
mineru CLI / MCP Server / MinerU.app
  -> mineru doclib
       - FastAPI + uvicorn
       - Routes -> Services -> Core
       - Background workers
       - SQLite + FTS5
  -> local parse-server
       - 独立进程
       - standard / pro tier
       - HTTP API
```

`doclib` 是默认常驻的本地文档库服务，负责文件入库、SHA256 去重、解析任务调度、缓存、搜索、配置和本地 API。`local parse-server` 是可选独立进程，负责加载较重模型并执行 standard/pro tier 解析。

Tier 的产品语义以 [解析 Tier](tiers.md) 为准；本文只描述它们在 doclib 和 ParseWorker 中的执行路径。

核心边界：

| 边界 | 选择 | 目的 |
|------|------|------|
| CLI/MCP/桌面端 -> doclib | HTTP + JSON over Unix Domain Socket | 本地安全、低延迟、避免端口冲突 |
| doclib -> local parse-server | HTTP + JSON over TCP loopback | 进程隔离，允许 GPU 模型崩溃不拖垮 doclib |
| doclib 存储 | SQLite WAL + FTS5 | 零运维、单文件、支持全文检索 |
| 任务队列 | DB rows + timestamp lock | 无外部依赖，崩溃后可恢复 |
| CPU 密集操作 | `asyncio.to_thread()` | 避免阻塞 async worker |

默认通道：

| 通道 | 地址 | 用途 |
|------|------|------|
| UDS | `$MINERU_HOME/doclib.sock` | CLI、MCP、桌面端访问 doclib |
| TCP | `127.0.0.1:15981` | doclib 调用 local parse-server |
| TCP 随机端口 | `127.0.0.1:<random>` | P1 Web UI |

## 2. 模块分层

doclib 内部采用三层结构：

```text
Routes
  -> Services
    -> Core
```

| 层级 | 职责 | 约束 |
|------|------|------|
| Routes | HTTP 路由、参数校验、响应格式化 | 不直接承载业务状态 |
| Services | 业务逻辑、状态管理、跨模块协调 | 组织 parse/search/config/cleanup 流程 |
| Core | SQLite、FTS、文件 IO、metadata 提取 | 不感知 CLI/API 产品语义 |

主要 service：

| Service | 职责 |
|---------|------|
| `ParseService` | 接收解析请求、同步入库、查缓存、创建解析任务 |
| `SearchService` | 文件名和内容搜索 |
| `ConfigService` | KV 配置、watch 目录、parsing-rules、exclude 规则 |
| `CleanupService` | 孤儿文档、删除文件和解析缓存清理 |

后台组件：

| 组件 | 并发 | 职责 |
|------|------|------|
| `WatchLoop` | 1 | 监听文件系统变化 |
| `IngestWorker` | 2 | 计算 SHA256、提取 metadata、写 files/docs、触发默认 parse |
| `ParseWorker` | 2 | 执行解析任务并写入产物 |
| `ParseServerHealthCheck` | 1 | 探测 local/remote parse-server 健康状态 |
| `DeviceMonitor` | 1 | 检测可插拔 watch 路径 |
| `Compaction` | 1 | 合并已完成 parse 批次 |

所有后台组件都是 asyncio task，随 server 启停。

## 3. 数据模型

doclib 的核心数据模型围绕三张表展开：

| 表 | 粒度 | 作用 |
|----|------|------|
| `files` | 文件路径实例 | 记录路径、文件状态、watch 来源和对应 `sha256` |
| `docs` | 文档内容 | 以 SHA256 去重，记录文档级 metadata |
| `parses` | 一次解析请求 | 记录 tier、page_range、privacy、状态和错误 |

### 3.1 `files`

`files` 表表示本机看到的文件实例。同一份内容可能有多个路径，因此 `files.sha256` 指向 `docs.sha256`。

关键字段：

| 字段 | 含义 |
|------|------|
| `path` | 本地文件绝对路径，唯一 |
| `filename` / `ext` / `size_bytes` / `mtime_ms` | 文件基础信息 |
| `sha256` | 入库后关联到 `docs` |
| `watch_id` | 来源 watch 目录 |
| `status` | `active` / `deleted` / `unreachable` |
| `locked_at` | IngestWorker 抢占锁 |
| `error_code` / `error_msg` | 文件级错误 |

`status` 表达路径可达性，而不是 doc 生命周期:

| 状态 | 含义 | 默认搜索可见 | 是否保护 doc |
|------|------|--------------|--------------|
| `active` | 路径当前存在且可访问 | 是 | 是 |
| `unreachable` | 所属 removable watch 当前不可达 | 否 | 是 |
| `deleted` | 路径已确认不存在，且不是设备整体不可达导致 | 否 | 是 |

只有当某个 doc 没有任何 file row 关联时，才算 orphan。`deleted` file row 在被清理前仍保留 `sha256` 并保护 doc。完整规则见 [ADR-0007](decisions/0007-doclib-file-availability-lifecycle.md)。

普通文档列表只展示 active docs，即至少被一个 `status=active` file row 引用的 doc。被 `deleted` / `unreachable` file row 保护的 doc 不算 orphan，但也不进入 `GET /docs` / `mineru list docs` 的默认结果。

### 3.2 `docs`

`docs` 表表示按内容去重后的文档。

关键字段：

| 字段 | 含义 |
|------|------|
| `sha256` | 文档内容主键 |
| `file_type` / `page_count` | 入库阶段尽量提取；`file_type` 是短类型名，例如 `pdf` / `docx` |
| `title` / `author` / `subject` / `keywords` | 文档 metadata，做保护性截断 |
| `is_image_based` | 文档是否以页面图片为主，缺少可用文字层 |
| `meta_tier` | 当前 metadata 来源 tier |
| `error_code` / `error_msg` | 内容身份级错误，例如加密、损坏、metadata 失败 |

metadata 更新规则：入库阶段写基础 metadata；解析完成后，只有当新结果的 tier 不低于当前 `meta_tier` 时才覆盖。

### 3.3 `parses`

`parses` 表表示解析任务和缓存记录。同一 `sha256 + tier` 可以有多行，用于增量页码解析和 force 重解析。

关键字段：

| 字段 | 含义 |
|------|------|
| `sha256` | 指向文档内容 |
| `tier` | 实体 tier：`flash` / `standard` / `pro`；未指定 tier 应在入队前或执行前解析为实体 tier |
| `page_range` | 正值页码范围字符串，如 `1~5,46~50` |
| `status` | `pending` / `parsing` / `done` / `failed` / `superseded` |
| `priority` | Agent 同步请求高于 watch 后台任务 |
| `privacy` | 用户请求偏好：`local` / `remote` |
| `via` | 实际执行路径：`local` / `remote` |
| `error_code` / `error_msg` | 解析级错误 |

缓存键为 `(sha256, tier)`，不区分 `privacy` 和 `via`。同一内容、同一 tier 的本地与远端产物应视为等价缓存。

### 3.4 辅助表

| 表 | 作用 |
|----|------|
| `fts_contents` | 内容全文搜索，只保留最高 tier 文本 |
| `fts_filenames` | 文件名搜索 |
| `watches` | 监控目录、可插拔设备状态 |
| `exclude_rules` | exclude 路径规则 |
| `parsing_rules` | parsing-rule 路径规则 |
| `config` | SQLite KV 配置 |
| `_migrations` | schema 版本追踪 |

SQLite 运行在 WAL 模式，使用 FTS5 提供搜索能力。每次 DB 操作打开独立 `aiosqlite` 连接，提交后关闭，避免跨 worker 游标冲突。

## 4. 数据流

### 4.1 两阶段模型

所有内容提取都归入 parse 概念，但内部拆成两个阶段：

```text
阶段 1: 入库
  -> 计算 SHA256
  -> 提取基础 metadata
  -> 写入 files/docs
  -> 写入 fts_filenames
  -> 根据 watch 默认 tier 或 parsing-rules 创建 parses 任务

阶段 2: 解析
  -> flash: 本地轻量解析
  -> 默认选择: 通过 parse-server 能力发现解析为 standard/pro
  -> standard/pro: local 或 remote parse-server
  -> 写 parsed artifacts
  -> 更新 fts_contents
  -> 更新 parses/docs 状态
```

这样用户无需理解“索引”和“解析”两个概念；不同 tier 只是解析深度和执行路径不同。

注意：`flash` 是 watch 自动发现和搜索索引的低成本 tier，不应作为用户主动阅读文档时的默认最终质量。主动阅读未指定 tier 时应使用默认选择策略，解析为当前可发现的最高非 `flash` tier，具体语义见 [解析 Tier](tiers.md)。

### 4.2 文件发现到入库

文件可以来自 watch，也可以来自显式 `mineru parse <path>`、`mineru show file <path>`、`GET /docs/by-path?path=...` 或 `invalidate(path=...)`。

所有 source file path 入口在使用 `files.sha256` 前，都必须先执行统一文件发现/刷新步骤:

1. 按扩展名白名单和 exclude 规则过滤。
2. 对 path 执行轻量 stat，读取 `size_bytes` 和 `mtime_ms`。
3. 如果 path 不存在于 active `files`，插入记录并设置 `sha256=NULL`。
4. 如果 path 已存在且 `mtime_ms`、`size_bytes` 均未变化，不做重新入库。
5. 如果 path 已存在但 `mtime_ms` 或 `size_bytes` 变化，更新文件 stat，清空 `files.sha256`、锁和错误字段。
6. `IngestWorker` 通过原子 SQL 抢占 `status=active AND sha256 IS NULL` 的文件。

`birthtime_ms` 暂不进入 P0。文件创建时间在不同平台上的语义和可用性不一致，目前还没有明确的产品或 Agent 行为依赖它；P0 不记录该字段，也不使用它参与文件变化判定。
7. 计算 SHA256，提取 metadata。
8. `INSERT OR IGNORE` 到 `docs`。
9. 更新 `files.sha256`，释放锁。
10. 更新文件名索引，并根据文件类型、watch 默认策略或 parsing-rules 创建 `parses` 任务。

必须执行文件变化检查的用户 path 操作:

| 操作 | 入口 |
|------|------|
| 主动解析 | `POST /parses` / SDK `ensure_parse()` / `mineru parse <path>` |
| 文件信息 | `GET /files/by-path?path=...` / SDK `get_file_by_path()` / `mineru show file <path>` |
| 按路径取文档 | `GET /docs/by-path?path=...` / SDK `get_doc_by_path(path)` |
| 按路径失效解析 | `POST /invalidate { path }` / SDK `invalidate(path=...)` / `mineru invalidate <path>` |

watch 流程也调用同一发现/刷新步骤:

1. `WatchLoop` 收到文件事件或执行初始扫描。
2. 按扩展名白名单和 exclude 规则过滤。
3. 调用统一文件发现/刷新步骤。

删除和不可达检测同样属于统一发现/刷新:

- watch file event、所有 scan 操作和上述 4 个同步 path 操作，都必须检查 path 当前是否存在。
- 如果 path 不存在，且所属 removable watch root 不可达，则标记为 `unreachable`。
- 如果 path 不存在，且不是设备整体不可达导致，则标记为 `deleted`，写入 `deleted_at`，保留 `sha256`。
- 标记 `deleted` 或 `unreachable` 时不清理 FTS；find 默认只返回 active file，search 默认优先返回 active file paths，只有在已索引 doc 没有任何 active file 时才 fallback 返回非 active file paths。

失败时按错误绑定对象写入对应层级:

- 路径实例当前不可读、权限不足、stat / SHA256 读取失败: 写入 `files.error_code` / `files.error_msg`。
- 已经计算出 SHA256，但内容本身无法完成基础 metadata 识别、加密或损坏: 写入 `docs.error_code` / `docs.error_msg`。
- 某次 parse batch 执行失败: 写入 `parses.error_code` / `parses.error_msg`。

### 4.3 解析请求到任务

显式解析请求流程：

1. CLI 或 SDK 调用 doclib `POST /parses`。
2. `ParseService` 先执行文件发现/刷新，确认 path 当前状态。
3. 如文件需要重新入库，则等待或同步完成入库，获得当前 `sha256`。
4. 查询 `(sha256, tier)` 下已完成批次，判断 page_range 是否覆盖请求范围。
5. 缓存命中则返回 `cache_hit=true` 和空 `wait_parse_ids`。
6. 已有 active parse 覆盖的页复用并提升 `priority`。
7. 未覆盖页插入新的 `parses` 记录，Agent 请求使用更高 `priority`。
8. `ParseWorker` 抢占任务并执行。

`--force` 对本次请求不复用 done 缓存，但可以复用已存在的 active parse。它不删除、不作废旧的 done 记录。增量页码解析和 force 重解析都通过多行 `parses` 表示。读取时只合并有效 done 批次；同一页出现多次时，按 `done_at` 选择最新有效页面。

`POST /invalidate` 是独立的缓存生命周期动作。第一版要求 `target="parses"`。被 invalidate 的记录不参与缓存覆盖判断、读取合并、搜索索引刷新和 compaction；对应 JSON 文件的物理删除由 cleanup 或 `Compaction` 后台完成。

### 4.4 ParseWorker 路由

ParseWorker 按 `tier`、`privacy` 和 parse-server 健康状态路由：

| 条件 | 执行路径 |
|------|----------|
| 未指定 tier | 通过当前目标 parse-server 能力发现解析为 `standard` 或 `pro`；不可解析为 `flash` |
| `tier=flash` | 直接本地调用轻量 parser，`via=local` |
| `tier=standard/pro` 且 `privacy=remote` | 优先调用 config 中的远端地址或默认 `mineru.net/api` |
| remote 不可用 | 尝试 fallback 到 local parse-server |
| `privacy=local` 且 local parse-server disabled | 返回 `no_engine` |
| local parse-server 配置但探活失败 | 返回可重试错误 |
| parse-server 不支持请求 tier | 返回 `tier_mismatch` |

关键约束：

- 用户显式 `--remote` 才允许上传文档。
- 未指定 tier 使用默认选择策略，但永远不会降级到 `flash`。
- 远端解析失败且属于文件损坏、加密等非网络错误时，不 fallback 到本地。
- 重试时保持原始 `privacy` 不变。
- `via` 只在解析完成后写入，避免失败记录误导。

## 5. 产物与搜索

解析产物按 `sha256`、实际使用的 `tier` 和解析页码批次隔离存储。doclib 的 `parsed/` 目录只持久化 Middle JSON 批次文件：

```text
~/.mineru/
  config.yaml
  doclib.sock
  doclib.db
  doclib.log
  data/
    temp/
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

每个 JSON 文件表示一次解析批次，文件名由页码范围和完成时间组成。每个 tier 独立目录，互不覆盖。remote 解析完成后也写入同一结构，供 CLI、SDK、搜索和缓存统一读取。

doclib 不在 `parsed/` 中保存 Markdown、Content List 或 HTML。它们都从 Middle JSON 读取时转换得到；搜索索引可以使用临时渲染出的 Markdown 文本，但不把这份 Markdown 保存为产物文件。

搜索策略：

- `fts_filenames` 用于文件名搜索。
- `fts_contents` 用于内容搜索，同一文档只保留最高 tier 文本。
- 内容超过 30K 字符时，保留 head + tail，完整内容仍从解析产物读取。
- 中文分词采用 jieba + FTS5 `unicode61` 的组合方案。

## 6. Server 生命周期

### 6.1 启动

启动顺序：

1. 创建 `data_dir`。
2. 初始化 SQLite schema、migration 和种子配置。
3. 清理 stale ingest 锁。
4. 将崩溃前 `parsing` 状态任务标记为 `failed` 并释放锁。
5. 创建 services。
6. managed 模式下启动 local parse-server。
7. 启动后台 asyncio tasks。
8. 启动 UDS 上的 uvicorn。

### 6.2 关闭

关闭顺序：

1. 通知后台任务停止，等待当前任务收尾。
2. managed 模式下终止 local parse-server。
3. 关闭数据库资源。
4. 删除 socket 文件。

### 6.3 崩溃恢复

server 启动时执行恢复：

| 恢复动作 | 目的 |
|----------|------|
| 清空 `files.locked_at` | 释放未完成 ingest 锁 |
| `parsing -> failed` | 避免解析任务永久卡住 |
| 检查已入库文件是否仍存在 | 标记 deleted 或 unreachable |
| 检查 managed parse-server | 不存在则自动拉起，连续失败降级 disabled |

## 7. 配置管理

doclib 启动期配置由 YAML / 环境变量管理，运行期策略配置由 SQLite `config` 表和 `mineru config` 管理。

基础配置：

| key | 默认值 | 说明 |
|-----|--------|------|
| `data_dir` | `~/.mineru/data` | 数据目录 |
| `scan_interval_sec` | `300` | watch 全量扫描间隔 |
| `ingest_lock_timeout_sec` | `60` | ingest 锁超时 |
| `parse_lock_timeout_sec` | `1800` | parse 锁超时 |

parse-server 配置：

| key | 默认值 | 说明 |
|-----|--------|------|
| `parse_server.local.mode` | `disabled` | `disabled` / `managed` / `self_hosted` |
| `parse_server.local.managed_tier` | `standard` | managed 模式启动的 tier |
| `parse_server.local.self_hosted_url` | 无 | self-hosted 地址 |
| `parse_server.remote.url` | `https://mineru.net/api` | 远端地址 |
| `parse_server.remote.api_key` | 无 | 远端 API Key |

API Key 优先级：环境变量 `MINERU_API_KEY` 高于 config 表中的 API Key。

配置优先级统一为：CLI 参数 > 环境变量 > 文件配置 / SQLite 配置 > 内置默认值。启动前文件配置和 SQLite 配置不应定义同一配置项；需要在 doclib server 启动前确定的配置使用文件配置，运行时可变配置使用 SQLite。

Watch、parsing-rules 和 exclude 规则也由 CLI 写入 SQLite。Parsing-rules 不得导致静默上传：只有规则显式允许 remote 时，才可以走远端解析。

## 8. 文件类型与处理策略

Watch 默认发现文档、Office、电子书、Apple 文档、文本、网页和邮件格式。图片不在 watch 白名单中，但可以通过显式 `mineru parse image.png` 触发。

处理路径：

| 类型 | 策略 |
|------|------|
| 纯文本 | 直接读取，无需模型解析 |
| Office / HTML | 本地 CPU 全量解析 |
| PDF / Image | 按 tier 路由到默认选择 / flash / standard / pro |

## 9. 错误与恢复

### 9.1 可插拔设备恢复

`DeviceMonitor` 只对 `removable=1` 的 watch target 做设备可达性判断。

设备不可达时:

1. watch target 标记为 `unreachable`。
2. 该 watch 下 `active` files 标记为 `unreachable`。
3. 不删除 file row、doc、parse cache 或 FTS。

设备恢复时:

1. watch target 标记为 `active`。
2. 该 watch 下 `unreachable` files 标记为 `active`。
3. 立即对该 watch 执行一次 scan: 文件仍不存在则标记 `deleted`；文件 stat 变化则清空 `sha256` 等待 ingest；文件未变化则保持当前绑定。

### 9.2 Cleanup 边界

deleted file cleanup 与 orphan doc cleanup 是两个独立动作。

- deleted file cleanup 只删除 `status='deleted'` 的 file row，并清理对应 `fts_filenames`。
- 手动 cleanup deleted 立即删除所有 deleted file rows。
- 后台 cleanup deleted 固定保留 7 天后删除。
- deleted cleanup 不自动删除 docs、parses、parsed JSON 或 `fts_contents`。
- orphan doc cleanup 只处理完全没有任何 file row 关联的 docs；执行时才删除 docs、parses、`fts_contents` 和 parsed JSON。

`fts_filenames` 生命周期跟随 `files` row；`fts_contents` 生命周期跟随 `docs` row。

### 9.3 Forget Path

`forget` 是显式移除 doclib path 记录的维护动作。

- `forget` 删除匹配的 `files` row，并清理对应 `fts_filenames`。
- `forget` 不删除真实磁盘文件。
- `forget` 不删除 `docs`、`parses`、parsed JSON 或 `fts_contents`。
- 如果最后一个 file row 被 forget，对应 doc 会成为 orphan，由 `cleanup orphan-docs` 后续处理。
- 如果 path 是目录，P0 永远递归匹配该目录树下的 file rows，不提供 `recursive` 参数。
- 如果 path 是已配置 watch root，默认拒绝；应先执行 watch remove。
- 如果 path 位于 active watch root 下，允许 forget，但后续 scan 可能重新发现。
- `forget` 不是 ignore rule，也不是 invalidate。

完整规则见 [ADR-0008](decisions/0008-doclib-forget-path.md)。

### 9.4 Scan Task

`scan` 是 doclib server 侧后台任务。

核心边界:

- CLI / SDK / HTTP 只创建 scan task 和查询状态，不直接执行扫描。
- watch initial scan、watch rescan、设备恢复后的 watch scan 都复用 `ScanService + ScanWorker`。
- `scans` 表同时作为任务表和轻量 scan log。
- scan 不同步执行 ingest；新文件或变化文件通过 `files.sha256=NULL` 进入 ingest worker。
- scan status 为 `pending` / `running` / `done` / `failed`，P0 不做 cancelled。
- scan kind 为 `manual` / `watch`。
- source 是 best-effort 记录字段，P0 不作为行为依据。

P0 API:

- `POST /scans`
- `GET /scans`
- `GET /scans/{scan_id}`

完整规则见 [ADR-0009](decisions/0009-doclib-scan-task.md)。

### 9.5 错误分类

错误按层级存储和恢复：

| 层级 | 示例 | 存储位置 | 恢复策略 |
|------|------|----------|----------|
| File | 路径不存在、权限不足、文件被锁、SHA256 读取失败 | `files.error_code` | 清除锁后可重试 |
| Doc | 文件损坏、加密、metadata 读取失败、内容不可识别 | `docs.error_code` | 不自动重试，除非文件内容变化或用户显式重试 |
| Parse | 引擎崩溃、超时、OOM | `parses.error_code` | 标记 failed，可由用户或策略重试 |
| Parse-server | 探活失败、服务不可用 | `parses.error_code` | 可重试，保留 privacy |

错误归属按对象划分，而不是按代码调用栈划分:

- `files.error_*`: 这个 path 实例当前有问题。
- `docs.error_*`: 这个 `sha256` 内容身份本身有问题。
- `parses.error_*`: 这一次 tier/page_range parse batch 有问题。

如果 ParseWorker 根据 `sha256` 找不到任何 active file，当前 parse batch 应写 `parses.error_code=no_accessible_file`。如果它找到了 file row 但 path 已不可读，还应同步更新 file 的可达性或 file 级错误。

parse-server 相关错误：

| code | 含义 | 可重试 |
|------|------|:--:|
| `quality_tier_unavailable` | 主动阅读需要质量 tier，但只有 `flash` 可用 | 否 |
| `no_engine` | 本地请求 standard/pro，但 local parse-server disabled | 否 |
| `engine_unavailable` | local parse-server 已配置但探活失败 | 是 |
| `parse_server_unavailable` | remote 不可用且 local fallback 不可用 | 是 |
| `tier_mismatch` | parse-server 不支持请求 tier | 否 |
| `parse_failed` | 文档损坏、加密等解析失败 | 否 |
| `internal_error` | parse-server 500 | 否 |

完整错误模型、`retryable` 和 `user_action` 见 [错误码体系](errors.md)。重试次数、退避算法和触发方仍需后续定稿。

## 10. 依赖边界

新增或重点依赖：

| 包 | 用途 |
|----|------|
| `typer` | CLI |
| `httpx` | Product SDK HTTP 客户端，支持 UDS |
| `uvicorn` / `fastapi` | doclib 和 parse-server HTTP |
| `aiosqlite` | 异步 SQLite |
| `watchfiles` | 文件系统监控 |
| `jieba` | 中文分词 |
| `pypdfium2` | PDF metadata 和 flash 解析辅助 |

对外 API 层应遵守惰性加载原则，避免导入 SDK 或 CLI 时触发 GPU、torch、transformers 等重依赖。

## 与其他文档的关系

- 产品优先级见 [产品路线图](roadmap.md)。
- CLI 如何触发本地能力见 [CLI 规格](cli.md)。
- API 如何暴露解析任务见 [Unified API](api.md)。
- SDK 分层见 [SDK 设计](sdk.md)。
- 解析 tier 语义见 [解析 Tier](tiers.md)。
- middle_json 结构见 [Middle JSON](middle-json.md)。
- 错误分类见 [错误码体系](errors.md)。
- 配置项收敛见 [配置体系](config.md)。
- 未决问题集中维护在 [开放问题清单](open-questions.md)。
