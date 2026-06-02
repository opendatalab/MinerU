# MinerU 技术设计文档

> **状态**：初稿，基于 everydoc 架构适配 MinerU 需求。
>
> 本文档覆盖 `mineru server` 及相关组件的内部技术设计。CLI 参数设计见 NEXT-CLI.md，REST API 规范见 NEW-API.md，错误码见 NEXT-ERROR.md，SDK 接口与包结构见 NEXT-SDK.md。

---

## 1. 系统架构

```
┌──────────────┐   Unix Domain Socket    ┌───────────────────────────────────┐
│  mineru CLI  │ ──── HTTP + JSON ────→  │         mineru server             │
│  (typer)     │ ←─── HTTP + JSON ────   │         (FastAPI + uvicorn)       │
└──────────────┘                         │                                   │
                                         │  ┌─ Routes ── Services ── Core   │
┌──────────────┐                         │  │                               │
│  MCP Server  │ ──── HTTP + JSON ────→  │  └─ Background                   │
│  (P1)        │                         │     ├─ WatchLoop                  │
└──────────────┘                         │     ├─ RegistrationWorker         │
                                         │     ├─ ParseWorker                │
┌──────────────┐                         │     └─ DeviceMonitor              │
│  MinerU.app  │ ──── HTTP + JSON ────→  │                                   │
│  (P1)        │                         └────────────┬──────────────────────┘
└──────────────┘                                      │
                                                 ┌────┴────┐
                                                 │ SQLite  │
                                                 │ + FTS5  │
                                                 └─────────┘
```

### 1.1 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 进程模型 | 单进程 asyncio | 简单可靠，避免 IPC 复杂度 |
| CLI ↔ Server 通信 | HTTP + JSON over Unix Domain Socket | 安全（权限 0600）、低延迟、无端口冲突 |
| 数据库 | SQLite (WAL) + FTS5 | 零运维、单文件、全文搜索内置 |
| 任务队列 | DB rows + 时间戳锁 | 无外部依赖，崩溃恢复简单 |
| CPU-bound offload | `asyncio.to_thread()` | SHA-256 计算、文件解析等交给线程池 |
| 文件监控 | watchfiles (FSEvents/inotify) | 跨平台、高性能 |

### 1.2 Socket 与端口

| 通道 | 路径 / 地址 | 权限 | 用途 |
|------|-----------|------|------|
| UDS | `/tmp/mineru.sock` | `0600` | CLI / MCP / 桌面端通信 |
| TCP | `127.0.0.1:<随机端口>` | loopback only | 可选，供浏览器访问 Web UI (P1) |

Server 启动后写入 lock 文件 `~/MinerU/server.lock`，内容为 PID 和 TCP 端口。CLI 通过此文件发现 server 状态。

---

## 2. 模块分层

```
Routes（HTTP 路由处理，参数校验，响应格式化）
  │
  ▼
Services（业务逻辑，状态管理，跨模块协调）
  │
  ▼
Core（数据层：DB、FTS、文件 IO）
```

### 2.1 Routes → Services 映射

本地 Server 的 HTTP 接口是面向 CLI/Agent 的内部协议，与 `parser/api_server.py`（无状态解析 API）和 mineru.net/api 完全独立，不需要格式一致性。

| Route 文件 | 端点 | 调用的 Service |
|-----------|------|---------------|
| `routes/parse.py` | `POST /parse`, `GET /parse/status` | `ParseService` |
| `routes/search.py` | `GET /search`, `GET /find` | `SearchService` |
| `routes/info.py` | `GET /info` | `SearchService` / DB 直查 |
| `routes/config.py` | `GET/POST /config/*` | `ConfigService` |
| `routes/cleanup.py` | `POST /cleanup/*` | `CleanupService` |
| `routes/server.py` | `GET /server/status` | 直接读取 AppState |

响应格式：成功时直接返回数据对象（无 envelope），失败时返回 OpenAI 兼容的 `{"error": {"type": ..., "code": ..., "message": ..., "param": ...}}`。

### 2.2 Services

| Service | 依赖 | 职责 |
|---------|------|------|
| `ParseService` | DB, FTS, ConfigService | 接收解析请求、文件注册（SHA-256 + metadata）、分配任务、管理 per-tier 解析状态 |
| `SearchService` | DB, FTS | 内容搜索 (fts_index)、文件名搜索 (fts_filenames) |
| `ConfigService` | DB | KV 配置读写、Watch 目录 CRUD、Parsing-Rules CRUD、路径排除判断 |
| `CleanupService` | DB | 孤儿文档清理、已删除文件清理、解析缓存清理 |

### 2.3 Core

| 模块 | 职责 |
|------|------|
| `db.py` | SQLite 连接管理、schema 初始化、migration、CRUD 操作 |
| `fts.py` | FTS5 索引的插入/更新/搜索/删除 |
| `file_io.py` | SHA-256 计算（线程池）、文件 stat、metadata 提取、文本预览提取 |

---

## 3. 数据库 Schema

### 3.1 核心表

#### files — 文件实例

```sql
CREATE TABLE files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    ext             TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    mtime_ms        INTEGER NOT NULL,
    sha256          TEXT    REFERENCES docs(sha256),
    watch_id        INTEGER REFERENCES watch_targets(id),
    scan_status     TEXT    NOT NULL DEFAULT 'active',  -- active / deleted / unreachable
    reg_locked_at   INTEGER,          -- 注册锁，Unix epoch ms
    reg_error       TEXT,
    first_seen_at   INTEGER NOT NULL,  -- Unix epoch ms
    updated_at      INTEGER NOT NULL   -- Unix epoch ms
);

CREATE INDEX idx_files_sha256 ON files(sha256);
CREATE INDEX idx_files_watch_id ON files(watch_id);
CREATE INDEX idx_files_scan_status ON files(scan_status);
```

#### docs — 文档内容（SHA-256 去重）

```sql
CREATE TABLE docs (
    sha256          TEXT    PRIMARY KEY,
    size_bytes      INTEGER NOT NULL,
    mime_type       TEXT,
    page_count      INTEGER,
    lang            TEXT,
    title           TEXT,
    author          TEXT,
    is_encrypted    INTEGER NOT NULL DEFAULT 0,
    first_seen_at   INTEGER NOT NULL,  -- Unix epoch ms
    updated_at      INTEGER NOT NULL   -- Unix epoch ms
);
```

> **注意**：`docs` 表不含解析状态字段（无 `parse_status`、`preview_text` 等），解析状态全部在 `doc_parses` 表中按 tier 独立管理。注册阶段仅填充 metadata（SHA-256、page_count、mime_type 等），不提取预览文本——预览文本由 flash tier 的解析产出。

#### doc_parses — 每个 tier 独立的解析记录

```sql
CREATE TABLE doc_parses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256          TEXT    NOT NULL REFERENCES docs(sha256),
    tier            TEXT    NOT NULL,  -- flash / standard / pro
    parse_status    TEXT    NOT NULL DEFAULT 'pending',  -- pending / parsing / done / error
    parsed_pages    TEXT,              -- JSON array, e.g. [1,2,3,4,5]
    total_pages     INTEGER,
    parse_locked_at INTEGER,           -- Unix epoch ms
    parse_error     TEXT,
    parse_path      TEXT,              -- 解析产物存储路径
    parsed_at       INTEGER,           -- Unix epoch ms
    created_at      INTEGER NOT NULL,  -- Unix epoch ms
    updated_at      INTEGER NOT NULL,  -- Unix epoch ms
    UNIQUE(sha256, tier)
);

CREATE INDEX idx_doc_parses_status ON doc_parses(parse_status);
```

> **设计要点**：每个 tier 的解析记录独立存储在 `doc_parses` 表中，同一文档可同时拥有 flash、standard、pro 三份解析结果。`parsed_pages` 记录已解析的页码列表，支持增量解析。`--force` 时删除对应 tier 的记录后重建。

#### fts_index — 全文搜索索引（flash 解析文本）

```sql
CREATE VIRTUAL TABLE fts_index USING fts5(
    doc_sha256 UNINDEXED,
    indexed_text,
    title,
    author,
    filename,
    tokenize='unicode61'
);
```

#### fts_parsed — 全文搜索索引（standard/pro 完整解析文本）

```sql
CREATE VIRTUAL TABLE fts_parsed USING fts5(
    doc_sha256 UNINDEXED,
    full_text,
    title,
    author,
    filename,
    tokenize='unicode61'
);
```

#### fts_filenames — 文件名搜索

```sql
CREATE VIRTUAL TABLE fts_filenames USING fts5(
    file_id UNINDEXED,
    filename_text,
    ext,
    tokenize='unicode61'
);
```

#### watch_targets — 监控目录

```sql
CREATE TABLE watch_targets (
    id              INTEGER PRIMARY KEY,  -- 路径的 hash，稳定 ID
    path            TEXT    NOT NULL UNIQUE,
    label           TEXT,
    removable       INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    recursive       INTEGER NOT NULL DEFAULT 0,
    watch_status    TEXT    NOT NULL DEFAULT 'active',  -- active / unreachable
    unreachable_at  INTEGER,           -- Unix epoch ms
    last_scan_at    INTEGER,           -- Unix epoch ms
    last_scan_files INTEGER DEFAULT 0
);
```

#### rules — 解析规则与排除规则

```sql
CREATE TABLE rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    rule_type       TEXT    NOT NULL,  -- exclude / parsing_rule
    pattern         TEXT    NOT NULL,  -- fnmatch glob pattern
    tier            TEXT,              -- parsing_rule: flash / standard / pro
    pages           TEXT,              -- parsing_rule: page range, e.g. "all"
    remote          INTEGER NOT NULL DEFAULT 0,  -- parsing_rule: 是否允许远端
    enabled         INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 0,
    hit_count       INTEGER NOT NULL DEFAULT 0
);
```

#### config — KV 全局配置

```sql
CREATE TABLE config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

#### _migrations — Schema 版本追踪

```sql
CREATE TABLE _migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  INTEGER NOT NULL,  -- Unix epoch ms
    description TEXT
);
```

### 3.2 SQLite Pragmas

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA mmap_size = 268435456;       -- 256 MB
PRAGMA cache_size = -20480;         -- 20 MB
PRAGMA temp_store = MEMORY;
PRAGMA wal_autocheckpoint = 1000;
```

### 3.3 默认配置种子数据

```sql
INSERT INTO config (key, value) VALUES
    ('data_dir',              '~/MinerU'),
    ('default_tier',          'flash'),
    ('scan_interval_sec',     '300'),
    ('reg_lock_timeout_sec',  '60'),
    ('parse_lock_timeout_sec', '1800'),
    ('device_check_interval_sec', '5');
```

---

## 4. 后台任务

### 4.1 组件职责

| 组件 | 职责 | 并发数 |
|------|------|--------|
| WatchLoop | 文件系统事件监控（watchfiles），发现文件变更 | 1 |
| RegistrationWorker | 计算 SHA-256、提取 metadata、写入 files/docs、FTS 索引文件名、触发默认 flash 解析 | 2 |
| ParseWorker | 执行解析任务（调用 `mineru.parser`），写入 FTS | 2 |
| DeviceMonitor | 定期检测 removable watch 路径的可达性 | 1 |

所有组件均为 asyncio task，在 server 启动时创建，关闭时 graceful stop。

### 4.2 Worker 任务获取

无外部消息队列。Worker 通过原子 SQL 获取任务：

```sql
-- RegistrationWorker 获取任务
UPDATE files
SET reg_locked_at = ?
WHERE id = (
    SELECT id FROM files
    WHERE sha256 IS NULL
      AND scan_status = 'active'
      AND (reg_locked_at IS NULL OR reg_locked_at < ?)
    ORDER BY first_seen_at ASC
    LIMIT 1
)
RETURNING *;

-- ParseWorker 获取任务
UPDATE doc_parses
SET parse_locked_at = ?, parse_status = 'parsing'
WHERE id = (
    SELECT id FROM doc_parses
    WHERE parse_status = 'pending'
      AND (parse_locked_at IS NULL OR parse_locked_at < ?)
    ORDER BY created_at ASC
    LIMIT 1
)
RETURNING *;
```

锁超时：Registration 60 秒，Parse 30 分钟。超时后锁可被其他 worker 窃取。

### 4.3 per-request DB 连接

每次 `execute()` / `fetchone()` / `fetchall()` 打开独立的 `aiosqlite` 连接，操作完成后提交并关闭。避免跨 worker 的游标冲突。

---

## 5. 数据流

### 5.1 两阶段模型

```
阶段 1: 注册（RegistrationWorker）
  - 计算 SHA-256
  - 提取 metadata (mime_type, page_count, title, author)
  - FTS 索引文件名
  - 触发默认 flash 解析（INSERT INTO doc_parses, tier='flash', parse_status='pending'）

阶段 2: 解析（ParseWorker）
  - flash tier：提取首尾各 5 页文本，写入 fts_index
  - standard/pro tier：完整解析，写入 fts_parsed
```

> **设计简化**：everydoc 的 L1 Index（注册 + 预览文本提取）在 MinerU 中被拆为：注册（仅 SHA-256 + metadata）+ flash 解析（首尾页文本）。用户无需理解"索引"和"解析"两个概念——所有内容提取都是 parse，只是 tier 不同。

### 5.2 文件发现 → 注册

```
Watch 检测到文件事件
  → 过滤（ext 白名单 + exclude 规则）
  → INSERT OR IGNORE INTO files (path, filename, ext, size_bytes, mtime_ms, sha256=NULL)
  → RegistrationWorker.acquire_task(): UPDATE files SET reg_locked_at WHERE sha256 IS NULL
  → process_file():
       1. compute SHA-256 (asyncio.to_thread)
       2. INSERT OR IGNORE INTO docs (sha256, size_bytes)
       3. extract metadata (title, author, page_count, mime_type)
       4. FTS insert fts_filenames
       5. UPDATE files SET sha256=?, reg_locked_at=NULL
       6. check parsing-rules → 命中时用规则指定的 tier，否则用 default_tier (flash)
       7. INSERT INTO doc_parses (sha256, tier, parse_status='pending')
  → 失败时: UPDATE files SET sha256=NULL, reg_error=?（可重试）
```

### 5.3 解析请求

```
用户请求: mineru parse doc.pdf --tier standard --pages 1~10
  → CLI → Product SDK (client.py) → Server POST /parse
  → ParseService.request_parse():
       1. 查 files 表获取 sha256（如果文件未注册，先同步注册）
       2. 查 doc_parses WHERE sha256=? AND tier='standard'
       3. 已有记录且 parsed_pages 覆盖请求范围 → 直接返回缓存
       4. 否则 → INSERT OR UPDATE doc_parses (parse_status='pending', parsed_pages 合并)
       5. --force 时 → DELETE + INSERT
  → ParseWorker.acquire_task()
  → process_doc():
       1. 从 doc_parses 读取 tier、待解析页码
       2. resolve_engine(tier) 确定 backend
       3. 调用 mineru.parser.parse(path, tier=..., pages=...)
       4. 写入解析产物到 ~/MinerU/parsed/{sha256[:2]}/{sha256}/{tier}/
       5. flash → UPDATE fts_index；standard/pro → UPDATE fts_parsed
       6. UPDATE doc_parses SET parse_status='done', parsed_pages=?, parsed_at=?
  → 失败时: UPDATE doc_parses SET parse_status='error', parse_error=?
```

### 5.3 解析产物存储

```
~/MinerU/
  mineru.db
  mineru.log
  parsed/
    ab/                             # sha256 前 2 字符
      ab3f...7e2d/                  # 完整 sha256
        flash/
          output.md
          middle.json
        standard/
          output.md
          middle.json
          images/
        pro/
          output.md
          middle.json
          images/
```

每个 tier 独立目录，互不覆盖。

### 5.4 搜索

```
用户请求: mineru search "关键词"
  → SearchService.search():
       1. 分词（jieba / unicode61）
       2. FTS5 MATCH on fts_parsed（优先）+ fts_index（补充）
       3. JOIN files/docs 获取元数据
       4. 按 doc_sha256 去重（同一文档可能有多个路径）
       5. 结果标注解析状态（flash / standard / pro）
```

### 5.5 可插拔设备

```
DeviceMonitor 每 5 秒检测:
  → os.stat(watch_path)
  → 成功（设备插入/恢复）:
       UPDATE watch_targets SET watch_status='active', unreachable_at=NULL
       UPDATE files SET scan_status='active' WHERE watch_id=? AND scan_status='unreachable'
       触发增量扫描
  → 失败（设备拔出）:
       UPDATE watch_targets SET watch_status='unreachable', unreachable_at=?
       UPDATE files SET scan_status='unreachable' WHERE watch_id=? AND scan_status='active'
       正在进行的解析任务标记特殊错误（非永久失败，设备恢复后可重试）
```

---

## 6. Server 生命周期

### 6.1 启动

```python
async def startup():
    # 1. 创建数据目录
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # 2. 初始化数据库（建表 + migration + 种子数据）
    await db.initialize()

    # 3. 崩溃恢复：清理 stale 锁
    await db.execute("UPDATE files SET reg_locked_at = NULL WHERE reg_locked_at IS NOT NULL")
    await db.execute("UPDATE doc_parses SET parse_locked_at = NULL, parse_status = 'error' WHERE parse_status = 'parsing'")

    # 4. 创建 services
    config_svc = ConfigService(db)
    parse_svc = ParseService(db, fts, config_svc)
    search_svc = SearchService(db, fts)

    # 5. 启动后台任务
    asyncio.create_task(watch_loop.run())
    for _ in range(2):
        asyncio.create_task(registration_worker.run())
        asyncio.create_task(parse_worker.run())
    asyncio.create_task(device_monitor.run())

    # 6. 写入 lock 文件
    write_lock_file(pid=os.getpid(), port=tcp_port)
```

### 6.2 关闭

```python
async def shutdown():
    # 1. 停止所有后台任务（设置 stop event，等待当前任务完成）
    await watch_loop.stop()
    await registration_worker_pool.stop()
    await parse_worker_pool.stop()
    await device_monitor.stop()

    # 2. 关闭数据库连接
    await db.close()

    # 3. 删除 lock 文件和 socket 文件
    remove_lock_file()
    remove_socket()
```

### 6.3 崩溃恢复

Server 启动时执行以下恢复操作：

| 操作 | 目的 |
|------|------|
| 清除所有 `reg_locked_at` | 释放因崩溃而未释放的注册锁 |
| 清除所有 `parse_locked_at`，将 `parsing` 状态改为 `error` | 释放解析锁，标记为可重试 |
| 检查已注册文件的活性（`os.path.exists`） | 标记已删除的文件 |

---

## 7. 配置管理

### 7.1 存储

所有配置存储在 SQLite `config` 表中（KV store），不使用外部配置文件。CLI 通过 `mineru config` 命令管理。

### 7.2 配置项

| key | 默认值 | 说明 |
|-----|--------|------|
| `data_dir` | `~/MinerU` | 数据目录根路径 |
| `default_tier` | `flash` | Watch 自动解析的默认 tier |
| `scan_interval_sec` | `300` | Watch 全量扫描间隔（秒） |
| `index_lock_timeout_sec` | `60` | 注册锁超时 |
| `parse_lock_timeout_sec` | `1800` | 解析锁超时（30 分钟） |
| `fts_preview_kb` | `30` | 预览文本截取大小（KB） |

### 7.3 Watch 目录

通过 `watch_targets` 表管理，CLI 操作：

```bash
mineru config watch add ~/Documents
mineru config watch add /Volumes/SSD --removable
mineru config watch list
mineru config watch rm ~/Documents
```

约束：不允许嵌套、必须绝对路径。

### 7.4 Parsing-Rules

通过 `rules` 表管理，使用 `fnmatch` glob 匹配：

```bash
mineru config parsing-rules add "*/论文/*" --tier standard --pages all
mineru config parsing-rules add "*/合同/*" --tier pro --remote
mineru config parsing-rules list
mineru config parsing-rules rm <id>
```

规则命中时，系统检查是否具备对应 tier 的能力（本地引擎可用 或 规则包含 `--remote`）。不会因规则配置而静默上传文件到远端。

### 7.5 排除规则

```bash
mineru config exclude add "*.tmp"
mineru config exclude list
mineru config exclude rm <id>
```

默认排除（硬编码，不可通过 config 修改）：

```
*/Library/*, */.git/*, */node_modules/*, */vendor/*, */go/pkg/*,
*/__pycache__/*, */.venv/*, */miniconda3/*, */.nvm/*, */.docker/*,
*/target/*, */dist/*, */build/*
```

---

## 8. 文件类型与处理策略

### 8.1 Watch 白名单

Watch 自动发现以下类型的文件：

| 类别 | 扩展名 |
|------|--------|
| 文档 | pdf, doc, docx, xls, xlsx, ppt, pptx, csv, rtf, odt, ods |
| 电子书 | epub, mobi |
| Apple | pages, key, numbers |
| 文本 | txt, md, markdown, rst, tex |
| 网页 | html, htm |
| 邮件 | eml, mbox |

图片（jpg, png 等）不在 Watch 白名单中，但可通过 `mineru parse image.png` 显式触发。

### 8.2 解析路由

```
发现文件 → 检查路径 → 计算 SHA-256 → 检查 SHA-256 去重
                                        ↓
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                     纯文本文件      Office 文档     PDF / Image
                     (txt, md...)   (docx, pptx,   (pdf, jpg...)
                          │         xlsx, html)          │
                          ▼             │                ▼
                     直接读取           ▼           按 tier 解析
                     (无需解析)    本地全量解析      flash / standard / pro
                                  (成本低，CPU)
```

---

## 9. FTS 分词策略

### 9.1 中文分词

使用 jieba 分词 + unicode61 tokenizer 的组合方案：

- **索引阶段**：jieba 切词后用 ``（Unit Separator）连接，输入 FTS5
- **查询阶段**：jieba 切词后用空格连接
- **展示阶段**：`strip_sep()` 去除分隔符

### 9.2 两张 FTS 表

| 表 | 数据来源 | 时机 |
|---|---------|------|
| `fts_index` | flash tier 解析的首尾页文本 | flash 解析完成时 |
| `fts_parsed` | standard/pro tier 的完整解析文本 | standard/pro 解析完成时 |

搜索时优先查 `fts_parsed`（覆盖更全），`fts_index` 作为补充（仅 flash 解析的文件也能搜到首尾页内容）。

---

## 10. 错误分类

错误分为三个层级，对应不同的处理策略：

| 层级 | 示例 | 存储位置 | 恢复策略 |
|------|------|---------|---------|
| File 级 | 路径不存在、权限不足 | `files.reg_error` | 清除 sha256，可重试 |
| Doc 级 | 文件损坏、加密 | `files.reg_error` | 不自动重试，需用户介入 |
| Parse 级 | 引擎崩溃、超时、OOM | `doc_parses.parse_error` | 标记 `parse_status='error'`，可通过 `--force` 重试 |

---

## 11. 依赖清单

### 11.1 新增依赖

| 包 | 用途 |
|---|------|
| `typer` | 新 CLI 框架（替代 click） |
| `httpx` | Product SDK 的 HTTP 客户端（支持 UDS） |
| `uvicorn` | ASGI 服务器 |
| `fastapi` | Server HTTP 框架（已有依赖） |
| `aiosqlite` | 异步 SQLite |
| `watchfiles` | 文件系统监控（FSEvents / inotify） |
| `jieba` | 中文分词（FTS 索引） |

### 11.2 现有依赖（保持不变）

| 包 | 用途 |
|---|------|
| `pydantic` | 数据模型 |
| `rich` | CLI 格式化输出 |
| `loguru` | 日志 |
