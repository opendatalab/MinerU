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

Server 启动后 UDS sock 文件位于 `/tmp/mineru.sock`（权限 0600）。CLI 通过 sock 连接检测 server 状态。TCP 端口按 config 中 `http.enabled` 决定是否启动。

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
| `SearchService` | DB, FTS | 内容搜索 (fts_contents)、文件名搜索 (fts_filenames) |
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
    birthtime_ms    INTEGER,           -- 文件创建时间，Unix epoch ms
    sha256          TEXT    REFERENCES docs(sha256),
    watch_id        INTEGER REFERENCES watch_targets(id),
    scan_status     TEXT    NOT NULL DEFAULT 'active',  -- active / deleted / unreachable
    locked_at       INTEGER,           -- Unix epoch ms
    error_code      TEXT,              -- 错误码，e.g. "permission_denied"
    error_msg       TEXT,              -- 人类可读错误描述
    deleted_at      INTEGER,           -- 标记删除时间，Unix epoch ms
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
    subject         TEXT,
    keywords        TEXT,
    is_encrypted    INTEGER NOT NULL DEFAULT 0,
    is_scanned      INTEGER NOT NULL DEFAULT 0,
    meta_tier       TEXT,               -- 当前 metadata 的来源 tier，NULL 表示注册阶段
    first_seen_at   INTEGER NOT NULL,  -- Unix epoch ms
    updated_at      INTEGER NOT NULL   -- Unix epoch ms
);
```

> **docs 元数据填充策略**：
> - **注册阶段**（register_file）：SHA-256 已全量读文件，顺手通过 pypdfium2 / python-docx 提取 metadata，成本可忽略。page_count 准确（pypdfium2 仅读 Catalog），title/author/subject/keywords 取自文件 metadata（可能不准，做保护性截断：title≤500、author≤200、subject≤1000、keywords≤1000）。`meta_tier=NULL`
> - **解析完成后**（ParseWorker）：引擎产出的 metadata 仅在 `tier ≥ meta_tier`（或 meta_tier 为 NULL）时 UPDATE docs 覆盖，同时更新 `meta_tier`。同一 tier 的多次批次不会重复更新
> - 解析状态全部在 `parses` 表中按 tier 独立管理

#### parses — 解析记录（每次解析请求为一行）

```sql
CREATE TABLE parses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT    NOT NULL REFERENCES docs(sha256),
    tier        TEXT    NOT NULL,         -- flash / standard / pro
    pages       TEXT    NOT NULL,         -- 页码范围，正值已展开，e.g. "1~5,46~50"
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending / parsing / done / failed
    priority    INTEGER NOT NULL DEFAULT 0,
    locked_at   INTEGER,                 -- Unix epoch ms
    error_code  TEXT,                   -- 错误码，e.g. "parse_timeout"
    error_msg   TEXT,                   -- 人类可读错误描述
    output_path TEXT,                    -- 解析产物存储路径
    done_at     INTEGER,                 -- Unix epoch ms
    created_at  INTEGER NOT NULL,        -- Unix epoch ms
    updated_at  INTEGER NOT NULL         -- Unix epoch ms
);

CREATE INDEX idx_parses_status ON parses(status, priority DESC, created_at ASC);
CREATE INDEX idx_parses_doc ON parses(sha256, tier);
```

**pages 字段**：
- 用 range 字符串描述，如 `"1~5"`、`"1~5,46~50"`、`"1~1000"`
- 仅存正值——用户输入的负索引（如 `-5~-1`）和 `all` 在插入前根据 `docs.page_count` 展开为正值
- 不存储为逐页展开的数组（1000 页全解析存 `"1~1000"`，不是 `"1,2,3,...,1000"`）

**并发**：Worker 获取单条 batch：
```sql
UPDATE parses
SET locked_at=?, status='parsing'
WHERE id = (
    SELECT id FROM parses
    WHERE status='pending' AND (locked_at IS NULL OR locked_at<?)
    ORDER BY priority DESC, created_at ASC LIMIT 1
)
RETURNING *;
```

**增量解析**：同一 sha256+tier 可有多行（如 1~5 和 6~10 是两个 batch），各自独立执行。Agent 等待时 `priority=1`，Watch 后台 `priority=0`。

**Force 重解析**：直接 `INSERT` 新行即可（允许与已有行 pages 重叠）：

```python
# --force --pages 4~7
await db.execute(
    "INSERT INTO parses (sha256, tier, pages, status, priority, created_at, updated_at) "
    "VALUES (?, ?, ?, 'pending', 1, ?, ?)",
    (sha256, tier, "4~7", now_ms(), now_ms())
)
```

**缓存命中**（`request_parse()` 时）：查该 (sha256, tier) 所有 done 行，按 `done_at DESC` 排序取最新覆盖：

```python
def pages_covered(request_pages: str, done_rows: list[dict]) -> bool:
    """检查请求的 pages 是否已被 done 批次完全覆盖。
    按 done_at 倒序，取最新 batch 的覆盖范围。"""
    needed = parse_range_set(request_pages)      # → {1,2,3,4,5}
    covered = set()
    for r in sorted(done_rows, key=lambda r: r["done_at"], reverse=True):
        covered |= parse_range_set(r["pages"])
        if needed <= covered:
            return True
    return False
```

**Compaction**（后台定时任务）：合并同一 (sha256, tier) 的 done 行，消除重叠和邻接区间，减少 parses 表记录数：

```
合并前                         合并后
"1~5" + "6~10"             → "1~10"
"1~5" + "4~7"              → "1~7"
"1~5" + "6~10" + "12~15"   → "1~10" + "12~15"

算法：
1. 取出所有 done 行，展开 pages 为整数集合
2. 排序后合并邻接区间（e.g. [1..5]+[6..10] → "1~10"）
3. DELETE 旧 done 行 + INSERT 合并后的新行
4. done_at 取被合并行中的最大值
```

Compaction 不在解析 critical path，作为低优先级后台周期性任务运行。

#### fts_contents — 全文搜索索引

```sql
CREATE VIRTUAL TABLE fts_contents USING fts5(
    sha256 UNINDEXED,
    tier UNINDEXED,             -- 当前内容的来源 tier
    text,
    title,
    author,
    filename,
    tokenize='unicode61'
);
```

**FTS 更新策略**：保留最高 tier 的内容。

```python
TIER_ORDER = {"flash": 0, "standard": 1, "pro": 2}
MAX_FTS_CHARS = 30_000   # 超出则 head+tail 截断（前 15K + 后 15K）

async def update_fts(sha256, text, title, author, filename, tier):
    existing = await db.fetchone(
        "SELECT tier FROM fts_contents WHERE sha256=?", (sha256,))
    if existing and TIER_ORDER[existing["tier"]] >= TIER_ORDER[tier]:
        return  # 跳过——当前已有更高或同 tier 的内容

    text = truncate_head_tail(text, MAX_FTS_CHARS)
    await db.execute(
        "INSERT OR REPLACE INTO fts_contents (sha256, text, title, author, filename, tier) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sha256, text, title or "", author or "", filename, tier))
```

- flash 先完成 → 写入；standard 后完成 → 覆盖；pro 后完成 → 再覆盖
- pro 先完成 → 写入；flash 后完成 → 跳过（pro > flash）
- 搜索只查这一张表，完整内容从解析产物文件读取

#### fts_filenames — 文件名搜索

```sql
CREATE VIRTUAL TABLE fts_filenames USING fts5(
    file_id UNINDEXED,
    filename,
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
SET locked_at = ?
WHERE id = (
    SELECT id FROM files
    WHERE sha256 IS NULL
      AND scan_status = 'active'
      AND (locked_at IS NULL OR locked_at < ?)
    ORDER BY first_seen_at ASC
    LIMIT 1
)
RETURNING *;

-- ParseWorker 获取任务
UPDATE parses
SET locked_at = ?, status = 'parsing'
WHERE id = (
    SELECT id FROM parses
    WHERE status = 'pending'
      AND (locked_at IS NULL OR locked_at < ?)
    ORDER BY priority DESC, created_at ASC
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
  - 触发默认 flash 解析（INSERT INTO parses (sha256, tier, pages, status='pending', priority=0)）

阶段 2: 解析（ParseWorker）
  - flash tier：提取首尾各 5 页文本，写入 fts_contents
  - standard/pro tier：完整解析，写入 fts_contents（覆盖 flash 的）
```

> **设计简化**：everydoc 的 L1 Index（注册 + 预览文本提取）在 MinerU 中被拆为：注册（仅 SHA-256 + metadata）+ flash 解析（首尾页文本）。用户无需理解"索引"和"解析"两个概念——所有内容提取都是 parse，只是 tier 不同。

### 5.2 文件发现 → 注册

```
Watch 检测到文件事件
  → 过滤（ext 白名单 + exclude 规则）
  → INSERT OR IGNORE INTO files (path, filename, ext, size_bytes, mtime_ms, sha256=NULL)
  → RegistrationWorker.acquire_task(): UPDATE files SET locked_at WHERE sha256 IS NULL
  → process_file():
       1. compute SHA-256 (asyncio.to_thread)
       2. INSERT OR IGNORE INTO docs (sha256, size_bytes, page_count, title, author, subject, keywords, is_encrypted)
       3. FTS insert fts_filenames
       4. UPDATE files SET sha256=?, locked_at=NULL
       5. check parsing-rules
       6. INSERT INTO parses (sha256, tier, pages, status='pending', priority=0)
  → 失败时: UPDATE files SET sha256=NULL, error_code=?, error_msg=?（可重试）
```

### 5.3 解析请求

```
用户请求: mineru parse doc.pdf --tier standard --pages 1~10
  → CLI → Product SDK (client.py) → Server POST /parse
  → ParseService.request_parse():
       1. 查 files 表获取 sha256（如果文件未注册，先同步注册）
       2. 查 parses WHERE sha256=? AND tier=? AND status='done'
       3. 已有 done 批次的 pages 覆盖请求范围 → 直接返回缓存
       4. 否则 → INSERT INTO parses (sha256, tier, pages, priority=1)
       5. --force 时 → DELETE FROM parses WHERE sha256=? AND tier=?，再 INSERT
  → ParseWorker.acquire_task()
  → process_doc():
       1. 从 parses 读取 sha256、tier、pages
       2. resolve_engine(tier) 确定 backend
       3. 调用 mineru.parser.parse(path, tier=..., pages=...)
       4. 写入解析产物到 ~/MinerU/parsed/{sha256[:2]}/{sha256}/{tier}/
       5. INSERT OR REPLACE fts_contents（仅当新 tier ≥ 当前 tier；截断到 30K head+tail）
       6. UPDATE parses SET status='done', done_at=?
       7. UPDATE docs（引擎能拿到更好的 title/author/subject/keywords/lang/is_scanned 时覆盖）
  → 失败时: UPDATE parses SET status='failed', error_code=?, error_msg=?
```

### 5.4 解析产物存储

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

### 5.5 搜索
```
用户请求: mineru search "关键词"
  → SearchService.search():
       1. 分词（jieba / unicode61）
       2. FTS5 MATCH on fts_contents
       3. JOIN files/docs 获取元数据
       4. 按 sha256 去重（同一文档可能有多个路径）
       5. 结果标注解析状态（flash / standard / pro）
```

### 5.6 可插拔设备

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
    await db.execute("UPDATE files SET locked_at = NULL WHERE locked_at IS NOT NULL")
    await db.execute("UPDATE parses SET locked_at = NULL, status = 'failed' WHERE status = 'parsing'")

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

    # 6. 启动 uvicorn（UDS + 可选 TCP）
    await uvicorn_server.serve()
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

    # 3. 删除 socket 文件
    remove_socket()
```

### 6.3 崩溃恢复

Server 启动时执行以下恢复操作：

| 操作 | 目的 |
|------|------|
| `UPDATE files SET locked_at=NULL` | 释放因崩溃而未释放的注册锁 |
| `UPDATE parses SET locked_at=NULL, status='failed' WHERE status='parsing'` | 释放解析锁，标记为可重试 |
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
| `reg_lock_timeout_sec` | `60` | 注册锁超时 |
| `parse_lock_timeout_sec` | `1800` | 解析锁超时（30 分钟） |

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

### 9.2 FTS 截断策略

| 表 | 数据来源 | 截断规则 |
|---|---------|---------|
| `fts_contents` | 解析产出的文本 | ≤30K 字符存全文；>30K 取前 15K + 后 15K（对齐段落边界） |

flash 的首尾各 5 页文本通常在 30K 限制内，不会触发截断。standard/pro 的完整解析文档若超出限制，用 head+tail 保留首尾搜索能力。

同一文档永远只有一行 FTS 记录，无需跨表合并。完整内容从解析产物 markdown 文件中读取。

---

## 10. 错误分类

错误分为三个层级，对应不同的处理策略：

| 层级 | 示例 | 存储位置 | 恢复策略 |
|------|------|---------|---------|
| File 级 | 路径不存在、权限不足 | `files.error_code` | 清除 sha256，可重试 |
| Doc 级 | 文件损坏、加密 | `files.error_code` | 不自动重试 |
| Parse 级 | 引擎崩溃、超时、OOM | `parses.error_code` | 标记 `status='failed'`，可重试 |

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
| `pypdfium2` | PDF page_count / metadata 读取 + flash 引擎文本提取 |

### 11.2 现有依赖（保持不变）

| 包 | 用途 |
|---|------|
| `pydantic` | 数据模型 |
| `rich` | CLI 格式化输出 |
| `loguru` | 日志 |
