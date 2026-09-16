# ADR-0009: Doclib Scan 后台任务

状态: Accepted
日期: 2026-06-12
相关文档: ../architecture.md, ../workflows.md, ../cli/mineru-library.md, 0006-doclib-file-change-detection.md, 0007-doclib-file-availability-lifecycle.md

## 背景

doclib 需要一个显式的 `scan` 能力，用于让用户、Agent 或系统主动检查本地文件系统，并把结果写入 doclib 状态。

这个能力需要覆盖:

- 新文件首次被 doclib 发现。
- 已知文件 stat 变化后清空 `files.sha256`，交给 ingest 重新识别。
- 已知文件缺失后标记为 `deleted` 或 `unreachable`。
- 目录扫描时发现新文件、变化文件和删除文件。
- watch initial scan、watch rescan、设备恢复后的 watch scan。

过去 watch 自己执行 initial scan；手动 path 操作又通过 `refresh_file()` 触发局部刷新。继续保留多套扫描实现会让 deleted / unreachable / exclude / 统计 / 日志语义分裂。

因此 scan 必须成为 server 侧后台任务，CLI 只负责提交任务和查询状态。CLI 断开不得中断 scan。

## 决策

引入统一的 `ScanService + ScanWorker`。

所有完整 scan 都必须复用同一套实现:

- `mineru scan <path>`。
- SDK / HTTP API scan。
- watch initial scan。
- watch rescan。
- removable 设备恢复后的 watch scan。

watch event 的单文件变化可以在 P0 继续直接调用 `refresh_file()`，因为它是轻量事件处理，不属于完整 scan。后续如果需要合并大量事件，可以再把目录级或批量事件转成 scan task。

## 执行模型

scan 是后台任务:

```text
CLI / SDK / HTTP
  -> POST /scans
  -> 写入 scans(status=pending)
  -> ScanWorker 领取任务
  -> ScanService.run_scan(scan_id)
  -> scans(status=done|failed)
```

CLI 默认可以等待一段时间，但等待只是轮询 scan task 状态。CLI 退出不影响 server 侧任务。

P0 不支持取消 scan，因此没有 `cancelled` 状态。

## scans 表

新增 `scans` 表，同时作为任务表和轻量 scan log:

```sql
CREATE TABLE scans (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    path                TEXT    NOT NULL,
    kind                TEXT    NOT NULL,              -- manual / watch
    source              TEXT    NOT NULL DEFAULT 'unknown',
    watch_id            INTEGER REFERENCES watches(id),
    status              TEXT    NOT NULL DEFAULT 'pending',
    locked_at           INTEGER,
    created_at          INTEGER NOT NULL,
    started_at          INTEGER,
    finished_at         INTEGER,
    updated_at          INTEGER NOT NULL,

    files_seen          INTEGER NOT NULL DEFAULT 0,
    files_refreshed     INTEGER NOT NULL DEFAULT 0,
    files_new           INTEGER NOT NULL DEFAULT 0,
    files_changed       INTEGER NOT NULL DEFAULT 0,
    files_deleted       INTEGER NOT NULL DEFAULT 0,
    files_unreachable   INTEGER NOT NULL DEFAULT 0,
    files_error         INTEGER NOT NULL DEFAULT 0,
    files_unsupported   INTEGER NOT NULL DEFAULT 0,
    files_excluded      INTEGER NOT NULL DEFAULT 0,

    error_code          TEXT,
    error_msg           TEXT
);

CREATE INDEX idx_scans_status ON scans(status, created_at);
CREATE INDEX idx_scans_kind_path_status ON scans(kind, path, status);
CREATE INDEX idx_scans_watch_id_status ON scans(watch_id, status);
```

`scans` 不记录 per-file 明细。P0 只做 summary。更细的 refresh log 可作为后续能力另行设计。

## kind

`kind` 是执行语义字段，不是遥测字段。

取值:

```text
manual | watch
```

### manual

显式 scan 请求。目标由 request path 指定，可以是:

- 当前存在的文件。
- 当前存在的目录。
- 当前不存在但 DB 中有历史 file row 的 path。
- 当前不存在但 DB 中有历史 prefix rows 的目录 path。

`manual` 不创建 watch。`scan` 和 `watch` 是两个概念；watch 的一部分能力依赖 scan，但 scan 本身不是持续监控。

### watch

watch 产生的完整 scan。必须绑定 `watch_id`，path 使用 watch root。

watch scan 负责:

- watch initial scan。
- 用户或 Agent 触发的 watch rescan。
- 设备恢复后立即执行的 watch scan。

watch scan 使用 watch 的 enabled/status/removable 语义，并更新 `watches.last_scan_at` / `last_scan_files`。

## source

`source` 是 best-effort 记录字段，用于本地诊断和未来 telemetry，不作为 P0 行为依据。

允许取值建议:

```text
unknown | cli | sdk | api | watch | system
```

P0 不强求 `cli` / `sdk` / `api` 区分准确。没有明确来源时写 `unknown` 或 `api` 均可，但同一实现中应保持稳定。

## status

scan status:

```text
pending | running | done | failed
```

语义:

- `pending`: 已创建，等待 ScanWorker。
- `running`: ScanWorker 正在执行。
- `done`: scan 已完成。业务上发现 root unreachable 也可以是 done。
- `failed`: scan 代码路径异常，无法完成任务。

P0 不做:

- `cancelled`
- `superseded`
- `paused`

## 去重

P0 不允许同一个目标同时存在多个 pending / running scan。

去重键:

```text
kind + normalized path
```

规则:

- 创建 scan 时，如果同一 `kind + normalized path` 已有 `pending` 或 `running` 任务，直接返回已有 scan。
- P0 不提供 `force`。
- watch scan 的 path 是 watch root；同时必须绑定对应 `watch_id`。

## ScanService 职责

`ScanService` 负责:

- 创建 scan task。
- 查询 scan task。
- 列出最近 scan tasks。
- 执行单个 scan task。
- 维护 scan status、锁、开始时间、完成时间、统计字段和错误字段。
- 对文件调用 `ParseService.refresh_file()`。
- 对目录执行统一两阶段扫描。
- 对 watch scan 更新 watch scan stats。

`ScanService` 不负责:

- 计算 SHA256。
- 提取 metadata。
- 创建 docs。
- 创建 parse batch。
- 执行 parse。

这些属于 `IngestWorker` / `ParseWorker` 职责。

## ScanWorker 职责

`ScanWorker` 负责:

- 循环领取 `pending` scan。
- 将任务置为 `running`，写 `locked_at` / `started_at`。
- 调用 `ScanService.run_scan(scan_id)`。
- 成功时写 `done` 和 summary。
- 异常时写 `failed`、`error_code`、`error_msg`。

P0 使用单 worker、单任务并发。后续如果需要并发，应先定义同 watch / 同 path 的并发策略。

ScanWorker 在 doclib app startup 中启动，与 ingest worker、parse worker、watch loop、device monitor 同级。

## scan 与 ingest 的边界

scan 和 ingest 必须职责清晰:

- scan 负责文件系统状态发现和 `files` row 刷新。
- ingest 负责对 `files.sha256 IS NULL` 的 active files 计算 SHA256、提取 metadata、写入 docs、写入 filename/content FTS，并按规则创建初始 parse batch。

scan 不同步等待 ingest 完成。

manual scan 或 watch scan 发现新文件 / 变化文件时:

- `refresh_file()` 插入或更新 `files` row。
- `files.sha256` 保持或变为 `NULL`。
- 该 file 进入 ingest worker 的处理范围。
- scan summary 统计它，但 scan task 可以在 ingest 完成前结束。

因此 `scan done` 表示“扫描和状态刷新完成”，不表示“入库、metadata、parse 都完成”。

## manual scan 语义

### 文件存在

如果 path 是文件:

- 不应用 exclude rules。
- 调用 `refresh_file(path)`。
- 支持首次发现、已知文件变化、已知文件未变化、stat error。

原因: `mineru parse <file>` 作为显式用户意图目前也不考虑 exclude rule；`mineru scan <file>` 同样是显式点名文件。

### 目录存在

如果 path 是目录:

- 应用 exclude rules。
- 执行两阶段扫描:
  1. 先读取 DB 中该目录前缀下 `status=active` 的已知 file rows，并逐个 `refresh_file()`，用于发现 deleted / unreachable / stat error。
  2. 再执行 `os.walk()`，对当前文件系统中支持的文件逐个 `refresh_file()`，用于发现新文件和变化文件。
- 统计 exclude 命中的文件为 `files_excluded`。
- 统计 unsupported extension 为 `files_unsupported`。

manual directory scan 不创建 watch，不更新 watch scan stats。

### path 不存在

如果 path 不存在:

- DB 中精确 file path 存在时，刷新该 path，按 deleted / unreachable / stat error 规则处理。
- DB 中存在 `path/` 前缀的 rows 时，按目录历史记录处理，刷新这些 rows。
- 两者都没有时，scan `done`，summary 为 0。

## watch scan 语义

watch scan 必须绑定 watch。

执行规则:

1. 读取 watch target。
2. 如果 watch 不存在或 disabled，scan `failed`，写 `error_code`。
3. 如果 watch root 不可达:
   - 更新 `watches.status=unreachable`。
   - scan status 写 `done`，不是 `failed`。
   - 不在同步 scan 中逐个刷新该 watch 下所有 files。
   - 批量 active -> unreachable 收敛仍由 DeviceMonitor 或后台策略处理。
4. 如果 watch root 可达:
   - 更新 watch status 为 `active`。
   - 执行两阶段扫描。
   - 应用 exclude rules。
   - 更新 `watches.last_scan_at` / `last_scan_files`。

watch scan 与 manual directory scan 的主要区别:

- watch scan 必须有 `watch_id`。
- watch scan 使用 watch root。
- watch scan 维护 watch status 和 scan stats。
- watch scan 是 watch 生命周期的一部分。

## exclude 规则

P0 规则:

| 场景 | 是否应用 exclude |
|------|------------------|
| manual scan 文件 | 否 |
| manual scan 目录 | 是 |
| watch scan | 是 |
| watch event 单文件 | 是 |
| parse 单文件 | 否 |

理由:

- 显式点名文件表示用户/Agent 明确要处理该文件。
- 目录扫描和 watch 自动发现属于批量发现行为，应遵守 exclude rules。

## summary 字段

scan summary 字段语义:

| 字段 | 含义 |
|------|------|
| `files_seen` | 当前文件系统 walk 中看到的支持文件数量，不含 DB 历史 missing rows |
| `files_refreshed` | 调用 `refresh_file()` 的次数 |
| `files_new` | `refresh_file()` 返回 `new` |
| `files_changed` | `refresh_file()` 返回 `changed` |
| `files_deleted` | `refresh_file()` 返回 `deleted` |
| `files_unreachable` | `refresh_file()` 返回 `unreachable` |
| `files_error` | `refresh_file()` 返回 `error` |
| `files_unsupported` | 扩展名不支持或 refresh 返回 `unsupported` |
| `files_excluded` | 目录 / watch scan 中被 exclude rules 跳过的文件数 |

`known` 不单独设字段；可由 `files_refreshed - new - changed - deleted - unreachable - error - unsupported` 推导。

## API

P0 API:

```http
POST /scans
GET  /scans
GET  /scans/{scan_id}
```

`POST /scans` 请求:

```json
{
  "path": "/Users/me/Documents",
  "kind": "manual",
  "source": "unknown"
}
```

watch scan 请求可以使用:

```json
{
  "path": "/Users/me/Documents",
  "kind": "watch",
  "watch_id": 123,
  "source": "watch"
}
```

如果已有同一 `kind + normalized path` 的 pending/running scan，返回已有 scan。

`GET /scans` 支持 P0 参数:

```text
limit
status
kind
watch_id
```

## CLI

P0 CLI:

```bash
mineru scan <path>
mineru scan <path> --no-wait
mineru scan <path> --wait 30
mineru scan status <scan_id>
mineru scan list
```

默认行为:

- 创建 manual scan。
- 默认等待一段时间，建议 30 秒。
- 如果超时仍 pending/running，输出 scan id 和当前状态。
- `--no-wait` 立即返回 scan id。

`mineru scan status <scan_id>` 查询单个 scan。

`mineru scan list` 查询最近 scan tasks。

watch rescan 命令使用:

```bash
mineru watch rescan <watch-path-or-id>
```

它创建 `kind=watch` 的 scan task，复用同一套 ScanWorker。`mineru scan <path>` 保持一次性 path scan 语义，不更新 watch stats。

## 与 G-006 的关系

`scans` 表同时是 scan task 表和轻量 scan log。

因此 G-006 P0 不再需要单独设计一张 scan log 表。

P0 retention 策略:

- scan worker 每完成一个 scan 后触发一次轻量 cleanup。
- 保留最近 1000 条 terminal scan tasks。
- terminal scan tasks 指 `done` / `failed`。
- 不清理 `pending` / `running` scan tasks。
- P0 使用条数 retention，不使用时间 retention。

P0 server status:

- `server status` 展示最近 5 条 scan tasks。
- 展示字段保持摘要级别: `id`、`path`、`kind`、`source`、`status`、`started_at`、`finished_at`、summary counters、`error_code`。
- 不展示 `error_msg`，避免 status 输出过长和泄漏异常细节。

P0 refresh log:

- 不做 per-file refresh log。
- `files` 表只维护每个 path 的当前状态。
- 后续如果需要诊断到每个 path 的刷新过程，可在 P1 新增 path-level refresh log。

## 替代方案

### CLI 直接执行 scan

拒绝。目录 scan 可能耗时较长，CLI 断开不应中断任务；scan 写 DB，也应由 doclib server 统一管理。

### watch initial scan 保留独立实现

拒绝。这样会让手动 scan 与 watch scan 的 deleted / unreachable / exclude / summary 语义分裂。

### scan 同步执行 ingest

拒绝。scan 与 ingest 职责不同。scan done 不应等待 SHA、metadata 或 parse 完成。

### `kind=path`

拒绝。`path` 是参数类型，不是任务类型。P0 使用 `manual | watch`。

### manual scan 文件应用 exclude rules

拒绝。显式点名文件与 `parse <file>` 一样，代表明确处理意图，不受批量发现规则影响。

## 影响

- 需要新增 `scans` 表。
- 需要新增 `ScanService` 与 `ScanWorker`。
- 需要调整 watch initial scan / watch rescan / device recovery scan，使其创建 scan task，而不是直接扫描。
- 需要新增 doclib Interface / Client / Server scan API。
- 需要新增 CLI `mineru scan`。
- `server status` 展示最近 5 条 scan tasks。
- ScanWorker 需要在 scan terminal 后执行 retention cleanup，保留最近 1000 条 terminal scan tasks。

## 后续动作

1. 在 migration 001 中加入 `scans` 表和索引。
2. 在 doclib types 中增加 scan request / response / info schema。
3. 在 Interface、Client、Server 增加 scan API。
4. 实现 `ScanService`。
5. 实现 `ScanWorker` 并在 app startup 中启动。
6. 改造 watch initial scan 使用 scan task。
7. 改造设备恢复后的 watch scan 使用 scan task。
8. 增加 CLI `mineru scan`。
9. 增加测试:
   - manual file scan。
   - manual directory scan。
   - historical missing path scan。
   - watch scan root unreachable。
   - pending/running 去重。
   - scan summary 计数。
   - CLI no-wait / status / list。
