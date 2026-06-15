# ADR-0007: Doclib 文件可达性生命周期

状态: Accepted
日期: 2026-06-12
相关文档: ../architecture.md, ../workflows.md, ../cli/mineru-library.md, 0006-doclib-file-change-detection.md

## 背景

doclib 不只需要判断“同一路径内容是否变化”，还需要判断“已知文件路径当前是否仍可读取”。这影响:

- watch 删除事件和 scan 发现的缺失文件。
- 可插拔设备拔出与恢复。
- `info(path)`、`parse(path)`、`list_docs(path=...)`、`invalidate(path=...)` 等同步 path 操作。
- `cleanup deleted` 与 `cleanup orphans` 的边界。
- `fts_filenames` 与 `fts_contents` 的生命周期。

如果不统一建模，系统可能把 removable 设备拔出误判为文件删除，或在 cleanup 时误删仍有历史 file row 关联的 doc。

## 决策

### 文件状态

`files.status` 使用三种状态:

| 状态 | 含义 | 默认 find | 默认 search | 是否保护 doc |
|------|------|-----------|----------------|--------------|
| `active` | 路径当前存在且可访问 | 是 | 优先返回 | 是 |
| `unreachable` | 所属 removable watch 当前不可达 | 否 | 无 active file 时 fallback 返回 | 是 |
| `deleted` | 路径已确认不存在，且不是设备整体不可达导致 | 否 | 无 active file 时 fallback 返回 | 是 |

`deleted` file row 仍保留 `sha256`，并继续保护对应 doc。只有完全没有任何 file row 关联的 doc 才是 orphan。

### 删除 / 变更检测入口

所有会观察 source file path 当前状态的入口都必须执行统一文件发现/刷新逻辑。

第一版包括:

1. watch file event。
2. 所有 scan 操作:
   - watch 初始扫描。
   - watch 周期 scan。
   - removable 设备恢复后的 scan。
   - 未来手动 refresh / rescan。
3. 所有同步涉及 source file path 的 4 个操作:
   - `POST /parses` / SDK `ensure_parse(path)` / `mineru parse <path>`。
   - `GET /info?path=...` / SDK `get_file_info(path)` / `mineru info <path>`。
   - `GET /docs?path=...` / SDK `list_docs(path=...)`。
   - `POST /invalidate { path }` / SDK `invalidate(path=...)` / `mineru invalidate <path>`。

### 文件缺失判断

当某个已知 path 不存在时，不能直接删除 row。

判断规则:

1. 如果该 file 属于 removable watch，且 watch root 当前不可达，标记为 `unreachable`。
2. 否则标记为 `deleted`，写入 `deleted_at`，保留 `sha256`。
3. 标记 `deleted` 或 `unreachable` 时不清理 FTS。

### 可插拔设备

`DeviceMonitor` 只对 `removable=1` 的 watch target 做设备可达性判断。

设备不可达:

1. watch target 标记为 `unreachable`。
2. 该 watch 下 `active` files 标记为 `unreachable`。
3. 不清理 file row、doc、parse cache 或 FTS。

设备恢复:

1. watch target 标记为 `active`。
2. 该 watch 下 `unreachable` files 标记为 `active`。
3. 立即对该 watch 执行一次 scan:
   - 文件仍不存在: 标记 `deleted`。
   - 文件 stat 变化: 清空 `sha256`，等待或触发 ingest。
   - 文件未变化: 保持当前绑定。

### Cleanup 边界

deleted file cleanup 与 orphan doc cleanup 是两个独立动作，互不连带。

deleted file cleanup:

- 只删除 `status='deleted'` 的 file row。
- 手动命令立即删除所有 deleted file rows。
- 后台任务只删除保留超过 7 天的 deleted file rows。
- 删除 file row 时清理对应 `fts_filenames`。
- 不自动删除 docs、parses、parsed JSON 或 `fts_contents`。

orphan doc cleanup:

- 只删除完全没有任何 file row 关联的 docs。
- 判断条件不区分 file 的 `status`:

```sql
SELECT d.*
FROM docs d
WHERE NOT EXISTS (
  SELECT 1 FROM files f
  WHERE f.sha256 = d.sha256
);
```

- 执行 orphan cleanup 时，才删除对应 `docs`、`parses`、`fts_contents` 和 parsed JSON。

### FTS 生命周期

FTS 生命周期跟随其对应主表记录:

| FTS 表 | 主表记录 | 生命周期 |
|--------|----------|----------|
| `fts_filenames` | `files` | file row 删除时删除 |
| `fts_contents` | `docs` | doc row 删除时删除 |

因此:

- 标记 `deleted` 时不删除 `fts_filenames`。
- 标记 `unreachable` 时不删除 `fts_filenames`。
- 默认 `find` 只返回 active file。
- 默认 `search` 优先返回 active file paths；如果某个已索引 doc 没有任何 active file，则 fallback 返回 non-active file paths。

### CLI 参数

为简化第一版语义，`mineru cleanup deleted-files` 不接收 `--older-than`。

- 手动 cleanup deleted: 立即删除所有 deleted file rows。
- 后台 deleted cleanup: 固定保留 7 天后删除。

## 替代方案

### 方案 A: deleted file 不保护 doc

拒绝。用户删除路径不代表内容缓存应立即失效；删除路径记录前仍应保留 doc 的可追溯关联。

### 方案 B: cleanup deleted 后自动 cleanup orphans

拒绝。清理路径记录和清理内容缓存是两个不同生命周期动作，应由不同命令和后台策略控制，避免隐式连带删除。

### 方案 C: 标记 deleted 时立即删除 filename FTS

拒绝。`fts_filenames` 的生命周期应与 `files` row 一致。默认 find 通过 `status='active'` 控制可见性即可。

### 方案 D: search 默认返回 unreachable / deleted

部分拒绝。第一版默认 find 只返回当前可访问文件。search 则优先返回 active file paths；只有在某个已索引 doc 没有任何 active file 时，才 fallback 返回 unreachable / deleted paths，避免已索引历史文档完全不可定位。

## 影响

- `CleanupService.find_orphan_docs()` 必须改为“没有任何 file row 关联”，不能只看 active file。
- `cleanup_deleted_files()` 不应自动触发 orphan cleanup。
- `mineru cleanup deleted-files` 立即删除所有 deleted file rows。
- watch scan 和同步 path 操作需要能把缺失文件标记为 `deleted` 或 `unreachable`。
- DeviceMonitor 恢复后需要触发该 watch 的 scan。
- `fts_filenames` 清理必须绑定到 file row 删除，而不是 scan status 变化。

## 后续动作

1. 修改文件发现/刷新逻辑，支持 path 不存在时按 watch 状态标记 `deleted` 或 `unreachable`。
2. 增加 watch 周期 scan 或等价 scan 入口，补齐删除检测。
3. 修改 DeviceMonitor，恢复后触发该 watch scan。
4. 修改 cleanup deleted 行为，去掉 `older_than` 参数并停止连带 orphan cleanup。
5. 修改 orphan cleanup 判断。
6. 补齐删除 file row 时清理 `fts_filenames`。
7. 增加测试:
   - deleted file 保留 `sha256` 并保护 doc。
   - orphan doc 只在没有任何 file row 时出现。
   - cleanup deleted 不自动 cleanup orphan。
   - removable 不可达不会造成 deleted。
   - removable 恢复后 scan 能把真实缺失文件标记为 deleted。
