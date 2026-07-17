# ADR-0006: Doclib 文件变化检测与重新入库语义

状态: Proposed
日期: 2026-06-11
相关文档: ../architecture.md, ../workflows.md, ../sdk/doclib-client.md, 0002-force-vs-invalidate.md

## 背景

doclib 同时以两个维度管理文件:

- `files.path`: 本地文件路径实例。
- `docs.sha256`: 文件内容身份。

同一个内容可以出现在多个路径下；同一个路径也可能在不同时间指向不同内容。当前文档只描述了 watch 发现文件并写入 `files`，没有清楚定义以下情况:

- 文件路径不变，但文件内容变化。
- 文件路径不变，但文件大小、mtime 或基础 metadata 变化。
- watch 发现变化与用户主动 `parse(path)`、`get_file_by_path(path)` 的行为是否一致。
- 旧 `sha256`、旧 parse cache、FTS 索引和新内容之间如何切换。

如果不定义这部分，系统可能把新文件内容写入旧 `sha256` 的 parse cache，导致 DB、parsed JSON、FTS 和真实文件内容不一致。

## 决策

doclib 引入统一的轻量文件发现/刷新步骤，建议命名为 `discover_file(path, watch_id=None)` 或等价 service 方法。watch、主动 parse 和需要刷新文件状态的入口都应复用该方法。

### 文件变化判断

文件发现阶段只做轻量 stat，不立即计算 SHA256。

对同一路径的 active `files` 记录:

1. 如果 `mtime_ms` 和 `size_bytes` 都与当前文件一致，认为路径未变化。
2. 如果任一字段变化，认为路径可能指向了新内容。
3. 发生变化时，更新 `files.size_bytes`、`files.mtime_ms`、`files.updated_at`，清空 `files.sha256`，清空可重试错误字段和锁字段。

`birthtime_ms` 暂不进入 P0。文件创建时间在不同平台上的语义和可用性不一致，目前还没有明确的产品或 Agent 行为依赖它；P0 不记录该字段，也不使用它参与文件变化判定。

`files.sha256 IS NULL` 是“需要重新入库 / 重新识别内容”的队列标记。

### 重新入库

`IngestWorker` 只处理 `status=active AND sha256 IS NULL` 的文件。

处理步骤:

1. 计算当前路径内容的 SHA256。
2. 提取 metadata。
3. `INSERT OR IGNORE` 写入 `docs`。
4. 更新 `files.sha256` 指向新的 `docs.sha256`。
5. 更新文件名 FTS。
6. 根据文件类型和规则决定是否创建 parse batch。

如果新 SHA256 已经存在于 `docs`，不重复解析已有内容；只把当前 path 绑定到已有 doc。是否需要新增 parse batch 仍按当前 doc 的 parse 覆盖情况、watch 默认策略和 parsing-rule 判断。

### Watch 入口

watch 初始扫描和文件系统事件不直接 `INSERT OR IGNORE` 到 `files`，而是调用统一发现方法。

watch 对路径变化的职责:

- 新路径: 插入 `files`，`sha256=NULL`。
- 已知路径且 stat 未变: 不做事。
- 已知路径且 stat 变化: 更新文件 stat，清空 `sha256`，让 ingest worker 重新处理。

watch scan 的 P0 策略是不新增 `files.last_seen_at` 字段，而是执行两阶段刷新:

1. scan 开始时先检查 watch root 是否可达。
2. 如果 watch root 不可达，只更新 `watches.status=unreachable`，不逐个刷新该 watch 下的所有 files。
3. 如果 watch root 可达，先读取该 watch 下 `status=active` 的已知 file paths，逐个调用统一发现方法；这一步用于发现已删除文件。
4. 然后再执行 `os.walk()`，对当前文件系统中存在的文件逐个调用统一发现方法；这一步用于发现新文件和变化文件。

这个方案会让已知且仍存在的文件在一轮 scan 中可能被 stat 两次，但它避免了:

- 在 `files` 表中增加 `last_seen_at` 字段。
- 每轮 scan 为所有 known files 写入 seen 时间戳。
- 在 watch scan 中实现一套独立于统一发现方法的 deleted 判断逻辑。

### 文件缺失、设备不可达与 stat 错误

统一发现方法必须区分三类情况:

1. 文件确实缺失。
2. removable watch root 不可达，文件状态暂时无法判断。
3. 文件存在性无法判断，但原因是权限或 stat 错误。

规则如下:

- `FileNotFoundError` 表示 path 缺失，进入 deleted / unreachable 判断。
- `PermissionError` 不表示 deleted，应写入 `files.error_code=file_permission_denied` 和 `files.error_msg`，保留当前 `status`。
- 其他 stat `OSError` 不表示 deleted，应写入 `files.error_code=stat_failed` 和 `files.error_msg`，保留当前 `status`。
- 当文件被标记为 `deleted` 或 `unreachable` 时，应清空 `files.error_code` / `files.error_msg`。
- 已经是 `deleted` 的 file，后续即使所属 removable watch root 不可达，也不应改为 `unreachable`。
- `deleted` file 可以重新变为 `active`。只要同一路径重新可读，应重置 `sha256=NULL`，让 ingest 重新确认内容身份。

当同步 path 调用发现文件缺失，且所属 removable watch root 不可达时:

- 允许立即把当前 `watches.status` 更新为 `unreachable`，并写入 `unreachable_at`。
- 只允许把当前 path 对应的 file 标记为 `unreachable`。
- 不允许在同步调用中批量更新该 watch 下的所有 active files，避免一次用户请求触发长时间 DB 写入。

批量状态收敛属于后台职责:

- `DeviceMonitor` 发现 removable watch root 不可达时，可以异步把该 watch 下的 active files 批量改为 `unreachable`。
- `DeviceMonitor` 发现 root 恢复时，可以把该 watch 下的 `unreachable` files 恢复为 `active`，并触发后续 scan。
- watch scan 发现 root 不可达时，也应只更新 watch 状态，并交给后台收敛文件级状态。

删除 removable watch 时:

- active files: `watch_id=NULL`，`status` 保持 `active`。
- deleted files: `watch_id=NULL`，`status` 保持 `deleted`。
- unreachable files: `watch_id=NULL`，`status` 改为 `deleted`，写入 `deleted_at`。
- 不立即删除 file row。file row 和 filename FTS 的生命周期仍然保持一致，清理交给 cleanup。

### 必须检查文件变化的 path 操作

所有以本地 source file path 为输入的用户操作，都必须在使用 DB 中的 `files.sha256` 之前执行统一发现方法。

第一版包括 4 个操作:

| 操作 | API / SDK | CLI | 要求 |
|------|-----------|-----|------|
| 主动解析 | `POST /parses` / `ensure_parse(ParseRequest.path)` | `mineru parse <path>` | 判断 path 是否变化；变化后重新入库，再基于新 `sha256` 处理 cache 与 parse batch |
| 文件信息 | `GET /files/by-path?path=...` / `get_file_by_path(path)` | `mineru show file <path>` | 判断 path 是否变化；变化后可返回 `sha256=None`，表示等待重新入库 |
| 按路径取文档 | `GET /docs/by-path?path=...` / `get_doc_by_path(path)` | 后续可由 CLI 暴露 | 判断 path 是否变化；避免返回旧 `sha256` 对应 doc |
| 按路径失效解析 | `POST /invalidate { path }` / `invalidate(path=...)` | `mineru invalidate <path>` | 判断 path 是否变化；避免误 invalidate 已不再属于该 path 的旧 `sha256` |

watch 初始扫描和文件系统事件也必须调用同一发现方法，但 watch 是后台来源，不属于用户输入操作。

以下 path 不属于 source file path，不进入文件变化检测:

- watch target path: `add_watch()` / `remove_watch()`。
- 输出路径: `get_doc_content(output=...)` / `mineru parse -o <output>`。
- 配置路径: socket、log、SQLite、data_dir 等 server 配置。
- glob pattern: exclude rule 和 parsing-rule pattern。

### 主动 parse(path) 入口

`POST /parses` / SDK `ensure_parse()` / CLI `mineru parse <path>` 在使用 path 查询缓存前，必须先执行统一发现方法。

这样可以保证:

- 如果 path 内容没变，继续使用已有 `sha256` 和 parse cache。
- 如果 path 内容变了，先清空旧绑定并重新 ingest，再按新 `sha256` 判断 cache 和创建 parse batch。
- `force=True` 作用于“当前 path 重新识别后的当前 `sha256`”，不能把新文件内容写到旧 `sha256` 的 parsed 目录下。

### get_file_by_path(path) 入口

`get_file_by_path(path)` 也必须先执行统一发现方法。CLI 层对应入口是 `mineru show file <path>`。

原因:

- `show file` 是用户显式以 path 查询当前文件状态的入口。
- 如果 path 内容或元数据已经变化，返回旧 `sha256`、旧 size 或旧 parse 状态会误导用户和 Agent。
- `show file` 只需要做轻量 stat/discover；如果发现变化，清空 `files.sha256` 并让 ingest worker 重新处理。它不需要同步完成 SHA256 计算和 parse。

因此 `get_file_by_path(path)` 的返回可以出现 `file.sha256=None`、`doc=None`、`active_parses=[]`，表示该路径刚被发现为变化状态，正在等待重新入库。

### ParseWorker 文件选择保护

Parse batch 绑定的是 `sha256`，不是 path。

`ParseWorker` 执行任务前必须通过 `sha256` 查找当前 active file:

```sql
SELECT * FROM files WHERE sha256=? AND status='active' LIMIT 1
```

如果 path 已变化并被清空 `sha256`，旧 parse batch 不得继续读取这个 path 的新内容。此时:

- 如果还有其他 active path 指向旧 `sha256`，可以继续解析旧内容。
- 如果没有 active path 指向旧 `sha256`，该 parse batch 应失败或等待重试，错误可记为 `no_accessible_file`。

### 旧 doc 与旧 parse cache

文件变化不等于 invalidate 旧 doc。

当 path 从旧 `sha256` 切到新 `sha256`:

- 旧 `docs` 和旧 `parses` 仍然按内容身份保留。
- 如果旧 `sha256` 没有任何 file row 引用，它会成为 orphan doc。
- orphan doc 的清理由 cleanup 机制负责。
- `invalidate` 仍然只表示用户或系统显式要求某个 doc/tier 的 parse cache 退出覆盖、读取和搜索。

这保持了两个动作的边界:

- 文件变化: path 重新绑定到新的内容身份。
- invalidate: 某个内容身份下的 parse cache 生命周期管理。

### FTS 更新

文件名 FTS:

- 新文件入库后写入或更新 `fts_filenames`。
- 同 SHA 新路径绑定时也必须写入该 path 对应的 filename FTS。
- 删除 `files` row 时必须清理对应 `fts_filenames`。仅标记 `deleted` 时不清理 filename FTS。

内容 FTS:

- 可解析文档由 parse 成功后更新 `fts_contents`。
- invalidate 后按剩余最高有效 tier 重建或删除 `fts_contents`。
- 纯文本文件不创建 parse batch，但入库时应直接读取文本并以 `tier=NULL` 写入 `fts_contents`，使 `GET /search` 能搜索文本内容。

### 纯文本文件

纯文本文件属于“无需 parse，但需要内容索引”的文件。

第一版支持范围由 `TEXT_EXTENSIONS` 定义。纯文本入库后:

- 写入 `files` 和 `docs`。
- 写入 `fts_filenames`。
- 读取文本内容，截断到 FTS 上限后写入 `fts_contents`。
- 不创建 `parses` 记录。
- `GET /docs/{sha256}/content` 是否直接返回原文可以另行设计，不作为本 ADR 的必需项。

## 替代方案

### 方案 A: 每次 parse/show file 都重新计算 SHA256

拒绝。SHA256 对大文件成本较高，且 `show file` 作为状态查询不应默认产生昂贵 IO。

### 方案 B: 只依赖 watch 事件，不做 stat 比较

拒绝。文件系统事件可能合并、丢失或只表示目录变化。即使收到事件，也需要通过 stat 判断 path 是否真的变化。

### 方案 C: path 变化时立即 invalidate 旧 parse cache

拒绝。parse cache 属于内容身份 `sha256`，不是路径。旧内容可能仍被其他 path 引用，自动 invalidate 会破坏内容去重和缓存复用。

### 方案 D: path 变化时直接删除旧 doc

拒绝。删除旧 doc 需要确认没有任何 file row 引用。应交给 orphan cleanup 统一处理。

## 影响

- 需要把 watch 的 `_discover_file()` 改为调用统一发现方法。
- 主动 parse 和 show file 需要先 refresh/discover path，再读取 `files.sha256`。
- `ingest_file()` 不能在发现 `existing_path.sha256` 后直接返回，必须以发现阶段的 stat 判断为准。
- ParseWorker 必须坚持按 `sha256` 找 active file，避免读取已变化 path 的新内容。
- FTS filename 需要补齐同 SHA 新路径和删除清理场景。
- 纯文本内容搜索需要在 ingest 阶段直接写 `fts_contents`。

## 后续动作

1. 在 doclib service 层新增统一 `discover_file()`。
2. 修改 watch 初始扫描和 watch event 入口，统一调用 `discover_file()`。
3. 修改 `ensure_parse()` / `request_parse()`，在缓存判断前调用 `discover_file()`。
4. 修改 `get_file_by_path(path)`，在查询返回前调用 `discover_file()`。
5. 修改 ingest 逻辑，支持 path 变化后的重新入库。
6. 补齐 `fts_filenames` 清理和同 SHA 新路径写入。
7. 为纯文本文件写入 `fts_contents`。
8. 增加测试覆盖:
   - path 未变化不重复入库。
   - path 内容变化后 `files.sha256` 被清空并重新绑定新 SHA。
   - `force=True` 不会把新内容写入旧 SHA。
   - 同 SHA 多路径都会进入 filename FTS。
   - 删除 files row 会清理 filename FTS。
   - 纯文本入库后可被 `search` 命中。
