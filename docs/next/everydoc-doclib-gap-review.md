# EveryDoc 与 MinerU Doclib 差异回收清单

状态: Draft
日期: 2026-06-12
读者: 核心开发者、负责 doclib / CLI / SDK / Agent 能力实现的编程 Agent
范围: 对比 `~/everydoc` 的设计与实现、当前 MinerU Next 文档、当前 `mineru/doclib` 实现，整理 everydoc 中值得讨论是否回收的能力
非目标: 直接给出最终方案；替代 ADR；修改当前接口或代码

## 1. 背景

`everydoc` 是早期用于验证 MinerU doclib 可行性的 demo 项目。MinerU 后续调整了若干产品语义，例如:

- 去掉 everydoc 中显式暴露给用户的 `index` 概念。
- 在 doclib 中用“入库 + flash parse”替代 everydoc 的 lightweight index。
- 将解析能力统一到 `flash` / `standard` / `pro` 解析档位。
- 将解析结果存储收敛为 Middle JSON，Markdown / Content List 等格式读取时即时转换。

尽管如此，everydoc 仍有一些已经设计清楚、甚至已经实现的 doclib 类能力。最近补齐文件变化检测时发现，MinerU doclib 曾遗漏 everydoc 中命名清晰的 `discover_file` / change detection 机制，因此有必要系统性回看 everydoc 中还有哪些设计值得回收。

## 2. 已基本吸收的设计

以下能力 MinerU doclib 已经采用或已有等价设计，本文件不作为重点讨论对象:

| 领域 | everydoc 设计 | MinerU 当前状态 |
|------|---------------|----------------|
| `files` / `docs` 分离 | 文件路径实例与内容 SHA 分离 | 已采用 |
| SHA256 去重 | 多路径共享同一内容记录 | 已采用 |
| 文件变化/删除检测 | `mtime_ms + size_bytes` 判断，缺失文件标记 `deleted` | 已补入 doclib 设计和实现 |
| watch 发现 | watchfiles + 初始扫描 | 已采用 |
| removable 设备 | watch / files 进入 `unreachable` | 已采用并补齐 cleanup 边界 |
| FTS 文件名索引 | 单独 `fts_filenames` | 已采用 |
| 后台任务锁 | timestamp lock + 超时接管 | 已采用 |
| cleanup | orphan-docs / deleted-files / temp | 已采用，并拆分 deleted files 与 orphan docs 生命周期 |
| rules | exclude + 自动解析规则 | MinerU 拆为 `exclude_rules` / `parsing_rules` |

## 3. 待讨论清单

### G-001 文件删除检测与活性验证

状态: Done。
结论文档: [ADR-0006](decisions/0006-doclib-file-change-detection.md), [ADR-0007](decisions/0007-doclib-file-availability-lifecycle.md), [workflows.md](workflows.md)。
实现位置: `ParseService.refresh_file()`、`WatchLoop._initial_scan()`。

来源:

- everydoc PRD / DESIGN 设计了目录 mtime 选择性验证、缺失文件标记 `deleted`、搜索命中时惰性活性验证。

处理结果:

- 数据模型有 `files.status = active / deleted / unreachable`。
- `refresh_file()` 统一处理 new / changed / known / deleted / unreachable / stat error。
- 删除/变更检测入口包括 watch event、watch scan，以及 4 个同步 source file path 操作: parse、info、docs?path、invalidate(path)。
- watch scan 采用两阶段刷新: 先 refresh DB 中该 watch 下的 active paths，再 `os.walk()` 发现当前文件系统中的新路径和变化路径。
- path 缺失且不属于设备整体不可达时，标记为 `deleted`，写入 `deleted_at`，保留 `sha256`，清空 `error_code` / `error_msg` / `locked_at`。
- 默认 `find` 只返回 active file；默认 `search` 优先返回 active file paths，如果某个 doc 已有内容索引但没有任何 active file，则 fallback 返回非 active file paths。
- `PermissionError` 与其他 stat `OSError` 不表示删除，分别写入 `file_permission_denied` / `stat_failed`。

后续优化:

- 大目录场景可引入目录 mtime、scan token 或 `last_seen_at` 优化；P0 暂不增加字段。
- 后台 deleted file cleanup 的调度 worker 仍需要在任务拆分中落地。

### G-002 `unreachable` 与 cleanup orphan 的边界

状态: Done。
结论文档: [ADR-0007](decisions/0007-doclib-file-availability-lifecycle.md)。
实现位置: `DeviceMonitor`、`CleanupService`、`mineru cleanup deleted-files` / `orphan-docs`。

来源:

- everydoc 将 removable 设备拔出建模为 `unreachable`，不是永久删除。

处理结果:

- `DeviceMonitor` 会把 removable watch 下文件从 `active` 切到 `unreachable`。
- 同步 path 调用发现 removable watch root 不可达时，只更新当前 file 和 watch status，不批量更新同 watch 下所有 files。
- `active`、`unreachable` 和 `deleted` file row 都保护 doc。
- 只有 doc 完全没有任何 file row 关联时，才算 orphan。
- deleted file cleanup 和 orphan doc cleanup 分开执行，互不连带。
- 手动 `mineru cleanup deleted-files` 立即删除所有 deleted file rows。
- 后台 deleted cleanup 固定保留 7 天后清理。
- 标记 `deleted` 或 `unreachable` 时不清理 FTS；`fts_filenames` 跟随 file row 生命周期，`fts_contents` 跟随 doc 生命周期。
- removable 设备恢复后，watch 和 files 恢复为 `active`，并立即对该 watch 执行 scan。
- deleted cleanup 删除 file row 时清理对应 `fts_filenames`，但不自动删除 docs、parses、parsed JSON 或 `fts_contents`。
- orphan doc cleanup 执行时才删除 docs、parses、`fts_contents` 和 parsed JSON。

后续优化:

- DeviceMonitor 恢复设备后触发指定 watch scan 的调度细节仍需要在后台任务实现中确认。

### G-003 watch remove 语义

状态: Done。
结论文档: [ADR-0006](decisions/0006-doclib-file-change-detection.md)。
实现位置: `ConfigService.remove_watch()`。

来源:

- everydoc PRD 设计: 移除 watch 后，文件记录保留，`watch_id = NULL`，不再参与后续监控。

处理结果:

- `remove_watch(path)` 删除 watch target 前先处理关联 file rows。
- active files: `watch_id=NULL`，`status` 保持 `active`。
- deleted files: `watch_id=NULL`，`status` 保持 `deleted`。
- unreachable files: `watch_id=NULL`，`status` 改为 `deleted`，写入 `deleted_at`。
- 不立即删除 file row，也不清理 FTS。

后续优化:

- 如果未来需要 watch 历史统计，可再讨论软删除 watch；P0 直接删除 watch target。

### G-004 手动 refresh / retry failed / remove file 能力

状态: In progress。
已定部分:

- `forget path`，见 [ADR-0008](decisions/0008-doclib-forget-path.md)。
- `scan` 后台任务，见 [ADR-0009](decisions/0009-doclib-scan-task.md)。
- 顶层 `watch` CLI 与 `watch rescan` 边界，见 [ADR-0010](decisions/0010-doclib-watch-cli.md)。

来源:

- everydoc 有 `index <path>`、`index --full`、`index/retry`、`unindex`。

MinerU 当前状态:

- MinerU 不再对用户暴露 `index` 概念。
- 同步 path 入口包括 parse、info、docs?path、invalidate。
- 已决定用 `forget` 表达“从 doclib 移除某路径记录”，不使用 `unindex` 或 `remove-file`。
- 已决定用 `scan` 表达“显式扫描文件或目录并更新 doclib 状态”，不使用 `sync` / `refresh` / `index`。
- 已决定将 watch 管理从 `mineru config watch` 提升为顶层 `mineru watch`。
- 已决定使用 `mineru watch rescan <watch-path-or-id>` 表达对已配置 watch target 的完整重扫。
- 已决定 P0 不提供 `retry_ingest_failures()` 独立能力。

风险:

- Agent 很难主动修复文件库状态。
- 入库失败后只能等待隐式路径操作或 watch 再次触发。
- `forget` 落地后，用户可以明确把某个路径从 doclib 中移除。
- `scan` 落地后，Agent 可以主动触发后台扫描，不依赖 CLI 长时间运行。

已定 forget 规则:

- CLI 使用 `mineru forget <path>`。
- 支持文件和目录；目录默认且永远递归。
- P0 不提供 `recursive` / `no-recursive` 参数。
- 不删除真实文件。
- 只删除 `files` row 和对应 `fts_filenames`。
- 不删除 `docs`、`parses`、parsed JSON 或 `fts_contents`。
- path 是 watch root 时默认拒绝，应先执行 watch remove。
- path 位于 active watch 下时允许执行，但返回 warning，说明后续 scan 可能重新发现。
- 默认 dry-run，执行需要 `--no-dry-run`。

P0 结论:

- `retry_ingest_failures()` 不进入 P0。
- P0 中已有错误的 file 不会被 ingest worker 自动重试。
- `scan` / `refresh` 只重新判断文件系统状态；如果 `mtime` 和 `size` 没变，保留已有错误态。
- 用户如需重新处理同一份未变化文件，P0 可通过 `forget` 后再 `scan` 的方式显式重建路径记录。

后续优化:

- P1 再讨论显式 retry ingest 能力。
- 显式 retry 需要单独定义可重试错误集合、path/watch/all 范围、dry-run、telemetry 和 CLI/API 入口。

初步优先级: P0。

### G-005 Server status 与 watch 维度可观测性

状态: Done。
实现位置: `ServerStatusResponse.watch_stats`、`ServerStatusResponse.error_summary`、`DoclibServer.get_server_status()`、`mineru server status`。

来源:

- everydoc `server status` 会按 watch 输出 total / indexed / parsed / errors。

处理结果:

- `ServerStatusResponse` 增加 `watch_stats`，按 watch 输出文件、入库、文档和解析任务聚合。
- 每个 watch stats 包括 total / active / deleted / unreachable files。
- 每个 watch stats 包括 pending ingest files、file error count、doc count。
- 每个 watch stats 包括 parse pending / parsing / failed / done count。
- 每个 watch stats 保留 watch path、label、removable、status、last_scan_at、last_scan_files。
- `ServerStatusResponse` 增加 `error_summary`，按 `files` / `docs` / `parses` 聚合 `error_code` 数量。
- `mineru server status` 在默认输出中增加 Watch Stats 和 Error Summary 表。
- JSON 模式返回完整结构。

边界:

- 不返回 error message 明细，避免 status 输出过长和泄漏路径/异常细节。
- 不增加 scan log 表；scan / refresh 历史追踪留给 G-006。
- 不按 watch 返回 error_code 分布；P0 只提供全局 error summary。

后续优化:

- 如果 watch 数量很多，可以考虑 status 分页或只返回异常 watch。

### G-006 scan log / refresh log

状态: P0 scan summary 已由 [ADR-0009](decisions/0009-doclib-scan-task.md) 覆盖。

来源:

- everydoc schema 有 `scan_log`，记录 scan 类型、watch、起止时间、总数、新增、更新、跳过、错误等。

MinerU 当前状态:

- 已决定新增 `scans` 表，同时作为 scan task 表和轻量 scan log。
- telemetry 设计会记录产品统计，但不是本地调试日志。

风险:

- watch 或 refresh 的行为不可追溯。
- Agent 无法回答“上一次扫描发生了什么”“为什么这个目录没入库”。

P0 结论:

- `scans` 表同时作为 scan task 表和轻量 scan log。
- P0 不新增独立 scan log 表。
- P0 不做 per-file refresh log。
- scan worker 每完成一个 scan 后触发一次轻量 cleanup。
- scan log retention 使用条数策略，保留最近 1000 条 terminal scan tasks。
- cleanup 只清理 `done` / `failed` scan，不清理 `pending` / `running` scan。
- `server status` 展示最近 5 条 scan tasks，字段保持摘要级别，不展示 `error_msg`。
- scan log 与 telemetry 的边界: `scans` 用于本地诊断，telemetry 用于匿名产品统计。

后续优化:

- P1 再讨论 per-file refresh log。
- 如果需要 per-file refresh log，应优先考虑只记录 error refresh 或状态变化 refresh，避免高频写放大。

初步优先级: P0。

### G-007 tags / auto-tag

状态: Deferred。
P0 结论: 不做。

来源:

- everydoc 有 `tags`、`file_tags`、`TagService` 和简单 auto-tag。

MinerU 当前状态:

- 当前 Next 文档和接口没有标签能力。

风险:

- 本地文档库缺少用户可控的轻量组织维度。
- Agent 只能基于路径、文件名、全文搜索和 parse 状态筛选文档。

后续再讨论:

- 第一版是否只做 manual tags，不做 auto-tag。
- tags 绑定 `files` 还是 `docs`，或者二者都支持。

后续优先级: P2 或更晚。

### G-008 搜索过滤与结果可信度

状态: P0 decision ready。
待实现位置: `SearchService.search()`、`SearchService.search_filenames()`、doclib interface/client/server、`mineru search` / `mineru find`。

来源:

- everydoc search 支持 `parsed`、`file_type`、时间范围等过滤。
- everydoc V2 提出搜索结果显式标注可信度。

MinerU 当前状态:

- `search()` 支持 `file_type`、`limit`、`offset`。
- `find()` 是文件名搜索。
- 搜索结果返回 `tier`，但没有系统性表达可信度、mtime 范围、watch 过滤、min tier 等。

风险:

- Agent 很难按解析档位和文件类型控制搜索范围。
- 用户无法快速判断结果来自低成本 flash、standard 还是 pro 的充分解析。

P0 结论:

- `search()` 增加 `tier`、`min_tier`、`file_type` 过滤。
- `tier` 表示只返回指定解析档位索引结果。
- `min_tier` 表示返回不低于该解析档位的索引结果，档位顺序为 `flash < standard < pro`。
- 如果 `tier` 与 `min_tier` 同时出现，二者同时生效；不兼容时返回空结果。
- 时间范围过滤、watch 过滤放到 P1。
- 搜索结果可信度不新增 `confidence` 或 `quality` 字段，只使用 `tier` 表达。
- 搜索结果默认优先返回 active files；由于一个 doc SHA256 可对应多个 file rows，结果中的 `paths` 是所选 file rows 的路径列表。
- 如果某个 doc 已有内容索引但没有任何 active file，则 fallback 返回非 active file paths，使 Agent 仍能定位 deleted / unreachable 的历史文档。
- `find()` 是文件名搜索，增加 `ext` 过滤；它仍默认只返回 active files。

后续优化:

- P1 再讨论 `mtime_after` / `mtime_before` / `watch_id`。
- 如果未来 Agent 需要区分 fallback path 的状态，可在结果中增加结构化 file refs；P0 先保持 `paths` 列表。

### G-009 Agent citation 与更细粒度索引

来源:

- everydoc V2 提出 Agent Citation 标准、section/page/snippet index。

MinerU 当前状态:

- Middle JSON 文档已有 locator 方向。
- 当前 doclib FTS 仍主要是文档级 `fts_contents`。

风险:

- Agent 搜索命中后只能拿到文档级 snippet，难以稳定引用 page / block / section。
- “Agent 时代的文档入口”目标需要更强的可验证引用。

建议讨论:

- 第一版是否需要 page-level 或 block-level FTS。
- citation id 使用 Middle JSON locator 还是独立 search hit id。
- FTS 表是否从 doc-level 扩展到 section/page/snippet-level。

初步优先级: P1。

### G-010 parser version / schema version 追踪

来源:

- everydoc V2 提出 `parse_version`，用于 parser 升级后识别旧缓存。

MinerU 当前状态:

- Middle JSON 文档讨论了 schema version。
- 当前 DB 的 `docs` / `parses` 还没有明确记录 parser version / schema version。

风险:

- parser 或 Middle JSON schema 升级后，无法系统性识别需要重新解析的旧缓存。
- Agent locator 可能因历史数据结构差异而不稳定。

建议讨论:

- version 记录在 `parses` 表、JSON envelope，还是两边都记录。
- 字段命名: `parser_version`、`schema_version`、`middle_json_version`。
- version 变化时是否自动 invalidate，还是提供 migration / refresh 工具。

初步优先级: P1。

## 4. 建议讨论顺序

G-001、G-002、G-003、G-005 已完成。剩余建议讨论顺序:

1. G-004 手动 refresh / retry failed / remove file 能力。
2. G-006 scan log / refresh log。
3. G-008 搜索过滤与结果可信度。
4. G-009 Agent citation 与更细粒度索引。
5. G-010 parser version / schema version 追踪。

排序理由:

- G-004 仍影响 doclib 数据生命周期和用户文件状态，错误实现可能造成缓存错用或误删。
- G-006 影响 Agent 和用户诊断历史扫描行为的能力，应在 P0 实现任务拆分前定稿。
- G-008 到 G-010 影响 Agent-native 体验和长期兼容性，适合在主链路稳定后细化。
- G-007 已明确 P0 不做，只作为 P2 或更晚的组织能力保留。

## 5. 待定输出

逐条讨论后，建议把结论分别落到:

| 结论类型 | 目标文档 |
|----------|----------|
| 数据生命周期规则 | `architecture.md` / `workflows.md` |
| API / SDK 能力变更 | `decisions/` / `sdk/doclib-client.md` |
| CLI 命令变更 | `cli/mineru-library.md` / `cli.md` |
| 任务拆分 | `implementation-plan.md` |
| 重大取舍 | 新增 ADR |
