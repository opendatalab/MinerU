# mineru library

状态: Draft
读者: CLI 使用者、Agent skill 作者、核心开发者
范围: `mineru` 本地文档库相关命令：search、find、list、show、config、watch、scan、invalidate、forget、cleanup
非目标: SQLite schema 字段级定义；完整配置优先级规范
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru` 不只是一次性解析命令，也维护本地文档库。本页描述文档库相关 CLI 能力。

详细数据模型见 [系统架构](../architecture.md)，配置主题见 [配置体系](../config.md)。

## 2. search

`mineru search` 面向已解析内容检索。

```bash
mineru search "keyword"
mineru search "keyword" --tier medium
mineru search "keyword" --min-tier medium --type pdf
```

行为：

- 查询 `fts_contents`，未解析或未建立内容索引的文件不会命中正文检索。
- 按文档 SHA256 去重。
- 支持按 `file_type`、`tier`、`min_tier` 过滤。
- 返回文件名、文件大小、页数、命中片段和 snippet 来源 tier。
- 默认优先返回 active file paths；如果某个已索引 doc 没有任何 active file，则 fallback 返回非 active file paths。
- 通过 `--json` 提供结构化输出。

JSON 输出中的每条结果至少包含:

| 字段 | 说明 |
|------|------|
| `filename` | 文件名，不含完整路径。 |
| `size_bytes` | 文件大小。 |
| `page_count` | 文档页数；未知时为 `null`。 |
| `snippet` | 命中的内容片段。 |
| `tier` | snippet 来源 tier。 |

`search` 不应在 telemetry 中记录 query 或 snippet，但 CLI JSON 可以返回 snippet 供用户和 Agent 使用。

## 3. find

`mineru find` 面向文件名定位，不搜索正文内容。

```bash
mineru find "report"
mineru find "report" --ext pdf
```

行为：

- 查询 `fts_filenames` 和文件 metadata。
- 支持按 `ext` 过滤。
- 可返回同一 SHA256 的多个路径。
- 可用于解析前确认目标文件。
- 当前 `find` 只暴露 `--ext`、`--limit/-n` 和 `--json`。

`find` JSON 输出中的每条结果至少包含:

| 字段 | 说明 |
|------|------|
| `filename` | 文件名。 |
| `size_bytes` | 文件大小。 |
| `page_count` | 文档页数；未知时为 `null`。 |

## 4. list

`mineru list` 列出 doclib 资源集合：

```bash
mineru list parses
mineru list scans
mineru list files
mineru list docs
```

当前过滤能力：

| 命令 | 过滤参数 |
|------|----------|
| `mineru list parses` | `--status`、`--tier`、`--limit/-n`、`--offset` |
| `mineru list scans` | `--status`、`--kind`、`--watch-id`、`--limit/-n`、`--offset` |
| `mineru list files` | `--status`、`--ext`、`--watch-id`、`--limit/-n`、`--offset` |
| `mineru list docs` | `--file-type`、`--limit/-n`、`--offset` |

四个子命令都支持 `--json`。

## 5. show

`mineru show file <path>` 查看本地路径对应的 file/doc/parse 状态。

当前 `show` 子命令：

```bash
mineru show file <path>
mineru show doc <doc-id-or-sha256>
mineru show parse <parse-id>
mineru show scan <scan-id>
```

`show file` 只接受文件路径。文档、parse task、scan task 等其他对象使用各自明确的 `show <resource>` 入口。四个子命令都支持 `--json`。

输出应区分实际 tier、缓存状态、产物路径和错误状态。已解析结果使用实际 tier。

JSON 输出至少包含:

| 字段 | 说明 |
|------|------|
| `filename` | 文件名。 |
| `size_bytes` | 文件大小。 |
| `page_count` | 文档页数；未知时为 `null`。 |
| `tiers` | 各 tier 已解析页码范围。 |
| `active_parses` | 当前 pending / parsing 的解析任务摘要。 |

`tiers` 应表达当前哪些页已经被哪些 tier 解析，例如:

```json
[
  {"tier": "flash", "page_range": "1~20"},
  {"tier": "medium", "page_range": "1~5,18~20"}
]
```

`active_parses` 应表达当前文档是否有正在进行中的解析任务，例如:

```json
[
  {"id": 123, "tier": "high", "page_range": "6~17", "status": "parsing"}
]
```

## 6. config

`mineru config` 管理本地配置。

常见范围：

| 范围 | 示例 |
|------|------|
| 基础配置 | 扫描间隔、锁超时、worker 数 |
| parse-server | local mode、managed tier、self-hosted URL、remote URL、API Key |
| watch | 通过独立 `mineru watch` 命令添加、列出、删除监控目录 |
| parsing-rules | 按路径规则触发 tier、page_range、remote |
| exclude | 排除路径模式 |

配置不应导致静默上传。只有规则或命令显式允许 remote 时，才可走远端解析。

当前命令：

```bash
mineru config show
mineru config get <key>
mineru config set <key> <value>
mineru config unset <key>
mineru config exclude-rules add <pattern> [--priority N]
mineru config exclude-rules list
mineru config exclude-rules remove <rule-id>
mineru config parsing-rules add <pattern> [--tier <tier>] [--pages <range>] [--remote] [--name <name>]
mineru config parsing-rules list
mineru config parsing-rules remove <rule-id>
```

`show`、`get`、规则 `add/list` 支持 `--json`；`set/unset/remove` 当前输出人类可读文本。

## 7. watch

watch 用于自动发现文件并建立本地索引。

关键语义：

- watch 默认使用 `flash`。
- watch 的目标是发现和搜索，不是最终阅读质量。
- Agent 或用户主动读取文档时，应通过 `mineru parse` 使用默认选择策略或显式 tier。
- 可插拔设备不可达时，watch 标记为 `unreachable`，该 watch 下的 active 文件标记为 `unreachable`，而不是永久删除。
- 可插拔设备恢复时，watch 和文件恢复为 `active`，并立即对该 watch 执行一次 scan。
- scan 发现文件真实缺失时，文件标记为 `deleted`，保留 `sha256`。

## 8. cleanup

`mineru cleanup` 管理本地文档库中的历史记录和缓存。第一版把 deleted file cleanup 与 orphan doc cleanup 分开，两个动作互不连带。

```bash
mineru cleanup deleted-files
mineru cleanup orphan-docs
mineru cleanup temp --older-than 7
```

deleted file cleanup:

- 立即删除所有 `status=deleted` 的 file row。
- 删除 file row 时删除对应 `fts_filenames`。
- 不自动删除 docs、parses、parsed JSON 或 `fts_contents`。
- 第一版不提供 `--older-than` 参数；后台任务固定保留 deleted file row 7 天后再清理。
- 默认 dry-run，实际执行需要 `--no-dry-run`。

orphan doc cleanup:

- 只处理完全没有任何 file row 关联的 docs。
- `active`、`unreachable` 和 `deleted` file row 都会保护 doc。
- 执行时才删除 docs、parses、`fts_contents` 和 parsed JSON。
- 默认 dry-run，实际执行需要 `--no-dry-run`。

temp cleanup:

- 删除 `doclib/temp` 下早于 `--older-than` 天的临时文件。
- `--older-than` 默认 7，且必须为非负数。
- 当前不提供 dry-run。

三个子命令都支持 `--json`。

## 9. forget

`mineru forget` 让 doclib 忘记某个 path 的本地记录。

```bash
mineru forget ~/Documents/a.pdf
mineru forget ~/Documents/project
mineru forget ~/Documents/project --no-dry-run
```

关键语义:

- 不删除磁盘上的真实文件或目录。
- 不删除 `docs`、`parses`、parsed JSON 或 `fts_contents`。
- 删除匹配的 `files` row，并清理对应 `fts_filenames`。
- 如果 path 是文件，只忘记该文件。
- 如果 path 是目录，默认并且永远递归忘记该目录树下的已入库文件；P0 不提供 `--recursive` / `--no-recursive`。
- 如果 path 不存在，但 DB 中存在精确 file path 或 `path/` 前缀匹配，也可以忘记历史记录。
- 如果 path 是已配置 watch root 且存在匹配文件，当前实现会拒绝；应先使用 `mineru watch remove <path>`。
- 如果 path 位于 active watch 下，允许执行，但返回 warning: 后续 scan 可能重新发现。
- 默认 dry-run，实际执行需要 `--no-dry-run`。
- 支持 `--json`。

`forget` 不是 ignore rule。它不会阻止 watch 未来重新发现同一路径。

## 10. scan

`mineru scan` 提交一个后台 scan task，让 doclib 扫描文件或目录并更新本地状态。

```bash
mineru scan ~/Documents/a.pdf
mineru scan ~/Documents/project
mineru scan ~/Documents/project --no-wait
mineru show scan 123
mineru list scans
```

关键语义:

- scan 在 doclib server 后台执行，CLI 断开不影响任务。
- scan 是一次性任务，不是 watch，不会持续监控。
- scan 不创建 watch。
- scan 不修改源文件，不上传远端。
- scan 只负责发现和刷新 `files` 状态，不同步等待 ingest 或 parse 完成。
- 新文件或变化文件会进入 ingest worker 的处理范围。
- 文件 scan 是显式点名文件，不应用 exclude rules。
- 目录 scan 应用 exclude rules。
- `mineru scan <path>` 是一次性 path scan，不创建 watch，也不更新 watch stats。
- watch initial scan、`mineru watch rescan`、设备恢复后的 watch scan 复用同一套 ScanWorker。
- 当前 `--wait` 默认 30 秒；`--no-wait` 创建任务后立即返回。
- 支持 `--json`。

P0 API / CLI 支持:

- 创建 scan。
- 查询单个 scan。
- 列出最近 scans。

完整规则见 [ADR-0009](../decisions/0009-doclib-scan-task.md)。

## 11. watch

`mineru watch` 管理持久 watch target:

```bash
mineru watch add ~/Documents
mineru watch add /Volumes/SSD --removable
mineru watch list
mineru watch remove ~/Documents
mineru watch rescan ~/Documents
mineru watch rescan 3
```

`mineru watch rescan <watch-path-or-id>` 只接受已配置 watch root 或 watch id。它创建 `kind=watch` 的后台 scan task，会更新 `watches.last_scan_at` / `last_scan_files`。

如果只想扫描 watch 下的某个子目录或任意 path，使用 `mineru scan <path>`。

`watch add/list/remove/rescan` 都支持 `--json`；`rescan` 还支持 `--wait` 和 `--no-wait`。

## 12. invalidate

`mineru invalidate` 把已有 done parse batch 标记为 superseded，源文件和 parsed JSON 之外的其它记录不应被直接删除。

```bash
mineru invalidate ~/Documents/a.pdf
mineru invalidate ~/Documents/a.pdf --tier flash
```

关键语义:

- 不删除磁盘源文件。
- 省略 `--tier` 时作废该文档所有 tier 的 done batch。
- 指定 `--tier` 时只作废对应 tier。
- 作废后再次 `mineru parse` 会重新创建或复用新的解析任务。
- 当前命令不暴露 `--json`。

## 13. telemetry

`mineru telemetry` 管理 doclib telemetry：

```bash
mineru telemetry status
mineru telemetry preview
mineru telemetry enable
mineru telemetry disable
mineru telemetry flush
```

所有 telemetry 子命令都支持 `--json`。详细数据边界见 [Telemetry 设计](../telemetry.md)。

## 14. parsing-rules

parsing-rules 用于按路径规则指定自动解析策略。

示例：

```bash
mineru config parsing-rules add "*/论文/*" --tier medium --pages all
mineru config parsing-rules add "*/合同/*" --tier high --remote
```

规则命中时，系统必须检查本地或远端能力是否支持对应 tier。不满足时应记录错误或提示用户，不静默降级到 `flash`。

parsing-rules 允许不指定 tier。执行时必须解析成实际实体 tier，并记录实际 tier；默认选择不能解析为 `flash`。

## 已收敛规则

P0 不做基于启发式的 watch 自动提示或自动排队升级。watch 默认使用 `flash`；后台自动升级只由用户显式配置的 parsing-rules 触发。
