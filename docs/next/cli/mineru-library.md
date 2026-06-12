# mineru library

状态: Draft
读者: CLI 使用者、Agent skill 作者、核心开发者
范围: `mineru` 本地文档库相关命令：search、find、info、config、watch、parsing-rules
非目标: SQLite schema 字段级定义；完整配置优先级规范
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru` 不只是一次性解析命令，也维护本地文档库。本页描述文档库相关 CLI 能力。

详细数据模型见 [系统架构](../architecture.md)，配置主题见 [配置体系](../config.md)。

## 2. search

`mineru search` 面向内容检索。

```bash
mineru search "keyword"
mineru search "keyword" --tier standard
mineru search "keyword" --min-tier standard --type pdf
```

行为：

- 查询 `fts_contents`。
- 按文档 SHA256 去重。
- 支持按 `file_type`、`tier`、`min_tier` 过滤。
- 返回文件名、文件大小、页数、命中片段和 snippet 来源 tier。
- 默认优先返回 active file paths；如果某个已索引 doc 没有任何 active file，则 fallback 返回非 active file paths。
- 对 Agent 应提供结构化输出模式。

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

`mineru find` 面向文件名、路径或文档定位。

```bash
mineru find "report"
mineru find "report" --ext pdf
```

行为：

- 查询 `fts_filenames` 和文件 metadata。
- 支持按 `ext` 过滤。
- 可返回同一 SHA256 的多个路径。
- 可用于解析前确认目标文件。

JSON 输出中的每条结果至少包含:

| 字段 | 说明 |
|------|------|
| `filename` | 文件名。 |
| `size_bytes` | 文件大小。 |
| `page_count` | 文档页数；未知时为 `null`。 |

## 4. info

`mineru info` 查看文件、文档或服务状态。

可能对象：

- 文件路径。
- SHA256 文档。
- parse task。
- 本地 doclib。
- parse-server。

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
  {"tier": "flash", "pages": "1~20"},
  {"tier": "standard", "pages": "1~5,18~20"}
]
```

`active_parses` 应表达当前文档是否有正在进行中的解析任务，例如:

```json
[
  {"id": 123, "tier": "pro", "pages": "6~17", "status": "parsing"}
]
```

## 5. config

`mineru config` 管理本地配置。

常见范围：

| 范围 | 示例 |
|------|------|
| 基础配置 | `data_dir`、`watch_default_tier`、扫描间隔 |
| parse-server | local mode、managed tier、self-hosted URL、remote URL、API Key |
| watch | 添加、列出、删除监控目录 |
| parsing-rules | 按路径规则触发 tier、pages、remote |
| exclude | 排除路径模式 |

配置不应导致静默上传。只有规则或命令显式允许 remote 时，才可走远端解析。

## 6. watch

watch 用于自动发现文件并建立本地索引。

关键语义：

- watch 默认使用 `flash`。
- watch 的目标是发现和搜索，不是最终阅读质量。
- Agent 或用户主动读取文档时，应通过 `mineru parse` 使用默认选择策略或显式 tier。
- 可插拔设备不可达时，watch 标记为 `unreachable`，该 watch 下的 active 文件标记为 `unreachable`，而不是永久删除。
- 可插拔设备恢复时，watch 和文件恢复为 `active`，并立即对该 watch 执行一次 scan。
- scan 发现文件真实缺失时，文件标记为 `deleted`，保留 `sha256`。

## 7. cleanup

`mineru cleanup` 管理本地文档库中的历史记录和缓存。第一版把 deleted file cleanup 与 orphan doc cleanup 分开，两个动作互不连带。

```bash
mineru cleanup deleted-files
mineru cleanup orphan-docs
mineru cleanup temp
```

deleted file cleanup:

- 立即删除所有 `scan_status=deleted` 的 file row。
- 删除 file row 时删除对应 `fts_filenames`。
- 不自动删除 docs、parses、parsed JSON 或 `fts_contents`。
- 第一版不提供 `--older-than` 参数；后台任务固定保留 deleted file row 7 天后再清理。

orphan doc cleanup:

- 只处理完全没有任何 file row 关联的 docs。
- `active`、`unreachable` 和 `deleted` file row 都会保护 doc。
- 执行时才删除 docs、parses、`fts_contents` 和 parsed JSON。

## 8. forget

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
- 如果 path 是已配置 watch root，默认拒绝；应先使用 `mineru watch remove <path>`。
- 如果 path 位于 active watch 下，允许执行，但返回 warning: 后续 scan 可能重新发现。
- 默认 dry-run，实际执行需要 `--no-dry-run`。

`forget` 不是 ignore rule。它不会阻止 watch 未来重新发现同一路径。

## 9. scan

`mineru scan` 提交一个后台 scan task，让 doclib 扫描文件或目录并更新本地状态。

```bash
mineru scan ~/Documents/a.pdf
mineru scan ~/Documents/project
mineru scan ~/Documents/project --no-wait
mineru scan status 123
mineru scan list
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

P0 API / CLI 支持:

- 创建 scan。
- 查询单个 scan。
- 列出最近 scans。

完整规则见 [ADR-0009](../decisions/0009-doclib-scan-task.md)。

## 10. watch

`mineru watch` 管理持久 watch target:

```bash
mineru watch add ~/Documents
mineru watch add /Volumes/SSD --removable
mineru watch list
mineru watch remove ~/Documents
mineru watch rescan ~/Documents
mineru watch rescan 3
```

`mineru watch rescan <watch-path-or-id>` 只接受已配置 watch root 或 watch id。它创建 `kind=watch` 的后台 scan task，会更新 `watch_targets.last_scan_at` / `last_scan_files`。

如果只想扫描 watch 下的某个子目录或任意 path，使用 `mineru scan <path>`。

## 11. parsing-rules

parsing-rules 用于按路径规则指定自动解析策略。

示例：

```bash
mineru config parsing-rules add "*/论文/*" --tier standard --pages all
mineru config parsing-rules add "*/合同/*" --tier pro --remote
```

规则命中时，系统必须检查本地或远端能力是否支持对应 tier。不满足时应记录错误或提示用户，不静默降级到 `flash`。

parsing-rules 允许不指定 tier。执行时必须解析成实际实体 tier，并记录实际 tier；默认选择不能解析为 `flash`。

## 未决问题

watch 升级提示集中维护在 [开放问题清单](../open-questions.md)。
