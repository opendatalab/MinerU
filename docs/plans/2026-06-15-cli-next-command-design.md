# MinerU CLI Next 命令设计草案

**日期**: 2026-06-15
**状态**: 设计阶段（待评审）
**范围**: `mineru/cli_next` 面向 doclib server/client 的命令结构
**非目标**: `mineru-kit` 专家参数、parse-server HTTP API 细节、旧命令兼容层

## 1. 背景

当前 `cli_next` 已经具备本地 doclib 的核心能力：单文件解析、scan task、watch target、搜索、文件信息、配置、清理、forget、server 生命周期管理等。

主要问题不是能力缺失，而是命令分组还没有稳定：

- `mineru parse <file>` 与 `mineru parse list` / `parses` 这类命名如果同时存在，容易让用户分不清“发起解析动作”和“查看解析任务集合”。
- `mineru scan <path>`、`mineru scan status <id>`、`mineru scan list` 把动作和资源查询混在同一个顶层动词下，当前实现也需要手动解析伪子命令。
- `mineru info <path>` 保留为文件信息短别名，但不继续扩展到 parse、scan、doc、server 等其他对象。
- `mineru config parse-server ...` 可以被通用 `mineru config set <key> <value>` 覆盖，专用子树会增加重复入口。
- `config show` 需要展示哪些值来自代码默认值，哪些值来自用户 override。

MVP 还在开发阶段，因此除 `info <path>` 作为 `show file <path>` 的短别名外，本设计不保留旧命令兼容入口。

## 2. 设计原则

1. **动作和资源分离**
   - `parse`、`scan` 保留为发起动作。
   - `list`、`show` 负责查询已有资源。

2. **单复数不混用**
   - 不设计 `parse` 和 `parses` 两个并列顶层命令。
   - 资源集合统一放到 `list <resource>`，单对象详情统一放到 `show <resource> <id>`。

3. **对象类型显式**
   - `show parse <parse_id>` 比 `info <id>` 清楚。
   - `show scan <scan_id>` 比 `scan status <scan_id>` 更清楚，因为它不会和“发起 scan”抢同一个命令空间。

4. **配置走 KV**
   - `config show/get/set/unset` 是配置 KV 的唯一入口。
   - parse-server 相关配置使用 `parse_server.*` key，不再提供 `config parse-server` 快捷命令。

5. **暂不增加内容读取子命令**
   - 暂不加入 `mineru content <sha256> --tier standard --pages 1~5`。
   - 暂不加入 `mineru show doc <sha256> --content --tier standard --pages 1~5`。
   - 解析内容读取继续由 `mineru parse <file>` 的输出行为承担。

## 3. 推荐命令树

### 3.1 顶层命令

| 命令 | 职责 |
|------|------|
| `mineru parse <file>` | 发起或复用单文件解析，并输出内容或任务状态 |
| `mineru scan <path>` | 发起一次性 scan task |
| `mineru search <query>` | 搜索已解析内容 |
| `mineru find <query>` | 按文件名或路径定位 doclib 文件记录 |
| `mineru list <resource>` | 列出资源集合 |
| `mineru show <resource> <id>` | 查看资源详情 |
| `mineru info <path>` | 查看文件信息，作为 `show file <path>` 的短别名 |
| `mineru watch ...` | 管理 watch target |
| `mineru config ...` | 管理运行时配置和规则 |
| `mineru cleanup ...` | 清理 doclib 历史记录和临时文件 |
| `mineru forget <path>` | 从 doclib 忘记路径记录，不删除源文件 |
| `mineru invalidate <path>` | 作废已有解析缓存，使后续 parse 重新运行 |
| `mineru server ...` | 管理本地 doclib server 生命周期 |

## 4. Parse 相关命令

### 4.1 发起解析

```bash
mineru parse <file> [--tier flash|standard|pro] [--pages <range>] [--format markdown|text|json|html] [--force] [--remote] [--wait <sec>] [--no-wait] [-o <path>] [--json] [-v]
```

语义：

- 只负责“用户主动要求解析这个文件”。
- 命中 done cache 时直接输出内容。
- 未命中时创建或复用 parse task。
- `--no-wait` 或超时后返回可继续查询的 parse 状态。

不在 `parse` 下增加 `list/status/show` 子命令。

### 4.2 列出解析任务

```bash
mineru list parses [--status pending|parsing|done|failed|superseded] [--tier flash|standard|pro] [--limit 50] [--json]
```

说明：

- `list parses` 使用复数资源名，因为它返回集合。
- 这里的 `parses` 只出现在 `list` 的资源位置，不作为顶层命令。

### 4.3 查看解析任务详情

```bash
mineru show parse <parse_id> [--json]
```

说明：

- 用于查看单个 parse task 的详细状态、错误、tier、pages、关联 sha256、产物路径等。
- 不读取正文内容。

## 5. Scan 相关命令

### 5.1 发起 scan

```bash
mineru scan <path> [--wait <sec>] [--no-wait] [--json]
```

语义：

- 创建一次性 scan task。
- 不创建 watch。
- 不上传远端。
- 目录 scan 应用 exclude rules；显式文件 scan 不应用 exclude rules。

### 5.2 列出 scan tasks

```bash
mineru list scans [--status pending|running|done|failed] [--kind manual|watch] [--watch-id <id>] [--limit 50] [--json]
```

替代当前：

```bash
mineru scan list
```

### 5.3 查看 scan 详情

```bash
mineru show scan <scan_id> [--json]
```

说明：

- 展示 scan 计数、路径、kind、watch_id、错误信息和时间戳。

## 6. Doc/File 相关命令

### 6.1 文件信息

```bash
mineru show file <path> [--json]
mineru info <path> [--json]
```

说明：

- `file` 代表路径记录，展示 path、sha256、大小、mtime、status、parse 概要。
- `info <path>` 先保留，作为 `show file <path>` 的短别名。
- `info` 只面向 path/file，不扩展为多对象查询命令。
- 如果文件不存在但 DB 中有历史记录，应展示历史状态，而不是只报 file not found。

### 6.2 文件列表

```bash
mineru list files [--status active|deleted|unreachable] [--ext <ext>] [--watch-id <id>] [--limit 200] [--offset 0] [--json]
```

说明：

- `list files` 返回 doclib 中的文件路径记录，而不是去磁盘重新扫描。
- 默认列出 active files。
- 默认排序为 `updated_at DESC`，便于查看最近入库或变化的文件。
- `--status deleted|unreachable` 用于排查 scan/watch 状态。
- 当前 server/client 暂无独立 list files endpoint，MVP 实现时需要新增。

### 6.3 文档信息

```bash
mineru show doc <sha256> [--json]
```

说明：

- 仅展示 doc metadata、关联路径、已解析 tier/pages、active parses、FTS 状态。
- 不加 `--content`。
- 内容读取命令暂不纳入本轮设计。

### 6.4 文档列表

```bash
mineru list docs [--path <path>] [--file-type <type>] [--limit 200] [--json]
```

说明：

- `list docs` 返回已入库文档集合，以 sha256 去重。
- 支持按 `docs.file_type` 过滤，例如 `pdf`、`docx`。
- 现有 doclib server/client 已有 `GET /docs` / `DoclibClient.list_docs(path=...)` 能力，但需要补 `file_type` / `limit` 参数。
- `--path <path>` 可用于按文件路径反查关联 doc。

## 7. Search 和 Find

保留现有顶层命令：

```bash
mineru search <query> [--type <ext>] [--tier <tier>] [--min-tier <tier>] [--limit 20] [--offset 0] [--json]
mineru find <query> [--ext <ext>] [--limit 50] [--json]
```

理由：

- `search` 和 `find` 是用户高频动作，不需要强行归到 `list` 或 `show`。
- `search` 查内容，`find` 查文件名/路径，语义清楚。

## 8. Watch

保留资源管理子树：

```bash
mineru watch add <path> [--removable] [--label <label>] [--json]
mineru watch list [--json]
mineru watch remove <path-or-id> [--json]
mineru watch rescan <path-or-id> [--wait <sec>] [--no-wait] [--json]
```

说明：

- `watch rescan` 是针对已配置 watch target 的动作，保留在 `watch` 子树下合理。
- 任意路径的一次性扫描仍使用 `mineru scan <path>`。

不支持 `mineru list watches`。watch 是明确的管理子树，集合查询保留为 `mineru watch list`。

## 9. Config

### 9.1 KV 配置

```bash
mineru config show [--json]
mineru config get <key> [--json]
mineru config set <key> <value>
mineru config unset <key>
```

语义：

- 默认值必须在代码中定义，不存储在 DB 中。
- DB `config` 表只存用户 override。
- `config get <key>` 对未知 key 返回错误。
- `config unset <key>` 删除 override，使该 key 回到代码默认值。
- `config show` 展示最终生效值，并标注来源。

人类可读输出示例：

```text
[Config]
  data_dir = ~/MinerU  [default]
  default_tier = flash  [default]
  parse_server.local.mode = managed  [override]
  parse_server.remote.api_key = sk-abc******xyz789  [override]
```

JSON 输出建议：

```json
{
  "config": {
    "data_dir": "~/MinerU",
    "parse_server.local.mode": "managed",
    "parse_server.remote.api_key": "sk-abc******xyz789"
  },
  "sources": {
    "data_dir": "default",
    "parse_server.local.mode": "override",
    "parse_server.remote.api_key": "override"
  }
}
```

敏感 key 在人类可读输出和 JSON 输出中均脱敏处理。脱敏规则为首尾各保留 6 个字符，中间替换为 `******`；长度不足以保留首尾时按安全优先原则整体替换为 `******`。

### 9.2 parse-server 配置

不再提供：

```bash
mineru config parse-server local.mode managed
mineru config parse-server remote.url https://...
```

改为：

```bash
mineru config set parse_server.local.mode managed
mineru config set parse_server.local.managed_tier standard
mineru config set parse_server.local.self_hosted_url http://127.0.0.1:8000
mineru config set parse_server.local.self_hosted_api_key <key>
mineru config set parse_server.remote.url https://mineru.net/api
mineru config set parse_server.remote.api_key <key>
```

### 9.3 Rules

保留在 `config` 子树下：

```bash
mineru config exclude-rules add <pattern> [--priority 0]
mineru config exclude-rules list [--json]
mineru config exclude-rules remove <rule_id>

mineru config parsing-rules add <pattern> [--tier <tier>] [--pages <range>] [--remote] [--name <name>]
mineru config parsing-rules list [--json]
mineru config parsing-rules remove <rule_id>
```

说明：

- exclude-rules 和 parsing-rules 是配置的一部分，不需要提升为顶层命令。
- rules 删除统一使用 `remove`，避免 `rm` 缩写降低自解释性。

### 9.4 Doclib HTTP Path 命名

MVP 阶段不保留旧 HTTP path 兼容入口，直接把配置子资源从 `/config/...` 下移到资源自身路径。

| 能力 | 新 HTTP path | 替换旧 HTTP path |
|------|--------------|------------------|
| KV config 列表 / 展示 | `GET /configs` | `GET /config` |
| KV config 单 key 查询 | `GET /configs/{key}` | 新增 |
| KV config 单 key 设置 | `PUT /configs/{key}` | `POST /config` |
| KV config 单 key unset | `DELETE /configs/{key}` | 新增 |
| watch 添加 | `POST /watches` | `POST /config/watch` |
| watch 列表 | `GET /watches` | `GET /config/watch` |
| watch 删除 | `DELETE /watches/{watch_id}` | `DELETE /config/watch` |
| exclude rule 添加 | `POST /exclude-rules` | `POST /config/exclude` |
| exclude rule 列表 | `GET /exclude-rules` | `GET /config/exclude` |
| exclude rule 删除 | `DELETE /exclude-rules/{rule_id}` | `DELETE /config/exclude/{rule_id}` |
| parsing rule 添加 | `POST /parsing-rules` | `POST /config/parsing-rules` |
| parsing rule 列表 | `GET /parsing-rules` | `GET /config/parsing-rules` |
| parsing rule 删除 | `DELETE /parsing-rules/{rule_id}` | `DELETE /config/parsing-rules/{rule_id}` |
| server shutdown | `POST /server/shutdown` | `POST /shutdown` |
| doc 内容读取 | `GET /docs/{sha256}/content` | 保留，但必须只读 |
| doc 内容导出 | `POST /docs/{sha256}/exports` | 从 `GET /docs/{sha256}/content?output=...` 拆出 |

`PUT /configs/{key}` 的 request body 只包含 value，不重复携带 key：

```json
{
  "value": "managed"
}
```

响应返回脱敏后的生效值和来源：

```json
{
  "key": "parse_server.local.mode",
  "value": "managed",
  "source": "override"
}
```

`GET /docs/{sha256}/content` 必须保持只读，不支持 `output` 参数。需要让 server 写出文件时使用 `POST /docs/{sha256}/exports`。导出 request body：

```json
{
  "tier": "standard",
  "pages": "1~5",
  "format": "markdown",
  "output": "/absolute/or/server-visible/path.md",
  "no_marker": false
}
```

导出响应：

```json
{
  "sha256": "<sha256>",
  "tier": "standard",
  "output": "/absolute/or/server-visible/path.md"
}
```

## 10. Cleanup、Forget、Invalidate、Server

### 10.1 Cleanup

保留现有命令：

```bash
mineru cleanup deleted-files [--dry-run|--no-dry-run] [--json]
mineru cleanup orphan-docs [--dry-run|--no-dry-run] [--json]
mineru cleanup temp [--older-than <days>] [--json]
```

### 10.2 Forget

保留现有命令：

```bash
mineru forget <path> [--dry-run|--no-dry-run] [--json]
```

### 10.3 Invalidate

保留现有命令：

```bash
mineru invalidate <path> [--tier <tier>]
```

说明：

- `invalidate` 是明确的缓存操作，作为顶层命令可以接受。
- 如果后续命令数量过多，可并入 `mineru cache invalidate <path>`，但 MVP 不建议扩散。

### 10.4 Server

保留现有命令：

```bash
mineru server start
mineru server stop
mineru server restart
mineru server status [--json]
```

## 11. 不纳入本轮的命令

暂不设计：

```bash
mineru content <sha256> --tier standard --pages 1~5
mineru show doc <sha256> --content --tier standard --pages 1~5
mineru parse list
mineru parses ...
mineru scan list
mineru scan status <scan_id>
mineru status parse <parse_id>
mineru status scan <scan_id>
mineru status server
mineru config parse-server ...
```

其中内容读取命令需要先明确两个问题：

- 用户是想按 `sha256` 读取已有解析产物，还是想沿用路径作为主入口。
- 内容读取是否需要支持 page range、format、marker、offset，以及这些参数是否应与 `parse` 完全一致。

## 12. 从当前实现迁移的改动清单

### 12.1 CLI 命令结构

新增 Typer 子树：

- `list`
- `show`

调整：

- `mineru scan <path>` 继续保留为顶层动作。
- 删除 `scan_cmd` 中对 `status` / `list` 的手动伪子命令解析。
- `mineru list scans` 调用现有 `DoclibClient.list_scans()`。
- `mineru list docs` 调用 `DoclibClient.list_docs()`，并补齐 `file_type` / `limit` 参数。
- `mineru list files` 需要新增 server/client/interface endpoint。
- `mineru show scan <id>` 调用现有 `DoclibClient.get_scan()`。
- `mineru list parses` 调用现有 `DoclibClient.list_parses()`，并补齐 `limit` 参数透传。
- `mineru show parse <id>` 调用现有 `DoclibClient.get_parse()`。
- `mineru show file <path>` 复用现有 `get_file_info()`。
- `mineru info <path>` 复用 `show file <path>` 的实现。
- `mineru show doc <sha256>` 调用现有 `DoclibClient.get_doc(expand_files=True)`。

删除：

- `mineru scan list`
- `mineru scan status <id>`
- `mineru config parse-server ...`

### 12.2 Config 默认值和 unset

实现要求：

- 新增代码级 `CONFIG_DEFAULTS`。
- DB 初始化不再把默认值写入 `config` 表。
- 启动或 migration 时清理历史上与默认值完全相同的 seeded rows。
- `ConfigService.get(key)` 返回 override，否则返回代码默认值。
- `ConfigService.get_all()` 返回默认值与 override 合并后的生效配置。
- `ConfigResponse` 增加 `sources: dict[str, "default" | "override"]`。
- `mineru config show` 人类可读输出标注 `[default]` / `[override]`。
- `mineru config unset <key>` 删除 override。

### 12.3 Doclib HTTP route 结构

直接修改 server/client/interface route，不保留旧 path：

- `GET /config` 改为 `GET /configs`。
- 新增 `GET /configs/{key}`，支持 `mineru config get <key>`。
- `POST /config` 改为 `PUT /configs/{key}`，request body 为 `{"value": "<value>"}`。
- 新增 `DELETE /configs/{key}`，支持 `mineru config unset <key>`。
- `POST /config/watch` 改为 `POST /watches`。
- `GET /config/watch` 改为 `GET /watches`。
- `DELETE /config/watch` 改为 `DELETE /watches/{watch_id}`；CLI 的 `watch remove <path-or-id>` 如果收到 path，需要先从 `GET /watches` 解析出 `watch_id`。
- `POST /config/exclude` 改为 `POST /exclude-rules`。
- `GET /config/exclude` 改为 `GET /exclude-rules`。
- `DELETE /config/exclude/{rule_id}` 改为 `DELETE /exclude-rules/{rule_id}`。
- `POST /config/parsing-rules` 改为 `POST /parsing-rules`。
- `GET /config/parsing-rules` 改为 `GET /parsing-rules`。
- `DELETE /config/parsing-rules/{rule_id}` 改为 `DELETE /parsing-rules/{rule_id}`。
- `POST /shutdown` 改为 `POST /server/shutdown`，与 `GET /server/status` 放在同一 server 子树下。
- `GET /docs/{sha256}/content` 移除 `output` 参数，保持只读。
- 新增 `POST /docs/{sha256}/exports`，用于需要 server 写文件的导出场景。

### 12.4 文档同步

设计确认后需要同步：

- `docs/next/cli/mineru.md`
- `docs/next/cli/mineru-library.md`
- `docs/next/sdk/doclib-client.md`
- route contract / interface contract tests

## 13. 完整 MVP 命令列表

```bash
mineru parse <file>
mineru scan <path>
mineru search <query>
mineru find <query>

mineru list parses
mineru list scans
mineru list files
mineru list docs

mineru show parse <parse_id>
mineru show scan <scan_id>
mineru show file <path>
mineru show doc <sha256>
mineru info <path>

mineru watch add <path>
mineru watch list
mineru watch remove <path-or-id>
mineru watch rescan <path-or-id>

mineru config show
mineru config get <key>
mineru config set <key> <value>
mineru config unset <key>
mineru config exclude-rules add <pattern>
mineru config exclude-rules list
mineru config exclude-rules remove <rule_id>
mineru config parsing-rules add <pattern>
mineru config parsing-rules list
mineru config parsing-rules remove <rule_id>

mineru cleanup deleted-files
mineru cleanup orphan-docs
mineru cleanup temp

mineru forget <path>
mineru invalidate <path>

mineru server start
mineru server stop
mineru server restart
mineru server status
```

## 14. 待确认问题

暂无。
