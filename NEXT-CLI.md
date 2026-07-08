# MinerU 仓库提供的命令行工具

本仓库提供两个命令行工具，分别是:
- mineru
- mineru-kit

## 两者关系

| 维度 | mineru | mineru-kit |
|------|--------|-----------|
| 定位 | 本地文档解析和管理中心 | 纯文档解析工具 |
| 受众 | Agent 系统 / 少量文档的非技术用户 | 大规模数据处理的开发人员 |
| 数据库 | 维护本地 doclib.db，去重 + 可搜索 | 不涉及数据库 |
| 输入 | 单文件（未来多文件/目录/URL） | 文件/目录 |
| 输出默认 | STDOUT（渐进式，默认有限页） | 文件（-o 必填） |
| 参数风格 | 精简，隐藏专家选项 | 完整，暴露所有引擎/并发/冲突策略参数 |

mineru-kit parse 中未出现在 mineru parse 的参数（如 `--backend`、`--tier`、`--remote-url`、`--api-key` 等）均为有意不暴露。

---

# mineru

定位为用户/Agent系统的本地文档解析和管理中心。
面向普通技术背景弱，仅仅是有文档需要解析的用户。
维护本地数据库。
所有解析过的文件均记录在数据库中，能被管理和搜索。
数据库可以规避相同内容的文件重复解析。
与桌面客户端数据互通。

子命令：
- mineru server start/stop/restart
- mineru parse
- mineru read
- mineru search


文档处理流程:

Found -> Check path -> calc SHA256 -> Check sha256 -> Plain Text                  -> just read
                                                      Word/PPT (pure CPU convert) -> parse
                                                      PDF/Image/HTML              -> light-parse

# 本地文档库的存储结构

MinerU home：默认是 `$HOME/.mineru` ，可通过 `MINERU_HOME` 配置。

[MINERU_HOME]
  config.yaml
  doclib.sock
  doclib.db
  logs/
    doclib.log
    doclib.access.log
    doclib.stdout.log
    doclib.stderr.log
  doclib/
    parsed/
    temp/


# mineru parse

## 概述

`parse` 是 `mineru` 的文档解析子命令。解析结果自动入库（doclib.db），同时输出到 STDOUT 或指定文件。支持本地引擎和远端 API 两种执行模式。

设计原则：
- **隐私优先**：不会在用户不知情的情况下将文件上传到远端，也不会静默降级解析质量
- **Agent-native**：默认输出有限页数 + 结构标记（marker），agent 可渐进式请求更多内容
- **先入库后输出**：所有解析结果先写入本地数据库，再输出到 STDOUT 或文件
- **去重**：相同 SHA-256 的文件不重复解析（除非 `--force`）

## Usage

```
mineru parse <file> [flags]
```

> 当前仅支持单个本地文件路径。未来计划支持多文件、目录路径和 URL。

## 参数

### 执行位置

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--remote` | bool | false | 切换为远端模式。远端地址来自 config；API Key 优先来自 config，未配置时使用环境变量 |

### 解析配置

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | — | string | (见下文) | 解析档位: `flash`, `medium`, `high` |
| `-p, --pages` | `-p` | string | `1~10` | 页码范围。推荐用 `~` 分隔（如 `1~5,-5~-1`），也支持 `-`（如 `1-5,-5--1`）。`all` 表示全部页 |
| `--language` | — | string | — | 文档语言提示。部分 tier 设置此参数可改善输出质量。[PDF Only]（逐渐不再推荐使用） |
| `--force` | — | bool | false | 忽略数据库缓存，强制重新解析 |

### 输出配置

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `-o, --output` | `-o` | path | `-` (STDOUT) | 输出路径。`-` 或省略表示输出到 STDOUT |
| `-f, --format` | `-f` | string | `markdown` | 输出格式。当前仅支持 `markdown`。 |
| `--no-marker` | — | bool | false | 输出的 markdown/text 中不含文档结构标记 |

### 同步控制

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--wait` | int | 60 | 同步等待的最大秒数。本地和远端模式均有效 |
| `--no-wait` | bool | false | 不等待。若文件已在数据库中完整解析过则直接返回结果；否则返回状态信息供后续查询 |

### 通用

| Flag | 简写 | 说明 |
|------|------|------|
| `-v, --verbose` | `-v` | 显示详细过程日志 |
| `-V, --version` | `-V` | 显示版本号 |

## Tier 档位

| 档位 | 定位 | 对应引擎 | 运行环境 |
|------|------|---------|---------|
| `flash` | 极速预览 | 轻量解析引擎（开发中） | 仅本地，仅 CPU |
| `medium` | 日常使用 | Pipeline（PP-DocLayoutV2 + PaddleOCR + 表格/公式模型） | 本地 / 远端 |
| `high` | 高质量 | VLM（MinerU2.5 端到端） | 本地 / 远端 |

> **注意**：更高档位不保证在所有场景下效果更好，但通常 flash 的质量上限低于 medium/high。

**默认值规则**：本地模式下，用户启动了哪个 tier 的引擎配置，就默认使用该 tier。远端模式下，不指定 `--tier` 时 CLI 不发送 `tier` 字段，由服务端使用默认选择策略。

**设计约束**：
- CLI 只暴露三个值：`flash` / `medium` / `high`。省略 `--tier` 表示使用默认选择策略
- 档位含义固定，不可通过配置修改映射关系
- 本地 high 与远端 high 效果目标相同
- 引擎名称不在 `mineru parse` 接口中暴露，用户只需选择档位

**CLI ↔ API 对齐**：
- `--remote` 模式下，`--tier` 参数直接映射到 API 的 `tier` 字段（值相同：`medium` / `high`）
- 远端模式下，CLI 统一请求 JSON 格式产物，markdown/text/html 由 CLI 本地从 JSON 生成
- local parse-server 提供与 mineru.net 完全相同的 API（Files/Uploads/Jobs），ParseWorker 根据 `--remote` 选择 base URL

> **团队讨论项**：是否为 `medium` 提供 `core` 作为短别名。

## 隐私优先：默认行为决策链

mineru 是一款强调用户隐私的工具。显式调用 `mineru parse` 时，系统按以下逻辑决策：

```
1. 用户是否指定了 --remote？
   ├─ 是 → 优先使用远程解析（使用 config 指定远端或默认 mineru.net）
   │        远程不可用时 fallback 到本地 parse-server（如果可用）
   │        本地也不可用 → 报错
   └─ 否 →
       2. 是否有本地 parse-server 可用（managed 或 self-hosted 模式探活成功）？
          ├─ 是 → 使用本地 parse-server 解析，正常执行
          └─ 没有 →
              3. 用户是否指定了 --tier flash？
                 ├─ 是 → 使用本地 flash 引擎（仅 CPU），正常执行
                 └─ 否 → 报错，提示可选方案
```

**报错输出示例**：

```
Error: 本地未检测到 medium/high 解析引擎。

可选方案：
  1. mineru parse doc.pdf --remote          使用远端解析（文件将上传至 mineru.net）
  2. mineru parse doc.pdf --tier flash      使用本地轻量解析（仅 CPU，质量较低）
  3. 配置并启动本地解析服务后重试               参考: mineru-kit api-server --help
```

**设计要点**：
- 不会在用户不知情的情况下把文件发到远端。必须显式 `--remote` 才会触发上传
- 不会在用户不知情的情况下降级到 flash。默认选择不可用时报错而非静默降级
- `--remote` 远程不可用时 fallback 到本地 parse-server——用户已接受上传，远程偶尔不可用本地兜底
- agent 收到此报错后可自行判断文件敏感性，决定选 `--remote` 还是 `--tier flash`

## 输出格式说明

| 格式 | 说明 |
|------|------|
| `markdown` | 带格式的 Markdown 文本，含标题、列表、表格等结构 |
| `text` | 纯文本，等价于 markdown 去掉所有格式修饰符 |
| `json` | middle_json 结构（完整的解析中间表示） |
| `html` | HTML 格式输出 |

## 文档结构标记（Marker）

输出的 markdown/text 中默认包含 HTML 注释形式的结构标记，用于辅助 AI agent 定位当前阅读位置：

```markdown
<!-- page 1 of 50 -->
# Introduction
...

<!-- image: page 1, bbox [72, 200, 523, 450] -->

<!-- page 5 of 50 -->
...end of page 5 content...

<!-- pages 6~45 not parsed. Use: mineru parse report.pdf --pages 6~10 -->

<!-- page 46 of 50 -->
...
```

标记类型：
- **页码标记**：`<!-- page N of M -->` 标识当前内容所在页
- **内容标记**：`<!-- image: page N, bbox [...] -->` 标识图片等非文本内容的位置信息
- **未解析提示**：`<!-- pages X~Y not parsed. Use: mineru parse ... --pages X~Y -->` 提示 agent 可请求更多页

使用 `--no-marker` 关闭所有标记输出。

## 渐进式阅读协议

默认 `--pages 1~10` 只解析有限页数。输出末尾的 marker 提示 agent 如何获取更多内容：

### PDF（物理页码）

```bash
$ mineru parse report.pdf
# → 输出第 1~10 页
# → marker: <!-- next pages available. Use: mineru parse report.pdf --pages 11~20 -->

$ mineru parse report.pdf --pages 11~20
# → 输出第 11~20 页
# → marker: <!-- next pages available. Use: mineru parse report.pdf --pages 21~30 -->
```

### 非分页文档（Word/PPT 等）

非分页文档当前不引入单独的 `--offset` 参数，而是和分页文档一样复用服务端返回的 continuation cursor。CLI 通过 `--after` 继续读取下一段内容：

```bash
$ mineru parse long.docx
# → 输出第一段内容
# → marker: <!-- next content available.
#            Use: mineru parse long.docx --after doc:ab12cd3/tier:medium/page:1/block:12/char:520 -->

$ mineru parse long.docx --after doc:ab12cd3/tier:medium/page:1/block:12/char:520
# → 输出后续内容
```

这意味着非分页文档的增量读取当前以 cursor 为正式协议，字符 offset 只作为 cursor 内部语义的一部分存在，不作为 `parse` 的公开 CLI 参数暴露。

## STDOUT 输出行为

当输出到 STDOUT（默认）时：
- 默认只输出 `--pages` 指定范围的内容（默认前 10 页）
- 图片不输出二进制数据，仅输出位置信息（页码、坐标）作为 marker
- 末尾包含提示信息，引导用户/agent 获取更多内容

当输出到文件（`-o path`）时：
- 输出 `--pages` 指定范围的完整内容，无额外截断
- 图片信息同样以 marker 形式嵌入（图片文件单独存储在数据库中）

当使用 `--json` 时，`mineru parse` 输出命令级 JSON envelope，而不是直接输出裸内容对象：

```json
{
  "parse": { "... parse summary ..." },
  "content": { "... DocContentResponse ..." } | null
}
```

其中：
- `parse` 始终存在，描述本次 `parse` 请求的解析状态摘要。
- `content` 仅在结果已经可读取时存在值；`--no-wait` 或等待超时但尚未完成时为 `null`。
- 错误场景仍输出结构化错误 JSON。

## 同步等待行为

| 场景 | 默认行为 |
|------|---------|
| 单文件 | `--wait 60`：同步等待最多 60 秒 |
| 多文件/目录（未来） | `--no-wait`：提交后立即返回状态信息 |

**超时行为**：`--wait` 指定的时间内未完成时，输出提示信息（解析仍在后台继续），包含后续查询命令。

**`--no-wait` 行为**：
- 若文件已在数据库中完整解析过（且请求的 page_range 已覆盖）→ 直接返回结果
- 否则 → 不等待，返回状态信息（行为同 `--wait` 超时）

## 示例

```bash
# 基本用法（输出到终端，默认前 10 页）
mineru parse doc.pdf

# 全文解析，输出到文件
mineru parse doc.pdf --pages all -o doc.md

# 指定页码范围
mineru parse doc.pdf --pages 10~20

# 使用 flash 档快速预览
mineru parse doc.pdf --tier flash

# 使用 high 档高质量解析
mineru parse doc.pdf --tier high --pages all -o doc.md

# 远端模式（使用默认 mineru.net）
mineru parse doc.pdf --remote

# 输出纯文本
mineru parse doc.pdf -f text

# 输出 middle_json
mineru parse doc.pdf -f json -o doc.json

# 强制重新解析（忽略缓存）
mineru parse doc.pdf --force --pages all -o doc.md

# 不等待，获取状态信息
mineru parse huge.pdf --pages all --no-wait

# 不含结构标记
mineru parse doc.pdf --no-marker
```


# mineru read

## 概述

`read` 是 `mineru` 的 locator-first 读取子命令。它读取 doclib 中已有的解析结果，不负责 discover、scan、ingest，也不默认创建 parse task。

与 `parse` 的边界：

```text
parse(path) = ensure document is parsed, then read default content
read(locator) = read existing parsed content by stable locator
```

## Usage

```bash
mineru read <locator> [flags]
```

当前 P0 支持：

```bash
mineru read <locator> [--format markdown|image] [--limit 30000] [--context N] [--output PATH] [--json] [--no-marker]
```

## Locator

```text
doc:{short_id}
doc:{short_id}/tier:{tier}
doc:{short_id}/tier:{tier}/page:{page_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

其中：
- `page_no` 和 `block_no` 使用 1-based 编号
- `char:{offset}` 使用 block 渲染文本内的 0-based 字符 offset

## 输出

- `--format markdown`：默认读取文本内容
- `--format image`：读取 page 或 block 的图像输出
- `--json`：输出完整 `DocContentResponse`
- `--output`：由 CLI 在本地写入 markdown 文件或 copy image asset

## Image 约束

- PDF：
  - page locator 支持 image
  - block locator 仅在 block 有非空 bbox 时支持 image
- Office：
  - 仅 image block 支持 image
- doc locator 和 doc/tier locator 不支持 image
- 不支持多页 image

## Continuation

`read` 的 continuation 使用 locator：

```text
<!-- Next: mineru read doc:ab12cd3/tier:medium/page:5 -->
```

## 示例

```bash
# 读取某页
mineru read doc:ab12cd3/tier:medium/page:4

# 读取某个 block 及其上下文
mineru read doc:ab12cd3/tier:medium/page:4/block:12 --context 2

# 导出 page image
mineru read doc:ab12cd3/tier:medium/page:4 --format image --output page4.jpg

# JSON 输出
mineru read doc:ab12cd3/tier:medium/page:4 --json
```


# mineru server

## 概述

mineru server 是常驻后台进程，负责：
- 文件系统监控（Watch）
- 解析任务队列调度
- 搜索索引维护
- 为 CLI 提供 API 服务

## 架构

```
┌──────────────┐   Unix Domain Socket    ┌───────────────────────┐
│  mineru CLI  │ ─── HTTP + JSON ──→     │   mineru server       │
│  (typer)     │ ←── HTTP + JSON ──→     │   (FastAPI + asyncio) │
└──────────────┘                         └───────────────────────┘
                                               │
                                          ┌────┴────┐
                                          │ SQLite  │
                                          │ + FTS5  │
                                          └─────────┘
```

- 单进程，asyncio 事件循环调度
- CLI ↔ Server 通信：HTTP + JSON over Unix Domain Socket
- 数据存储：SQLite（WAL 模式）+ FTS5 全文搜索
- Socket 路径：默认是 `$MINERU_HOME/doclib.sock`，权限 `0600`

## 生命周期

```bash
mineru server start       # 后台启动
mineru server stop        # 优雅关闭
mineru server restart     # 重启
mineru server status      # 运行状态 + 路径/HTTP/SQLite 摘要 + worker 状态 + parse-server 探活状态
```

## 内部组件

| 组件 | 职责 |
|------|------|
| Watch Loop | 文件系统事件监控（watchfiles），发现新文件/变更文件 |
| Registration Worker | 计算 SHA-256、记录文件元数据、文件名进 FTS |
| Parse Worker Pool | 执行解析任务（调用本地引擎或远端 API） |
| Device Monitor | 可插拔设备（外接硬盘）检测与恢复 |

## 崩溃恢复

Server 启动时：
- 清空所有未释放的锁（`*_locked_at` 字段）
- 重置处于 `parsing` 状态的任务为 `error`（可重试）
- 对已入库文件进行活性验证（检测已删除文件）


# Watch 与 Parsing-Rules

## Watch 机制

用户配置 mineru 监控哪些目录，系统自动发现并入库：

```bash
mineru watch add ~/Documents
mineru watch add /Volumes/SSD --removable
mineru watch list
mineru watch remove ~/Documents
mineru watch rescan ~/Documents
```

**Watch 行为**：
- 发现文件 → 入库（计算 SHA-256、记录 path/size/mtime、文件名进 FTS）
- 入库后默认使用 **flash** tier 解析首尾各 5 页
- 文件类型决定解析方式：office 文档直接本地全量解析（成本低），PDF/image 用 flash

**文件类型白名单**（Watch 自动发现的范围）：

```
文档类: pdf, doc, docx, xls, xlsx, ppt, pptx, csv, rtf, odt, ods
电子书: epub, mobi
Apple:  pages, key, numbers
文本类: txt, md, markdown, rst, tex
网页类: html, htm
邮件类: eml, mbox
```

> **图片类**（jpg, png 等）不在 Watch 白名单中——Watch 扫描到的图片默认不算做文档。但用户可通过 `mineru parse image.png` 显式触发图片解析。

**约束**：
- Watch 目录不允许嵌套
- 必须是绝对路径
- 默认排除：`*/Library/*`, `*/.git/*`, `*/node_modules/*`, `*/vendor/*`, `*/go/pkg/*`, `*/__pycache__/*`, `*/.venv/*`, `*/miniconda3/*`, `*/.nvm/*`, `*/.docker/*`, `*/target/*`, `*/dist/*`, `*/build/*`

## 可插拔设备

> 设计 P0（架构需提前考虑），实现 P1。

支持外接硬盘、U 盘等可插拔存储设备的 Watch：

```bash
mineru watch add /Volumes/SSD --removable
```

**设计要点**：
- Server 定期对 `--removable` 的 watch 根路径执行 `os.stat()` 检测设备状态
- 设备拔出 → watch 状态标记为 `unreachable`，该 watch 下所有文件标记为 `unreachable`（非 `deleted`）
- 设备插回 → 恢复为 `active`，触发增量扫描，重试之前因拔出而失败的任务
- 正在进行中的解析任务遇到设备拔出 → 标记特殊错误（留有恢复机会），不视为永久失败
- 第一版默认搜索结果不返回 `unreachable` 文件；后续可增加显式参数返回并标注"设备未连接"

## Parsing-Rules（解析规则）

Parsing-rules 决定**哪些文件在 watch 自动触发时升级到更高 tier 或解析更多页数**：

```bash
mineru config parsing-rules add "*/论文/*" --tier medium --pages all
mineru config parsing-rules add "*/合同/*" --tier high --remote
mineru config parsing-rules list
mineru config parsing-rules rm <id>
```

### 规则生效前提

规则命中时，系统检查是否具备对应能力：

| 规则要求 | 前提条件 | 不满足时 |
|---------|---------|---------|
| `--tier medium` | 本地 medium 服务已配置 | 规则不生效，保持 flash |
| `--tier high` | 本地 high 服务已配置 | 规则不生效，保持 flash |
| `--tier medium --remote` | 允许远端解析 | 使用远端 medium |
| `--tier high --remote` | 允许远端解析 | 使用远端 high |

**核心原则**：不会因为规则配置而静默上传文件到远端。规则中必须显式包含 `--remote` 才会使用远端解析。

### 与 everydoc auto-parse 的区别

| | everydoc auto-parse | mineru parsing-rules |
|--|---|---|
| 决定什么 | "是否 parse"（二选一） | "用什么 tier / 解析多少页"（连续光谱） |
| 前提 | 无 | 必须有对应能力（本地服务 或 允许 remote） |
| 默认状态 | 文件未 parse | 文件已用 flash 解析（至少有预览） |


# mineru 其他子命令

## mineru search

搜索已入库/已解析的文档内容：

```bash
mineru search "关键词"                   # 搜索文档内容
mineru search "关键词" --type pdf         # 限定文件类型
mineru search "关键词" --tier medium    # 限定索引来源 tier
mineru search "关键词" --min-tier medium # 限定最低索引来源 tier
mineru search "关键词" --limit 10         # 限制结果数
mineru search "关键词" --json             # JSON 输出（给 agent）
```

搜索结果标注 snippet 来源 tier（agent 可据此判断可信度）：
- flash 解析：仅搜索到首尾页内容
- medium/high 解析：搜索全文内容

搜索结果默认优先返回 active file paths。如果某个已索引 doc 没有任何 active file，则 fallback 返回非 active file paths，避免历史文档完全不可定位。

## mineru find

仅搜索文件名（覆盖所有已入库文件，不论是否解析过）：

```bash
mineru find "report"
mineru find "2024" --ext pdf --json
```

## mineru show file

查看文件详情：

```bash
mineru show file ~/Documents/report.pdf
```

输出：文件路径、类型、大小、SHA-256、解析状态、使用的 tier、已解析页范围、所属 watch 目录。

## mineru config

配置管理：

```bash
mineru config show                             # 查看全部配置
mineru watch add/list/remove/rescan            # Watch 目录管理
mineru config exclude add/list/rm              # 排除规则（glob pattern）
mineru config parsing-rules add/list/rm        # 解析规则
mineru config parse-server local.mode <mode>   # 本地 parse-server 模式: disabled | managed | self_hosted
mineru config parse-server local.managed-tier <tier>  # managed 模式 tier: medium | high
mineru config parse-server local.self_hosted_url <url>  # self_hosted 模式的 parse-server 地址
mineru config parse-server local.self-hosted-api-key <key>  # self_hosted 模式的 API Key（可选）
mineru config parse-server remote.url <url>    # 远程 parse-server 地址（默认 mineru.net）
mineru config parse-server remote.api-key <key>  # 远程 API Key
```

**parse-server 配置说明**：

| 配置项 | 默认值 | 说明 |
|------|------|------|
| `parse_server.local.mode` | `disabled` | `disabled` — 无本地 parse-server；`managed` — doclib 自动拉起；`self_hosted` — 用户管理 |
| `parse_server.local.managed_tier` | `medium` | managed 模式启动的 tier |
| `parse_server.local.self_hosted_url` | — | self_hosted 模式的 parse-server HTTP 地址 |
| `parse_server.local.self_hosted_api_key` | — | self_hosted 模式的 API Key（可选） |
| `parse_server.remote.url` | `https://mineru.net/api` | 远程 parse-server 地址 |
| `parse_server.remote.api_key` | — | 远程 API Key。未配置时可回退读取环境变量 `MINERU_API_KEY` |


# mineru 数据库设计概要

## 核心表

```
files              文件实例（path, sha256, size, mtime, watch_id, status）
docs               文档内容（sha256 主键, page_count, metadata）
parses             解析批次（sha256, tier, page_range, status）
fts_index          全文搜索索引（解析产出的文本）
fts_filenames      文件名搜索索引
watches            监控目录配置
rules              解析规则（parsing-rules + exclude）
config             KV 全局配置
```

## 关键设计

- **File/Doc 分离**：多个路径（备份副本）可指向同一个 doc（SHA-256 去重）
- **增量解析**：每个 tier 独立追踪已完成 parse batch 的 `page_range` 覆盖范围，互不影响。同一 tier 内只解析未覆盖的页（增量），不同 tier 之间不存在覆盖关系。`--force` 忽略 done cache，但仍可复用 active parse batch
- **多 tier 结果共存**：同一文件用 flash 解析过，再用 high 解析，两份结果共存（不覆盖）。搜索索引保留当前最高 tier 的内容，用户可用 `--tier` / `--min-tier` 过滤搜索结果
- **任务队列**：无实体队列，靠 DB 查询 `WHERE status='pending' AND lock_expired`
- **锁机制**：时间戳锁，超时自动释放（parse 锁 30 分钟，registration 锁 60 秒）
- **错误分类**：File 级（路径/权限）、Doc 级（内容损坏/加密）、Parse 级（API 超时/引擎失败）



# mineru-kit

定位为纯文档解析工具。
面向本地部署mineru，有大量数据要处理的开发者。
仅解析，不建索引，无法根据内容搜索。
不会用到数据库。

子命令：
- mineru-kit models（模型下载、配置查看与校验）
- mineru-kit parse（单个文件、文件夹的解析）
- mineru-kit api-server（本地启动的端到端解析服务，实现 NEXT-API.md 规范，提供 medium / high tier 的解析能力）

mineru-kit parse 替换：
- mineru -p xxx.pdf -o xxx.md
- mineru-open-api extract
- mineru-open-api flash-extract

mineru-kit api-server 替换：
- mineru-api 命令

mineru-kit vlm-server 替换：
- mineru-vllm-server
- mineru-lmdeploy-server
- mineru-openai-server

mineru-kit models 替换：
- mineru-models-download

## mineru-kit models

`models` 是 `mineru-kit` 的模型管理命令组。第一阶段只提供三个子命令：

- `mineru-kit models download`
- `mineru-kit models show`
- `mineru-kit models verify`

第一阶段继续使用旧配置文件体系：

- 默认 `~/mineru.json`
- 或 `MINERU_TOOLS_CONFIG_JSON` 指定的路径

`download` 用位置参数显式指定 bundle，不提供默认 bundle：

```bash
mineru-kit models download pipeline
mineru-kit models download vlm --source modelscope
mineru-kit models download all
```

规则：

- `--source` 默认 `huggingface`
- 下载完成后默认更新配置文件
- 不支持 `--config`
- 不支持 `--no-config`
- 不支持显式设置模型目录
- 当前保持现有下载行为：由 Hugging Face / ModelScope 下载器自行决定缓存路径，MinerU 只记录最终模型目录

`show` 用于查看当前模型配置和基本状态，`verify` 用于轻量校验模型配置与关键路径。

## mineru-kit vlm-server

`vlm-server` 是未来唯一正式的本地 VLM 服务启动入口。它部署的是与 mineru.net chat API 同类的文档理解 VLM 模型，主要用于 OCR、布局理解、页面理解和文档局部问答，不承诺通用聊天能力。

它可以成为 `mineru-kit api-server` 的后端，尤其服务于 `vlm-http-client` / `hybrid-http-client`。

稳定提供的协议范围：

- `GET /v1/health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`/v1/responses` 不作为当前稳定承诺。

统一参数仅有：

```bash
mineru-kit vlm-server --engine auto
mineru-kit vlm-server --engine vllm
mineru-kit vlm-server --engine lmdeploy
mineru-kit vlm-server --engine sglang
mineru-kit vlm-server --engine mlx
```

除 `--engine` 外，其余参数原样透传到底层 engine server。

## mineru-kit api-server

`api-server` 是未来唯一正式的 parse-server 启动入口。它启动 self-hosted HTTP 解析服务，实现 v1 API（非 doclib API）中的绝大多数 path。它在功能上与 mineru.net 的 parse API 对齐，区别仅在于 base URL 不同（`http://127.0.0.1:16580` vs `https://mineru.net/api`）。

### 与 mineru doclib 的协作

```
mineru doclib（纯 CPU 进程）
  │
  ├─ flash tier → 直接调用 mineru.parser.parse()
  │
  └─ medium / high tier → HTTP 调用
       ├─ 未指定 --remote → mineru-kit api-server (127.0.0.1:16580)
       └─ 指定 --remote    → config 指定远端或 mineru.net/api
```

- **managed**：doclib 启动时自动拉起 api-server，停止时一起关闭；这不是用户直接执行的命令模式
- **self-hosted**：用户独立启动 api-server，通过 config 告知 doclib 地址
- **API 格式一致**：local parse-server 和 mineru.net 使用同一套 v1 parse API，ParseWorker 仅切换 base URL

### Usage

```bash
mineru-kit api-server --tier medium --port 16580
mineru-kit api-server --tier high --port 15982
mineru-kit api-server --tier medium --language en --ocr-mode ocr --disable-table
```

规则：

- 一个进程只服务一个 tier
- `--tier` 默认值为 `medium`
- `--backend` 是高级覆盖参数
- `--tier` 与 `--backend` 同时出现且不兼容时，启动直接报错
- 启动后的 HTTP API 不公开 backend；`GET /v1/tiers` 也不新增 backend 字段

稳定公开参数：

- `--host`
- `--port`
- `--tier`
- `--api-key`

稳定解析参数：

- `--language`
- `--ocr-mode`
- `--disable-table`
- `--disable-formula`
- `--disable-image-analysis`
- `--concurrency`
- `--upload-dir`
- `--url-timeout`
- `--max-wait`

专家参数：

- `--backend`

不进入正式命令设计：

- `--reload`


# mineru-kit parse

## 概述

`parse` 是 `mineru-kit` 的通用文档解析子命令。它是无状态批处理命令，支持 local 和 remote 两种执行模式，输入只支持文件和目录。

## Usage

```
mineru-kit parse [input...] [flags]
```

## 输入

至少指定一个输入。当前只支持 `<input>` 位置参数，不支持 stdin 或路径列表。

```
<input>...              文件路径或目录路径，可混合多个
```

### 目录展开规则

- 默认仅展开一层
- 不支持递归展开
- 自动跳过不支持的文件类型
- 按文件名排序，保证可复现
- 目录下无任何支持的文件则报错退出

## 输出

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--output` | `-o` | path | — | 输出路径，必填。单文件可为文件路径或目录路径；多文件只能为目录路径 |
| `--format` | `-f` | string | `markdown` | 输出格式: `markdown`, `middle_json`, `zip` |

### 命名与冲突

单文件且 `--output` 为目录路径时：

- `markdown` -> `<stem>.md`
- `middle_json` -> `<stem>.json`
- `zip` -> `<stem>.zip`

多文件时，命名方式相同。

如果一个批次内多个输入映射到同一个输出路径，直接报错，整个批次都不解析。

示例：

```
# 不同目录同名文件
mineru-kit parse a/report.pdf b/report.pdf -o out/
# Error: output filename collision detected:
#   report.md
#     ← a/report.pdf
#     ← b/report.pdf
# Entire batch aborted before parsing.
```

## 执行模式

### 全局控制

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--remote` | — | bool | false | 连接 `mineru.net` 官方解析服务 |
| `--remote-url` | — | url | — | 连接指定解析服务 |
| `--api-key` | — | string | — | remote 模式使用的 API Key |
| `--verbose` | `-v` | bool | false | 详细输出 (远端: HTTP 调试; 本地: 引擎日志) |

## 文档参数

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--language` | — | string | `ch` | 文档语言。支持与 `mineru-kit api-server` 一致的语言集合 |
| `--pages` | — | string | — | 页码范围，如 `"1-10,15"` (默认全部) |
| `--disable-formula` | — | bool | false | 禁用公式识别 |
| `--disable-table` | — | bool | false | 禁用表格识别 |

## 本地模式专属

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | — | string | — | 解析档位: `flash`, `medium`, `high`；默认选择与 `api-server` 一致，但不会默认落到 `flash` |
| `--backend` | `-b` | string | — | 本地后端；允许与 `--tier` 同时出现，但不兼容时应报错 |
| `--ocr-mode` | — | string | `auto` | OCR / text extraction mode: `auto`, `txt`, `ocr` |
| `--disable-image-analysis` | — | bool | false | 禁用图片/图表分析（VLM/hybrid 生效） |

## 远端模式专属

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
remote 模式允许传 `--tier`，但禁止传 `--backend`。

未传 `--tier` 时，使用目标服务提供的最高 tier。

传了 `--tier` 但目标服务不提供时，直接报错。

## 示例

### 本地模式

```bash
# 单文件
mineru-kit parse doc.pdf -o out/

# 扫描件 OCR
mineru-kit parse scanned.pdf -o out/ --ocr-mode ocr

# 英文文档，指定后端
mineru-kit parse report.pdf -o out/ --backend pipeline --language en

# 显式使用 flash
mineru-kit parse report.pdf -o out/ --tier flash

# 目录批量
mineru-kit parse ./invoices/ -o out/
```

### 远端模式

```bash
# 官方远端
mineru-kit parse doc.pdf -o out/ --remote

# 官方远端，显式要求 high
mineru-kit parse doc.pdf -o out/ --remote --tier high --api-key xxx

# 自定义远端
mineru-kit parse doc.pdf -o out/ --remote-url https://example.com/api --tier medium --api-key xxx
```

### 输出冲突处理

```bash
# 报错退出
mineru-kit parse a/report.pdf b/report.pdf -o out/
# Error: output filename collision detected:
#   report.md
#     ← a/report.pdf
#     ← b/report.pdf
# Entire batch aborted before parsing.
```
