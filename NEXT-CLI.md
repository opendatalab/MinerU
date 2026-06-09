# MinerU 仓库提供的命令行工具

本仓库提供两个命令行工具，分别是:
- mineru
- mineru-kit

## 两者关系

| 维度 | mineru | mineru-kit |
|------|--------|-----------|
| 定位 | 本地文档解析和管理中心 | 纯文档解析工具 |
| 受众 | Agent 系统 / 少量文档的非技术用户 | 大规模数据处理的开发人员 |
| 数据库 | 维护本地 mineru.db，去重 + 可搜索 | 不涉及数据库 |
| 输入 | 单文件（未来多文件/目录/URL） | 文件/目录/URL/文件列表/stdin 混合 |
| 输出默认 | STDOUT（渐进式，默认有限页） | 文件（-o 必填） |
| 参数风格 | 精简，隐藏专家选项 | 完整，暴露所有引擎/并发/冲突策略参数 |

mineru-kit parse 中未出现在 mineru parse 的参数（如 `--backend`、`--method`、`--on-collision`、`--stdin`、`--concurrency` 等）均为有意不暴露。

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
- mineru info
- mineru search


文档处理流程:

Found -> Check path -> calc SHA256 -> Check sha256 -> Plain Text                  -> just read
                                                      Word/PPT (pure CPU convert) -> parse
                                                      PDF/Image/HTML              -> light-parse

# 本地文档库的存储结构

data_dir：默认是 `$HOME/MinerU` ，允许用户配置。

[data_dir]
  mineru.db
  mineru.log


# mineru parse

## 概述

`parse` 是 `mineru` 的文档解析子命令。解析结果自动入库（mineru.db），同时输出到 STDOUT 或指定文件。支持本地引擎和远端 API 两种执行模式。

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
| `--remote` | url (可选) | — | 切换为远端模式。不带参数时使用默认地址 mineru.net；带参数时使用指定地址 |
| `--api-key` | string | — | API Key（也从配置文件 `parse_server.remote.api_key` 和环境变量 `MINERU_API_KEY` 读取）|

### 解析配置

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | — | string | (见下文) | 解析档位: `flash`, `standard`, `pro` |
| `-p, --pages` | `-p` | string | `1~5,-5~-1` | 页码范围。推荐用 `~` 分隔（如 `1~5,-5~-1`），也支持 `-`（如 `1-5,-5--1`）。`all` 表示全部页 |
| `--language` | — | string | — | 文档语言提示。部分 tier 设置此参数可改善输出质量。[PDF Only]（逐渐不再推荐使用） |
| `--force` | — | bool | false | 忽略数据库缓存，强制重新解析 |

### 输出配置

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `-o, --output` | `-o` | path | `-` (STDOUT) | 输出路径。`-` 或省略表示输出到 STDOUT |
| `-f, --format` | `-f` | string | `markdown` | 输出格式: `markdown`, `text`, `json`, `html`。单选 |
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
| `standard` | 日常使用 | Pipeline（PP-DocLayoutV2 + PaddleOCR + 表格/公式模型） | 本地 / 远端 |
| `pro` | 高质量 | VLM（MinerU2.5 端到端） | 本地 / 远端 |

> **注意**：更高档位不保证在所有场景下效果更好，但通常 flash 的质量上限低于 standard/pro。

**默认值规则**：本地模式下，用户启动了哪个 tier 的引擎配置，就默认使用该 tier。远端模式下，不指定 `--tier` 时 CLI 向 API 发送 `tier: auto`（由服务端选择最佳方案）。

**设计约束**：
- CLI 只暴露三个值：`flash` / `standard` / `pro`。不暴露 `auto`——省略 `--tier` 即为 auto 行为
- 档位含义固定，不可通过配置修改映射关系
- 本地 pro 与远端 pro 效果目标相同
- 引擎名称不在 `mineru parse` 接口中暴露，用户只需选择档位

**CLI ↔ API 对齐**：
- `--remote` 模式下，`--tier` 参数直接映射到 API 的 `tier` 字段（值相同：`standard` / `pro`）
- 远端模式下，CLI 统一请求 JSON 格式产物，markdown/text/html 由 CLI 本地从 JSON 生成
- local parse-server 提供与 mineru.net 完全相同的 API（Files/Uploads/Jobs），ParseWorker 根据 `--remote` 选择 base URL

> **团队讨论项**：是否为 `standard` 提供 `core` 作为短别名。

## 隐私优先：默认行为决策链

mineru 是一款强调用户隐私的工具。显式调用 `mineru parse` 时，系统按以下逻辑决策：

```
1. 用户是否指定了 --remote？
   ├─ 是 → 优先使用远程解析（--remote 不带参数时默认 mineru.net，带参数时使用指定地址）
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
Error: 本地未检测到 standard/pro 解析引擎。

可选方案：
  1. mineru parse doc.pdf --remote          使用远端解析（文件将上传至 mineru.net）
  2. mineru parse doc.pdf --tier flash      使用本地轻量解析（仅 CPU，质量较低）
  3. 配置并启动本地解析服务后重试               参考: mineru-kit api-server --help
```

**设计要点**：
- 不会在用户不知情的情况下把文件发到远端。必须显式 `--remote` 才会触发上传
- 不会在用户不知情的情况下降级到 flash。`auto` tier 不可用时报错而非静默降级
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

默认 `--pages 1~5,-5~-1` 只解析有限页数。输出末尾的 marker 提示 agent 如何获取更多内容：

### PDF（物理页码）

```bash
$ mineru parse report.pdf
# → 输出第 1~5 页和第 46~50 页
# → marker: <!-- pages 6~45 not parsed. Use: mineru parse report.pdf --pages 6~10 -->

$ mineru parse report.pdf --pages 6~10
# → 输出第 6~10 页
# → marker: <!-- pages 11~45 not parsed. Use: mineru parse report.pdf --pages 11~15 -->
```

### 非分页文档（Word/PPT 等）

非分页文档的渐进式阅读方案待团队讨论确定，以下为两个候选方案：

**方案 A：虚拟分页（统一 `--pages` 语义）**

系统内部按段落边界将文档切分为虚拟页（每页约 3000~5000 字符），对外暴露统一的页码接口：

```bash
$ mineru parse long.docx
# → 输出虚拟页 1~5 和倒数 5 页
# → marker: <!-- pages 6~20 not shown. Use: mineru parse long.docx --pages 6~10 -->
```

优势：agent 协议完全统一，不区分文件类型。
劣势：需要解释"虚拟页"概念。

**方案 B：字符偏移（`--offset`）**

截断位置对齐到段落边界，marker 中给出字符偏移：

```bash
$ mineru parse long.docx
# → 输出前 ~30K 字符（对齐到段落边界）
# → marker: <!-- truncated at paragraph boundary (char ~30000 of ~150000).
#            Next: mineru parse long.docx --offset 30000 -->

$ mineru parse long.docx --offset 30000
# → 输出 30K~60K 字符
# → marker: <!-- Next: mineru parse long.docx --offset 60000 -->
```

优势：无新概念，实现简单。
劣势：PDF 用 `--pages`，Word 用 `--offset`，agent 需处理两种参数。

> **团队讨论项**：选择方案 A 或方案 B。

## STDOUT 输出行为

当输出到 STDOUT（默认）时：
- 默认只输出 `--pages` 指定范围的内容（默认前后各 5 页）
- 图片不输出二进制数据，仅输出位置信息（页码、坐标）作为 marker
- 末尾包含提示信息，引导用户/agent 获取更多内容

当输出到文件（`-o path`）时：
- 输出 `--pages` 指定范围的完整内容，无额外截断
- 图片信息同样以 marker 形式嵌入（图片文件单独存储在数据库中）

## 同步等待行为

| 场景 | 默认行为 |
|------|---------|
| 单文件 | `--wait 60`：同步等待最多 60 秒 |
| 多文件/目录（未来） | `--no-wait`：提交后立即返回状态信息 |

**超时行为**：`--wait` 指定的时间内未完成时，输出提示信息（解析仍在后台继续），包含后续查询命令。

**`--no-wait` 行为**：
- 若文件已在数据库中完整解析过（且请求的 pages 范围已覆盖）→ 直接返回结果
- 否则 → 不等待，返回状态信息（行为同 `--wait` 超时）

## 示例

```bash
# 基本用法（输出到终端，默认前后各5页）
mineru parse doc.pdf

# 全文解析，输出到文件
mineru parse doc.pdf --pages all -o doc.md

# 指定页码范围
mineru parse doc.pdf --pages 10~20

# 使用 flash 档快速预览
mineru parse doc.pdf --tier flash

# 使用 pro 档高质量解析
mineru parse doc.pdf --tier pro --pages all -o doc.md

# 远端模式（使用默认 mineru.net）
mineru parse doc.pdf --remote

# 远端模式（自定义地址）
mineru parse doc.pdf --remote https://my.server.com

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
- Socket 路径：`/tmp/mineru.sock`（权限 `0600`，仅当前用户）

## 生命周期

```bash
mineru server start       # 后台启动
mineru server stop        # 优雅关闭
mineru server restart     # 重启
mineru server status      # 运行状态 + 索引统计 + 队列长度 + parse-server 探活状态
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
mineru config watch add ~/Documents
mineru config watch add /Volumes/SSD --removable
mineru config watch list
mineru config watch rm ~/Documents
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
mineru config watch add /Volumes/SSD --removable
```

**设计要点**：
- Server 定期对 `--removable` 的 watch 根路径执行 `os.stat()` 检测设备状态
- 设备拔出 → watch 状态标记为 `unreachable`，该 watch 下所有文件标记为 `unreachable`（非 `deleted`）
- 设备插回 → 恢复为 `active`，触发增量扫描，重试之前因拔出而失败的任务
- 正在进行中的解析任务遇到设备拔出 → 标记特殊错误（留有恢复机会），不视为永久失败
- 搜索结果中 `unreachable` 文件仍可命中，但标注"设备未连接"

## Parsing-Rules（解析规则）

Parsing-rules 决定**哪些文件在 watch 自动触发时升级到更高 tier 或解析更多页数**：

```bash
mineru config parsing-rules add "*/论文/*" --tier standard --pages all
mineru config parsing-rules add "*/合同/*" --tier pro --remote
mineru config parsing-rules list
mineru config parsing-rules rm <id>
```

### 规则生效前提

规则命中时，系统检查是否具备对应能力：

| 规则要求 | 前提条件 | 不满足时 |
|---------|---------|---------|
| `--tier standard` | 本地 standard 服务已配置 | 规则不生效，保持 flash |
| `--tier pro` | 本地 pro 服务已配置 | 规则不生效，保持 flash |
| `--tier standard --remote` | 允许远端解析 | 使用远端 standard |
| `--tier pro --remote` | 允许远端解析 | 使用远端 pro |

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
mineru search "关键词" --limit 10         # 限制结果数
mineru search "关键词" --json             # JSON 输出（给 agent）
```

搜索结果标注解析状态（agent 可据此判断可信度）：
- flash 解析：仅搜索到首尾页内容
- standard/pro 解析：搜索全文内容

## mineru find

仅搜索文件名（覆盖所有已入库文件，不论是否解析过）：

```bash
mineru find "report"
mineru find "2024" --type pdf --json
```

## mineru info

查看文件详情：

```bash
mineru info ~/Documents/report.pdf
```

输出：文件路径、类型、大小、SHA-256、解析状态、使用的 tier、已解析页范围、所属 watch 目录。

## mineru config

配置管理：

```bash
mineru config show                             # 查看全部配置
mineru config watch add/list/rm                # Watch 目录管理
mineru config exclude add/list/rm              # 排除规则（glob pattern）
mineru config parsing-rules add/list/rm        # 解析规则
mineru config parse-server local.mode <mode>   # 本地 parse-server 模式: disabled | managed | self_hosted
mineru config parse-server local.managed-tier <tier>  # managed 模式 tier: standard | pro
mineru config parse-server local.self_hosted_url <url>  # self_hosted 模式的 parse-server 地址
mineru config parse-server local.self-hosted-api-key <key>  # self_hosted 模式的 API Key（可选）
mineru config parse-server remote.url <url>    # 远程 parse-server 地址（默认 mineru.net）
mineru config parse-server remote.api-key <key>  # 远程 API Key
```

**parse-server 配置说明**：

| 配置项 | 默认值 | 说明 |
|------|------|------|
| `parse_server.local.mode` | `disabled` | `disabled` — 无本地 parse-server；`managed` — doclib 自动拉起；`self_hosted` — 用户管理 |
| `parse_server.local.managed_tier` | `standard` | managed 模式启动的 tier |
| `parse_server.local.self_hosted_url` | — | self_hosted 模式的 parse-server HTTP 地址 |
| `parse_server.local.self_hosted_api_key` | — | self_hosted 模式的 API Key（可选） |
| `parse_server.remote.url` | `https://mineru.net/api` | 远程 parse-server 地址 |
| `parse_server.remote.api_key` | — | 远程 API Key（环境变量 `MINERU_API_KEY` 也可设置）|


# mineru 数据库设计概要

## 核心表

```
files              文件实例（path, sha256, size, mtime, watch_id, scan_status）
docs               文档内容（sha256 主键, parse_status, parse_tier, parsed_pages）
fts_index          全文搜索索引（解析产出的文本）
fts_filenames      文件名搜索索引
watch_targets      监控目录配置
rules              解析规则（parsing-rules + exclude）
config             KV 全局配置
```

## 关键设计

- **File/Doc 分离**：多个路径（备份副本）可指向同一个 doc（SHA-256 去重）
- **增量解析**：每个 tier 独立追踪 `parsed_pages`，互不影响。同一 tier 内只解析未覆盖的页（增量），不同 tier 之间不存在覆盖关系。`--force` 忽略所有 tier 的缓存，强制重新解析
- **多 tier 结果共存**：同一文件用 flash 解析过，再用 pro 解析，两份结果共存（不覆盖）。搜索时优先使用最高 tier 的结果，用户可按 tier 查询历史解析
- **任务队列**：无实体队列，靠 DB 查询 `WHERE status='pending' AND lock_expired`
- **锁机制**：时间戳锁，超时自动释放（parse 锁 30 分钟，registration 锁 60 秒）
- **错误分类**：File 级（路径/权限）、Doc 级（内容损坏/加密）、Parse 级（API 超时/引擎失败）



# mineru-kit

定位为纯文档解析工具。
面向本地部署mineru，有大量数据要处理的开发者。
仅解析，不建索引，无法根据内容搜索。
不会用到数据库。

子命令：
- mineru-kit parse（单个文件、文件夹的解析）
- mineru-kit api-server（本地启动的端到端解析服务，实现 NEXT-API.md 规范，提供 standard / pro tier 的解析能力）

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

## mineru-kit api-server

`api-server` 是本地部署的 HTTP 解析服务，实现 NEXT-API.md 定义的 Files/Uploads/Jobs 端点。它在功能上与 mineru.net 等价，区别仅在于 base URL 不同（`http://127.0.0.1:15981` vs `https://mineru.net/api`）。

### 与 mineru doclib 的协作

```
mineru doclib（纯 CPU 进程）
  │
  ├─ flash tier → 直接调用 mineru.parser.parse()
  │
  └─ standard / pro tier → HTTP 调用
       ├─ 未指定 --remote → mineru-kit api-server (127.0.0.1:15981)
       └─ 指定 --remote    → mineru.net/api （或 --remote 指定的地址）
```

- **managed 模式**：doclib 启动时自动拉起 api-server，停止时一起关闭
- **self_hosted 模式**：用户独立启动 api-server，通过 config 告知 doclib 地址
- **API 格式完全一致**：local parse-server 和 mineru.net 使用同一套 API，ParseWorker 仅切换 base URL

### Usage

```bash
mineru-kit api-server --tier standard --port 15981
mineru-kit api-server --tier pro --port 15981
```

| Flag | 默认 | 说明 |
|------|------|------|
| `--tier` | `standard` | 服务提供的解析档位：`standard` / `pro` |
| `--port` | `15981` | 监听端口（仅 127.0.0.1） |


# mineru-kit parse

## 概述

`parse` 是 `mineru-kit` 的通用文档解析子命令。支持本地引擎和远端 API 两种执行模式，通过 `--server` 自动切换。输入支持文件路径、目录路径和 URL 的灵活混合，批量处理时提供可配置的输出消歧策略。

## Usage

```
mineru-kit parse [input...] [flags]
```

## 输入

至少指定一种输入来源。`<input>` 位置参数、`--list`、`--stdin-list` 三种来源合并为一个统一输入列表。`--stdin` 为独占模式，与其他输入方式互斥。

```
<input>...              文件路径、目录路径或 URL，可混合多个
--list <file>           从文件追加输入（每行一个，# 开头为注释）
--stdin-list            从 stdin 追加输入（每行一个）
--stdin                 独占模式：从 stdin 读取文件原始字节
--stdin-name <name>     stdin 模式的文件名/后缀，用于推断格式 (默认: stdin.pdf)
```

### 目录展开规则

- 默认仅展开一层（`glob("*")`）
- `-r` / `--recursive` 开启递归展开
- 自动跳过隐藏文件和不支持的文件类型
- 按文件名排序，保证可复现
- 目录下无任何支持的文件则报错退出

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--recursive` | `-r` | bool | false | 递归展开目录 |

## 输出

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--output` | `-o` | path | — | 输出路径。本地模式必填（视为目录）；远端单文件可选（省略则 stdout）；远端批量必填（视为目录） |
| `--format` | `-f` | string | `md` | 输出格式，逗号分隔: `md`, `json`, `html`, `latex`, `docx`。仅远端模式支持多格式 |
| `--on-collision` | — | string | `fail` | 输出文件名冲突策略 (详见下文) |

### 冲突处理

所有输入展开后，预处理阶段全部 stem 完成冲突检测。冲突发生时不静默处理，由 `--on-collision` 决定策略：

| 值 | 行为 |
|----|------|
| `fail` | 报错退出，列出冲突详情和解决建议 (默认) |
| `rename` | 自动加父目录前缀消歧；前缀不够加多层；根目录仍冲突时回退数字后缀 |
| `path` | 按输入相对路径镜像输出目录结构，无冲突 |

示例：

```
# 不同目录同名文件
mineru-kit parse a/report.pdf b/report.pdf -o out/ --on-collision rename
# → out/a_report.md + out/b_report.md

mineru-kit parse a/report.pdf b/report.pdf -o out/ --on-collision path
# → out/a/report.md + out/b/report.md
```

## 执行模式

### 全局控制

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--server` | — | url | — | 远端 API 地址。指定则切换为远端模式，否则使用本地引擎 |
| `--api-key` | — | string | — | API Key (覆盖环境变量 `MINERU_API_KEY`) |
| `--verbose` | `-v` | bool | false | 详细输出 (远端: HTTP 调试; 本地: 引擎日志) |

### 同步/异步行为

| 场景 | 默认行为 | 覆盖 |
|------|---------|------|
| 本地 单文件 | 同步阻塞，进度条 | — |
| 本地 目录/批量 | 同步阻塞，多任务进度条 | — |
| 远端 单文件 | 同步等待 (轮询 + 进度条) | `--async` 立即返回 task-id |
| 远端 批量 | 异步 (提交全部，列 task-id 清单) | `--wait` 同步等到全部完成 |
| 远端 `--stdin` | 同步等待 | `--async` |
| 远端 `--fast` | 同步等待 | `--async` |

## 文档参数

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--lang` | `-l` | string | `ch` | 文档语言。支持: `ch`, `ch_server`, `ch_lite`, `en`, `korean`, `japan`, `chinese_cht`, `latin`, `arabic`, 等 |
| `--pages` | — | string | — | 页码范围，如 `"1-10,15"` (默认全部) |
| `--formula` | — | bool | true | 公式识别 (`--no-formula` 关闭) |
| `--table` | — | bool | true | 表格识别 (`--no-table` 关闭) |

## 本地模式专属

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--method` | `-m` | string | `auto` | 解析方法: `auto`, `txt`, `ocr` |
| `--backend` | `-b` | string | `hybrid-auto-engine` | 后端引擎: `pipeline`, `vlm-http-client`, `hybrid-http-client`, `vlm-auto-engine`, `hybrid-auto-engine` |
| `--vlm-url` | `-u` | url | — | VLM 推理服务地址 (http-client 后端需要) |
| `--image-analysis` | — | bool | true | 图片/图表分析 (`--no-image-analysis` 关闭，VLM/hybrid 后端生效) |

## 远端模式专属

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--model` | — | string | `auto` | 模型选择: `auto`, `vlm`, `pipeline`, `html` |
| `--ocr` | — | bool | false | OCR 模式，用于扫描件 (`--no-ocr` 关闭) |
| `--fast` | — | bool | false | 轻量模式: 无需认证, 仅 md, ≤10MB/20页 (覆盖 `--format`, `--model`, `--formula`, `--table`) |
| `--async` | — | bool | false | 提交后立即返回 task-id，不等待结果 |
| `--wait` | — | bool | false | 批量模式下同步等待全部完成 (覆盖远端批量的默认异步行为) |

## 超时与并发

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--timeout` | — | int | 0 | 最大等待秒数。0 表示使用模式默认值: 远端单文件 300s, 远端批量 1800s |
| `--concurrency` | — | int | 0 | 并发处理数。本地模式默认 1 (GPU 串行)；远端模式默认由服务端控制 |

## 参数适用性矩阵

| Flag | 本地模式 | 远端模式 |
|------|:---:|:---:|
| `--output` / `-o` | ● 必填 | ● 单文件可选 |
| `--format` / `-f` | ○ | ● |
| `--on-collision` | ● | ● |
| `--lang` / `-l` | ● | ● |
| `--pages` | ● | ● |
| `--formula` / `--no-formula` | ● | ● |
| `--table` / `--no-table` | ● | ● |
| `--timeout` | ○ | ● |
| `--method` / `-m` | ● | ○ |
| `--backend` / `-b` | ● | ○ |
| `--vlm-url` / `-u` | ● | ○ |
| `--image-analysis` | ● | ○ |
| `--model` | ○ | ● |
| `--ocr` / `--no-ocr` | ○ | ● |
| `--fast` | ○ | ● |
| `--async` | ○ | ● |
| `--wait` | ○ | ● |
| `--api-key` | ○ | ● |
| `--concurrency` | ● | ● |

- ● 有效 / ○ 忽略

## 示例

### 本地模式

```bash
# 单文件
mineru-kit parse doc.pdf -o out/

# 扫描件 OCR
mineru-kit parse scanned.pdf -o out/ -m ocr

# 英文文档，指定后端
mineru-kit parse report.pdf -o out/ -b pipeline -l en

# 目录批量
mineru-kit parse ./invoices/ -o out/

# 递归目录
mineru-kit parse ./archive/ -r -o out/

# 从文件列表批量
mineru-kit parse --list files.txt -o out/

# 混合输入
mineru-kit parse doc.pdf ./invoices/ --list extra.txt -o out/
```

### 远端模式

```bash
# 单文件同步 (stdout)
mineru-kit parse doc.pdf --server https://api.mineru.net

# 多格式输出
mineru-kit parse doc.pdf --server ... -o out/ -f md,docx

# 异步提交
mineru-kit parse huge.pdf --server ... --async
# → task-id: abc-123-def

# 批量文件 (默认异步)
mineru-kit parse *.pdf --server ... -o results/
# → Submitted 47 tasks. batch-id: xyz-456

# 批量同步等待
mineru-kit parse *.pdf --server ... -o results/ --wait

# 快速模式 (无需认证)
mineru-kit parse doc.pdf --server ... --fast

# 扫描件 OCR
mineru-kit parse scanned.pdf --server ... --ocr

# 文件列表批量
mineru-kit parse --list files.txt --server ... -o out/
```

### stdin 管道

```bash
# 管道传文件内容
curl https://example.com/doc.pdf | mineru-kit parse --stdin --server ... -o out/
cat doc.pdf | mineru-kit parse --stdin --stdin-name report.pdf -o out/

# 管道传文件路径列表
find . -name '*.pdf' | mineru-kit parse --stdin-list -o out/
```

### 输出冲突处理

```bash
# 报错退出 (默认)
mineru-kit parse a/report.pdf b/report.pdf -o out/
# Error: output filename collision detected:
#   report.md
#     ← a/report.pdf
#     ← b/report.pdf
#   Use --on-collision rename or --on-collision path.

# 自动加前缀消歧
mineru-kit parse a/report.pdf b/report.pdf -o out/ --on-collision rename
# → out/a_report.md + out/b_report.md

# 镜像目录结构
mineru-kit parse a/report.pdf b/report.pdf -o out/ --on-collision path
# → out/a/report.md + out/b/report.md
```

