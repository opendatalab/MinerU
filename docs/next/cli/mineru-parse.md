# mineru parse

状态: Draft
读者: Agent skill 作者、CLI 使用者、核心开发者
范围: `mineru parse` 的默认行为、参数、隐私、tier、输出、等待和缓存
非目标: 批量目录解析；解析 backend 专家参数
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru parse` 是 `mineru` 的主动文档解析入口。它面向“用户或 Agent 决定读取某个文档”的场景，因此默认应追求可阅读质量，而不是只做低成本索引。

设计原则：

- 隐私优先：不显式 `--remote` 就不上传文档。
- 质量优先：未指定 tier 时使用默认选择策略，不会解析为 `flash`。
- Agent-native：默认适合 STDOUT、有限上下文和渐进式阅读。
- 先入库后输出：解析结果先写入本地文档库，再输出。
- 去重缓存：相同 SHA256 和实体 tier 可复用 done 缓存；`--force` 跳过 done 缓存，但可复用 active parse，不作废旧缓存。

## 2. Usage

```bash
mineru parse <file> [flags]
```

当前设计以单个本地文件为主。多文件、目录、文件列表和 stdin 混合输入属于 `mineru-kit parse` 的职责。

## 3. 核心参数

### 执行位置

| Flag | 类型 | 说明 |
|------|------|------|
| `--remote` | bool | 显式启用远端解析；远端地址和 API Key 来自 config 或环境变量 |

### 解析配置

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--tier` | `flash` / `standard` / `pro` | 不传 | 解析 tier；省略时使用默认选择策略；语义见 [解析 Tier](../tiers.md) |
| `-p, --pages` | range | `1~10` | 分页文档的页码范围；`all` 表示全部页 |
| `--offset` | integer | `0` | 非分页长文档的继续读取 offset |
| `--language` | string | 自动 | 文档语言提示，逐步弱化为高级选项 |
| `--force` | bool | false | 跳过 done 缓存；复用 active parse 或为未覆盖页创建新 parse；不删除或作废旧缓存 |

### 输出配置

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `-o, --output` | path | `-` | 输出路径；`-` 或省略表示 STDOUT |
| `-f, --format` | markdown / text / middle-json / content-list / structured-content / html | markdown | 输出格式 |
| `--no-marker` | bool | false | 不输出结构 marker |

### 同步控制

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--wait` | seconds | 60 | 同步等待解析完成 |
| `--no-wait` | bool | false | 不等待；未命中缓存时返回任务状态 |

## 4. Tier 行为

`mineru parse` 未指定 `--tier` 时使用默认选择策略。

默认选择策略通过当前目标 parse-server 的能力发现，选择最高可用的非 `flash` tier。如果只发现 `standard`，则使用 `standard`；如果发现 `pro`，则使用 `pro`。如果 doclib 通过多个 parse-server 发现 `standard` 和 `pro`，也选择 `pro`。如果找不到 `standard` 或 `pro`，返回可解释错误。

`flash` 只有在用户显式指定 `--tier flash` 时才作为最终解析结果返回。

## 5. 隐私决策链

默认执行路径：

1. 不带 `--remote`：只使用本地能力。
2. 带 `--remote`：允许上传文档到 config 指定远端或默认 `mineru.net/api`。
3. 本地能力不足：返回错误和修复建议，不静默上传。
4. 用户显式选择 `--tier flash`：允许返回 `flash` 结果。
5. 未指定 tier：使用默认选择策略，不可降级到 `flash`。

当本地能力不足、默认选择无法解析、远端未显式允许或 parse-server 不支持请求 tier 时，错误码见 [错误码体系](../errors.md)。

## 6. 渐进式阅读

默认 STDOUT 输出应适合 Agent context。长文档不一次性输出全文，而是输出有限范围，并通过 marker 指示如何继续。

分页文档默认读取范围固定为 `1~10`。这个默认值用于让 Agent 以线性、可预测的方式渐进阅读文档，避免默认读取尾页后在后续续读中重复返回相同页面。用户可以通过 `--pages all` 读取全文，或通过显式页码范围读取任意页段。

PDF 使用物理页码：

```text
<!-- next pages available. Use: mineru parse report.pdf --pages 11~20 -->
```

非分页文档可以使用虚拟页或字符 offset：

```text
<!-- truncated at paragraph boundary (char ~30000 of ~150000). Next: mineru parse long.docx --offset 30000 -->
```

marker 是 Agent 的控制协议，不应依赖自然语言猜测。

非分页长文档正式支持 `--offset`。`--offset` 表示从指定字符或逻辑片段位置继续读取；具体 offset 单位和 marker 语法后续在 Agent marker 设计中定稿，但 CLI 参数本身进入正式设计。

## 7. STDOUT 与文件输出

默认输出到 STDOUT，便于 Agent 直接消费。指定 `--output` 时写入文件。

结构化或可机器消费格式应尽量保持稳定：

| 格式 | 用途 |
|------|------|
| markdown | 默认阅读输出 |
| text | 纯文本阅读 |
| middle-json | 完整 Middle JSON，中间结构和高级调试 |
| content-list | 扁平内容列表 |
| structured-content | 面向 Agent 和新客户端的结构化内容 JSON |
| html | 人类浏览或调试 |

`json`、`content-list-v2` 不是 NEXT 版正式格式名，不进入公开格式集合。命名决策见 [ADR-0001](../decisions/0001-json-output-formats.md)。

结构化输出或 marker 中如果暴露 tier，应暴露实际使用的实体 tier。

## 8. 等待与缓存

`mineru parse` 先查本地文档库缓存。命中缓存时可直接返回。

未命中时：

- `--wait N`：等待最多 N 秒。
- `--no-wait`：立即返回任务状态。
- 超过等待时间：返回可继续查询的状态，而不是丢失任务。

缓存实体应落到解析后的实体 tier，例如 `standard` 或 `pro`。

## 未决问题

Agent marker 语法集中维护在 [开放问题清单](../open-questions.md)。
