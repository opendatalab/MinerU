# mineru parse

状态: Draft
读者: Agent skill 作者、CLI 使用者、核心开发者
范围: `mineru parse` 的默认行为、参数、隐私、tier、输出、等待和缓存
非目标: 批量目录解析；解析 backend 专家参数
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru parse` 是 `mineru` 的主动文档解析入口。它面向“用户或 Agent 决定读取某个文档”的场景，因此默认应追求可阅读质量，而不是只做低成本索引。

与 [mineru read](mineru-read.md) 的边界：

```text
parse(path) = ensure document is parsed, then read default content
read(locator) = read existing parsed content by stable locator
```

`parse` 负责从文件 path 出发，把文档纳入 doclib 并返回第一段可读内容。后续继续阅读、按页读取、按 block 读取或导出 page/block 图像，应转到 `mineru read`。

设计原则：

- 隐私优先：不显式 `--remote` 就不上传文档。
- 质量优先：PDF/image 未指定 tier 时使用默认选择策略，不会解析为 `flash`；Office/HTML 未指定 tier 时归一为 `flash`。
- 文本直读：`.txt`、`.md`、`.markdown`、`.csv`、`.rst` 和 `.tex` 无需解析，应直接读取源文件。
- Agent-native：默认适合 STDOUT、有限上下文和渐进式阅读。
- 先入库后输出：解析结果先写入本地文档库，再输出。
- 去重缓存：相同 SHA256 和实体 tier 可复用 done 缓存；`--force` 跳过 done 缓存，但可复用 active parse，不作废旧缓存。

## 2. Usage

```bash
mineru parse <file> [flags]
```

当前设计以单个本地文件为主。多文件和目录输入属于 `mineru-kit parse` 的职责。

## 3. 核心参数

### 执行位置

| Flag | 类型 | 说明 |
|------|------|------|
| `--remote` | bool | 显式启用远端解析；远端地址来自 config，API Key 优先来自 config，未配置时使用环境变量 |

### 解析配置

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--tier` | `flash` / `medium` / `high` / `xhigh` | 不传 | 解析 tier；省略时由服务端决定；语义见 [解析 Tier](../tiers.md) |
| `-p, --pages` | range | 不传 | 分页文档的页码范围；`all` 表示全部页 |
| `--after` | cursor | 不传 | 从服务端返回的 cursor 继续读取 |
| `--limit` | int | `30000` | STDOUT 内容软字符上限 |
| `--force` | bool | false | 跳过 done 缓存；复用 active parse 或为未覆盖页创建新 parse；不删除或作废旧缓存 |

### 输出配置

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `-o, --output` | path | `-` | 输出路径；`-` 或省略表示 STDOUT |
| `-f, --format` | markdown | markdown | 输出格式。当前仅支持 `markdown`。 |
| `--json` | bool | false | 输出命令级 JSON；顶层包含 `{ "parse": ..., "content": ... }`，失败或超时时可包含 `error` |
| `--no-marker` | bool | false | 不输出结构 marker |

### 同步控制

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--wait` | seconds | 60 | 同步等待解析完成 |
| `--no-wait` | bool | false | 不等待；未命中缓存时返回任务状态 |

## 4. Tier 行为

`mineru parse` 未指定 `--tier` 时使用默认选择策略。

PDF/image 的默认选择策略通过当前目标 parse-server 的能力发现，按 [解析 Tier](../tiers.md) 定义的 `high` -> `xhigh` -> `medium` 顺序选择。如果找不到可用质量 tier，返回可解释错误。Office/HTML 归一为 `flash` 语义，详见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。文本文件不创建 parse 任务或 Middle JSON，显式请求返回 `parse_not_required`。

`flash` 只有在用户显式指定 `--tier flash` 时才作为最终解析结果返回。

## 5. 隐私决策链

默认执行路径：

1. 不带 `--remote`：只使用本地能力。
2. 带 `--remote`：允许上传文档到 config 指定远端或默认 `mineru.net/api`。
3. 本地能力不足：返回错误和修复建议，不静默上传。
4. 用户显式选择 `--tier flash`：允许返回 `flash` 结果。
5. 未指定 tier：PDF/image 使用默认选择策略，不可降级到 `flash`；Office/HTML 归一为 `flash`。
6. 文本文件不上传也不解析，直接返回 `parse_not_required`，提示调用方读取源文件。

当本地能力不足、默认选择无法解析、远端未显式允许或 parse-server 不支持请求 tier 时，错误码见 [错误码体系](../errors.md)。

## 6. 渐进式阅读

默认 STDOUT 输出应适合 Agent context。长文档不一次性输出全文，而是输出有限范围，并通过 marker 指示如何继续。

当前 CLI 不在命令层硬编码默认页码范围；不传 `--pages` 时由 doclib 的内容读取计划决定首次返回范围。用户可以通过 `--pages all` 读取全文，或通过显式页码范围读取任意页段。

分页文档使用物理页码：

```text
<!-- next pages available. Use: mineru parse report.pdf --pages 11~20 -->
```

非分页文档当前也使用 `--after` cursor 继续读取：

```text
<!-- next content available. Use: mineru parse long.docx --after doc:ab12cd3/tier:flash/page:1/block:12/char:520 -->
```

marker 是 Agent 的控制协议，不应依赖自然语言猜测。

当用户后续已经拥有 locator 时，推荐直接使用 `mineru read <locator>`，而不是继续依赖 path + `--pages` / `--after`。

非分页长文档当前正式使用 `--after` cursor 继续读取。`after` 的值来自服务端返回的结构化 `next_request.after`，CLI marker 只负责把这个值渲染为下一条命令。

## 7. STDOUT 与文件输出

默认输出到 STDOUT，便于 Agent 直接消费。指定 `--output` 时写入文件。

`--json` 时，`mineru parse` 的顶层返回命令级 envelope：

```json
{
  "parse": { "... parse summary ..." },
  "content": { "... DocContentResponse ..." } | null,
  "error": { "... ErrorInfo ..." } | null
}
```

规则如下：

- 当解析结果已可读取时，`content` 为 `DocContentResponse`。
- 当使用 `--no-wait` 或等待超时、解析尚未完成时，`content` 为 `null`。
- 当命令需要保留 parse 状态供调用方继续轮询或诊断时，失败或等待超时可以在同一 envelope 顶层包含 `error`。
- `parse.status=parsing` 是 CLI 等待超时时的命令级状态；服务端 `ParseResponse.status` 只使用 `pending` 或 `done`。
- 直接请求错误仍返回 [错误码体系](../errors.md) 定义的结构化错误 JSON。

当前 `mineru parse -f/--format` 只接受 `markdown`。其它结构化或可机器消费产物仍属于设计方向，命名决策见 [ADR-0001](../decisions/0001-json-output-formats.md)。

结构化输出或 marker 中如果暴露 tier，应暴露实际使用的实体 tier。

## 8. 等待与缓存

`mineru parse` 先查本地文档库缓存。命中缓存时可直接返回。

未命中时：

- `--wait N`：等待最多 N 秒。
- `--no-wait`：立即返回任务状态。
- 超过等待时间：返回可继续查询的状态，而不是丢失任务。

缓存实体应落到解析后的实体 tier，例如 `medium`、`high` 或 `xhigh`。

## 未决问题

Agent marker 语法集中维护在 [开放问题清单](../open-questions.md)。
