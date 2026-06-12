# ADR-0001: JSON 输出格式命名

状态: Accepted
日期: 2026-06-09
相关文档: ../cli/mineru-parse.md, ../api/parse-jobs.md, ../sdk/parse-result.md, ../middle-json.md

## 背景

MinerU 当前同时存在三类 JSON 结构:

1. Middle JSON: 保真的中间结构，按 page / block / line / span 组织。
2. Content List v1: 旧的扁平内容列表。
3. Content List v2: 新的按页组织、语义更强的结构化内容列表。

如果 CLI、API 或 SDK 只暴露 `json` 这个格式名，调用方无法知道返回的是哪一种 JSON。随着 Agent、SDK 和第三方客户端开始依赖这些输出，`json` 会成为不稳定的契约名称。

同时，`content-list-v2` 也不适合作为长期公开名称。`v2` 是历史演进状态，不应出现在面向用户的长期格式名中；版本应由 schema version 表达。

## 决策

公开输出格式使用语义名称，不使用 `json` 表示具体产物。

正式格式名:

| 语义 | CLI 格式名 | API / SDK 格式名 | 说明 |
|------|------------|------------------|------|
| Markdown | `markdown` | `markdown` | 默认阅读输出。 |
| 纯文本 | `text` | `text` | 可选阅读输出。 |
| Middle JSON | `middle-json` | `middle_json` | 完整中间结构，用于缓存、调试、高级 SDK 和重新 render。 |
| Content List v1 | `content-list` | `content_list` | 扁平内容列表。 |
| Structured Content | `structured-content` | `structured_content` | 面向 Agent 和新客户端的结构化内容 JSON，替代公开名称 `content-list-v2`。 |
| HTML | `html` | `html` | 从 Middle JSON 或 Markdown 派生的 HTML 输出。 |

`Structured Content` 是 Content List v2 这类结构的公开产品名。当前代码中的 `content_list_v2` 函数名属于实现现状，不进入 NEXT 版公开 CLI/API/SDK 格式契约。

`json` 不作为正式格式名。`content_list_v2` / `content-list-v2` 不进入公开格式集合。NEXT 版尚未实现，没有历史兼容负担；除当前代码内部命名外，新实现和新文档只使用正式格式名。

## 输出边界

### Middle JSON

Middle JSON 是 doclib 唯一持久化的解析内容产物。它应保留足够信息，以支持:

- read-time render Markdown。
- read-time render Content List v1。
- read-time render Structured Content。
- 按页增量解析和批次合并。
- 后续 schema migration。

Middle JSON 适合核心开发者、SDK 高级用户和需要重新渲染的系统使用，不作为 Agent 默认结构化消费格式。

### Content List v1

Content List v1 是扁平数组，适合简单消费，但不承载新的 Agent-native 语义承诺。

### Structured Content

Structured Content 是面向 Agent、新客户端和结构化下游的推荐 JSON。它应强调:

- 按页组织。
- 标题、段落、列表、公式、图片、表格、代码等语义类型。
- 可选 bbox / locator / chunk id。
- 稳定 schema version。

Structured Content 的版本演进通过 schema version 表达，而不是把公开格式名改成 `structured-content-v2`、`structured-content-v3`。

## 替代方案

### 方案 A: 保留 `json`

不采用。`json` 是编码方式，不是产物语义。继续用它会让 Middle JSON、Content List v1 和 Structured Content 三者在 CLI/API/SDK 中混淆。

### 方案 B: 继续公开 `content-list-v2`

不采用。`v2` 暴露了历史实现细节，不适合长期公开契约。一旦出现下一版结构，用户会被迫理解 `v2/v3` 的命名迁移，而不是理解稳定语义。

### 方案 C: 命名为 `agent-json`

不采用。Structured Content 适合 Agent，但不只服务 Agent。把格式绑定到 Agent 会限制 SDK、Web UI 和第三方系统的使用场景。

### 方案 D: 命名为 `semantic-content`

不采用。它强调语义，但比 `structured-content` 更抽象。`structured-content` 更容易让开发者理解这是可机器消费的结构化文档内容。

## 影响

- CLI `--format` 应支持 `middle-json`、`content-list`、`structured-content`。
- API `output_formats` 应支持 `middle_json`、`content_list`、`structured_content`。
- 当前代码里的 `content_list_v2()` / `render_content_list_v2()` 需要迁移到 Structured Content 语义；公开接口只使用正式格式名。
- doclib 的 `parsed/` 目录仍只持久化 Middle JSON；Content List v1 和 Structured Content 都是读取时派生结果。

## 后续动作

1. 更新 CLI/API/SDK 文档中的格式名示例。
2. 为 Structured Content 定义 schema version 字段和 validator。
3. 将现有 `render_content_list_v2()` 的文档表述逐步迁移为 Structured Content render。
4. 在实现计划中把 `content_list_v2` backend-specific converter 收敛任务改写为 Structured Content schema 收敛任务。
