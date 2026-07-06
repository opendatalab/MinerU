# ADR-0012: Doclib Block Locator

状态: Accepted
日期: 2026-06-15
相关文档: 0011-doclib-doc-short-id.md, ../middle-json/agent-gaps.md, ../middle-json/structured-content-schema.md, ../workflows.md

## 背景

Agent 读取文档后，需要稳定地引用文档中的具体内容块。引用必须同时满足:

- 对 Agent 和开发者可读。
- 可从引用回查到 doc、tier、page 和 block。
- 不泄露本地绝对路径或 filename。
- 不依赖 render 输出是否隐藏某些 block。
- 能在 Markdown marker、Structured Content 和 citation record 中复用。

此前部分文档把稳定引用描述成额外 ID。本 ADR 将 P0 的主引用模型收敛为局部 locator 与全局 block reference。

## 决策

P0 定义两级定位:

1. 局部 locator:

   ```text
   page:{page_no}/block:{block_no}
   ```

2. 全局 block reference:

   ```text
   doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
   ```

示例:

```text
page:1/block:3
doc:ab12cd3/tier:medium/page:1/block:3
```

字段规则:

| 字段 | 说明 |
|------|------|
| `short_id` | `docs.short_id`，见 [ADR-0011](0011-doclib-doc-short-id.md)。 |
| `tier` | 实际解析档位: `flash` / `medium` / `high`。 |
| `page_no` | 1-based 页号。 |
| `block_no` | 1-based block 号。 |

## Block 编号来源

`block_no` 来自 Middle JSON 中 block 自身的稳定编号，或 canonical Middle JSON page 内的稳定 block 序列。

核心规则:

- `block_no` 不是 render 阶段生成的。
- `block_no` 不是“第几个可见输出块”。
- 即使某个 block 在某种输出格式中不显示，也仍然有编号。
- header / footer / page_number 参与编号。
- locator 指向原始结构中的 block，而不是 Markdown / Structured Content 中第几个可见 item。

因此，不同 render profile 不能因为隐藏 block 而重编号。

## Render Profile

`block_ref` 不包含 render profile:

```text
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
```

原因:

- 编号在原始 / canonical Middle JSON 中已经存在或可稳定生成。
- 编号不依赖 Markdown、Structured Content、HTML 等 render 输出。
- 输出格式是否隐藏 header/footer/page_number 不影响 block_no。

如果未来 schema major version 改变导致 locator 规则不兼容，应通过 schema migration、schema version 或后续补充字段处理，而不是在 P0 block_ref 中加入 profile。

## Span / Cell / List Item

P0 不定义 span、cell 或 list-item locator。

原因:

- P0 以 block 为最小公开引用单元。
- span/cell 不单独暴露为公开 citation 目标。
- OCR span 粒度、表格 cell 合并、Office list 嵌套路径等差异会显著增加复杂度。

后续如需引用 list item 或其他嵌套 child，可在 P1+ 增加:

```text
page:{page_no}/block:{block_no}/child:{child_path}
```

但这些不是 P0 契约。

## 与 `short_id` 的关系

`block_ref` 的跨文档唯一性依赖 `docs.short_id UNIQUE`。

`short_id` 的职责是让 doc reference 短、稳定、可读；block locator 的职责是定位到该 doc 的具体内容块。二者分层:

- [ADR-0011](0011-doclib-doc-short-id.md) 定义 `docs.short_id` 如何生成、持久化和保持稳定。
- 本 ADR 定义如何使用 `short_id`、`tier`、`page_no`、`block_no` 组成 block reference。

完整 `sha256` 仍应在需要严格校验的结构化响应顶层返回。`block_ref` 不替代 `sha256`。

## 输出形态

### Structured Content

Structured Content item 可包含:

```json
{
  "locator": "page:1/block:3",
  "block_ref": "doc:ab12cd3/tier:medium/page:1/block:3",
  "page": 1,
  "block": 3,
  "type": "paragraph"
}
```

### Markdown Locator Marker

Markdown 可选 locator marker 可包含:

```html
<!-- mineru:block {"ref":"doc:ab12cd3/tier:medium/page:1/block:3","locator":"page:1/block:3","type":"paragraph"} -->
```

规则:

- 默认 Markdown 可以不输出 block marker。
- Agent / SDK 可显式开启。
- marker 中不得包含本地绝对路径或 filename。

### Citation Record

Agent citation record 可包含:

```json
{
  "block_ref": "doc:ab12cd3/tier:medium/page:4/block:12",
  "locator": "page:4/block:12",
  "page": 4,
  "block": 12,
  "bbox": [10.0, 20.0, 300.0, 160.0],
  "bbox_known": true,
  "type": "table",
  "text": "visible excerpt...",
  "source_sha256": "...",
  "source_short_id": "ab12cd3"
}
```

## 回查

`doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}` 回查流程:

1. 使用 `short_id` 精确查找 `docs.short_id`。
2. 使用 `tier` 选择对应解析结果。
3. 使用 `page_no` 定位 page。
4. 使用 `block_no` 定位该页 block。

P0 可以不支持短于 `short_id` 的 prefix lookup。调用方传入不完整 doc 前缀时，可以返回 `doc_ref_not_found` 或后续定义的 `ambiguous_doc_ref`。

## 替代方案

### 方案 A: 使用 0-based page/block

拒绝。0-based 与内部 `page_idx` 一致，但对 Agent 输出、CLI、日志和人类调试不直观。P0 public locator 使用 1-based。

### 方案 B: 只给可见输出 block 编号

拒绝。这样 locator 会依赖 render profile。隐藏 header/footer/page_number 或切换输出格式时，block 编号可能变化。

### 方案 C: 在 block_ref 中加入 render profile

拒绝。P0 block 编号不由 render 产生，因此不需要 profile。加入 profile 会让引用更长，也会误导调用方认为 locator 依赖输出格式。

### 方案 D: P0 暴露 span/cell/list-item locator

拒绝。P0 不把 span/cell/list item 作为公开引用目标。细粒度定位后续再设计。

### 方案 E: 使用额外 hash ID 作为主引用

拒绝作为 P0 主方案。P0 主引用使用可读的 `block_ref`，避免同时维护两套含义接近的引用 ID。

## 影响

- Middle JSON normalization 必须确保每页 block 编号稳定。
- header/footer/page_number 等默认隐藏块仍参与编号。
- render 层不得因隐藏 block 而重编号。
- Structured Content 和 Markdown marker 应复用同一 locator helper。
- `block_ref` helper 应依赖 `docs.short_id`、实体 tier、1-based page_no 和 1-based block_no。

## 后续动作

1. 定义 helper:

   ```python
   locator_for_block(page_no: int, block_no: int) -> str
   block_ref(short_id: str, tier: Tier, page_no: int, block_no: int) -> str
   ```

2. 确认 Middle JSON 中 block 稳定编号字段来源。
3. 在 Structured Content 输出中增加 `locator` / `block_ref`。
4. 为 Markdown 增加可选 block marker。
5. 增加回查测试:
   - 1-based locator 正确。
   - hidden block 参与编号。
   - 同一 doc/tier/page/block 生成稳定 block_ref。
   - 不同 doc 的相同 page/block 因 `short_id` 不同而不冲突。
