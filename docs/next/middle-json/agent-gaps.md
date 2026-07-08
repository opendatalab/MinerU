# Agent-native Gap

状态: Draft
读者: Agent 能力开发者、SDK 开发者、backend 开发者
范围: Agent 使用 Middle JSON 时需要的稳定引用、定位和隐私能力
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 目标

Agent 读取文档时，不只是需要 markdown 文本，还需要可以回溯的证据结构。Middle JSON 必须支持:

- 稳定定位到文档中的块。
- 从回答引用回到 page / bbox / source。
- 跨进程、跨机器复现同一引用。
- 在 schema 迁移后尽量保持引用可恢复。

## 当前缺口

| 能力 | 当前状态 | 影响 |
|------|----------|------|
| block ref | 已决策，未实现 | Agent 还不能用稳定可读 ID 引用解析块。 |
| locator | 已决策，未实现 | 已有统一 page/block 地址格式，是 P0 主引用模型。 |
| source hash | doclib 有 sha256，但 Middle JSON envelope 未携带 | 跨文件引用不稳定。 |
| bbox known flag | 代码已有 `bbox_known()` helper；citation 输出仍未定稿 | Office/HTML 的 `EMPTY_BBOX` 不应被误解为真实坐标。 |
| schema version | 当前 `ParseResult.to_dict()` 已写 `schema_version` | 历史缓存仍缺完整 migration。 |
| child/list-item locator | 暂缓 | P0 不把嵌套 child 或 list item 作为公开引用目标。 |
| span/cell locator | 不进入公开契约 | span/cell 不单独暴露为公开 citation 目标。 |

## P0 Block Reference / Locator

P0 的局部 locator 与全局 block reference 由 [ADR-0012](../decisions/0012-doclib-block-locator.md) 定义，并使用
[ADR-0011](../decisions/0011-doclib-doc-short-id.md) 定义的 `docs.short_id`。

局部 locator:

```text
page:{page_no}/block:{block_no}
```

全局 block reference:

```text
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
```

示例:

```text
doc:ab12cd3/tier:medium/page:1/block:3
```

规则:

- `short_id` 来自 `docs.short_id`，默认 7 位 SHA256 前缀，冲突时递增长度，并保持持久稳定。
- `page_no` 使用 1-based 页号。
- `block_no` 使用 1-based 页内 block 号，来自 Middle JSON / canonical Middle JSON 的稳定 block 编号。
- `block_no` 不是 render 阶段生成的，也不是“第几个可见输出块”。
- header / footer / page_number 等默认隐藏块仍参与编号。
- `block_ref` 不包含本地绝对路径或 filename。
- 完整 `sha256` 仍应在需要严格校验的结构化响应顶层返回。

## Locator

P0 locator 只定义到 block:

```text
page:{page_no}/block:{block_no}
```

字段含义:

| 字段 | 说明 |
|------|------|
| `page_no` | 原文档 1-based 页号。 |
| `block_no` | Middle JSON / canonical Middle JSON 的页内稳定 block 编号，1-based。 |

设计要求:

- locator 在同一文档、同一 schema major version 内稳定。
- locator 不要求跨不同解析 tier 完全一致；跨文档和跨 tier 引用使用 `block_ref` 区分。
- locator 不直接包含文件路径或文件名。
- locator 不依赖 render profile；输出格式隐藏 block 时不能重编号。
- P0 不定义 span、cell 或 list-item locator。

## Citation Record

Agent 对外返回引用时，建议使用 citation record，而不是直接暴露内部 block:

```json
{
  "block_ref": "doc:ab12cd3/tier:medium/page:4/block:12",
  "locator": "page:4/block:12",
  "page": 4,
  "bbox": [10.0, 20.0, 300.0, 160.0],
  "bbox_known": true,
  "type": "table",
  "text": "visible excerpt...",
  "source_sha256": "...",
  "source_short_id": "ab12cd3"
}
```

字段:

| 字段 | 必带 | 说明 |
|------|:--:|------|
| `block_ref` | 是 | 稳定可读引用 ID。 |
| `locator` | 是 | 可解释路径。 |
| `page` | 是 | 1-based 页号。 |
| `bbox` | 否 | 可定位时返回。 |
| `bbox_known` | 是 | 区分真实 bbox 与 unknown。 |
| `type` | 是 | block/content 类型。 |
| `text` | 否 | 用于展示的短摘录。 |
| `source_sha256` | 是 | 原文件 hash。 |
| `source_short_id` | 是 | doclib 持久化短 ID。 |

## Unknown BBox

当前 Office/HTML 常用 `EMPTY_BBOX=(0,0,0,0)` 表示未知。Agent 场景必须避免误解。

目标规则:

- public citation 中必须有 `bbox_known`。
- `EMPTY_BBOX` 默认解释为 unknown，除非 validator 能证明它是合法真实 bbox。
- 对 unknown bbox，UI 可以显示页级引用，而不是框选区域。

## Block 粒度

Agent citation 默认以 block 为最小稳定单元:

| Block type | Agent 默认 citation unit |
|------------|------------------|
| `text` | 单 block。 |
| `title` | 单 block，并可作为后续 section context。 |
| `list` | list 容器；必要时可展开 list item。 |
| `table` | table 容器，保留 body/caption/footnote。 |
| `image` | image 容器，保留 caption/footnote 和 image path。 |
| `code` | code 容器。 |
| `header/footer/page_number` | 默认不作为 answer citation，除非用户明确查询。 |

Span 级引用作为增强能力，用于:

- OCR 词级/行级定位。
- 表格单元格定位。
- 超链接定位。
- 行内公式定位。

但 span/cell 不作为 P0 公开引用目标。后续如果需要 child 或 list item locator，应另行设计。

## 隐私边界

Agent citation 不应默认暴露:

- 本地绝对路径。
- 原始 filename。
- Office anchor，如果 anchor 中含用户内容且未脱敏。
- 内联 image base64。

允许暴露:

- `source_sha256`。
- `source_short_id`。
- page。
- bbox。
- block/content type。
- 短摘录。

`_meta.file.filename` 是否进入 public envelope 需要单独决策。默认建议可选，且 SDK/API 对外返回时可脱敏或省略。

## 工作项

P0:

1. 按 [ADR-0011](../decisions/0011-doclib-doc-short-id.md) 在 `docs` 表增加并持久化 `short_id`。
2. 按 [ADR-0012](../decisions/0012-doclib-block-locator.md) 定义 `locator_for_block(page_no, block_no)`。
3. 按 [ADR-0012](../decisions/0012-doclib-block-locator.md) 定义 `block_ref(short_id, tier, page_no, block_no)`。
4. 在 doclib 持久化时保存 `source_sha256`、`short_id` 与 schema version。
5. 在 SDK 中提供 citation 构造 helper。

P1:

1. 为 unknown bbox 输出 `bbox_known=false`。
2. 为 render markdown marker 增加可选 locator marker。
3. 如未来需要 child/list item locator，另行定义公开契约。

验收:

- 同一个文件重复解析产生相同 block_ref。
- 不同文件相同内容片段 block_ref 不冲突。
- Office unknown bbox 不会被 UI 当成真实框。
- Agent citation 可以从 block_ref 回查到 page/block。
