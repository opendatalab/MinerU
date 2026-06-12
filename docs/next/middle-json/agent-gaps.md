# Agent-native Gap

状态: Draft
读者: Agent 能力开发者、SDK 开发者、backend 开发者
范围: Agent 使用 Middle JSON 时需要的稳定引用、chunk id、定位和隐私能力
底稿: `../../../NEXT-JSON.md`

## 目标

Agent 读取文档时，不只是需要 markdown 文本，还需要可以回溯的证据结构。Middle JSON 必须支持:

- 稳定定位到文档中的块。
- 从回答引用回到 page / bbox / source。
- 跨进程、跨机器复现同一引用。
- 在 schema 迁移后尽量保持引用可恢复。

## 当前缺口

| 能力 | 当前状态 | 影响 |
|------|----------|------|
| chunk id | 未实现 | Agent 无法稳定引用解析块。 |
| locator | 未定义 | 无统一 page/block/span 地址格式。 |
| source hash | doclib 有 sha256，但 Middle JSON envelope 未携带 | 跨文件引用不稳定。 |
| bbox known flag | 未定义 | Office/HTML 的 `EMPTY_BBOX` 会被误解为真实坐标。 |
| schema version | 未实现 | 历史缓存无法安全迁移。 |
| parent/child path | 未定义 | 嵌套 list/table/image 引用不稳定。 |
| span address | 未定义 | 细粒度引用只能靠文本匹配。 |

## Locator

定义一个不物化或可选物化的 locator:

```text
page:{page_idx}/block:{block_index}
page:{page_idx}/block:{block_index}/span:{span_index}
page:{page_idx}/block:{block_index}/child:{child_path}
```

字段含义:

| 字段 | 说明 |
|------|------|
| `page_idx` | 原文档 0-based 页号。 |
| `block_index` | normalization 后的页内稳定 block index。 |
| `span_index` | normalization 后的页内稳定 span index。 |
| `child_path` | 嵌套 child 的路径，如 `0.2.1`。 |

设计要求:

- locator 在同一文档、同一 schema major version 内稳定。
- locator 不要求跨不同解析 tier 完全一致，但应能通过 chunk id 区分。
- locator 不直接包含文件路径或文件名。

## Chunk ID

chunk id 是函数输出，不要求物化进每个 block。推荐函数:

```text
chunk_id = hash(schema_major, file_sha256, tier, page_idx, locator)
```

输入:

| 输入 | 说明 |
|------|------|
| `schema_major` | schema 主版本，如 `1`。 |
| `file_sha256` | 原文件 SHA-256。 |
| `tier` | 产生该结果的 tier。 |
| `page_idx` | 页号。 |
| `locator` | block/span locator。 |

原因:

- `file_sha256` 保证同名不同文件不冲突。
- `tier` 区分 flash/standard/pro 的不同质量结果。
- `schema_major` 允许重大 schema 变化后重新生成。
- 不包含 filename，减少隐私泄漏。

## Citation Record

Agent 对外返回引用时，建议使用 citation record，而不是直接暴露内部 block:

```json
{
  "chunk_id": "ck_...",
  "locator": "page:3/block:12",
  "page_idx": 3,
  "bbox": [10.0, 20.0, 300.0, 160.0],
  "bbox_known": true,
  "type": "table",
  "text": "visible excerpt...",
  "source_sha256": "..."
}
```

字段:

| 字段 | 必带 | 说明 |
|------|:--:|------|
| `chunk_id` | 是 | 稳定引用 ID。 |
| `locator` | 是 | 可解释路径。 |
| `page_idx` | 是 | 页号。 |
| `bbox` | 否 | 可定位时返回。 |
| `bbox_known` | 是 | 区分真实 bbox 与 unknown。 |
| `type` | 是 | block/content 类型。 |
| `text` | 否 | 用于展示的短摘录。 |
| `source_sha256` | 是 | 原文件 hash。 |

## Unknown BBox

当前 Office/HTML 常用 `EMPTY_BBOX=(0,0,0,0)` 表示未知。Agent 场景必须避免误解。

目标规则:

- public citation 中必须有 `bbox_known`。
- `EMPTY_BBOX` 默认解释为 unknown，除非 validator 能证明它是合法真实 bbox。
- 对 unknown bbox，UI 可以显示页级引用，而不是框选区域。

## Block 粒度

Agent chunk 默认以 block 为最小稳定单元:

| Block type | Agent 默认 chunk |
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

## 隐私边界

Agent citation 不应默认暴露:

- 本地绝对路径。
- 原始 filename。
- Office anchor，如果 anchor 中含用户内容且未脱敏。
- 内联 image base64。

允许暴露:

- `source_sha256`。
- page_idx。
- bbox。
- block/content type。
- 短摘录。

`_meta.file.filename` 是否进入 public envelope 需要单独决策。默认建议可选，且 SDK/API 对外返回时可脱敏或省略。

## 工作项

P0:

1. 定义 `locator_for_block(page, block)`。
2. 定义 `locator_for_span(page, block, span)`。
3. 定义 `chunk_id(meta, locator)`。
4. 在 doclib 持久化时保存 `source_sha256` 与 schema version。
5. 在 SDK 中提供 citation 构造 helper。

P1:

1. 为 nested block 定义 child path。
2. 为 span 分配稳定 index。
3. 为 unknown bbox 输出 `bbox_known=false`。
4. 为 render markdown marker 增加可选 locator marker。

验收:

- 同一个文件重复解析产生相同 chunk id。
- 不同文件相同内容片段 chunk id 不冲突。
- Office unknown bbox 不会被 UI 当成真实框。
- Agent citation 可以从 chunk id 回查到 page/block。
