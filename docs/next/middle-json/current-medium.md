# 当前事实标准

状态: Draft
读者: backend 开发者、SDK 开发者、输出开发者
范围: 当前 `mineru/types.py` 中 document model 的字段契约和注意事项
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 标准来源

当前事实标准来自 `mineru/types.py`:

- `PageInfo`
- `Block`
- `Line`
- `Span`
- `ContentItem`
- `BlockType`
- `ContentType`
- `ContentTypeV2`

这些类型已具备 `to_dict()` / `from_dict()` 能力，能够递归处理嵌套 dataclass list，并把 JSON 中的 list bbox / page_size 转回 tuple。

当前 `ParseResult.to_dict()` 输出顶层 `schema_version + pages`，并在所有 page backend 一致时保留顶层 `_backend` 作为临时兼容 metadata。`ParseResult.from_dict()` / `from_json()` 已可恢复 `pages` envelope，但不接受旧 `pdf_info` envelope。

## `PageInfo`

字段:

| 字段 | 类型 | 当前要求 | 说明 |
|------|------|----------|------|
| `page_idx` | int | 必填 | 0-based 页号。 |
| `page_size` | tuple[int, int] 或 null | 建议必填 | PDF/VLM/Hybrid 通常有；Office/HTML 可能为空。 |
| `preproc_blocks` | list[Block] | 默认空 | 预处理 block。Pipeline/VLM/Hybrid 中较常见。 |
| `para_blocks` | list[Block] | 默认空 | 主阅读流。render 主要消费它。 |
| `discarded_blocks` | list[Block] | 默认空 | 页眉、页脚、页码等边缘内容。 |
| `_backend` | internal | 临时 | 当前 render facade 仍依赖它 dispatch。长期应移入 `_meta.backend`。 |

目标要求:

- `page_idx` 必须稳定对应原文档页号。
- `para_blocks` 必须按 reading order 排序。
- `discarded_blocks` 不应混入主阅读流，但需要保留以支持搜索和引用。
- `_backend` 不进入长期 public schema。

## `Block`

字段:

| 字段 | 类型 | 当前要求 | 说明 |
|------|------|----------|------|
| `index` | int | 必填 | 页内排序标识。当前仍有 backend 差异。 |
| `type` | str | 必填 | 来自 `BlockType`。 |
| `bbox` | tuple[float, float, float, float] | 必填 | 当前类型必填；未知时常用 `EMPTY_BBOX`。 |
| `lines` | list[Line] | 默认空 | 叶子内容。 |
| `blocks` | list[Block] | 默认空 | 嵌套结构，如 list、image/table/code 容器。 |
| `angle` | int 或 null | 可选 | 文本角度。 |
| `score` | float 或 null | 可选 | 置信度。 |
| `level` | int 或 null | 可选 | 标题层级。 |
| `sub_type` | str | 可选 | 代码等细分类型。 |
| `guess_lang` | str | 可选 | 语言猜测。 |
| `merge_prev` | bool | 可选 | 与前一 block 合并。 |
| `section_number` | str | 可选 | Office 标题编号等。 |
| `html` | str | 可选 | 表格或富内容。 |
| `text` | str | 可选 | 兼容字段。 |
| `latex` | str | 可选 | 公式内容。 |
| `anchor` | str | Office | 目录/标题 anchor。 |

目标要求:

- 一个 block 可以同时有 `lines` 和 `blocks`，但公开消费方应优先按 block type 理解结构。
- `index` 必须在 normalization 阶段变成 Agent 可依赖的 reading order。
- `bbox=EMPTY_BBOX` 表示未知，不等于真实页面左上角零面积框。

## `Line`

字段:

| 字段 | 类型 | 当前要求 | 说明 |
|------|------|----------|------|
| `bbox` | tuple[float, float, float, float] | 必填 | 当前类型必填；未知时使用 `EMPTY_BBOX`。 |
| `spans` | list[Span] | 默认空 | 行内内容。 |

内部字段:

- `_is_list_start`
- `_is_list_end`
- `_code_type`
- `_code_guess_lang`

这些字段不应进入 public schema。

## `Span`

字段:

| 字段 | 类型 | 当前要求 | 说明 |
|------|------|----------|------|
| `type` | str | 必填 | 来自 `ContentType` 或兼容字符串。 |
| `bbox` | tuple[float, float, float, float] | 必填 | 当前类型必填；未知时使用 `EMPTY_BBOX`。 |
| `content` | str | 默认空 | 文本、公式或图表描述。 |
| `score` | float | 默认 0 | OCR 置信度等。 |
| `image_path` | str | 默认空 | 图片产物路径。 |
| `image_base64` | str | 默认空 | 内联图像。 |
| `html` | str | 默认空 | 表格 HTML。 |
| `latex` | str | 默认空 | 公式 LaTeX。 |

内部字段:

- `_cross_page`
- `_np_img`
- `_url`
- `_style`
- `_children`
- `_extra`

Office 解析已经使用 `_url`、`_style`、`_children` 表达超链接和文本样式。是否公开这些字段需要单独决策。

## Type 集合

`BlockType` 已覆盖:

- 基础正文: `text`、`title`、`list`、`index`
- 视觉容器: `image`、`table`、`chart`、`code`
- 视觉子块: `image_body`、`table_body`、`chart_body`、`code_body` 等
- 特殊正文: `algorithm`、`ref_text`、`phonetic`、`abstract`
- 页边内容: `header`、`footer`、`page_number`、`aside_text`、`page_footnote`
- Pipeline 类型: `doc_title`、`paragraph_title`、`vertical_text`、`header_image`、`footer_image`、`formula_number`

`ContentType` 已覆盖:

- `text`
- `inline_equation`
- `interline_equation`
- `image`
- `table`
- `chart`
- `equation`
- `hyperlink`

当前 `Span` 没有公开 `index` 字段；span 级 locator 不属于 P0 public contract。

## Validator 现状

生产代码当前不提供 Middle JSON validator。原轻量页面树校验逻辑已退化为单测中的 test-local helper，用于约束当前 dataclass 结构的预期行为。

已覆盖:

- page 的 `page_idx` 与可选 `page_size`。
- block 的 `index`、`type`、`bbox`、title level。
- line 的 `bbox`、`spans`。
- span 的 `type`、`bbox` 和基础内容缺失 warning。
- `EMPTY_BBOX` 在单测 validator helper 中按 unknown bbox 处理，并作为 warning。

当前未覆盖完整 envelope 校验、JSON Schema 文件和跨文件 migration。

## 当前标准的缺口

当前 dataclass 已经是事实标准，但仍缺少:

- 顶层 `_meta`。
- 完整 canonical `_meta`。
- `file.sha256`。
- 统一 locator。
- 旧 `pdf_info` 到 `pages` 的 migration。
- 对 `EMPTY_BBOX` 和真实 bbox 的正式语义区分。
- 对内部字段公开策略的决策。
