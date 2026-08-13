# 当前 Middle JSON 事实标准

状态: Draft

读者: backend 开发者、SDK 开发者、输出开发者

范围: `doc_analyze()` 第一返回值及 `mineru/types.py` 中的公开 Middle JSON 对象

## 标准来源

当前事实标准由以下 Pydantic v2 模型定义：

- `MiddleJson`
- `PageInfo`
- `Block` discriminated union
- 各具体 `*Block` 模型

所有模型使用 `extra="forbid"` 和严格校验。反序列化入口是
`model_validate()` / `model_validate_json()`，序列化入口是 `to_dict()` /
`to_json()`。不再提供旧 `Block(...)` 通用构造器或 dict-like 兼容接口。

本轮未迁移的 `ParseResult`、renderer、LLM 标题修正和 Structured Content
不属于本文档描述的对象链路。

## `MiddleJson`

| 字段 | 类型 | 要求 |
|------|------|------|
| `pages` | `list[PageInfo]` | 必填，`page_idx` 唯一且严格递增 |
| `file_suffix` | `pdf/docx/pptx/xlsx` | 必填 |
| `effort` | `flash/low/medium/high/xhigh` | 必填 |
| `parse_mode` | `txt/ocr` | 必填 |
| `mineru_version` | 非空字符串 | 必填 |

当 `file_suffix="pdf"` 时，每个页面顶层 block 必须具有有效 bbox。Office
顶层 block 可以没有 bbox。

## `PageInfo`

| 字段 | 类型 | 要求 |
|------|------|------|
| `page_idx` | 非负整数 | 必填，使用实际页号映射 |
| `blocks` | `list[Block]` | 必填，默认为空列表 |

顶层 block 必须具有非负 `index`。同一页内的 index 必须唯一、严格递增，
但允许缺号或从非零值开始。数组位置不能替代 index。

## Block 公共字段

所有具体 block 只共享以下定位字段：

| 字段 | 类型 | 要求 |
|------|------|------|
| `type` | discriminator literal | 必填 |
| `index` | 非负整数或 null | 顶层必填，嵌套项可省略 |
| `bbox` | 4 个 float 的 tuple 或 null | PDF 顶层必填，Office/嵌套项可省略 |

bbox 必须是有限的 `[0, 1]` 归一化坐标，并满足 `x1 > x0`、`y1 > y0`。
每个具体 block 都显式声明必填 `content`，公共基类不再杂糅类型专属字段。

公开 discriminator 共 29 个：

```text
aside_text
chart, chart_body, chart_caption, chart_footnote
code, code_body, code_caption, code_footnote
doc_title, paragraph_title
footer, formula_number, header
image, image_body, image_caption, image_footnote
index, interline_equation, list
page_footnote, page_number, ref_text
table, table_body, table_caption, table_footnote
text
```

仅 raw 阶段允许的 `algorithm/caption/equation/footnote/title/phonetic` 不属于
公开 union。其它旧 `BlockType` 常量也不会被严格反序列化接受。

## 主要具体模型

| 模型 | `content` | 专属字段 |
|------|-----------|----------|
| `TextBlock` | `str` | `continues_prev`；不允许 `anchor` |
| `RefTextBlock` | `str` | 不允许 `continues_prev` |
| `DocTitleBlock` | `str` | `anchor` |
| `ParagraphTitleBlock` | `str` | `anchor`、正整数 `level` |
| `TableBlock` | 有序视觉子块列表 | `continues_prev` |
| `TableBodyBlock` | `str` | `cell_merge`、图片字段 |
| `CodeBlock` | 有序视觉子块列表 | `sub_type`、`guess_lang` |
| `ImageBlock/ChartBlock` | 有序视觉子块列表 | 开放字符串 `sub_type` |
| `ImageBodyBlock/ChartBodyBlock` | `str` 或 null | 图片字段 |
| 其它叶子 block | `str` | 仅各自声明的字段 |

`CodeBlock(sub_type="code")` 必须有非空 `guess_lang`；
`sub_type="algorithm"` 禁止 `guess_lang`。`continues_prev` 只允许出现在页面
顶层 `text/list/table`，任何嵌套项即使显式写成 null 也会被拒绝。

`merge_prev`、`is_numbered_style` 和 `section_number` 均不属于公开模型。
Office 自动标题编号已在对象化前写入 `ParagraphTitleBlock.content`。

## 递归容器

`ListBlock.content` 是保持原顺序的递归列表：

```text
TextBlock | RefTextBlock | ListBlock
```

`sub_type` 存在时，只约束当前层直接文本子项；嵌套 ListBlock 独立校验。

`IndexBlock.content` 同样递归：

```text
TextBlock | IndexBlock
```

Office 目录规范化会递归移除普通 TextBlock 的 `anchor`。递归结构始终序列化到
`content`，不产生通用 `blocks` 字段。

## 视觉容器

视觉父块的 `content` 使用上下文限定的有序子 block：

| 父块 | 允许的子块 |
|------|------------|
| `image` | `image_body/image_caption/image_footnote` |
| `table` | `table_body/table_caption/table_footnote` |
| `chart` | `chart_body/chart_caption/chart_footnote` |
| `code` | `code_body/code_caption/code_footnote` |

每个父块必须且只能有一个对应 body；caption 和 footnote 可以有多个。父块有
index 时 body 必须使用相同 index；父子同时有 bbox 时必须相等。

## 图片载荷与导出

以下 block 可以直接携带 `image_base64` 和 `image_path`：

- `image_body`
- `table_body`
- `chart_body`
- `interline_equation`

普通 `to_dict()` / `to_json()` 不执行文件 I/O；可通过
`exclude_block_fields={"image_base64"}` 递归排除图片字段。完整序列化保留
base64 并支持严格 round-trip。

`MiddleJson.export()` 会在深拷贝上完成图片外置并原子提交 JSON 与 sidecar：

```text
images/page_{page_idx}_{type}_{index}.{ext}
images/page_{page_idx}_{type}_{index}_{ordinal}.{ext}
```

第二种名称用于同一 HTML block 中的多张内嵌图片，ordinal 从 1 开始。导出
JSON 不包含 `image_base64` 或 `data:image/...`，原始 MiddleJson 对象保持不变。

## Analyze 私有对象

公开 Block/PageInfo 不引用 `Line` 或 `Span`。Analyze 的文字回填与组行使用
backend 私有 slotted draft，并只把最终 content 和临时行框写回 raw dict。
旧 `Line/Span/ContentItem` dataclass 已从 `types.py` 删除；旧 parser、renderer、
doclib 和 content-list 不属于当前保证可用的 Analyze/MiddleJson 边界。
