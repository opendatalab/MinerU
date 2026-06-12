# 当前 Content List v2 结构盘点

状态: Draft
读者: Structured Content schema 设计者、render 开发者、backend 开发者、编程 Agent
范围: 梳理当前代码中 `content_list_v2` 的事实输出结构，作为 NEXT 版 `structured_content` / `structured-content` schema 的起点
非目标: 定义最终 Structured Content schema；重写 render 实现

## 1. 定位

NEXT 版公开格式名已经定为 `structured_content` / `structured-content`。当前代码里的 `content_list_v2` 是它的事实起点，但不是 NEXT 版公开格式名。

本文只回答:

- 当前 `content_list_v2` 从哪里生成。
- 顶层结构是什么。
- 每类 item 当前有哪些字段。
- Pipeline、VLM/Hybrid、Office 之间有哪些差异。
- 哪些现状不应直接固化进 Structured Content schema。

## 2. 生成路径

当前统一入口:

```text
ParseResult.content_list_v2()
  -> render_content_list_v2(pages, img_bucket_path="")
    -> 根据 PageInfo._backend 分发
      -> pipeline: backend/pipeline/pipeline_middle_json_mkcontent.py
      -> vlm/hybrid: backend/vlm/vlm_middle_json_mkcontent.py
      -> office: render/office/output.py
```

当前调用点:

| 调用点 | 当前产物名 |
|--------|------------|
| `ParseResult.save()` | `{prefix}_content_list_v2.json` |
| 旧 CLI dump | `{pdf_file_name}_content_list_v2.json` |
| client-side output regeneration | `{doc_stem}_content_list_v2.json` |
| 当前 parse API server | `output_files.content_list_v2` |

这些是实现现状。NEXT 版公开 API / CLI / SDK 应使用 `structured_content` / `structured-content`。

## 3. 顶层结构

当前 `render_content_list_v2()` 返回:

```text
list[list[dict]]
```

含义:

```json
[
  [
    {"type": "title", "content": {}},
    {"type": "paragraph", "content": {}}
  ],
  [
    {"type": "table", "content": {}}
  ]
]
```

规则:

| 层级 | 含义 |
|------|------|
| 外层 list | 文档页序列。 |
| 内层 list | 当前页的内容 item 序列。 |
| item | 一个 block render 后的结构化内容。 |

当前缺失:

- 没有顶层 envelope。
- 没有 `schema_version`。
- 没有显式 `pages` 字段。
- 每页没有 `page_idx` / `page_size` / `page_number`。
- item 默认没有 `page_idx`；页号只能从外层数组位置推断。
- 没有 `chunk_id` / `locator`。

## 4. Item 通用形态

大多数 item 形态:

```json
{
  "type": "paragraph",
  "content": {},
  "bbox": [0, 0, 1000, 1000]
}
```

通用字段:

| 字段 | 类型 | 当前来源 | 说明 |
|------|------|----------|------|
| `type` | string | `ContentTypeV2` | item 类型。 |
| `content` | object | backend converter | 每类 item 的主体字段不同。 |
| `bbox` | array[int, int, int, int] | PDF backend | 可选。Pipeline/VLM/Hybrid 通常有，Office 通常没有。 |
| `sub_type` | string | 部分视觉块 | 可选。主要来自 image/chart/code 等 block 的 `sub_type`。 |
| `anchor` | string | Office | 可选。主要来自 DOCX 标题、目录或列表项。 |

### 4.1 BBox

当前 PDF backend 的 v2 item bbox 是归一化到 0-1000 的整数框:

```json
"bbox": [x0, y0, x1, y1]
```

现状问题:

- Pipeline/VLM/Hybrid 会输出 item 级 bbox。
- Office 通常不输出 bbox。
- v2 converter 没有统一过滤 `EMPTY_BBOX`；如果 source bbox 是 `(0, 0, 0, 0)`，当前可能输出 `[0, 0, 0, 0]`。
- Pipeline helper 返回 tuple，VLM 直接返回 list；JSON 序列化后都会变成 array。
- bbox 只在 item 级别出现；PDF span 通常没有 bbox，Office span 可能因为 `asdict(span)` 携带 bbox。

Structured Content 需要重新定义 unknown bbox，不应把 `[0, 0, 0, 0]` 误当成真实框。

## 5. Type 集合

当前 `ContentTypeV2` 定义了以下类型:

| 类别 | 值 |
|------|----|
| 代码与算法 | `code`, `algorithm` |
| 公式 | `equation_interline`, `equation_inline` |
| 视觉内容 | `image`, `table`, `chart` |
| 表格细分 | `simple_table`, `complex_table` |
| 列表 | `list`, `text_list`, `reference_list`, `index` |
| 文本结构 | `title`, `paragraph` |
| span | `text`, `phonetic`, `md`, `code_inline` |
| 页边内容 | `page_header`, `page_footer`, `page_number`, `page_aside_text`, `page_footnote` |

当前实现并没有使用全部类型:

- `md` 和 `code_inline` 当前没有在主要 v2 converter 中稳定产出。
- `hyperlink` 可能从 Office raw span 中出现，但它不在 `ContentTypeV2` 中。
- `index` 由 Pipeline 和 Office 输出；VLM converter 当前没有显式 `index` 分支。

## 6. Span 结构

### 6.1 PDF backend span

Pipeline、VLM、Hybrid 的文本类 content 通常由简化 span 组成:

```json
[
  {"type": "text", "content": "hello"},
  {"type": "equation_inline", "content": "x+y"},
  {"type": "phonetic", "content": "..."}
]
```

当前支持:

| span type | 含义 |
|-----------|------|
| `text` | 普通文本。 |
| `equation_inline` | 行内公式。 |
| `phonetic` | VLM phonetic / phonetic block 的文本。 |

PDF backend 会合并相邻同类 text span，并做换行、连字符和 CJK 空格处理。

### 6.2 Office span

Office v2 直接使用 `asdict(span)` 输出 span，因此字段更接近 Middle JSON 的 `Span`:

```json
{
  "type": "text",
  "bbox": [0, 0, 0, 0],
  "content": "hello",
  "score": 0,
  "image_path": "",
  "image_base64": "",
  "html": "",
  "latex": "",
  "_cross_page": false,
  "_np_img": null,
  "_url": "",
  "_style": ["bold"],
  "_children": [],
  "_extra": {}
}
```

Office inline equation 会把 `type` 从 `inline_equation` 改成 `equation_inline`。

Office hyperlink 可能输出:

```json
{
  "type": "hyperlink",
  "content": "link text",
  "_url": "https://example.com",
  "_style": ["underline"],
  "_children": []
}
```

现状问题:

- Office span 会泄漏 `_url`、`_style`、`_children`、`_extra` 等内部字段。
- Office span 的 `bbox` 通常是 unknown。
- Office span 比 PDF span 重得多，和 PDF span 结构不一致。

## 7. Item 类型结构

### 7.1 Paragraph

当前输出:

```json
{
  "type": "paragraph",
  "content": {
    "paragraph_content": []
  },
  "bbox": [0, 0, 1000, 1000]
}
```

来源:

| Backend | block type |
|---------|------------|
| Pipeline | `text`, `abstract` |
| VLM/Hybrid | `text`, `phonetic` |
| Office | `text` |

差异:

- VLM `phonetic` block 会输出 `paragraph` item，但内部 span type 可能是 `phonetic`。
- Pipeline `abstract` 被归入 `paragraph`。
- Office paragraph content 使用 raw Office span。

### 7.2 Title

当前输出:

```json
{
  "type": "title",
  "content": {
    "title_content": [],
    "level": 1
  },
  "bbox": [0, 0, 1000, 100]
}
```

规则:

- 如果 title level 为 0，Pipeline/VLM/Office 都可能降级输出为 `paragraph`。
- Office title 可能有顶层 `anchor`。
- Office title 可能在 `title_content` 前插入自动 `section_number` 文本 span。

### 7.3 Page Header / Footer / Number / Aside / Footnote

当前输出:

```json
{
  "type": "page_header",
  "content": {
    "page_header_content": []
  },
  "bbox": [0, 0, 1000, 80]
}
```

动态 content key:

| item type | content key |
|-----------|-------------|
| `page_header` | `page_header_content` |
| `page_footer` | `page_footer_content` |
| `page_number` | `page_number_content` |
| `page_aside_text` | `page_aside_text_content` |
| `page_footnote` | `page_footnote_content` |

差异:

- Pipeline/VLM 支持 `page_header`、`page_footer`、`page_number`、`page_aside_text`、`page_footnote`。
- Office 支持 `page_header`、`page_footer`、`page_footnote`，没有显式 `page_number` / `page_aside_text` 分支。

### 7.4 Interline Equation

Pipeline/VLM 输出:

```json
{
  "type": "equation_interline",
  "content": {
    "math_content": "E = mc^2",
    "math_type": "latex",
    "image_source": {"path": "images/eq_0.jpg"}
  },
  "bbox": [100, 200, 900, 260]
}
```

Office 输出:

```json
{
  "type": "equation_interline",
  "content": {
    "math_content": "E = mc^2",
    "math_type": "latex"
  }
}
```

差异:

- PDF backend 带 `image_source`。
- Office 不带 `image_source`。

### 7.5 Image

当前输出:

```json
{
  "type": "image",
  "content": {
    "image_source": {"path": "images/img_0.jpg"},
    "content": "optional image description",
    "image_caption": [],
    "image_footnote": []
  },
  "sub_type": "optional",
  "bbox": [100, 100, 900, 600]
}
```

差异:

| Backend | 字段差异 |
|---------|----------|
| Pipeline | `content` 仅在 image content 或 `sub_type` 存在时写入；支持 `image_footnote`；可写 `sub_type`。 |
| VLM/Hybrid | 总是写 `content`，空则为 `""`；支持 `image_footnote`；可写 `sub_type`。 |
| Office | 写 `image_source` 和 `image_caption`；通常不写 `content` / `image_footnote` / `sub_type`。 |

### 7.6 Table

当前输出:

```json
{
  "type": "table",
  "content": {
    "image_source": {"path": "images/table_0.jpg"},
    "table_caption": [],
    "table_footnote": [],
    "html": "<table>...</table>",
    "table_type": "simple_table",
    "table_nest_level": 1
  },
  "bbox": [100, 100, 900, 600]
}
```

`table_type`:

| 值 | 当前判定 |
|----|----------|
| `simple_table` | HTML 中没有 `colspan` / `rowspan`，且嵌套 table 数量不超过 1。 |
| `complex_table` | HTML 中有 `colspan` / `rowspan`，或嵌套 table 数量大于 1。 |

差异:

- Pipeline/VLM 带 `image_source` 和 `table_footnote`。
- Office 不带 `image_source` / `table_footnote`，但会格式化表格 HTML 中的图片路径和 `<eq>` 标签。

### 7.7 Chart

当前输出:

```json
{
  "type": "chart",
  "content": {
    "image_source": {"path": "images/chart_0.jpg"},
    "content": "",
    "chart_caption": [],
    "chart_footnote": []
  },
  "sub_type": "optional",
  "bbox": [100, 100, 900, 600]
}
```

差异:

| Backend | 字段差异 |
|---------|----------|
| Pipeline | `content` 固定为空字符串；支持 `chart_caption` / `chart_footnote`。 |
| VLM/Hybrid | `content` 使用 chart body content；支持 `chart_caption` / `chart_footnote`；可写 `sub_type`。 |
| Office | `content` 是格式化后的 chart HTML / 文本；支持 `chart_caption`；通常不写 `chart_footnote`。 |

### 7.8 Code

Pipeline/VLM 输出:

```json
{
  "type": "code",
  "content": {
    "code_caption": [],
    "code_content": [],
    "code_footnote": [],
    "code_language": "python"
  },
  "bbox": [100, 100, 900, 600]
}
```

差异:

- Pipeline 支持 `code_footnote`。
- VLM/Hybrid 当前不写 `code_footnote`。
- Office 当前没有 `code` 分支。

### 7.9 Algorithm

Pipeline/VLM 输出:

```json
{
  "type": "algorithm",
  "content": {
    "algorithm_caption": [],
    "algorithm_content": [],
    "algorithm_footnote": []
  },
  "bbox": [100, 100, 900, 600]
}
```

差异:

- Pipeline 支持 `algorithm_footnote`。
- VLM/Hybrid 当前不写 `algorithm_footnote`。
- Office 当前没有 `algorithm` 分支。

### 7.10 List

普通列表当前输出:

```json
{
  "type": "list",
  "content": {
    "list_type": "text_list",
    "attribute": "unordered",
    "list_items": [
      {
        "item_type": "text",
        "item_content": []
      }
    ]
  },
  "bbox": [100, 100, 900, 600]
}
```

引用列表当前输出:

```json
{
  "type": "list",
  "content": {
    "list_type": "reference_list",
    "list_items": [
      {
        "item_type": "text",
        "item_content": []
      }
    ]
  }
}
```

差异:

| Backend | 字段差异 |
|---------|----------|
| Pipeline | 普通 list 有 `attribute`，ref_text 会输出 `list_type=reference_list`。 |
| VLM/Hybrid | 普通 list 通常没有 `attribute`；`sub_type=ref_text` 时输出 `reference_list`。 |
| Office | 普通 list 有 `attribute`；每个 item 额外有 `ilevel`、`prefix`，可能有 `anchor`。 |

Office list item:

```json
{
  "item_type": "text",
  "ilevel": 0,
  "prefix": "-",
  "item_content": [],
  "anchor": "_Toc..."
}
```

### 7.11 Index

Pipeline/Office 输出:

```json
{
  "type": "index",
  "content": {
    "list_type": "text_list",
    "list_items": [
      {
        "item_type": "text",
        "item_content": []
      }
    ]
  }
}
```

差异:

- Pipeline index item 和 list item 形态接近。
- Office index item 复用 `_flatten_list_items_v2()`，因此可能带 `ilevel`、`prefix`、`anchor`。
- VLM converter 当前没有显式 `index` 分支。

## 8. Backend 差异汇总

| 维度 | Pipeline | VLM / Hybrid | Office |
|------|----------|--------------|--------|
| 顶层 | 按页二维数组 | 按页二维数组 | 按页二维数组 |
| item bbox | 有，0-1000 | 有，0-1000 | 通常无 |
| span 形态 | 简化 span | 简化 span | raw `Span` asdict |
| `paragraph` | text / abstract | text / phonetic | text |
| `page_number` | 支持 | 支持 | 不支持 |
| `page_aside_text` | 支持 | 支持 | 不支持 |
| `code` / `algorithm` | 支持 | 支持 | 不支持 |
| `index` | 支持 | 无显式分支 | 支持 |
| table footnote | 支持 | 支持 | 不支持 |
| chart footnote | 支持 | 支持 | 不支持 |
| code footnote | 支持 | 不支持 | 不支持 |
| top-level `anchor` | 不常见 | 不常见 | 支持 |
| list item `prefix` / `ilevel` | 不支持 | 不支持 | 支持 |

## 9. 当前结构的直接风险

这些现状不建议直接固化为 Structured Content schema:

1. 顶层是裸二维数组，缺少 version、metadata 和显式 page 对象。
2. 页号依赖外层数组下标，缺少稳定 `page_idx`。
3. item 缺少稳定 `id` / `chunk_id` / `locator`。
4. Office span 泄漏内部字段，且和 PDF span 结构不一致。
5. `bbox=[0,0,0,0]` 可能表示 unknown，而不是真实区域。
6. 同一语义在不同 backend 下字段不一致，例如 table/image/chart footnote、code footnote、list attribute。
7. media path 拼接规则不统一，空 path 可能产生不理想路径。
8. VLM converter 对未显式支持的 block type 没有统一返回 `None` 的规则。

## 10. 作为 Structured Content 起点的建议

后续设计 Structured Content schema 时，可以保留这些方向:

- 以 page 为组织单位。
- 以 item 表达标题、段落、列表、公式、图片、表格、图表、代码、页边内容。
- `content` 内继续按类型保存语义字段。
- 文本内容使用 span list，而不是直接拼成字符串。
- `table_type`、`table_nest_level`、`image_source`、caption / footnote 的语义可以继续保留。

需要重新定义这些部分:

- 顶层 envelope。
- page object。
- item id / locator / chunk id。
- span schema。
- unknown bbox。
- media source。
- backend 差异的 normalization 规则。
