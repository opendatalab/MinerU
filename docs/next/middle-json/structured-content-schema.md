# Structured Content Schema

状态: Draft
读者: Structured Content 实现者、API/SDK/CLI 开发者、Agent 能力开发者
范围: NEXT 版 `structured_content` / `structured-content` 的目标 schema 草案
非目标: 定义 Middle JSON；定义 Markdown/HTML 输出；覆盖所有历史 `content_list_v2` 细节

## 1. 定位

Structured Content 是从 Middle JSON render 得到的结构化内容 JSON。它面向 Agent、新客户端和下游程序消费。

它不是 doclib 的持久化源数据。doclib 的 `parsed/` 目录只保存 Middle JSON；Structured Content 在读取时由 Middle JSON 派生。

设计目标:

1. 比 Middle JSON 更适合直接消费。
2. 比 Markdown 更结构化。
3. 保留页、item、span、bbox 和 locator，支持 Agent 引用。
4. 以当前 `content_list_v2` 为起点，但不照搬其内部字段泄漏和 backend 差异。

## 2. 顶层结构

目标形态:

```json
{
  "schema": "mineru.structured_content",
  "schema_version": "1.0",
  "source": {
    "sha256": "ab3f...",
    "filename": "report.pdf",
    "mime_type": "application/pdf"
  },
  "parse": {
    "tier": "pro",
    "backend": "hybrid",
    "version": "3.1.14"
  },
  "pages": []
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `schema` | string | 是 | 固定为 `mineru.structured_content`。 |
| `schema_version` | string | 是 | Structured Content schema 版本。 |
| `source` | object | 否 | 源文件信息。不得包含本地绝对路径。 |
| `parse` | object | 否 | 实际解析信息。 |
| `pages` | array | 是 | 页面列表。 |

### 2.1 `source`

```json
{
  "sha256": "ab3f...",
  "filename": "report.pdf",
  "mime_type": "application/pdf"
}
```

约束:

- `sha256` 可用于缓存和引用。
- `filename` 可选；如果存在，只保存显示名，不保存本地绝对路径。
- `source` 不承载隐私敏感路径。

### 2.2 `parse`

```json
{
  "tier": "standard",
  "backend": "pipeline",
  "version": "3.1.14"
}
```

约束:

- `tier` 只记录实际使用的实体 tier: `flash` / `standard` / `pro`。
- 不记录 `requested_tier` / `resolved_tier`。
- `backend` 是实现信息，普通客户端不应依赖它做产品逻辑。

## 3. Page

目标形态:

```json
{
  "page_idx": 0,
  "page_number": 1,
  "page_size": {
    "width": 612,
    "height": 792
  },
  "items": []
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `page_idx` | integer | 是 | 0-based 页号。 |
| `page_number` | integer | 是 | 1-based 页号，便于人类显示。 |
| `page_size` | object/null | 是 | 页面尺寸。未知时为 `null`。 |
| `items` | array | 是 | 当前页内容 item 列表。 |

说明:

- 当前 `content_list_v2` 的外层数组下标应显式提升为 `page_idx`。
- Office/HTML 如果没有真实 page size，应输出 `page_size:null`，不要伪造尺寸。

## 4. Item 通用结构

目标形态:

```json
{
  "id": "p0_i3",
  "type": "paragraph",
  "locator": {
    "page_idx": 0,
    "item_idx": 3
  },
  "bbox": {
    "normalized": [100, 120, 900, 180]
  },
  "content": {}
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `id` | string | 是 | 当前 Structured Content 内稳定 item id。 |
| `type` | string | 是 | item 类型。 |
| `locator` | object | 是 | 可回溯位置。 |
| `bbox` | object/null | 是 | item 坐标。未知时为 `null`。 |
| `content` | object | 是 | 类型相关内容。 |

### 4.1 `id`

建议格式:

```text
p{page_idx}_i{item_idx}
```

例如:

```json
"id": "p2_i14"
```

约束:

- `id` 在同一 Structured Content 文档内唯一。
- `id` 不承诺跨不同解析版本稳定。
- 需要跨解析稳定引用时，使用后续 `chunk_id` 或 locator 规则。

### 4.2 `locator`

第一版最小形态:

```json
{
  "page_idx": 0,
  "item_idx": 3
}
```

可扩展形态:

```json
{
  "page_idx": 0,
  "item_idx": 3,
  "block_index": 12
}
```

约束:

- `locator` 不包含本地文件路径。
- `block_index` 只有在可从 Middle JSON 稳定映射时才写入。
- 后续可以增加 `chunk_id`，但第一版不强制。

### 4.3 `bbox`

目标形态:

```json
{
  "normalized": [100, 120, 900, 180]
}
```

规则:

- `normalized` 使用 0-1000 整数坐标 `[x0, y0, x1, y1]`。
- unknown bbox 使用 `null`。
- 不使用 `[0,0,0,0]` 表示 unknown。
- 如果确实存在真实零面积 bbox，应在 normalization 阶段视为无效并输出 `null`。

## 5. Item Type

第一版支持以下公开 item type:

| Type | 说明 |
|------|------|
| `paragraph` | 普通段落。 |
| `title` | 标题。 |
| `list` | 列表或引用列表。 |
| `index` | 目录。 |
| `equation_interline` | 独立公式。 |
| `image` | 图片。 |
| `table` | 表格。 |
| `chart` | 图表。 |
| `code` | 代码块。 |
| `algorithm` | 算法块。 |
| `page_header` | 页眉。 |
| `page_footer` | 页脚。 |
| `page_number` | 页码。 |
| `page_aside_text` | 旁注。 |
| `page_footnote` | 页脚注。 |

不进入第一版公开 item type:

- `content-list-v2` 不是 type。
- `md` 不是 item type。
- backend-specific block type 不直接泄漏到 item type；需要先归一化。

## 6. Span

目标形态:

```json
{
  "type": "text",
  "text": "hello",
  "marks": ["bold"],
  "url": "https://example.com"
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `type` | string | 是 | span 类型。 |
| `text` | string | 否 | 文本内容。 |
| `latex` | string | 否 | 公式内容。 |
| `marks` | array | 否 | 文本样式。 |
| `url` | string | 否 | 超链接地址。 |

第一版 span type:

| Type | 字段 | 说明 |
|------|------|------|
| `text` | `text` | 普通文本。 |
| `equation_inline` | `latex` | 行内公式。 |
| `phonetic` | `text` | 注音或音标文本。 |
| `link` | `text`, `url`, `marks?` | 超链接。 |

样式 marks:

| Mark | 说明 |
|------|------|
| `bold` | 加粗。 |
| `italic` | 斜体。 |
| `underline` | 下划线。 |
| `strikethrough` | 删除线。 |
| `superscript` | 上标。 |
| `subscript` | 下标。 |
| `emphasis` | 着重号等强调样式。 |

约束:

- 不输出 `_style`、`_url`、`_children`、`_extra`。
- Office hyperlink 应归一化为 `type:"link"`。
- PDF backend 的简化 span 和 Office raw span 都应归一化到同一 span schema。
- span bbox 第一版不强制；如后续需要可扩展。

## 7. Content Shapes

### 7.1 Paragraph

```json
{
  "type": "paragraph",
  "content": {
    "spans": [
      {"type": "text", "text": "A paragraph."}
    ]
  }
}
```

当前 v2 的 `paragraph_content` 统一改为 `spans`。

### 7.2 Title

```json
{
  "type": "title",
  "content": {
    "level": 1,
    "spans": [
      {"type": "text", "text": "Introduction"}
    ]
  }
}
```

规则:

- `level` 为正整数。
- Office `section_number` 应作为普通 text span 写入 `spans`，或后续定义专门字段；第一版按普通 span 处理。

### 7.3 Page Header / Footer / Number / Aside / Footnote

统一形态:

```json
{
  "type": "page_header",
  "content": {
    "spans": []
  }
}
```

当前 v2 的动态 key:

- `page_header_content`
- `page_footer_content`
- `page_number_content`
- `page_aside_text_content`
- `page_footnote_content`

统一改为:

```json
"spans": []
```

### 7.4 Interline Equation

```json
{
  "type": "equation_interline",
  "content": {
    "latex": "E = mc^2",
    "source": {
      "type": "image",
      "path": "images/eq_0.jpg"
    }
  }
}
```

规则:

- `latex` 来自当前 `math_content`。
- `source` 可选；Office 没有公式图片时省略或为 `null`。
- `math_type` 第一版不保留为字段；默认解释为 LaTeX。

### 7.5 Image

```json
{
  "type": "image",
  "content": {
    "source": {
      "type": "image",
      "path": "images/img_0.jpg"
    },
    "description": [
      {"type": "text", "text": "optional image description"}
    ],
    "caption": [],
    "footnote": []
  }
}
```

映射:

| 当前字段 | 目标字段 |
|----------|----------|
| `image_source` | `source` |
| `content` | `description` |
| `image_caption` | `caption` |
| `image_footnote` | `footnote` |

### 7.6 Table

```json
{
  "type": "table",
  "content": {
    "html": "<table>...</table>",
    "source": {
      "type": "image",
      "path": "images/table_0.jpg"
    },
    "caption": [],
    "footnote": [],
    "table_type": "simple_table",
    "table_nest_level": 1
  }
}
```

规则:

- `html` 保留。
- `source` 可选。Office 无 table image 时为 `null` 或省略。
- `table_type` 保留 `simple_table` / `complex_table`。
- `table_nest_level` 保留为 integer。

### 7.7 Chart

```json
{
  "type": "chart",
  "content": {
    "source": {
      "type": "image",
      "path": "images/chart_0.jpg"
    },
    "description": [],
    "caption": [],
    "footnote": []
  }
}
```

映射:

| 当前字段 | 目标字段 |
|----------|----------|
| `image_source` | `source` |
| `content` | `description` |
| `chart_caption` | `caption` |
| `chart_footnote` | `footnote` |

### 7.8 Code

```json
{
  "type": "code",
  "content": {
    "language": "python",
    "spans": [],
    "caption": [],
    "footnote": []
  }
}
```

映射:

| 当前字段 | 目标字段 |
|----------|----------|
| `code_language` | `language` |
| `code_content` | `spans` |
| `code_caption` | `caption` |
| `code_footnote` | `footnote` |

### 7.9 Algorithm

```json
{
  "type": "algorithm",
  "content": {
    "spans": [],
    "caption": [],
    "footnote": []
  }
}
```

映射:

| 当前字段 | 目标字段 |
|----------|----------|
| `algorithm_content` | `spans` |
| `algorithm_caption` | `caption` |
| `algorithm_footnote` | `footnote` |

### 7.10 List

```json
{
  "type": "list",
  "content": {
    "list_type": "text_list",
    "ordered": false,
    "items": [
      {
        "spans": [],
        "level": 0,
        "marker": "-"
      }
    ]
  }
}
```

规则:

- `list_type` 保留 `text_list` / `reference_list`。
- 当前 `attribute:"ordered"` 映射为 `ordered:true`。
- 当前 `attribute:"unordered"` 映射为 `ordered:false`。
- Office `ilevel` 映射为 `level`。
- Office `prefix` 映射为 `marker`。
- 当前 `item_content` 映射为 `spans`。

### 7.11 Index

```json
{
  "type": "index",
  "content": {
    "items": [
      {
        "spans": [],
        "level": 0,
        "target": {
          "anchor": "_Toc..."
        }
      }
    ]
  }
}
```

规则:

- 当前 index 先复用 list item 结构。
- Office `anchor` 映射为 `target.anchor`。
- 页码、目标页、目标 item 如果后续可从 Middle JSON 中稳定恢复，再扩展 `target`。

## 8. Media Source

统一形态:

```json
{
  "type": "image",
  "path": "images/img_0.jpg"
}
```

字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `type` | string | 是 | 第一版只定义 `image`。 |
| `path` | string | 是 | 相对路径或 API file path。 |

约束:

- 不输出本地绝对路径。
- 空 path 不应构造为 `images/`。
- 远端 API 可在 wrapper 层把 media source 解析为 file reference；Structured Content 内部先保持路径语义。

## 9. Normalization Rules

从当前 `content_list_v2` 到 Structured Content 的第一版 normalization:

1. 裸二维数组包装为 envelope。
2. 外层数组下标转成 `page_idx` 和 `page_number`。
3. 每个 item 生成 `id` 和 `locator`。
4. `bbox:[0,0,0,0]` 或缺失 bbox 转成 `bbox:null`。
5. 当前 `paragraph_content` / `title_content` / `*_content` 统一转成 `spans`。
6. 当前 `image_source` 统一转成 `source`。
7. 当前 caption / footnote 字段统一转成 `caption` / `footnote`。
8. Office raw span 归一化为公开 span。
9. `_` 开头字段不进入 Structured Content。
10. 未识别 item type 应进入 validation issue，不静默丢弃。

## 10. Validation

P0 validator 应检查:

- 顶层 `schema == "mineru.structured_content"`。
- `schema_version` 存在。
- `pages` 是 array。
- 每个 page 有 `page_idx`、`page_number`、`page_size`、`items`。
- 每个 item 有 `id`、`type`、`locator`、`bbox`、`content`。
- item id 在文档内唯一。
- item `locator.page_idx` 与所在 page 一致。
- bbox 只能是 `null` 或 0-1000 的四元整数数组。
- span 不包含 `_` 开头字段。
- `json`、`content_list_v2`、`content-list-v2` 不作为公开格式名出现。

## 11. Open Design Points

这版 schema 仍需讨论:

1. `chunk_id` 是否第一版必填。
2. `id` 是否需要跨同一文件、同一 tier、同一版本解析稳定。
3. `source.filename` 是否默认写入。
4. `parse.backend` 是否进入公开 Structured Content，还是只放在 API response metadata。
5. `bbox.normalized` 是否采用 object 包装，还是直接用 `bbox:[...]`。
6. `caption` / `footnote` 是否统一为 span list，还是 item list。
7. `description` 是否使用 span list，还是普通 string。
