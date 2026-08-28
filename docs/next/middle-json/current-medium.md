# 当前 Middle JSON 事实标准

状态: Accepted

读者: backend、SDK、renderer 与 Doclib 开发者

范围: Middle JSON schema 3.0、ModelJson 及公开 Block/InlineSpan 契约

## 标准来源

事实标准由 `mineru/types.py` 中的严格 Pydantic v2 模型定义：

- `ModelJson`、`MiddleJson`、`PageInfo`；
- `Block` / `PageBlock` discriminated union；
- `InlineSpan` discriminated union；
- 各具体 `*Block` 与 `*Span` 模型。

模型统一使用 `extra="forbid"`。`doc_analyze()` 与 `aio_doc_analyze()` 固定返回
`tuple[MiddleJson, ModelJson]`，`ParseResult.to_dict()` 输出
`schema_version="3.0"`。

Schema 3.0 是严格切换：缺失版本或版本为 1.0/2.0 的 API 结果、缓存和本地
payload 不自动迁移，调用方必须从源文件重新解析。

## ModelJson

| 字段 | 类型 | 要求 |
|------|------|------|
| `pages` | `list[list[dict]]` | Analyze 产生的 raw blocks |
| `page_index_map` | `list[int]` | 空列表表示整本；非空时与 pages 等长、唯一且严格递增 |
| `file_suffix` | `pdf/doc/docx/ppt/pptx/xls/xlsx/rtf/csv/epub/html/ofd/odt/ods/odp` | 必填 |
| `effort` | `flash/medium/high/xhigh` | 必填 |
| `parse_mode` | `txt/ocr` | 必填 |
| `mineru_version` | 非空字符串 | 必填 |

文本型 raw block 必须直接携带 InlineSpan 列表。校验失败会报告
`pages[页号][块号]`。PDF 的扁平 `list/index` layout block 在对象化前也可以携带
Span；形成递归容器后，其 `content` 改为子 block 列表。

## MiddleJson 与 PageInfo

| 字段 | 类型 | 要求 |
|------|------|------|
| `pages` | `list[PageInfo]` | `page_idx` 唯一且严格递增 |
| `is_full_document` | `bool` | 必填，不提供默认值 |
| `file_suffix` | 与 ModelJson 相同 | 必填 |
| `effort` | `flash/medium/high/xhigh` | 必填 |
| `parse_mode` | `txt/ocr` | 必填 |
| `mineru_version` | 非空字符串 | 必填 |

`PageInfo` 只包含 `page_idx` 与 `blocks`。顶层 block 必须具有唯一且严格递增的
`index`。PDF 与 OFD 属于固定版式输入，其顶层 block 还必须具有有效的 `[0, 1]`
归一化 bbox；其它格式可以省略 bbox。

## InlineSpan

自然语言不再使用 `<eq>/<text>/<hyperlink>` 字符串协议，而使用以下联合：

```json
[
  {"type": "text", "content": "你好呀，OFD Reader&Writer！", "styles": ["bold"]},
  {"type": "equation_inline", "content": "x<y"},
  {"type": "code_inline", "content": "print(x)"},
  {
    "type": "hyperlink",
    "url": "https://example.com",
    "content": [{"type": "text", "content": "链接"}]
  }
]
```

### TextSpan

- `content` 至少一个字符；纯空白和换行合法；
- `styles` 只允许 `bold/italic/underline/emphasis/strikethrough/superscript/subscript`；
- 样式自动去重，并按上述固定顺序规范化；
- `superscript` 与 `subscript` 互斥；
- 相邻同样式 TextSpan 在统一规范化边界合并。

### EquationInlineSpan 与 CodeInlineSpan

- 公式保存不含外层定界符的 LaTeX，禁止空白值；
- 行内代码禁止空字符串；
- 两者不携带字体、bbox、score、图片或块级字段。

### HyperlinkSpan

- `url` 必须通过共享安全策略，危险协议、本地路径、畸形 URL 和控制字符被拒绝；
- `content` 至少有一个 Text/Equation/Code 子 Span；
- 禁止嵌套 HyperlinkSpan；
- 自动链接只扫描 TextSpan，不扫描公式、代码或已有链接。

`&`、`<`、`>`、实体外观、Unicode 与完整 `<eq>`/`<script>` 字面量在
TextSpan 中保持原文。renderer 在目标格式边界执行转义，不反向解析这些字符串。

## 使用 InlineSpan 的 block

以下叶子 block 的 `content` 为 `list[InlineSpan]`：

- `text/ref_text/doc_title/paragraph_title`；
- `header/footer/page_number/aside_text/page_footnote`；
- list/index 的文本或标题叶子；
- `image/table/chart/code` 的 caption 与 footnote；
- `algorithm_body`。

`continues_prev` 只属于顶层 `text/ref_text/list/table`。标题仍使用全局层级：
`doc_title.level=1`，`paragraph_title.level=2..6`。

## 保持专用字符串的 block

| block | 字符串语义 |
|------|------------|
| `equation` | 行间 LaTeX |
| `code_body` | 普通代码正文 |
| `table_body` | 安全 HTML 或空间投影文本 |
| `image_body` | OCR/描述、Mermaid 等视觉专用内容 |
| `chart_body` | Chart HTML/文本 |

表格和 Chart HTML 中的公式继续使用受限安全 DOM 节点处理，不进入 InlineSpan
流水线。图片 sidecar、bbox、score 和字体信息也不属于 InlineSpan。

## 容器约束

公开 discriminator 共 29 个，其中新增 `algorithm_body`。视觉父块必须且只能有
一个 body：

| 父块 | body | annotation |
|------|------|------------|
| `image` | `image_body` | `image_caption/image_footnote` |
| `table` | `table_body` | `table_caption/table_footnote` |
| `chart` | `chart_body` | `chart_caption/chart_footnote` |
| `code(sub_type=code)` | `code_body` | `code_caption/code_footnote` |
| `code(sub_type=algorithm)` | `algorithm_body` | `code_caption/code_footnote` |

普通代码必须有非空 `guess_lang`；算法禁止 `guess_lang`。父块与 body 同时携带
index/bbox 时必须一致。

`ListBlock.content` 递归允许 `TextBlock | RefTextBlock | ListBlock`；
`IndexBlock.content` 递归允许 `TextBlock | DocTitleBlock | ParagraphTitleBlock |
IndexBlock`。

## 后处理与 renderer

`backend/postprocess/inline.py` 只提供结构操作：规范化、可见文本提取、相邻合并、
段落连接、裁剪与文字映射。不存在字符串标签 parser 或对应正则。

Markdown、HTML、DOCX 与 Structured Content renderer 按 Span discriminator 直接
分派。HTML versioned wire 使用明确 DOM metadata 往返恢复类型、样式、公式和
链接，不从渲染后的标签字符串猜测语义。

## 缓存与兼容

- `ParseResult.from_dict()` 仅接受 `schema_version="3.0"`；
- 1.0/2.0 与缺失版本返回“重新解析源文件”的明确错误；
- Doclib 将旧 batch/cache 视为 stale 并重新调度；
- `legacy_schema_adapter` 已删除；
- 不提供 `str | list[InlineSpan]` 联合，也不恢复旧版带 bbox/图片职责的 Line/Span。
