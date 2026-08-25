# 当前 Middle JSON 事实标准

状态: Draft

读者: backend 开发者、SDK 开发者、输出开发者

范围: `doc_analyze()` 返回值及 `mineru/types.py` 中的公开 Model/Middle JSON 对象

## 标准来源

当前事实标准由以下 Pydantic v2 模型定义：

- `ModelJson`
- `MiddleJson`
- `PageInfo`
- `Block` discriminated union
- 各具体 `*Block` 模型

所有模型使用 `extra="forbid"` 和严格校验。反序列化入口是
`model_validate()` / `model_validate_json()`，序列化入口是 `to_dict()` /
`to_json()`。不再提供旧 `Block(...)` 通用构造器或 dict-like 兼容接口。

本轮未迁移的 `ParseResult`、renderer 和 Structured Content 不属于本文档描述的对象链路。

`doc_analyze()` 与 `aio_doc_analyze()` 固定返回
`tuple[MiddleJson, ModelJson]`，不再把第二返回值暴露为裸 model list。

## `ModelJson`

| 字段 | 类型 | 要求 |
|------|------|------|
| `pages` | `list[list[dict]]` | 必填，保存 Analyze 产生的 raw blocks |
| `page_index_map` | `list[int]` | 必填；空列表表示整本默认顺序，非空时与 pages 等长且唯一递增 |
| `file_suffix` | `pdf/doc/docx/ppt/pptx/xls/xlsx` | 必填 |
| `effort` | `flash/medium/high/xhigh` | 必填，使用公开分析档位 |
| `parse_mode` | `txt/ocr` | 必填，使用分析后实际值 |
| `mineru_version` | 非空字符串 | 必填 |

`ModelJson.is_full_document` 是由 `page_index_map` 是否为空计算的运行时属性，
不进入 JSON；`resolved_page_indices` 同样不序列化，并把空映射展开为顺序页号。
任意非空映射均表示显式抽页，不根据映射值猜测是否覆盖整本。

文档级转换只接受完整 ModelJson：`model_json_to_pages(model_json)` 负责严格页面
对象化，`model_json_to_middle_json(model_json, llm_aided_config=...)` 负责构造
MiddleJson 并执行适用的 PDF 后处理。不再提供裸 model list 加独立页映射的转换入口。

## `MiddleJson`

| 字段 | 类型 | 要求 |
|------|------|------|
| `pages` | `list[PageInfo]` | 必填，`page_idx` 唯一且严格递增 |
| `is_full_document` | `bool` | 必填，保存整本或抽页语义，不提供默认值 |
| `file_suffix` | `pdf/doc/docx/ppt/pptx/xls/xlsx` | 必填 |
| `effort` | `flash/medium/high/xhigh` | 必填 |
| `parse_mode` | `txt/ocr` | 必填 |
| `mineru_version` | 非空字符串 | 必填 |

当 `file_suffix="pdf"` 时，每个页面顶层 block 必须具有有效 bbox。Office
顶层 block 可以没有 bbox。PDF 标题分级直接读取 `is_full_document`；缺少该字段的
旧 Middle JSON 无法通过严格解析，需要从 ModelJson 重新生成。

## `PageInfo`

| 字段 | 类型 | 要求 |
|------|------|------|
| `page_idx` | 非负整数 | 必填，使用实际页号映射 |
| `blocks` | `list[PageBlock]` | 必填，默认为空列表 |

顶层 block 必须具有非负 `index`。同一页内的 index 必须唯一、严格递增，
但允许缺号或从非零值开始。数组位置不能替代 index。

`PageBlock` 只允许 16 种页面根类型：`text/ref_text/doc_title/paragraph_title`、
`header/footer/page_number/page_footnote/aside_text`、`equation/list/index` 和
`image/table/chart/code`。视觉 `body/caption/footnote` 只能出现在对应父块内部；
通用 `Block` 联合仍可用于解析全部公开 discriminator。

## Block 公共字段

所有具体 block 只共享以下定位字段：

| 字段 | 类型 | 要求 |
|------|------|------|
| `type` | discriminator literal | 必填 |
| `index` | 非负整数或 null | 顶层必填，嵌套项可省略 |
| `bbox` | 4 个 float 的 tuple 或 null | PDF 顶层必填，Office/嵌套项可省略 |

bbox 必须是有限的 `[0, 1]` 归一化坐标，并满足 `x1 > x0`、`y1 > y0`。
每个具体 block 都显式声明必填 `content`，公共基类不再杂糅类型专属字段。

公开 discriminator 共 28 个：

```text
aside_text
chart, chart_body, chart_caption, chart_footnote
code, code_body, code_caption, code_footnote
doc_title, paragraph_title
footer, header
image, image_body, image_caption, image_footnote
equation, index, list
page_footnote, page_number, ref_text
table, table_body, table_caption, table_footnote
text
```

仅 raw 阶段允许的 `algorithm/caption/footnote/formula_number/phonetic` 不属于
公开 union。其它旧 `BlockType` 常量也不会被严格反序列化接受。

块级行间公式唯一使用 `equation`。Legacy Span 链中的
`ContentType.INTERLINE_EQUATION` 暂未迁移，不属于公开 Block discriminator。

## 主要具体模型

| 模型 | `content` | 专属字段 |
|------|-----------|----------|
| `TextBlock` | `str` | `continues_prev`；不允许 `anchor` |
| `RefTextBlock` | `str` | `continues_prev`；不允许 `anchor` |
| `DocTitleBlock` | `str` | `anchor`、必填 `level=1` |
| `ParagraphTitleBlock` | `str` | `anchor`、必填 `level>=2` |
| `PageAuxTextBlock` | `str` | 共享页眉、页脚、页码、边栏和页脚注释结构 |
| `TableBlock` | 有序视觉子块列表 | `continues_prev`、`cell_merge` |
| `TableBodyBlock` | `str` | 图片字段 |
| `CodeBlock` | 有序视觉子块列表 | `sub_type`、`guess_lang` |
| `ImageBlock/ChartBlock` | 有序视觉子块列表 | 开放字符串 `sub_type` |
| `EquationBlock/ImageBodyBlock/TableBodyBlock/ChartBodyBlock` | `str` | 共享图片载荷内容结构 |
| `*AnnotationBlock` | `str` | 每个视觉家族共享 caption/footnote 结构 |
| 其它叶子 block | `str` | 仅各自声明的字段 |

`TextBlock` 与 `RefTextBlock` 共同继承 `ContinuableTextBlockBase`。PDF 分析阶段
两者都必须具有合法的临时 `lines`，该字段只参与续段计算，不进入严格 Middle JSON。
`ref_text` 复用正文的终止符和几何规则，但不要求续段首行或首列顶起始边界，
也不因续段以数字或大写字符开头而拒绝合并。

`CodeBlock(sub_type="code")` 必须有非空 `guess_lang`；
`sub_type="algorithm"` 禁止 `guess_lang`。`continues_prev` 只允许出现在页面
顶层 `text/ref_text/list/table`，任何嵌套项即使显式写成 null 也会被拒绝。

`merge_prev`、`is_numbered_style` 和 `section_number` 均不属于公开模型。
标题使用全局文档层级：文档标题固定为一级，所有段落标题从二级开始。
Office 自动标题编号按 `level - 1` 的段落深度计算，并在对象化前写入
`ParagraphTitleBlock.content`。

## 递归容器

`ListBlock.content` 是保持原顺序的递归列表：

```text
TextBlock | RefTextBlock | ListBlock
```

`sub_type` 存在时，只约束当前层直接文本子项；嵌套 ListBlock 独立校验。

`IndexBlock.content` 同样递归：

```text
TextBlock | DocTitleBlock | ParagraphTitleBlock | IndexBlock
```

Office 目录规范化会按 document-wide anchor 把匹配到真实目标标题的目录叶子
转换为对应 TitleBlock，并继承目标 level；未匹配 anchor 的叶子降级为普通
TextBlock。递归结构始终序列化到 `content`，不产生通用 `blocks` 字段。

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
`cell_merge` 属于后一个 `table` 根块，描述其与前一个 table 的跨页视觉列续接；
`table_body` 只保存 HTML/文本和图片载荷。

## 图片载荷与导出

以下 block 可以直接携带 `image_base64` 和 `image_path`：

- `image_body`
- `table_body`
- `chart_body`
- `equation`

这四类图片载荷 block 的 `content` 都必须是字符串。Analyze 会将 image/chart
缺失的文本表示或 raw `null` 规范化为空字符串；严格 Middle JSON 不接受
`content: null`。

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

当前 schema 采用直接切换策略：缺失标题 `level`、`paragraph_title.level=1`、
`table_body.cell_merge` 和视觉载荷 `content: null` 均不会在严格反序列化时兼容。

## Analyze 私有对象

公开 Block/PageInfo 不引用 `Line` 或 `Span`。Analyze 的文字回填与组行使用
backend 私有 slotted draft，并只把最终 content 和临时行框写回 raw dict。
旧 `Line/Span/ContentItem` dataclass 已从 `types.py` 删除；旧 parser、renderer、
doclib 和 content-list 不属于当前保证可用的 Analyze/MiddleJson 边界。
