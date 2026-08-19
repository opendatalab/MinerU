# Rendering Contract

状态: Markdown / Content List / DOCX v1
读者: render 开发者、backend 开发者、SDK 开发者
范围: 严格 MiddleJson 到 Markdown、Content List 和 DOCX 的消费契约

## 公共入口

```python
from mineru.render import MarkdownRenderMode, render_markdown

markdown = render_markdown(
    middle_json,
    mode=MarkdownRenderMode.DEFAULT,
    asset_base_url="",
)
```

`render_markdown()` 只接受 `MiddleJson`，不兼容旧 dict、`list[PageInfo]`、
backend 参数或旧 `Line/Span` renderer。渲染过程基于深拷贝，不修改输入对象，
也不负责把图片写到文件系统。

图片优先使用 `image_path`，缺失时使用 `image_base64` data URI。
`asset_base_url` 只给相对 `image_path` 和 HTML 内相对 `img src` 添加前缀。

## 可编辑 DOCX

```python
from pathlib import Path

from mineru.render import RenderMode, render_docx

asset_dir = Path("output/document")
docx_bytes = render_docx(
    middle_json,
    mode=RenderMode.DEFAULT,
    asset_resolver=lambda relative_path: (asset_dir / relative_path).read_bytes(),
)
```

`render_docx()` 只接受严格 `MiddleJson`，返回完整 `.docx` bytes，不写文件、不读取 cwd，
也不访问网络。block 同时携带两种图片来源时优先解码 `image_base64`；仅有
`image_path` 时必须提供 `asset_resolver(relative_path) -> bytes`。必需图片缺失、路径
不安全、格式损坏或 SVG 输入会抛出带 `page_idx/block_index/block_type` 的
`DocxRenderError`；WebP 在内存中转为 PNG。

DOCX 输出固定使用 A4 纵向与 20 mm 页边距。标题是 Word Heading 1–9，anchor 是
document-wide bookmark，Index 标题叶子写成内部链接；正文直接消费共享 inline AST，
保留粗体、斜体、下划线、删除线、emphasis、上下标、外链及行内公式。列表保留 producer
已经内化的 marker，只用缩进表达层级，不重建 `numbering.xml`。

公式通过 `latex2mathml -> mathml2omml -> OMML` 输出为可编辑 Office Math。末端
`\tag{...}` 会从公式主体剥离并通过右对齐 tab 单独写编号；转换失败的块公式依次回退
公式图片和可见 LaTeX，行内公式回退可见 LaTeX，所有回退均记录 page/block 定位。

HTML table 使用 occupancy grid 生成原生 Word table，支持 `rowspan/colspan`、嵌套表格、
单元格富文本、公式、链接和图片。HTML 结构无效时只在存在表格图片时回退；空间投影
文本表格按 v1 契约只输出表格图片。Image body 文本写入图片 alt description；Chart
在图片后继续输出可用的 HTML 数据表。视觉父块仍严格保持 body/caption/footnote 原顺序。

`RenderMode.DEFAULT` 与下文 Markdown DEFAULT 使用同一份续段/续表 planner；
`RenderMode.FULL` 保留全部页面辅助块，不跨源页合并，并在相邻 `PageInfo` 间写硬分页。
`MarkdownRenderMode` 是 `RenderMode` 的既有公共别名。DOCX 是语义化可编辑输出，不承诺
复刻源 PDF/Office 的字体、分栏、断行和页数；CLI、doclib 与 v1 API 不在本契约范围。

## 树形 Markdown Content List

严格 render 层还提供不合并、不丢块的树形内容输出：

```python
from mineru.render import render_content_list

content_list = render_content_list(
    middle_json,
    asset_base_url="",
)
```

`render_content_list()` 只接受 `MiddleJson`，返回可以直接交给 `json.dumps()` 的
文档级 dict。顶层保留 `pages/file_suffix/effort/parse_mode/mineru_version`，每页保留
`page_idx/blocks`，页面和顶层 block 的数量及顺序不变。它不使用下文的
`MarkdownRenderMode` 计划，因此不隐藏页面辅助块，也不合并 `continues_prev` 文本、
列表或跨页表格；`continues_prev` 和 `cell_merge` 等消费提示仍保留在对应 block。

`content` 保存适合结构化消费的文本表示，而不是可直接拼接成整篇文档的完整视觉
Markdown。`list/index` 的递归子树会收敛为一个字符串；`image/table/chart/code` 的唯一
body 会提升为父块 `content`，但图片资源只放在 `image_source`，不在 `content` 重复
输出图片语法或 details。说明文本按源 `index` 排序后分别输出为 `captions: list[str]`
和 `footnotes: list[str]`，视觉数组始终存在，空说明保留为空字符串。

`doc_title/paragraph_title.content` 只包含行内 Markdown，不含 heading 标记或 HTML
anchor；原始 `level` 和可选 `anchor` 作为独立字段保留。`equation.content` 是不带行间
定界符的裸 LaTeX，公式图片同样只通过 `image_source` 表达。所有输出 block 都删除
`index/guess_lang/image_path/image_base64`，`guess_lang` 仍会先参与代码 Markdown 的生成。

```json
{
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "type": "paragraph_title",
          "anchor": "section-a",
          "level": 2,
          "content": "**章节标题**"
        },
        {
          "type": "equation",
          "content": "x^2+y^2=z^2",
          "image_source": "images/equation.png"
        }
      ]
    }
  ],
  "file_suffix": "pdf",
  "effort": "flash",
  "parse_mode": "txt",
  "mineru_version": "3.4.0"
}
```

`image/table/chart/equation` 有图片载荷时，会输出最终选择的 `image_source`：`image_path`
优先，`image_base64` 兜底，并应用 `asset_base_url` 和 Markdown 地址转义。即使 table
选择 GFM/HTML、equation 已有 LaTeX，图片来源也会保留且只出现一次。

如需完整可展示 Markdown，应调用 `render_markdown()`，不能简单拼接 content_list 中的
`content`。

该接口目前只属于严格 `mineru.render` 公共面；本契约不迁移旧 parser、CLI、API、
doclib 或历史扁平 `content_list.json` 产物。

## 展示模式

`MarkdownRenderMode.DEFAULT`:

- 隐藏 `header/footer/page_number/aside_text/page_footnote`。
- 合并页内和跨页 `text/ref_text.continues_prev`；ref_text 可跨过页面辅助块查找前序 ref_text。
- 合并页内和跨页 `list.continues_prev`；ref list 可跨过页面辅助块查找前序列表。
- 只合并跨页 `table.continues_prev`。
- 页面之间不输出分割线。

`MarkdownRenderMode.FULL`:

- 展示全部顶层 block。
- 只合并页内 `text/ref_text/list.continues_prev`。
- 不合并跨页 text/ref_text/list/table。
- 每两个相邻 `PageInfo` 之间输出 `---`，空白页边界同样保留。

text 合并允许跨越任意其他 block，后一个 text 的内容被吸收到前一个 text，
后块不再在原位置输出。ref_text 与 ref list 都只跨过页面辅助块，其他语义块
仍会分别阻断文本链和列表链；两种参考文献标记互不混用。普通 list 仍要求
物理相邻。table 只接受跨页合并；失败时保留两张原表。

## Block 规则

- `text/ref_text`: 解析行内公式、样式和超链接；普通 text 转义可能误触发的 Markdown block 前缀。
- `doc_title/paragraph_title`: 使用全局 `level`，Markdown 标题最多六级；anchor 输出为 HTML id。
- `list`: 递归缩进；普通列表直接使用 content 已内化的前缀。`sub_type=ref_text`
  时统计每个直属非空 item 的前五个可见字符，数字前缀未达到严格多数则给全部 item
  补 `- `，已有 `- ` 不重复；嵌套列表独立判定。
- `index`: 递归输出 GFM 列表；标题叶子使用 anchor 生成内部链接，不当作 heading 输出。
- `equation`: content 非空时输出行间 LaTeX，空 content 回退图片。
- `image/chart`: 图片优先，识别内容放入折叠 details；chart 的简单 HTML 表格转为 GFM，复杂表格保留 HTML，无图片时直接输出转换后的内容。
- `table`: 简单单层 HTML 转 GFM；合并单元格或不可无损结构保留 HTML；空间投影文本使用动态 fenced block；空 content 回退图片。
- `code`: 使用 `guess_lang` 和动态 fenced block。
- `algorithm`: 使用等宽、`white-space: pre-wrap` 的 HTML div，只把 `<eq>` 转为行内公式。

视觉父块严格按 `content` 原顺序渲染，body、重复 caption 和 footnote 不重新排序。

## 行内语义

行内 parser 只解释白名单标签，包括 `<eq>`、`<text style>`、
`<hyperlink>/<url>`、`<sup>/<sub>`。未知或损坏标签作为普通文本转义保留。
普通字体样式使用 Markdown；underline、emphasis、上下标和复杂组合使用安全 HTML。
underline/strikethrough 包裹的 ASCII 空格按 dev 规则转为 `_`/`-` marker；
整块 marker 会转义首字符，避免被识别为 Markdown 分割线。

LaTeX 定界符来自 `$MINERU_HOME/config.yaml`:

```yaml
render:
  latex_delimiters:
    display:
      left: "$$"
      right: "$$"
    inline:
      left: "$"
      right: "$"
```

## Index anchor

Office model-list 在对象化前按 document-wide anchor 查找真实目标标题。匹配成功的
目录 text 叶子改成相同的 `doc_title/paragraph_title` 类型与 level，同时保留目录
显示文本；未匹配 anchor 降级为普通 TextBlock。已有旧 MiddleJson 不补造已丢失的 anchor。
