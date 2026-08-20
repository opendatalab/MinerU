# Rendering Contract

状态: 统一入口 / Markdown / Content List / DOCX / HTML v1
读者: render 开发者、backend 开发者、SDK 开发者
范围: 严格 MiddleJson 到 Markdown、Content List、DOCX 和 HTML 的统一消费契约

## 统一入口

```python
from mineru.render import (
    MarkdownRenderOptions,
    RenderFormat,
    RenderMode,
    render,
)

markdown = render(
    middle_json,
    RenderFormat.MARKDOWN,
    options=MarkdownRenderOptions(
        mode=RenderMode.DEFAULT,
        asset_base_url="",
    ),
)
```

`render()` 每次接收一个严格 `MiddleJson` 和一个 `RenderFormat`，返回目标格式的原生结果：

| `RenderFormat` | Options | 返回类型 |
| --- | --- | --- |
| `MARKDOWN` | `MarkdownRenderOptions(mode, asset_base_url)` | `str` |
| `HTML` | `HtmlRenderOptions(mode, asset_base_url, standalone, document_title)` | `str` |
| `DOCX` | `DocxRenderOptions(mode, asset_resolver)` | `bytes` |
| `CONTENT_LIST` | `ContentListRenderOptions(asset_base_url)` | `dict[str, Any]` |

`options` 省略时使用对应格式的默认 Options。入口只接受 `RenderFormat` 枚举，格式与 Options
类型不匹配时抛出 `TypeError`，不归一化字符串格式，也不在单次调用中批量渲染。四个专用
`render_*()` 函数继续保留；统一入口只负责严格校验和分发，不改变底层渲染语义或异常类型。

所有入口均不修改传入的 `MiddleJson`。renderer 不负责把结果或图片写到文件系统；DOCX
需要读取相对图片 sidecar 时，只能通过显式 `asset_resolver` 获取字节。

### 内部依赖边界

`markdown.py`、`html.py`、`docx.py` 和 `content_list.py` 是稳定公共门面；实现代码位于非公共
`mineru.render._internal`。`common` 只保存跨格式 AST、解析、列表/目录语义和 render planner，
不得依赖任一格式实现。`markdown`、`html`、`docx` 子包只能依赖 `common` 与自身模块，彼此
不交叉导入。`_internal` 路径不属于 SDK 兼容承诺。

## Markdown

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
保留粗体、斜体、下划线、删除线、emphasis、上下标、外链及行内公式。emphasis 使用下方
着重号；带可见装饰的边界空格会等量写为 NBSP，避免 Word 隐藏下划线或删除线。列表保留
producer 已经内化的 marker，只用缩进表达层级，不重建 `numbering.xml`。

公式通过 `latex2mathml -> mathml2omml -> OMML` 输出为可编辑 Office Math。末端
`\tag{...}` 会从公式主体剥离并通过右对齐 tab 单独写编号；转换失败的块公式依次回退
公式图片和可见 LaTeX，行内公式回退可见 LaTeX，所有回退均记录 page/block 定位。
Office producer 生成的规范无横线 `\genfrac{}{}{0pt}{}{num}{den}` 会在 DOCX adapter 内
转换为双行 matrix，不改写 MiddleJson 中的原始 LaTeX。

HTML table 使用 occupancy grid 生成原生 Word table，支持 `rowspan/colspan`、嵌套表格、
单元格富文本、公式、链接和图片。HTML 结构无效时只在存在表格图片时回退；空间投影
文本表格在 content 非空时使用独立的等宽预格式样式，仅输出原始文本并保留空格、换行
与 Tab；content 为空或纯空白时改用其图片，两者都缺失才报错。Image body 文本写入图片
alt description；Chart
在图片后继续输出可用的 HTML 数据表。视觉父块仍严格保持 body/caption/footnote 原顺序。

`RenderMode.DEFAULT` 与下文 Markdown DEFAULT 使用同一份续段/续表 planner；
`RenderMode.FULL` 保留全部页面辅助块，不跨源页合并，并在相邻 `PageInfo` 间写硬分页。
`MarkdownRenderMode` 是 `RenderMode` 的既有公共别名。DOCX 是语义化可编辑输出，不承诺
复刻源 PDF/Office 的字体、分栏、断行和页数；CLI、doclib 与 v1 API 不在本契约范围。

## MinerU 独立 HTML

```python
from mineru.render import RenderMode, render_html

html_document = render_html(
    middle_json,
    mode=RenderMode.DEFAULT,
    asset_base_url="",
    standalone=True,
    document_title=None,
)
```

`render_html()` 只接受严格 `MiddleJson`，返回字符串且不写文件、不读取图片 sidecar、
不访问网络。`standalone=True` 输出完整 HTML5 文档；`False` 输出单根
`article.mineru-document` fragment，不包含 CSS、脚本或
`head`。显式标题缺失时，HTML title 使用首个非空 `doc_title` 的纯文本，再回退为
`MinerU Document`。完整文档使用 `body.mineru-html-body` 提供居中和响应式页面布局；
fragment 调用方可以自行决定外围容器尺寸。

HTML 与 DOCX/Markdown 共用 `RenderMode` 和续段/续表 planner。DEFAULT 输出无页面 wrapper
的连续阅读内容；FULL 为每个 `PageInfo` 输出一个 `section.mineru-page`，包括空页，并在
相邻页面之间输出 `hr.mineru-page-break`。每个顶层 block wrapper 保留来源 page/type/index
元数据。行内内容直接消费共享 AST：普通文本始终 HTML escape，未知标签保持可见；公式只
在 `mineru-math` carrier 内使用 `\(...\)` 或 `\[...\]`，不扫描普通正文中的美元符号。
HTML 的普通 `InlineText` 默认 linkify `http/https`、`www.`、邮箱和常见裸域名；裸域名统一补
`https://`，邮箱补 `mailto:`。无协议裸域名的 TLD 必须属于工程常用白名单：
`com cn org net edu gov io ai dev app de uk nl ru br fr au in eu jp`、
`ca hk tw sg kr nz info biz xyz site online tech cloud shop store`。显式 HTTP(S)、`www.` 和邮箱
不受该小白名单限制。已有 hyperlink、代码、算法、公式、raw HTML 及 Mermaid 源码不重复
识别，候选 href 仍通过统一 URL sanitizer。

完整文档只内联压缩后的 `mineru.min.css`。其可读源码为 `mineru.css`，所有文档规则均
限定在 `.mineru-document` 内，standalone 外围布局规则限定在 `.mineru-html-body`，
不会修改 fragment 宿主页的全局标签样式。样式使用系统字体，不加载外部字体。
image、table、chart 和 code 主体均从正文内容区左边界开始；图片保持固有尺寸和宽高比，
宽表格/流程图只在自身容器滚动。视觉 caption 与 footnote 使用 figure 可用宽度并统一左对齐，
多行说明的每一行均从同一左边界开始，body/caption/footnote 的原始顺序不变。
有公式时按需加载固定 MathJax 4.1.2 `tex-chtml.js`，启用 `ui/safe`、禁用 `require`，并
关闭菜单和 enrichment。有可高亮代码时按需加载固定 Prism 1.30.0 core 和 Autoloader，
语言组件根固定到同版本 jsDelivr。合法 flowchart 按需加载固定 Mermaid 11.16.1 UMD；
该约 3.57 MB bundle 只在含流程图的文档下载并可由浏览器缓存。三个入口脚本均携带 SRI；
动态 MathJax 扩展、字体和 Prism 语言组件没有逐文件 SRI。renderer 不注入 CSP，HTTP
宿主需要自行允许相应 jsDelivr script/font/style，并评估 CDN 供应链策略。

`ImageBlock(sub_type="flowchart")` 仅在 body 是完整 `mermaid` fence，且首个有效语句为
`graph|flowchart + TB|TD|BT|RL|LR` 时生成 `.mineru-flowchart`。frontmatter、init directive、
其他 Mermaid 图类型及超过 50,000 字符的源码沿用普通图片/details 路径，不加载 Mermaid。
合法流程图初始显示已有 raster；Mermaid 以 `securityLevel: "strict"`、禁用 HTML label、
`maxTextSize=50000`、`maxEdges=500` 逐图生成 SVG，成功后替换 raster。无 JS、CDN/语法失败
继续显示 raster；无 raster 时自动展开已转义源码。图源码只在浏览器本地参与渲染。

fragment 调用方需要加载随包提供的 `mineru.min.css`，并按实际内容加载同版本
MathJax、Prism 和 Mermaid。MathJax 必须在初始化前复用 standalone 的 delimiter、
`ignoreHtmlClass/processHtmlClass`、`ui/safe`、禁用 `require` 及 safeOptions 配置。动态插入后调用
`MathJax.typesetPromise([root])` 与 `Prism.highlightAllUnder(root)`；替换已有公式前先调用
`MathJax.typesetClear([root])`。fragment 内的相对图片以宿主页 URL 为基准，不能通过
fragment 自带 `base` 修正。

fragment 宿主的 MathJax 配置至少必须包含以下安全边界，并在加载 `tex-chtml.js` 前赋值：

```javascript
window.MathJax = {
  loader: {load: ["ui/safe"]},
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEnvironments: false,
    processRefs: false,
    packages: {"[-]": ["require"]},
  },
  options: {
    ignoreHtmlClass: "mineru-document",
    processHtmlClass: "mineru-math",
    enableMenu: false,
    enableEnrichment: false,
    safeOptions: {
      allow: {URLs: "none", classes: "none", cssIDs: "none", styles: "none"},
    },
  },
};
```

standalone 还会通过 `startup.ready` 把 menu/enrichment 选项重新施加到 live MathDocument；
fragment 宿主如需完全相同行为，也必须复用该 hook，而不能只复制上面的静态字段。
fragment 中的 flowchart 只包含 canvas、fallback 与源码 DOM；宿主必须复用 standalone 的
Mermaid 11.16.1、SRI、安全配置、顺序 `mermaid.render()` 和失败状态切换，且不得调用
`bindFunctions`。Mermaid 点击指令和 HTML label 在本契约中始终禁用。

HTML table/chart/image body 只在识别为 markup 时经过 `nh3` allowlist。活动标签、事件属性、
来源 style/class/id/data 属性和危险 URL 会被删除；链接只允许相对地址、fragment、
`http/https/mailto/tel`，图片只允许安全 sidecar、根相对路径、HTTP(S) 与签名匹配的 raster
data URI。安全 `<eq>` 在清洗后转换为公式 carrier。清洗后不可用的 table 优先回退整体
图片，否则以转义后的 `pre` 可见保留；普通 `<local_dir>` 等尖括号文本不会进入 sanitizer。
Chart 的 pipe table 只实现 GFM 列结构与反斜杠/pipe 解码，cell 内容继续消费 MiddleJson
行内标签；`**bold**`、反引号和 Markdown link 等任意 GFM inline 语法不会二次解析。

v1 只增加严格 `mineru.render` 公共面，不迁移旧 parser、CLI、API 或 doclib，也不提供
Markdown 字符串转 HTML、离线单文件、主题切换、侧边 TOC 或用户自定义 head/CSS/script。

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
