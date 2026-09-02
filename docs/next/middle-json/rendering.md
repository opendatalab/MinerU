# Rendering Contract

状态: Implemented
读者: render 开发者、backend 开发者、SDK 开发者
范围: 严格 MiddleJson 到 Markdown、Content List V1/V2、Structured Content、DOCX、EPUB、PDF 和 HTML 的统一消费契约

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
| `MARKDOWN` | `MarkdownRenderOptions(mode, asset_base_url, image_renderer)` | `str` |
| `HTML` | `HtmlRenderOptions(mode, asset_base_url, standalone, document_title)` | `str` |
| `DOCX` | `DocxRenderOptions(asset_resolver)` | `bytes` |
| `EPUB` | `EpubRenderOptions(title, authors, language, identifier, modified_at, asset_resolver)` | `bytes` |
| `PDF` | `PdfRenderOptions(asset_resolver, document_title)` | `bytes` |
| `STRUCTURED_CONTENT` | `StructuredContentRenderOptions(asset_base_url)` | `dict[str, Any]` |
| `CONTENT_LIST` | `ContentListRenderOptions(asset_base_url)` | `list[dict[str, Any]]` |
| `CONTENT_LIST_V2` | `ContentListV2RenderOptions(asset_base_url)` | `list[list[dict[str, Any]]]` |

`options` 省略时使用对应格式的默认 Options。入口只接受 `RenderFormat` 枚举，格式与 Options
类型不匹配时抛出 `TypeError`，不归一化字符串格式，也不在单次调用中批量渲染。各专用
`render_*()` 函数继续保留；统一入口只负责严格校验和分发，不改变底层渲染语义或异常类型。
只有 Markdown 与 HTML 的专用函数和 Options 暴露 `mode`；其它格式传入 `mode` 会按严格签名
直接抛出 `TypeError`，不会兼容或静默忽略。

所有入口均不修改传入的 `MiddleJson`。renderer 不负责把结果或图片写到文件系统；DOCX、
EPUB 与 PDF 需要读取相对图片 sidecar 时，只能通过显式 `asset_resolver` 获取字节。

### 内部依赖边界

`markdown.py`、`html.py`、`docx.py`、`epub.py`、`pdf.py`、`content_list.py`、`content_list_v2.py` 和
`structured_content.py` 是稳定公共门面；实现代码位于非公共
`mineru.render._internal`。`common` 只保存跨格式 AST、解析、列表/目录语义和 render planner，
不得依赖任一格式实现。各格式私有子包只能依赖 `common`、基础层与自身模块，彼此
不交叉导入。`_internal` 路径不属于 SDK 兼容承诺。

## Markdown

```python
from mineru.render import RenderMode, render_markdown

markdown = render_markdown(
    middle_json,
    mode=RenderMode.DEFAULT,
    asset_base_url="",
)
```

`render_markdown()` 只接受 `MiddleJson`，不兼容旧 dict、`list[PageInfo]`、
backend 参数或旧 `Line/Span` renderer。渲染过程基于深拷贝，不修改输入对象，
也不负责把图片写到文件系统。

图片按 `image_path`、`image_base64` data URI、受限 HTTP(S) `image_url` 的顺序选择。
`asset_base_url` 只给相对 `image_path` 和 HTML 内相对 `img src` 添加前缀。

`ImageRenderer` 是从 `mineru.render` 导出的 Block 级图片扩展契约。传入自定义 renderer 时，
image block 及无结构内容的 table/chart/equation 可以改用调用方提供的图片引用；结构化 table
仍保留文本或表格内容，但会移除由自定义 renderer 接管的内嵌 `<img>`。清理后只剩图片的
table 会回退到自定义 renderer。

## 可编辑 DOCX

```python
from pathlib import Path

from mineru.render import render_docx

asset_dir = Path("output/document")
docx_bytes = render_docx(
    middle_json,
    asset_resolver=lambda relative_path: (asset_dir / relative_path).read_bytes(),
)
```

`render_docx()` 只接受严格 `MiddleJson`，返回完整 `.docx` bytes，不写文件、不读取 cwd，
也不访问网络。block 同时携带多种图片来源时优先通过 `asset_resolver` 解码 `image_path`，
其次使用 `image_base64`；仅有 `image_url` 时写入可点击链接而不下载。仅有
`image_path` 时必须提供 `asset_resolver(relative_path) -> bytes`。必需图片缺失、路径
不安全、格式损坏或任意外部 SVG 输入会抛出带 `page_idx/block_index/block_type` 的
`DocxRenderError`；WebP 在内存中转为 PNG。MinerU 自己生成并通过安全子集校验的
WMF/EMF SVG 会写入 Office 2016 `asvg:svgBlip`，同时保留高密度 PNG fallback，现代
Word/LibreOffice 可使用矢量资源，旧版 Office 仍能显示同尺寸后备图。

DOCX 输出固定使用 A4 纵向与 20 mm 页边距。标题是 Word Heading 1–9，标题与
`page_footnote` anchor 是 document-wide bookmark，Index 标题叶子和正文 `#anchor`
写成内部链接；正文直接消费共享 inline AST，
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

DOCX 固定使用共享 planner 的 DEFAULT 连续语义：隐藏页面辅助块、合并跨页续段与续表，
不按 `PageInfo` 写硬分页或保留空源页；`page_footnote` 仍然输出。DOCX 是语义化可编辑输出，
不承诺复刻源 PDF/Office 的字体、分栏、断行和页数；CLI、doclib 与 v1 API 不在本契约范围。

## 单正文 EPUB 3.3

```python
from datetime import datetime, timezone

from mineru.render import EpubRenderOptions, RenderFormat, render

epub_bytes = render(
    middle_json,
    RenderFormat.EPUB,
    options=EpubRenderOptions(
        title=None,
        authors=("Alice",),
        language="zh-CN",
        modified_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        asset_resolver=lambda relative_path: (asset_dir / relative_path).read_bytes(),
    ),
)
```

`render_epub()` 只接受严格 `MiddleJson`，返回完整 EPUB 3.3 容器 bytes，不落盘、不读取 cwd、
不访问网络。容器固定包含首个且不压缩的 `mimetype`、`META-INF/container.xml`、一个 OPF、
一个 navigation document、静态 CSS、可用图片和唯一正文 `content.xhtml`。OPF spine 始终只有
这个正文项，不按 `PageInfo` 或标题拆章，也不生成 EPUB 2 NCX、封面或 page-list。

标题默认取首个非空 `doc_title`，否则回退为 `MinerU Document`；语言默认 `und`。未显式提供
identifier 时，由完整 MiddleJson、标题、作者和语言生成稳定 UUID URN。`modified_at` 默认使用
本次调用的 UTC 时间；显式值必须带时区，并同时控制 OPF 修改时间和 ZIP 成员时间，因此相同输入
与固定时间可得到相同 bytes。OPF 只输出 EPUB 要求的 title、identifier、language、modified 和
可选 creator 元数据，不把输入 EPUB 的原 OPF 元数据或 CSS 带入新容器。

navigation document 优先使用第一个至少含一个真实标题目标的 `IndexBlock`，保留有效项的顺序和
嵌套层级并丢弃悬空项；没有有效目录时按正文标题 `level` 建树；没有标题时使用书名链接到正文
起点。标题和 `page_footnote` 获得 document-wide 唯一 id，内部 fragment 会重写到这些真实目标，
脚注链接增加 EPUB `noteref` 语义。无包内目标的相对链接展开为普通文字，显式
`http/https/mailto/tel` 外链继续保留。

行内和行间 LaTeX 通过现有 `latex2mathml` 生成静态 Presentation MathML，不加载 MathJax；转换
失败时保留可见 LaTeX。只有正文实际写入 MathML 时，manifest 的正文项才声明 `mathml` property。
代码、算法、流程图和富 HTML 都使用无脚本静态表示；富 HTML 只复制 allowlist 标签和有界的
表格/列表属性，删除活动内容、事件、来源样式及无法重写的资源。

图片按 `image_path`、`image_base64`、`image_url` 的候选顺序处理。相对 `image_path` 和富 HTML
相对 `img src` 只能经 `asset_resolver` 读取；data URI 会校验 MIME、签名和完整图片；WebP、BMP、
TIFF 及安全 MinerU SVG fallback 规范化为 PNG，PNG/JPEG/GIF 保留，最终按内容 SHA-256 去重。
renderer 不下载 `image_url`。路径缺失、resolver 抛错、图片损坏或远程图片都不会中断输出：图片
主体省略，但已有结构内容、alt、caption、footnote 和识别文字继续可见，也不会生成虚构占位文案。

EPUB 固定使用共享 planner 的 DEFAULT 连续语义，在单个 `article.mineru-document` 中展平页面，
隐藏页面辅助块、合并跨页续段与续表，不生成源页 section、空页或分页标记；`page_footnote` 仍然
输出。该能力仅属于严格 `mineru.render` 公共面，不新增 ParseResult 方法，也不迁移 CLI、API、
doclib 或 parse-job `output_formats`。

## MinerU PDF

```python
from mineru.render import render_pdf

pdf_bytes = render_pdf(
    middle_json,
    asset_resolver=lambda relative_path: (asset_dir / relative_path).read_bytes(),
    document_title=None,
)
```

`render_pdf()` 只接受严格 `MiddleJson`，返回完整 PDF bytes，不写文件、不读取 cwd 且不访问网络。
它固定使用 A4 纵向和 20 mm 页边距，是面向阅读的语义化重排输出，不承诺复刻源文件坐标、
分栏、断行或原始物理页数。PDF 固定使用共享 planner 的 DEFAULT 连续语义：隐藏页面辅助块、
合并跨页续段与续表，不保留空源页或在 `PageInfo` 边界硬分页；内容仍会按 A4 可用空间自然分页，
`page_footnote` 仍然输出。

PDF 使用 ReportLab 生成，标题、正文、表格、图片、代码、caption 与 footnote 采用 MinerU HTML
主题的打印配色。标题与 `page_footnote` anchor 会建立 PDF destination，目录和 InlineSpan fragment
链接可跳转到已注册目标；HTTP(S)、mailto 等外链保持可点击。Latin 使用 Helvetica/Courier，
中文使用 `STSong-Light`，日文与韩文分别使用标准 CID 字体。

行内与行间 LaTeX 通过 ZiaMath 的 inline/display 模式转换为自包含 SVG path，再由 FontTools 的
SVG path parser 直接映射为 ReportLab 矢量路径。该转换器只接受 ZiaMath 固定输出的
`svg/g/path/rect` 与 `M/L/Q/Z` 子集，不用于 MiddleJson 的任意 SVG。行内公式携带真实
width/ascent/descent 参与段落换行和基线计算；行间公式居中并保留末端 `\\tag{...}` 编号。
转换失败时，行内公式回退可见 `$...$`，行间公式依次回退现有公式图片和可见 LaTeX。

图片仍按 `image_path`、`image_base64`、`image_url` 的优先级选择。相对 sidecar 只经
`asset_resolver` 获取，远程 URL 不下载；缺失、损坏、任意外部 SVG 或其它不支持的图片输出带
page/block 定位的浅色占位，远程 URL 同时保留可点击链接，图片错误不会中止 PDF。HTML table
物化为支持 rowspan/colspan、嵌套表、重复表头和跨页拆分的原生 ReportLab table；结构无效时
回退空间文本、整体图片或占位。

该接口只属于严格 `mineru.render` 公共面，不新增 ParseResult 方法，也不迁移 CLI、API 或 doclib。

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

HTML 与 Markdown 共用 `RenderMode` 和续段/续表 planner。DEFAULT 输出无页面 wrapper
的连续阅读内容；FULL 为每个 `PageInfo` 输出一个 `section.mineru-page`，包括空页，并在
相邻页面之间输出 `hr.mineru-page-break`。每个顶层 block wrapper 保留来源 page/type/index
元数据。`article.mineru-document` 同时声明 `data-mineru-html-version="1"` 和
`data-render-mode="default|full"`；顶层 block、visual body/caption/footnote、列表/目录叶子使用
原始下划线形式的 `data-block-type`，并按类型携带 `data-block-sub-type`、`data-guess-lang`、
`data-anchor` 或 `data-level`。这些 data 属性构成 renderer 到 HTML Flash parser 的版本化机器契约，
CSS class 不参与精确类型判定。

行内内容直接消费共享 AST：普通文本始终 HTML escape，未知标签保持可见；公式只在
`mineru-math` carrier 内使用 `\(...\)` 或 `\[...\]`，不扫描普通正文中的美元符号。每个公式
carrier 同时保存 `data-block-type="equation"`、`data-formula-display` 和裸 LaTeX
`data-mineru-latex`，供 HTML parser 无损恢复；可见 MathJax 定界符不属于 Middle JSON 内容。
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

## 3.4.5 兼容 Content List V1/V2

严格 render 层同时提供两个由当前 MiddleJson 派生的兼容结构：

```python
from mineru.render import render_content_list, render_content_list_v2

content_list = render_content_list(middle_json, asset_base_url="")
content_list_v2 = render_content_list_v2(middle_json, asset_base_url="")
```

`render_content_list()` 返回按页序扁平化的 `list[dict]`。每个 item 保留真实 `page_idx`，可用 bbox
转换为 0-1000 整数坐标；标题继续使用 `type: "text"` 和 `text_level`，视觉内容使用
`img_path`、caption、footnote 与 body 等 3.4.5 字段。

`render_content_list_v2()` 返回 `list[list[dict]]`，外层与输入页面一一对应，空页保留 `[]`；每个
item 使用 3.4.5 风格的 `type + content`，行内内容收敛为 `text`、`equation_inline`、
`code_inline` 和 `hyperlink` span。V2 不增加 envelope 或显式页号，因此抽页结果只能通过外层顺序消费。

两个 renderer 都只接受严格 `MiddleJson`，没有 `mode` 参数，保持页面和 block 阅读顺序，不使用展示型
planner，也不合并 `continues_prev` 文本或跨页表格。同页连续 `ref_text` 会合并为 reference list；列表、
目录按源顺序递归展平，目录删除可信的末尾页码并保留 anchor。图片按 `image_path`、`image_base64`、
`image_url` 的顺序选择唯一来源，相对 sidecar 与 HTML 内相对图片使用 `asset_base_url`。

这是 render 层兼容 API；本阶段不新增 ParseResult 方法，不迁移 CLI/API 格式名，也不新增自动落盘产物。
现有 `render_structured_content()` 仍是独立的树形文档级 dict，不是 V2 的别名。

## 树形 Markdown Structured Content

严格 render 层还提供不合并、不丢块的树形内容输出：

```python
from mineru.render import render_structured_content

structured_content = render_structured_content(
    middle_json,
    asset_base_url="",
)
```

`render_structured_content()` 只接受 `MiddleJson`，返回可以直接交给 `json.dumps()` 的
文档级 dict。顶层保留 `pages/is_full_document/file_suffix/effort/parse_mode/mineru_version`，每页保留
`page_idx/blocks`，页面和顶层 block 的数量及顺序不变。它没有 `mode` 参数，也不使用
Markdown/HTML 的展示型 planner，因此不隐藏页面辅助块，也不合并 `continues_prev` 文本、
列表或跨页表格；`continues_prev` 和 `cell_merge` 等消费提示仍保留在对应 block。

`content` 保存适合结构化消费的文本表示，而不是可直接拼接成整篇文档的完整视觉
Markdown。`list/index` 的递归子树会收敛为一个字符串；`image/table/chart/code` 的唯一
body 会提升为父块 `content`，但图片资源只放在 `image_source`，不在 `content` 重复
输出图片语法或 details。说明文本按源 `index` 排序后分别输出为
`captions/footnotes: list[{bbox?: [x0, y0, x1, y1], content: str}]`。`bbox` 直接保留
说明子块的归一化坐标，源坐标缺失时省略该键；不输出说明的 `type/index`。视觉数组始终
存在，空说明保留为 `{"content": ""}`。这是相对于原字符串数组结构的直接契约变更，
消费方必须通过对象的 `content` 字段读取说明文本。

```json
{
  "captions": [
    {
      "bbox": [0.073, 0.235, 0.485, 0.245],
      "content": "图表1：汽车指数上周下跌 2.29%"
    },
    {
      "content": "没有源坐标的 Office 说明"
    }
  ]
}
```

`doc_title/paragraph_title.content` 只包含行内 Markdown，不含 heading 标记或 HTML
anchor；原始 `level` 和可选 `anchor` 作为独立字段保留。`equation.content` 是不带行间
定界符的裸 LaTeX，公式图片同样只通过 `image_source` 表达。所有输出 block 都删除
`index/guess_lang/image_path/image_base64/image_url`，`guess_lang` 仍会先参与代码 Markdown 的生成。

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
优先，`image_base64` 与 `image_url` 依次兜底，并应用 `asset_base_url` 和 Markdown 地址转义。即使 table
选择 GFM/HTML、equation 已有 LaTeX，图片来源也会保留且只出现一次。

如需完整可展示 Markdown，应调用 `render_markdown()`，不能简单拼接 structured_content 中的
`content`。

该接口目前只属于严格 `mineru.render` 公共面；本契约不迁移旧 parser、CLI、API、
doclib 或历史扁平 `content_list.json` 产物。

## 展示模式

`RenderMode` 只适用于 Markdown 与 HTML。DOCX、EPUB、PDF 固定采用下述 DEFAULT 语义；
Structured Content 与 Content List V1/V2 不使用展示型 planner。

`RenderMode.DEFAULT`:

- 隐藏 `header/footer/page_number/aside_text`，保留独立 `page_footnote`。
- 合并页内和跨页 `text/ref_text.continues_prev`；ref_text 可跨过页面脚注与辅助块查找前序 ref_text。
- 合并页内和跨页 `list.continues_prev`；ref list 可跨过页面脚注与辅助块查找前序列表。
- 只合并跨页 `table.continues_prev`。
- 页面之间不输出分割线。

`RenderMode.FULL`:

- 展示全部顶层 block。
- 带 anchor 的 `page_footnote` 在 DEFAULT/FULL 中由 Markdown 输出 span id，HTML 输出元素 id。
- 只合并页内 `text/ref_text/list.continues_prev`。
- 不合并跨页 text/ref_text/list/table。
- 每两个相邻 `PageInfo` 之间保留格式对应的页面边界：Markdown 输出分隔线，HTML 输出 page section 与分页线。

text 合并允许跨越任意其他 block，后一个 text 的内容被吸收到前一个 text，
后块不再在原位置输出。ref_text 与 ref list 都只跨过页面脚注与辅助块，其他语义块
仍会分别阻断文本链和列表链；两种参考文献标记互不混用。普通 list 仍要求
物理相邻。table 只接受跨页合并；失败时保留两张原表。

## Block 规则

- `text/ref_text`: 直接渲染 InlineSpan 中的公式、样式和超链接；普通 TextSpan 转义可能误触发的 Markdown block 前缀。
- `doc_title/paragraph_title`: 使用全局 `level`，Markdown 标题最多六级；anchor 输出为 HTML id。
- `page_footnote`: 独立于页面辅助块，Markdown/HTML 的 DEFAULT/FULL 以及固定默认 DOCX/EPUB/PDF 都输出；Markdown 使用无可见标签的 `<small><span class="mineru-page-footnote" data-block-type="page_footnote" style="color:#6b7280">…</span></small>`，其余格式使用各自的脚注样式和 anchor。
- `list`: 递归缩进；普通列表直接使用 content 已内化的前缀。`sub_type=ref_text`
  时统计每个直属非空 item 的前五个可见字符，数字前缀未达到严格多数则给全部 item
  补 `- `，已有 `- ` 不重复；嵌套列表独立判定。
- `index`: 递归输出 GFM 列表；标题叶子使用 anchor 生成内部链接，不当作 heading 输出。
- `equation`: content 非空时输出行间 LaTeX，空 content 回退图片。
- `image/chart`: 图片优先，识别内容放入折叠 details；chart 的简单 HTML 表格转为 GFM，复杂表格保留 HTML，无图片时直接输出转换后的内容。
- `table`: 简单单层 HTML 转 GFM；`strong/em/s` 的简单样式组合使用 Markdown wrapper，
  `u/sup/sub` 或其它复杂组合整体保留标准 HTML 标签；合并单元格或不可无损结构保留
  HTML；空间投影文本使用动态 fenced block；空 content 回退图片。表格不解析旧
  `<text style="...">` 私有标签。
- `code`: 使用 `guess_lang` 和动态 fenced block。
- `algorithm`: `algorithm_body` 使用等宽、`white-space: pre-wrap` 的 HTML div，并按 EquationInlineSpan 渲染公式。

视觉父块严格按 `content` 原顺序渲染，body、重复 caption 和 footnote 不重新排序。

## 行内语义

renderer 直接按 Text/EquationInline/CodeInline/Hyperlink discriminator 分派，不解析
字符串标签。TextSpan 中任意 `<...>` 都作为普通文本转义保留。普通字体样式使用
Markdown；underline、emphasis、上下标和复杂组合使用安全 HTML。
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
