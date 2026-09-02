# ADR-0031: HTML 原生静态语义 Flash 解析

状态: Accepted，已实现
日期: 2026-08-27
相关文档: ../sdk/parser.md, ../cli/mineru-kit-parse.md, ../api/parse-jobs.md, 0024-file-type-tier-normalization.md

## 背景

文件类型、Doclib、API Server 与 mineru-kit 已把 `.html/.htm` 归入仅支持 Flash 的输入，但严格 `FileSuffix` 与
`doc_analyze()` 尚未接通，导致请求在高层被接受后才以 `Unsupported file type: html` 失败。旧 HTML parser 依赖
历史 `Line`/`Span` 结构，也无法复用 Middle JSON 2.0 的统一后处理与 renderer。

## 决策

- HTML 规范后缀统一为 `html`，使用 `flash/txt`、整本文档、单逻辑页、无 bbox 的 ModelJson/MiddleJson 2.0 路径。
- 只解析静态源码，不执行 JavaScript、不启动浏览器、不生成布局坐标。
- 默认固定使用保守 `auto` 正文选择：高置信候选进入正文投影，其余情况回退完整 body；首版不公开模式参数。
- MinerU 自身 renderer 输出版本化 `data-mineru-html-version="1"` 机器契约。只有当前 renderer 能生成的 canonical
  marker 树进入精确路径，parser 跳过 `auto` 裁剪并按 `data-block-type` 恢复顶层 block、visual child、列表/目录叶子及关键元数据。
- `data-block-type` 使用原始下划线形式的 `BlockType` 值，是新版 MinerU HTML 的唯一机器类型来源；
  `mineru-caption`、`mineru-footnote` 和 `mineru-figure--*` 等 class 只负责样式。
- 精确解码先把固定 DOM grammar 解析为无资源副作用的 typed plan，再加载图片并一次性物化 raw blocks。未知版本或
  非 canonical v1 均丢弃整条精确路径，从干净 DOM 进入通用解析；人工编辑后的机器 HTML 不承诺保留原 block 类型和元数据。
- Image/Chart body 的 renderer-owned 图片恢复为唯一主图片载荷；rich-content carrier 内的安全 HTML（包括嵌套图片）
  按语义规范化后保留在 `content`，不会提升为额外主图片载荷。
- 旧 MinerU HTML 与普通互联网 HTML 都走通用路径，只承诺正文、图片、表格、链接及可识别公式的内容兼容，
  不承诺按旧 class suffix 精确恢复 Chart、Index 或 annotation 类型。
- EPUB XHTML 与 standalone HTML 共用静态 markup projector，继续输出既有标题、正文、列表、表格、图片、代码、
  公式和页面脚注 block，不增加 HTML 专属 block。
- 新版公式 carrier 保存 `data-formula-display` 与 `data-mineru-latex`。网页公式按固定来源优先级收敛为裸 LaTeX：
  行间写入 `EquationBlock.content`，行内写入 `EquationInlineSpan`，不保留 `$...$`、`\(...\)` 或 `\[...\]` 外层定界符。
  无法安全转换的 MathML 与暂不支持的 AsciiMath 保留可见文本，不冒充 LaTeX。
- 有序列表只保留连续阿拉伯编号和单一列表级起始值。
- data URI 与安全本地图片进入 `image_base64`；本地相对路径不得逃逸 HTML 所在安全根目录。
- 远程图片不由 MinerU 下载，在 `ImagePayloadBlock.image_url` 中保存受限 HTTP(S) 绝对地址；Markdown、HTML、
  Structured Content 可引用该地址，DOCX 只写可点击链接。
- 输入、DOM 深度/节点、单图、图片总量、stylesheet 与生成内容均使用固定资源预算。

## 替代方案

- **直接依赖 Docling 或 magic-html**：未采用。前者引入独立文档 IR，后者主要输出正文 HTML，都会产生重复转换和
  新运行时依赖；本实现只借鉴其 DOM walker、正文评分和安全边界。
- **Playwright 渲染动态页面**：未采用。浏览器运行时、脚本隔离、超时和布局坐标属于独立能力。
- **默认强制正文抽取**：未采用。论坛、文档站、目录页和多 article 页面容易丢失有价值内容。
- **服务器下载远程图片**：未采用。避免引入 DNS、重定向、私网访问和总下载预算等 SSRF 风险面。

## 影响

- `.html/.htm` 可通过 SDK、API Server、mineru-kit 和 Doclib 本地 Flash 路径生成统一多格式输出。
- 新版 MinerU HTML 可按类型级往返；FULL 中页面辅助类型可恢复，但 HTML 输入仍固定投影为一个逻辑页，
  不依据 `data-page-idx` 重建原始多页。
- `image_url` 是 schema 2.0 内可选且默认省略的字段。
- 打开 Markdown 或 HTML 输出时，客户端可能访问原 HTML 中保留的远程图片地址；服务端解析阶段不会访问该地址。
- 所有非 PDF 页范围请求继续返回 `page_range_invalid`。

## 后续动作

- 如需公开 `auto/main/document` 或支持浏览器渲染，另行设计 API、配置、安全隔离和回归语料。
- 使用文章、文档站、论坛和本地资源页面持续校准正文选择阈值，低置信结果必须保持 body 回退。
