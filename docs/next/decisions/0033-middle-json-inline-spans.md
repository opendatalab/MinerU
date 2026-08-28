# ADR-0033: Middle JSON 3.0 结构化行内 Span

状态: Accepted
日期: 2026-08-28
相关文档: ../middle-json/current-medium.md, ../middle-json/rendering.md

## 背景

Schema 2.0 使用自然语言字符串中的 `<eq>`、`<text style>` 和
`<hyperlink>/<url>` 表达行内语义。这要求 parser、后处理和 renderer 反复匹配
标签外观文本，导致 `&` 实体双重转义、标签字面量误识别和损坏标记恢复歧义。

## 决策

- Schema 版本提升为 3.0；自然语言 leaf 的 `content` 统一改为
  `list[InlineSpan]`。
- InlineSpan 是严格的 Text/EquationInline/CodeInline/Hyperlink 联合；Span 不携带
  bbox、score、字体、图片或块级字段。
- parser 直接从 PDF 几何区间、Office 富文本、DOM 或 OFD TextLine 构造 Span；
  后处理只做结构化合并和裁剪；renderer 按 discriminator 分派。
- 算法正文新增 `algorithm_body` 并使用 Span；普通代码、行间公式、表格 HTML、
  Chart HTML、图片专用内容和 Mermaid 继续使用专用字符串。
- HTML v1 wire 使用明确 DOM metadata 精确恢复 Span，不从展示标签反推。
- 当前运行时仍严格使用 schema 3.0；仅对 MinerU 3.4.5 页面结构及对应的 1.0
  pages 包装保留 `legacy_schema_adapter`，先转换为结构化 raw ModelJson 再进入统一后处理。
- 其它未知旧 payload 仍必须从源文件重新解析，不恢复字符串与 Span 的联合公开类型。

## 结果

- `A&B`、`1<2`、实体外观与完整 `<script>`/`<eq>` 字面量可以无歧义保真；
- URL 安全策略只作用于 HyperlinkSpan.url；自动链接只作用于 TextSpan；
- parser、后处理和 renderer 之间不再存在自然语言私有标记协议；
- 这是有意的不兼容升级，调用方必须携带并校验 `schema_version="3.0"`。
