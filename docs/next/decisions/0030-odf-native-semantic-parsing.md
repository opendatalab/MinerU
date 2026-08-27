# ADR-0030: ODF 原生语义 Flash 解析

状态: Accepted
日期: 2026-08-26
相关文档: ../architecture.md, 0022-doclib-file-type-tier-remote-semantics.md, 0024-file-type-tier-normalization.md

## 背景

ODT、ODS、ODP 是 ZIP/XML 形式的 OpenDocument 文档。经 LibreOffice 转为 OOXML 会引入外部进程、部署差异和转换损失；把 ODF 交给 CSV、RTF 或 OOXML converter 也会破坏既有格式边界。

## 决策

- 在 `model/flash/office/odf/` 内直接解析 ODF package、样式、正文、表格、图片、MathML 和图表对象，不依赖外部 Office 套件或第三方文档转换运行时。
- `FileSuffix`、`OfficeSuffix` 和 `OFFICE_EXTENSIONS` 追加 `odt`、`ods`、`odp`；三个格式经独立 `OdtModel`、`OdsModel`、`OdpModel` 进入既有 `analyze_office()` 与 Middle JSON 2.0 链路。
- CSV 继续由独立 `analyze_csv()` 路由，RTF 继续使用原有 lexer/parser/converter；ODF 变更不得重构既有 converter。
- 内容识别顺序保持 PDF、RTF、OOXML 的既有优先级，再检查 ODF `mimetype` 和 manifest；ODF 是强内容类型，优先于 `.csv` 扩展名兜底。
- ODT 按显式分页和 master-page 切分逻辑页，ODP 每张 slide 一页，ODS 每个可见 sheet 一页。三者固定记录 `effort="flash"`、`parse_mode="txt"`。
- 加密包不支持；宏、脚本和外部对象不执行或下载。ZIP、XML、表格展开和资源字节均使用固定上限。

## 结构能力

- ODT 保留标题、段落、富文本、列表、链接、书签、表格、图片、MathML、脚注和页眉页脚。
- ODP 保留 slide 边界、标题、shape 文本、列表、表格、图片、图表和 speaker notes。
- ODS 保留可见 sheet、离散数据区、合并单元格、typed cached value、图片和图表；隐藏 sheet 不输出。
- 图表优先使用 series 的精确范围，只有唯一非空内嵌表时才安全回退；可恢复时同时保留预览和 HTML 数据表。

## 兼容性

- Middle JSON schema 版本、Block 类型、renderer 和 ParseResult 不变，只扩展合法 `file_suffix` 值。
- `ott/ots/otp`、`fodt/fods/fodp`、密码解密和像素级版式恢复不属于本决策。
- 现代 CLI、API Server、mineru-kit 和 doclib 自动继承 ODF；旧 `cli_old` 不在范围内。
