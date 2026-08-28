# ADR-0029: RTF 原生语义 Flash 解析

状态: Accepted
日期: 2026-08-26
相关文档: ../architecture.md, ../middle-json/current-medium.md, 0022-doclib-file-type-tier-remote-semantics.md

## 背景

RTF 是带有 group、control word、destination、代码页和可嵌入二进制素材的结构化文档格式。把它当作普通文本会丢失标题、列表、表格、图片、公式和脚注；调用 LibreOffice、Pandoc 或平台专属转换器又会引入部署、性能和跨平台差异。项目需要让 `.rtf` 与现有 Office 格式进入相同的稳定 `doc_analyze()` / Middle JSON 3.0 链路。

## 决策

- RTF 使用 `mineru/model/flash/office/rtf/` 下的纯 Python lexer、typed IR、状态机、Office Math 适配器和 converter，不依赖 anydoc、Pandoc、LibreOffice 或外部服务。
- `FileSuffix`、`OfficeSuffix` 和 `OFFICE_EXTENSIONS` 接受 `rtf`；`RtfModel -> analyze_office -> doc_analyze` 固定返回 `effort="flash"`、`parse_mode="txt"`。
- RTF 是可重排语义文档。输出固定为一个无 bbox 的逻辑页；`\page` / `\column` 保留为换行，显式 `page_range` 返回 `page_range_invalid`。
- Parser 覆盖代码页与 Unicode、继承样式、标题、嵌套列表、合并/嵌套表格、字段链接、脚注/尾注、页眉页脚、RTF Office Math、pict 图片和图片 MTEF 公式。当前 schema 没有 quote block，顶层引用样式降级为普通文本但保留内容。
- RTF header 是强内容签名，优先于扩展名和 CSV 兜底；允许 UTF-8 BOM、最多 64 个 ASCII 空白和大小写差异。
- 只保留安全相对链接、已解析标题 bookmark 以及 `http` / `https` / `mailto` / `tel`。对象数据、活动内容和外部图片不执行、不读取、不联网。
- 固定资源上限为：输入和单素材 128 MiB、素材总量 128 MiB、group 深度 256、token 1600 万、表格网格 400 万、列表 8 层、嵌套表格 4 层。资源超限和结构损坏分别使用已有 `LegacyOfficeResourceLimitError` 与 `LegacyOfficeMalformedError`。

## 恢复边界

- 缺少尾部右花括号但已经恢复出可信内容时保留内容并记录 warning。
- 截断 `\binN`、无 RTF header 或失去二进制游标边界时硬失败。
- 未知 `\*` destination、危险 URL、损坏图片和不受支持对象按局部降级处理，不能污染周围正文。
- Office Math 转换失败时保留首个安全 `mmathPict` 预览；解析成功时不重复输出公式预览图。

## 替代方案

### 依赖 anydoc

未采用。其统一文档模型和测试维度可作为实现参考，但核心 Rust wheel 会扩大 MinerU 的运行时依赖和平台发布矩阵。

### LibreOffice / Pandoc 转换

未采用。外部进程不属于稳定核心依赖，且转换耗时、可用性和结果会随平台与安装版本变化。

### 纯文本提取

未采用。它无法满足 MinerU 对表格、公式、图片和严格 Middle JSON 的结构化契约。

## 影响

- 现代 CLI、API Server、mineru-kit 和 doclib 自动接受 RTF，并按 Office 的本地 Flash-only tier/remote 规则处理。
- `file_suffix="rtf"` 成为 Middle JSON 3.0 的稳定公开值。
- RTF 不提供物理页、坐标或 OCR 版式语义；需要这些能力时应由调用方显式转换为 PDF 后另行解析。
