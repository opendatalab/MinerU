# ADR-0028: CSV 本地结构化 Flash 解析

状态: Accepted
日期: 2026-08-26
相关文档: ../cli/mineru-parse.md, ../cli/mineru-kit-parse.md, ../api/parse-jobs.md, 0022-doclib-file-type-tier-remote-semantics.md, 0024-file-type-tier-normalization.md

## 背景

CSV 原先归入普通 text 输入：可以被 Doclib 发现、入库和建立原始文本索引，但显式 parse 返回 `parse_not_required`，不会产生 Middle JSON、表格 block 或统一渲染结果。这会丢失 CSV 的行列结构，也使 SDK、parse-server、mineru-kit 与 Doclib 无法用同一套表格结果处理 CSV。

CSV 没有可靠的内容签名，同一个扩展名还可能包含不同编码、分隔符、引号内换行和不等宽记录。因此它需要独立的轻量解析边界，不能伪装成 XLSX，也不能重新放入 Office 转换器。

## 决策

### 1. 文件类型与路由

- `.csv` 从 `TEXT_EXTENSIONS` 移入独立的 `CSV_EXTENSIONS`，并加入 `FLASH_ONLY_PARSE_EXTENSIONS` 与 `PARSEABLE_EXTENSIONS`。
- `FileSuffix`、`ModelJson.file_suffix` 和 `MiddleJson.file_suffix` 接受 `csv`。
- CSV 由独立 `CsvModel` 和 `analyze_csv()` 生产 model-list；`doc_analyze()` 使用 `pdf -> csv -> office` 的显式路由，Office 类型和映射保持不变。
- CSV 没有可靠内容签名。路径解析先尊重 PDF、OOXML、OLE2 和图片等强内容类型，再以 `.csv` 扩展名兜底；无路径字节流必须显式指定 `file_suffix="csv"`。

### 2. 输出契约

- CSV 固定记录 `effort="flash"`、`parse_mode="txt"`，不进入 PDF/image 质量 tier。
- 一份 CSV 对应一个逻辑页面。非空输入产生一个 `TableBlock`，空输入产生一个空页面。
- 表格主体使用安全 HTML 表达，继续复用 Middle JSON 2.0 的 `TableBlock/TableBodyBlock` 和既有 Markdown、HTML、DOCX、Structured Content renderer；不新增 block 类型或 CSV 专属输出 schema。

### 3. 解析与数据完整性

- 支持 UTF-8、带 BOM 的 UTF-8/UTF-16、GB18030 和 Windows-1252；所有解码均为严格模式。
- 支持逗号、分号、Tab 和竖线，并识别合法的 Excel `sep=` 声明。
- 支持转义引号和引号内换行；真正损坏的 CSV 使整次解析失败，不跳过记录。列数不齐按最大列宽补空单元格。
- 所有字段保留字符串语义，不转换数字、日期或公式外观；HTML 投影必须转义不可信字段。
- 首行只在标签形态和后续字段类型提供足够证据时标记为表头。
- 解析器设置输入字节、行数、列数和规范化网格规模上限；不截断、不分页，也不返回部分表格。

### 4. Doclib、Tier 与 Remote

CSV 属于仅支持 flash tier 的结构化输入，行为与 Office/HTML 的 tier 归一组一致：

| 场景 | CSV 行为 |
|------|----------|
| 单文件未指定 tier 或指定 `flash` | 本地 `flash` 解析 |
| 单文件显式指定质量 tier | `tier_unsupported_for_file_type` |
| 单文件请求 remote | `remote_unsupported_for_file_type` |
| API Server、目录和多文件批量 | 文件级执行归一为 `flash` |
| parsing-rule | 忽略质量 tier 和 remote，创建本地 `flash` parse row |

Doclib 对新增或变更的 CSV 创建解析任务，解析完成后使用表格 Markdown 建立 FTS，并持久化 Middle JSON 缓存。已有 CSV 记录不自动批量回填；显式 force 解析或文件变化后升级。

## 替代方案

### 方案 A：继续把 CSV 当作普通文本

未采用。该方案无法保留行列结构，也不能提供统一 TableBlock 和渲染结果。

### 方案 B：把 CSV 伪装为 XLSX 或放入 Office 转换器

未采用。CSV 不是工作簿容器，没有工作表、样式、合并单元格或图片等 Office 语义，这会破坏模块职责和文件类型元数据。

### 方案 C：新增 delimiter、encoding、header 公开参数

首版未采用。现有统一 parse API 没有格式专属参数；首版使用确定性的自动策略，避免同时扩大 SDK、HTTP 和缓存键协议。

## 影响

- ADR-0022 和 ADR-0024 中把 `.csv` 列为无需解析 text 的部分被本 ADR 修订；其它 text 类型保持原行为。
- CSV 会产生一页 `flash` 缓存、locator 和 FTS 内容，依赖 `parse_not_required` 的 CSV 调用方需要改用结构化结果。
- `mineru/cli_old` 和旧 Gradio WebUI 不在本次接入范围，其上传格式列表不变。
- 实现必须覆盖编码、分隔符、表头、损坏输入、资源限制、同步/异步、API Server、mineru-kit 和 Doclib 生命周期测试。

## 后续动作

1. 若未来支持 `.tsv` 或公开 CSV 参数，应新增文件后缀与缓存键兼容设计，不能把 `.csv` 的自动决策隐式复用于其它格式。
2. 若需要自动回填历史 CSV，应单独设计可控的 Doclib 迁移或批处理，而不是在升级时自动创建大量任务。
