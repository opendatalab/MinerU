# ADR-0034: TSV 独立文件后缀

状态: Accepted，已实现
日期: 2026-09-16
相关文档: 0028-csv-structured-flash-parsing.md, 0022-doclib-file-type-tier-remote-semantics.md, 0024-file-type-tier-normalization.md, ../architecture.md, ../tiers.md

## 背景

ADR-0028 把 `.csv` 从普通 text 提升为仅支持 flash tier 的结构化输入，并在后续动作 1 中要求：未来支持 `.tsv` 时必须新增文件后缀与缓存键兼容设计，不能把 `.csv` 的自动决策隐式复用于其它格式。

在此之前 `.tsv` 只有偶然可用的行为：`CSV_EXTENSIONS` 早已包含 `tsv`，但共享协议 `FileSuffix` 没有 `tsv` 值，探测层 `_resolve_signatureless_csv_suffix` 把 `.tsv` 路径统一折叠为 `csv`。结果是一份 `.tsv` 文件能端到端解析，但持久化的 `metadata.file_suffix` 记为 `csv`，doclib `docs.file_type` 也是 `csv`，格式身份不真实；公开文档中的 CSV/TSV 表述得不到协议层支撑。

## 决策

### 1. 共享协议与探测

- DocVortex `FileSuffix` Literal 新增 `tsv`（位于 `csv` 之后），`FILE_SUFFIXES` 随 `get_args` 自动扩展；schema 版本维持 2.0，加法变更沿用既有先例。旧严格读取方无法接受 `tsv` 文档，需升级 DocVortex，与历次新增原生格式一致。
- 探测层扩展名兜底按扩展名返回独立后缀：`.csv` 路径返回 `csv`，`.tsv` 路径返回 `tsv`；强内容类型（PDF、OOXML、OLE2、图片等）继续优先于扩展名兜底。
- 无路径字节流不自动进入 tsv 解析：Magika 的 csv/tsv 标签在无扩展名时兜底为 `txt`，必须显式 `file_suffix="tsv"`（或 `"csv"`），两种分隔文本保持对称。

### 2. 解析与输出契约

- tsv 不新增解析器：DocVortex `model_types` 与 MinerU `doc_analyze` 路由均把 `tsv` 指向既有 `CsvModel` / `analyze_csv()`，继续使用自动分隔符嗅探（逗号、分号、Tab、竖线）与 `sep=` 指令，不公开格式专属参数（延续 ADR-0028 方案 C 结论）。
- 输出契约与 CSV 完全一致：固定 `effort="flash"`、`parse_mode="txt"`，一份输入一个逻辑页面、一个 `TableBlock`；`metadata.file_suffix` 如实记录 `tsv`。

### 3. 文件类型与 Doclib

- `FILE_TYPE_BY_EXTENSION["tsv"]` 从 `"csv"` 改为 `"tsv"`，与 `metadata.file_suffix` 一致；doclib `docs.file_type`、`mineru search --type` 过滤与 API Server 类型标记随之生效。
- 已入库旧 `.tsv` 行保留历史 `csv` 标记，不做批量迁移；文件内容变化重入库后自然更新为 `tsv`。
- tier、remote、page_range 语义不变：tsv 与 csv 同属仅支持 flash tier 的结构化输入组，显式质量 tier 报 `tier_unsupported_for_file_type`，remote 报 `remote_unsupported_for_file_type`，不接受 page_range。

### 4. 缓存键兼容

- Doclib 缓存键保持内容寻址 `(sha256, tier, page_range)`，后缀不进入键。相同字节的 `.csv` 与 `.tsv`（两种格式都合法的内容）共享同一条缓存与同一个 doc 行。
- 理由：两种分隔文本走同一引擎、同一自动嗅探，相同字节必然产出相同解析结果，按后缀分区只会复制等价缓存。缓存内 `metadata.file_suffix` 记录实际产出批次的格式；读取边界 `read_cached_middle_json` 不与当前文件扩展名交叉校验，与 PDF/图片现有语义一致。
- 这不是把 `.csv` 的决策隐式复用：tsv 的格式身份在协议、探测、路由与 file_type 四层都是显式独立的，只有结果缓存按内容共享。

## 替代方案

### 方案 A：继续把 `.tsv` 折叠为 `csv`

未采用。解析结果虽然正确，但持久化元数据与 doclib file_type 撒谎，公开文档的 CSV/TSV 支持表述没有协议依据。

### 方案 B：后缀进入缓存键

未采用。相同字节跨后缀必然产出相同结果，分区只增加 parses 表与落盘路径的迁移成本，不带来语义收益。

### 方案 C：为 tsv 新增独立解析器或公开分隔符参数

未采用。Tab 已在既有分隔符候选中，`sep=` 指令亦支持；独立解析器与格式参数会无谓扩大 SDK、HTTP 与缓存键协议面。

## 影响

- ADR-0028 后续动作 1 中 `.tsv` 部分由本 ADR 实现并修订；公开 CSV 参数部分仍维持不做。
- 已有 `.tsv` 的 doclib 记录在内容不变时继续命中旧 `csv` 标记缓存，属预期行为；重新解析后新批次记录 `tsv`。
- API Server、mineru-kit、WebUI、watch/scan 均通过既有 `PARSEABLE_EXTENSIONS`/`FLASH_ONLY_PARSE_EXTENSIONS` 派生集合自动接受 `.tsv`，无需单独接入。
- 实现测试覆盖：探测与强内容优先、同步/异步 analyze 契约、API Server job、doclib 生命周期（独立 file_type、flash 缓存、FTS）、页范围拒绝与 FileSuffix 枚举，位于两个仓库的 CSV/格式矩阵测试族。

## 后续动作

1. 若未来公开 CSV/TSV 专属参数（delimiter、encoding、header），仍须按 ADR-0028 的约束新增缓存键兼容设计。
2. 若引入更多无签名分隔文本后缀，应复用本 ADR 的模式：协议显式加值、探测按扩展名返回、共用引擎、缓存保持内容寻址。
