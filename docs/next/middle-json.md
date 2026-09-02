# Middle JSON

状态: Draft
读者: backend 开发者、Markdown/Structured Content 输出开发者、SDK 开发者、Agent 能力开发者
范围: Middle JSON schema 2.0 事实标准、Agent-native gap、统一 envelope、迁移策略和验收清单
非目标: 具体 OCR 或模型算法实现
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 当前定位

Middle JSON 已经收敛为 schema 2.0 的严格 Pydantic 模型：`mineru/types.py` 中的 `MiddleJson`、`PageInfo`、Block 联合与 InlineSpan 联合。自然语言使用结构化 Span，不再使用字符串标签协议。

## 目录

1. [总览](middle-json/README.md): 当前事实、目标和工作分层。
2. [当前事实标准](middle-json/current-medium.md): `MiddleJson` / `PageInfo` / `BlockBase` / `TextBlock` / `ImageBlock` 等 Pydantic 模型的现状。
3. [Backend 差异](middle-json/backend-gaps.md): `tier`（flash/basic/standard/advanced）在 `backend/analysis` 与 `backend/postprocess` 上的已解决和未解决问题。
4. [Agent-native Gap](middle-json/agent-gaps.md): 引用、定位、稳定性和隐私边界。
5. [Canonical Envelope](middle-json/envelope.md): 顶层结构、`_meta`、版本和兼容输入。
6. [当前 Content List v2 结构盘点](middle-json/structured-content-current.md): 当前 Structured Content 起点的事实结构。
7. [Structured Content Schema](middle-json/structured-content-schema.md): NEXT 版结构化内容 JSON 的目标 schema 草案。
8. [Rendering Contract](middle-json/rendering.md): Markdown / HTML / LaTeX / DOCX / Structured Content 如何通过 `render/api.py` 消费 middle structure。
9. [迁移计划](middle-json/migration.md): 可执行阶段、任务清单和验收标准。

## 整理原则

- 以当前代码中的 Pydantic 严格模型为起点（schema 2.0）。
- 区分“已经解决”、“部分解决”、“仍需工作”。
- Agent 引用与稳定 page/block locator 是 P0 目标。
- MinerU 3.4.5 页面及对应 1.0 pages 包装经 `legacy_schema_adapter` 转换；其它旧 payload
  和无法识别的缓存视为 stale，必须从源文件重新解析。
- render 统一不能只看入口 facade，还要收敛格式-specific 分支。

## 与其他文档的关系

- API 的 `middle_json` 输出见 [Unified API](api.md)。
- SDK 的 `ParseResult` 见 [SDK 设计](sdk.md)。
- 产品侧 Agent-native 目标见 [产品路线图](roadmap.md)。
- backend 处理边界见 [系统架构](architecture.md)。
