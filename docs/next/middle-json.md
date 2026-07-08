# Middle JSON

状态: Draft
读者: backend 开发者、Markdown/Content List 输出开发者、SDK 开发者、Agent 能力开发者
范围: Middle JSON 的当前事实标准、Agent-native gap、统一 envelope、迁移策略和验收清单
非目标: 具体 OCR 或模型算法实现
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 当前定位

Middle JSON 已经有一个事实标准: `mineru/types.py` 中的 `PageInfo`、`Block`、`Line`、`Span` dataclass。下一阶段的重点不是从零定义 schema，而是把现有类型、旧输出格式、各 backend 差异和 Agent-native 需求收敛成可执行的统一方案。

## 目录

1. [总览](middle-json/README.md): 当前事实、目标和工作分层。
2. [当前事实标准](middle-json/current-medium.md): `PageInfo` / `Block` / `Line` / `Span` 的现状。
3. [Backend 差异](middle-json/backend-gaps.md): Pipeline / VLM / Hybrid / Office / HTML 的已解决和未解决问题。
4. [Agent-native Gap](middle-json/agent-gaps.md): 引用、定位、稳定性和隐私边界。
5. [Canonical Envelope](middle-json/envelope.md): 顶层结构、`_meta`、版本和兼容输入。
6. [当前 Content List v2 结构盘点](middle-json/structured-content-current.md): 当前 Structured Content 起点的事实结构。
7. [Structured Content Schema](middle-json/structured-content-schema.md): NEXT 版结构化内容 JSON 的目标 schema 草案。
8. [Rendering Contract](middle-json/rendering.md): Markdown / Content List / Structured Content 如何消费 middle structure。
9. [迁移计划](middle-json/migration.md): 可执行阶段、任务清单和验收标准。

## 整理原则

- 以当前代码中的 typed dataclass 为起点。
- 区分“已经解决”、“部分解决”、“仍需工作”。
- Agent 引用与稳定 page/block locator 是 P0 目标。
- schema 版本和迁移函数必须能处理历史数据。
- render 统一不能只看入口 facade，还要收敛 backend-specific 分支。

## 与其他文档的关系

- API 的 `middle_json` 输出见 [Unified API](api.md)。
- SDK 的 `ParseResult` 见 [SDK 设计](sdk.md)。
- 产品侧 Agent-native 目标见 [产品路线图](roadmap.md)。
- backend 处理边界见 [系统架构](architecture.md)。
