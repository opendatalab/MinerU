# ADR-0020: Doclib Schema Stability Boundary

状态: Accepted
日期: 2026-06-18
相关文档: ../architecture.md, ../sdk/doclib-client.md, ../cli/mineru-server.md

## 背景

doclib 以 SQLite 作为本地存储实现，但 CLI、SDK、Agent 和未来外部 HTTP client 依赖的是 doclib 对外暴露的 JSON / Pydantic models。

此前存在一个未决问题：

- 哪些 SQLite 字段进入稳定 v1 schema，哪些仍视为内部或实验字段。

如果直接把 SQLite 表结构视为稳定契约，会带来两个问题：

1. 存储实现细节会被错误地锁死，后续迁移、压缩、归并列或调整索引成本过高。
2. 对外字段稳定性会继续模糊，调用方无法判断哪些响应字段可以长期依赖。

因此需要明确：稳定的是哪一层，如何分层，以及哪些字段仍不进入稳定承诺。

## 决策

v1 稳定 schema 绑定的是 **doclib public models 的字段**，而不是 SQLite 表结构或原始列名。

稳定性分三层：

1. `core stable`
2. `operational stable`
3. `diagnostic / internal`

### 1. Core stable

以下对象进入稳定 v1 schema，字段级稳定：

- `ParseResponse`
- `ParseInfo`
- `FileInfo`
- `DocInfo`
- `DocContentResponse`
- `SearchResult`
- `FindResult`
- `FileInfoResponse`
- `TierParseInfo`
- `WatchInfo`
- `ExcludeRuleInfo`
- `ParsingRuleInfo`
- `ConfigValueResponse`
- `ConfigResponse`
- `ForgetPathResponse`
- `InvalidateResponse`

这些对象承载用户主链路、SDK 主接口和 Agent 主消费面。字段可以新增可选项，但不应删除、重命名或改变既有语义。

### 2. Operational stable

以下对象也进入稳定 v1 schema，但归类为运维状态面：

- `ScanInfo`
- `WatchStats`
- `ErrorBucket`
- `ErrorSummary`
- `HTTPServerStatus`
- `WorkerStatus`
- `LocalParseServerStatus`
- `RemoteParseServerStatus`
- `ParseServerStatus`
- `ServerStatusResponse`

这些对象的目标是支持：

- `mineru server status`
- watch / scan / health 观察
- SDK 和桌面端的运行状态展示

它们属于稳定契约，但只覆盖长期有产品意义的状态字段，不要求为每个调试字段提供长期承诺。

### 3. Diagnostic / internal

以下字段不进入稳定承诺：

- 仅用于调试或环境自检的字段
- 容易随部署方式、日志策略或 SQLite 配置变化的字段
- 仅为了排障临时加入的运行时细节

当前明确归入 `diagnostic / internal` 的典型字段包括：

- `ServerStatusResponse.version`
- `ServerStatusResponse.python_version`
- `ServerStatusResponse.sqlite_size_bytes`
- `ServerStatusResponse.sqlite_journal_mode`
- `ServerStatusResponse.sqlite_wal_size_bytes`
- `ServerStatusResponse.watches`
- `ServerStatusResponse.recent_logs`

这些字段可以继续暴露，但调用方不得把它们当作稳定 v1 契约。

## 稳定边界说明

### SQLite 不是稳定对外 schema

SQLite 表、列名、索引、锁字段、计数缓存字段都属于内部实现。允许：

- 改列名
- 拆表或并表
- 调整索引
- 改写中间状态字段

前提是对外 public models 的语义保持兼容。

### Public models 的兼容规则

对于进入 `core stable` 和 `operational stable` 的字段：

- 可以新增可选字段
- 不应删除既有字段
- 不应重命名既有字段
- 不应改变字段语义
- 不应把原先稳定字段降级为仅诊断字段

如果必须做破坏性修改，应通过新的 ADR 和明确迁移说明处理。

## 替代方案

### 方案 A：SQLite 表结构直接作为稳定 schema

未采用。

原因：

- 锁死内部存储实现，后续演化成本过高。
- SQLite 列名与对外 API 命名层并不总是一一对应。

### 方案 B：只稳定主链路对象，不稳定运维对象

未采用。

原因：

- `mineru server status`、scan、watch 已进入 P0 主链路。
- 如果运维对象长期不稳定，CLI、桌面端和 SDK 状态展示都无法形成可靠契约。

### 方案 C：所有 `doclib/types.py` 字段全部稳定

未采用。

原因：

- 会把明显仍在演进的诊断字段也锁死。
- 不利于后续对状态输出和调试信息做迭代。

## 影响

### 对实现

- 后续存储层重构以 public models 兼容为边界，而不是以 SQLite 列兼容为边界。
- 新增字段时需要先判断属于 `core stable`、`operational stable` 还是 `diagnostic / internal`。

### 对 CLI / SDK / Agent

- 主链路对象和运维对象都有清晰的稳定层级。
- 调用方可以安全依赖 `core stable` 与 `operational stable` 字段。
- 调试字段应按 best-effort 使用，不应写死依赖。

### 对文档

- `open-questions.md` 中的 `OQ-B-003` 不再保留为 blocker。
- 后续状态类文档应在需要时标注哪些字段属于 diagnostic。

## 后续动作

1. 从 `open-questions.md` 中移除 `OQ-B-003`。
2. 在相关 SDK / CLI 文档中引用本 ADR 作为 schema 稳定性边界说明。
3. 后续若需要，可把 diagnostic 字段进一步收敛到单独命名空间，而不是继续平铺在顶层响应中。
