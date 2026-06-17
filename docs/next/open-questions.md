# 开放问题清单

状态: Draft
读者: 项目核心开发者、编程 Agent、文档维护者
范围: 集中维护下一版 MinerU 仍未定稿、需要产品或技术决策的问题
非目标: 重复已经写入专题文档正文的规则；替代实现计划

## 1. 使用方式

本文只记录真正未决的问题。已经由专题文档正文确定的规则，不应继续出现在开放问题中。

处理顺序建议:

1. 先处理 `Blocker`，否则实现任务会缺少完成边界。
2. 再处理 `P0`，确保主链路、API、CLI 和 SDK 的契约稳定。
3. `P1/P2` 可以随功能推进逐步决策。
4. 涉及长期架构选择的问题，沉淀为 [设计决策记录](decisions/README.md)。

## 2. 已收敛规则

以下内容不再作为开放问题讨论:

| 规则 | 当前结论 |
|------|----------|
| `parse-server` | 最终术语，中文为“解析服务”；此前误写的相关术语不作为项目概念使用。 |
| 默认选择策略 | 只会解析为 `standard` 或 `pro`，永远不会等价于 `flash`。 |
| 实际 tier 记录 | 任务、缓存、产物和 metadata 只记录实际使用的实体 tier，不记录 `requested_tier` / `resolved_tier`。 |
| `flash` | 长期存在，既是一个解析档位，也是 PDF 快速解析 backend 名称。 |
| doclib 产物 | `parsed/` 目录只持久化按页组织的 Middle JSON 批次文件；Markdown、Content List、HTML 读取时转换。 |
| 能力发现 | parse-server 的解析档位发现 endpoint 统一为 `GET /v1/tiers`。 |
| 搜索结果 tier | search result 应以机器可读字段显式返回索引来源 tier。 |
| remote fallback | 用户显式允许 remote 后，remote 失败可以 fallback 到 local；local 失败不能自动扩大到 remote。 |
| SDK tier 参数 | 普通 parser 入口应支持 `tier` 语义；`backend` 只作为专家层或过渡参数暴露。 |
| 错误映射 | `quality_tier_unavailable` 应进入稳定错误体系。 |
| JSON 输出格式 | 不把 `json` 作为正式产物名；正式格式为 `middle_json`、`content_list`、`structured_content`，详见 [ADR-0001](decisions/0001-json-output-formats.md)。 |
| Force 与 invalidate | `--force` 跳过 done cache，可复用 active parse，只为未覆盖页创建新 parse；invalidate 才改变旧缓存可用性，详见 [ADR-0002](decisions/0002-force-vs-invalidate.md)。 |
| doclib HTTP API | 本地 doclib HTTP API 使用 `/docs`、`/parses`、`/search` 和 `POST /invalidate`，详见 [ADR-0004](decisions/0004-doclib-http-api-resources.md)。 |
| 本地 `pro` | `pro` 本地运行是正式支持能力，不是实验能力；当前代码基本 ready。 |
| mineru.net tier | `mineru.net/api` 在相当长时间内只提供最高等级的 `pro` 解析。 |
| api-server tier | 一个 `mineru-kit api-server` 进程只服务一个 tier。 |
| managed 生命周期 | managed 模式下，模型下载、预热、重试和退避由 `mineru-kit api-server` 负责。 |
| 本地 api-server 安全 | 本地 api-server 默认监听 loopback；可通过 `--api-key` 设置固定 API Key，默认不设置。 |
| P0 主链路 | P0 包含完整 watch、rules、search，不再把它们标为可选主链路。 |
| 配置优先级 | CLI 参数 > 环境变量 > 文件配置 / SQLite 配置；启动前文件配置与 SQLite 配置不应定义同一配置项。 |
| watch 默认 tier | watch 自动解析默认 tier 暂不配置化，固定使用 `flash`。 |
| doclib API 与 SDK | doclib API 和 SDK 是同一套方法/协议；项目内部除 doclib client 外不直接依赖 HTTP，外部客户端未来可以使用 HTTP API。 |
| `mineru-kit vlm-server` | 纳入当前 CLI 文档第一阶段。 |
| `mineru-kit parse` 与 doclib | `mineru-kit parse` 不允许复用 `mineru` 的本地 doclib 缓存；kit 是纯工具，不感知 doclib。 |
| `mineru-kit` Agent 暴露 | `mineru-kit` 长期对 Agent skill 隐藏，只作为专家入口；Agent 默认使用 `mineru`。 |
| `mineru-kit` 参数稳定性 | 暂不划分 `stable` / `experimental` 等稳定性等级，先保持简单。 |
| `mineru-kit parse` 输入 | 当前只支持文件和目录输入；不支持 stdin、路径列表、URL 输入和递归目录。 |
| `mineru-kit parse` 输出 | `--output` 必填；单文件可输出到文件路径或目录路径；多文件只能输出到目录路径；同名冲突直接报错并终止整个批次。 |
| `mineru-kit parse` local/remote | local 支持 `tier/backend` 并校验兼容；remote 支持 `--remote`/`--remote-url`/`--api-key`，允许传 `tier`，禁止传 `backend`。 |
| parsing-rules 默认 tier | parsing-rules 允许不指定 tier；执行时必须解析为实体 tier，并只记录实际 tier；默认选择不能解析为 `flash`。 |
| Telemetry P0 | P0 必须实现 doclib server telemetry 状态、聚合、flush 和 CLI 管理入口；纯工具无 telemetry 能力。 |
| `mineru parse` 默认页码范围 | 分页文档默认读取 `1~10`。 |
| 渐进式阅读协议 | Server 返回结构化 `next_request`；CLI marker 只由 `next_request` 渲染，不作为协议源头，详见 [ADR-0013](decisions/0013-doc-content-progressive-reading.md)。 |
| 非分页文档增量读取 | 非分页长文档通过 `after` cursor 继续读取；当前不为 `mineru parse` 暴露单独的 `--offset` 参数。 |
| Doc short id | `docs.short_id` 按 SHA256 7 位前缀起步、冲突递增长度、写入后保持稳定，详见 [ADR-0011](decisions/0011-doclib-doc-short-id.md)。 |
| Block locator | Agent-native 整体是 P0。P0 public locator 使用 1-based `page_no` / `block_no`，形态为 `doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}`，详见 [ADR-0012](decisions/0012-doclib-block-locator.md)。 |
| `mineru read` | 新增 locator-first 顶级读取命令；`parse` 和 `read` 共享 `ReadPlan` 与 `execute_read_plan()`，`read --format image` 进入 P0，详见 [ADR-0014](decisions/0014-mineru-read-command.md)。 |
| `search/find/show file` JSON 输出 | `search` 输出文件名、文件大小、页数和 snippet；`find` 输出文件名、文件大小、页数；`show file` 输出文件大小、页数、文档 metadata 摘要、各 tier 已解析页和 active parse 摘要。 |
| Tool SDK `server_url` | Tool SDK 的 PDF VLM / hybrid parser 可以保留 `server_url` 作为 parser-layer 高级 backend 参数；API-backed parser 使用 `api_url`。 |
| `MinerUApiParser` 位置 | API-backed parser 保持现状，公开入口继续使用 `from mineru.parser import MinerUApiParser`；不重命名，不移动到 `mineru.parser.remote`。 |
| `ParseResult` backend 字段 | `ParseResult` 不再持有 `_backend` / `_version_name`；backend 仅允许作为 page / middle-json 内部来源信息存在。 |
| 其他语言 SDK | 其他语言 SDK 暂无开发计划；如果未来开发，只覆盖 v1 Unified API，不覆盖本地 doclib UDS 能力。 |
| `mineru server status` | `mineru server status` 使用 doclib `ServerStatusResponse`，不与 parse-server `/v1/health` 强行对齐；后续只需决定该 JSON schema 的稳定发布边界。 |
| `mineru-kit` 对外暴露 | `mineru-kit` 保留为专家工具入口；Agent 默认入口是 `mineru`，skill 不主动暴露 `mineru-kit`。 |
| Office/HTML unknown bbox | 公共 schema 允许 unknown bbox；不要求 Office/HTML 在 P0 估算真实 bbox，validator 应区分 unknown 与非法 bbox。 |
| `standard` 硬件基线 | `standard` 当前使用 pipeline 后端；Apple Silicon macOS 约使用 4GB 统一内存，Windows / Linux 需要 NVIDIA GPU 且至少 4GB 显存。详见 [解析 Tier](tiers.md#5-standard)。 |
| watch tier 升级 | P0 不做基于启发式的自动提示或自动排队升级。watch 默认使用 `flash`；后台自动升级只由用户显式配置的 parsing-rules 触发；用户或 Agent 主动读取时再按默认选择策略解析到 `standard` / `pro`。 |
| Middle JSON `schema_version` | 采用 `pages` 的 Middle JSON 顶层结构必须写 `schema_version`；旧 CLI `pdf_info` 结构暂不改。当前只增加 `schema_version` 和 `pages`，暂不增加 `_meta`；代码常量定义为 `mineru.schema.middle_json.MIDDLE_JSON_SCHEMA_VERSION`。 |

## 3. Blocker

| ID | 问题 | 决策产物 | 影响范围 |
|----|------|----------|----------|
| OQ-B-003 | 哪些 SQLite 字段进入稳定 v1 schema，哪些仍视为内部或实验字段。 | schema 稳定性说明 | doclib server、SDK、Agent |
| OQ-B-004 | CLI exit code、`error.code`、`retryable`、`user_action` 和 trace/request id 的稳定契约。 | 错误码补充规格 | CLI、API、SDK |

## 5. CLI 与 Agent 协议

| ID | 问题 | 建议归属 |
|----|------|----------|
| OQ-C-001 | Markdown block marker 作为 P0 Agent-native 输出的一部分时，是否只允许显式 opt-in，而默认只输出普通 Markdown 与 continuation marker。 | CLI / Agent |
| OQ-C-002 | `mineru` 与 `mineru-kit` 是否共享同一组选项命名。 | CLI |
| OQ-C-012 | `mineru server status` 的 JSON schema 是否作为稳定 doclib SDK/API schema 发布。 | CLI / API |
| OQ-C-013 | 多用户或多项目场景下 UDS 路径是否需要命名空间。 | server 生命周期 |

## 6. SDK

| ID | 问题 | 建议归属 |
|----|------|----------|
| OQ-S-001 | 是否将 `MineruClient` 作为 Python Product SDK 的唯一入口。 | Doclib SDK |
| OQ-S-002 | 是否提供顶层 `mineru.client` 作为 `mineru.doclib.client.MineruClient` 的稳定别名。 | SDK |
| OQ-S-003 | 是否需要 `AsyncMineruClient`。 | Doclib SDK |
| OQ-S-004 | `MineruClient.parse(format=...)` 是否改为 `output_format` 并与 v1 API 对齐。 | Doclib SDK |
| OQ-S-006 | 是否公开低层 `V1ApiClient`，让开发者直接操作 uploads/jobs/files。 | API SDK |
| OQ-S-007 | Product SDK 是否只面向 doclib，还是同时提供 cloud API client。 | SDK 分层 |
| OQ-S-010 | `parse_batch()` 是否需要统一进度回调接口。 | Tool SDK |
| OQ-S-012 | 是否增加 `page_count()`、`text()`、`html()` 等便利方法。 | ParseResult |
| OQ-S-013 | `ParseResult.save()` 的 writer protocol 是否正式定义为 `Protocol`。 | ParseResult |
| OQ-S-014 | 是否把 api-server 中的 Pydantic models 提取为共享 schema 模块。 | SDK / API |
| OQ-S-015 | SDK exception 是否暴露 `user_action`，用于 CLI 输出下一步建议。 | SDK errors |

## 7. Middle JSON

| ID | 问题 | 建议归属 |
|----|------|----------|
| OQ-M-002 | `filename` 是否默认写入 Middle JSON envelope。 | Middle JSON envelope |
| OQ-M-003 | `parsed_at` 是否默认写入；写入会降低跨机器 diff 稳定性。 | Middle JSON envelope |
| OQ-M-005 | backend-specific block type 的公开枚举、兼容策略和降级策略如何定稿。 | Middle JSON schema |

## 8. Roadmap 与评测

| ID | 问题 | 建议归属 |
|----|------|----------|
| OQ-R-001 | P0 / P1 / P2 是否绑定粗粒度日期承诺。 | Roadmap |
| OQ-R-002 | WAI 的一年和两年目标值如何设定。 | Roadmap |
| OQ-R-003 | 解析质量回归框架采用 OmniDocBench，还是另建内部评测集。 | Roadmap / eval |
| OQ-R-006 | VLM “第一梯队”如何定义 benchmark 和 gap 阈值。 | Roadmap / eval |
| OQ-R-007 | 对外公开路线图和内部执行路线图是否拆分。 | Roadmap |

## 9. ADR 候选

| ID | 主题 | 触发条件 |
|----|------|----------|
| ADR-C-001 | `doclib server + local parse-server` 进程边界 | 当实现需要改为单进程、插件式加载或多 parse-server 管理时。 |
| ADR-C-002 | SQLite-only 配置与项目级 `mineru.toml` | 当需要项目级可提交配置或启动前可编辑配置时。 |
| ADR-C-003 | 本地 API 稳定性等级 | 当第三方客户端开始依赖 doclib HTTP API 时。 |
| ADR-C-004 | Middle JSON schema versioning | 当 schema validator、跨语言 SDK 或公开样例开始稳定发布时。 |
