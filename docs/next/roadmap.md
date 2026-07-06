# 产品路线图

状态: Draft
读者: 产品、项目核心开发者、需要理解 Next MinerU 方向的贡献者
范围: 产品定位、能力边界、优先级、阶段节奏
非目标: 展开 API/CLI/SDK 的字段级规格
底稿: `../../NEXT-ROADMAP.md`

## 1. 产品定位

MinerU 的下一阶段定位是：

> MinerU = Agent 时代的文档入口。

MinerU 不做完整 RAG 平台，也不做企业交付套件。它的核心价值是把文档解析、结构化输出、本地缓存、远端高质量解析和 Agent-native 调用入口收敛成一个可靠的基础能力。

MinerU 面向四类产品界面：

| 界面 | 形态 | 面向 |
|------|------|------|
| `mineru` | PyPI 包、CLI、SDK、Server、MCP | 开发者与 Agent |
| `MinerU.app/MinerU.exe` | 桌面客户端 | 普通用户 |
| `mineru.net` | Web | 体验入口、账号与 API Key 管理 |
| `mineru.net/api` | SaaS API | 远程高质量解析后端 |

其中 `mineru` 是核心产品形态。桌面端是 `mineru` 的 GUI 壳；SaaS API 是 `mineru` 的远程高质量后端；Agent 默认不直接调用 SaaS API，而是通过 CLI + Skill 或 MCP Server 使用 MinerU。

## 2. 核心原则

### 2.1 Privacy First

文档隐私是硬约束：默认本地解析，文档内容不离开本机。只有用户显式指定 `--remote` 或等价远端配置时，才上传到 `mineru.net/api`。

使用统计与文档内容分开处理。遥测必须透明、可控、可关闭，不包含文档内容、文件名和路径。CLI/GUI 首次启动应要求用户选择；SDK、Server、CI 等无 tty 场景默认关闭。

### 2.2 For Agent

Agent 使用场景是下一阶段的设计原点。CLI、SDK、MCP 和输出格式都应优先支持确定性、结构化、可自动恢复的工作流。

关键要求包括：

- 结构化输出，而不是只能解析 free-text。
- 确定性错误码，并提供修复建议。
- 支持 token-budgeted 输出和 continuation。
- 支持可寻址 page/block locator，便于 citation 和回溯。
- 支持按页、区域、类型过滤，减少不必要的解析和上下文占用。

### 2.3 易用优先

开发者应能通过 `pip install mineru` 或 uv 快速安装。普通用户应能通过 Agent + Skill 或桌面端零配置使用。默认路径要简单，进阶能力通过显式参数开启。

### 2.4 SaaS 只做高质量后端

SaaS 不承担 MinerU 的全部产品体验。它的定位是最高质量、最快速度的远程解析能力，通过统一 API 为 CLI、SDK、MCP 和开发者服务。

### 2.5 数据驱动但不被指标绑架

路线图以 WAI（Weekly Active Installs，每周至少完成一次成功 parse 的 install 数）作为北极星指标，同时跟踪解析成功率、backend 分布、错误码、留存和调用来源。

解析质量、社区反馈和代码质量不能只靠 dashboard 判断，仍需要人工评测、review 和专项回归。

## 3. 能力边界

MinerU 坚守“文档入口”定位。以下能力不内建到核心项目：

| 不做 | 交给 |
|------|------|
| 文档 chunking 策略 | LangChain / LlamaIndex 等生态层 |
| embedding 与向量化 | 向量库和下游应用 |
| chat with PDF | Agent 或应用层 |
| RAG pipeline | 生态层 |
| 内容 summary | 下游 LLM |
| 向量库 | 生态层 |
| Agent 框架 | 生态层 |
| prompt 模板 | 生态层 |

商业交付也不作为当前路线图目标：

| 不做 | 说明 |
|------|------|
| Enterprise tier | 不做官方企业版产品分层 |
| 带 SLA 的私有化交付 | 开源可自部署，但官方不提供商业交付通道 |
| 合规认证 | 不承诺 SOC2、HIPAA、GDPR、等保等认证 |
| 企业定制支持 | 不作为当前产品路线的一部分 |

## 4. 产品界面与通道

### 4.1 Agent 双通道

Agent 使用 MinerU 有两条通道：

| 通道 | 定位 | 默认建议 |
|------|------|----------|
| CLI + Skill | Agent 直接操作本地文件系统时的推荐通道 | 默认使用 |
| MCP Server | 需要标准工具发现、跨平台协议或远程 Agent 调用时使用 | P1 补齐 |

两条通道底层共享同一 SDK。本地解析是默认行为，远程解析需要显式选择。

### 4.2 本地能力中心

`pip install mineru` 安装的是本地文档能力中心，包含：

| 能力 | 优先级 | 说明 |
|------|--------|------|
| CLI | P0 | `mineru` 与 `mineru-kit` 命令 |
| Python SDK | P0 | 本地解析与远端调用的编程接口 |
| 文档库 | P0 | SQLite + SHA256 去重缓存 |
| Server | P0 基础 / P1 完整 | 本地后台进程、任务队列、后续扩展 UI/MCP |
| 搜索 | P1 | 文件名与内容检索 |
| 桌面端 | P1 | CLI 的 GUI 壳 |

解析能力通过 `flash`、`medium`、`high` 暴露；用户未指定 tier 时使用默认选择策略。tier 的用户语义、硬件边界和质量策略见 [解析 Tier](tiers.md)。

命令分工：

| 命令 | 面向 | 定位 |
|------|------|------|
| `mineru` | 普通用户、Agent | 文档管理中心；叠加本地数据库、缓存、搜索和远程切换 |
| `mineru-kit` | 解析内核、批处理开发者 | 无状态解析工具和服务工具 |

Skill 默认只暴露 `mineru`，避免普通 Agent 用户过早接触底层工具命令。

### 4.3 SaaS 能力

SaaS 的首要目标是远程高质量解析。匿名用户和注册用户使用同一质量等级，差异只体现在配额和限速机制：

| 用户 | 身份识别 | 配额 |
|------|----------|------|
| 匿名用户 | IP | 较低基础配额 |
| 注册用户 | API Key | 更高配额 |

`--remote` 上传的数据默认不用于训练。上传文件和解析结果按有限留存期保存，训练授权和质量样本捐赠必须是用户主动 opt-in。

## 5. 技术主线

### 5.1 Agent-native 输出

P0 阶段需要优先交付 Agent 可可靠消费的输出能力：

- 统一 message contract。
- typed error code 和修复建议。
- token-budgeted 输出和 continuation。
- 稳定 page/block locator。
- 页面、区域和类型过滤。
- Skill 决策树。
- SHA256 内容寻址缓存。

这些能力依赖 middle_json 的最小统一子集：稳定 page/block locator、page、bbox 和 type。

### 5.2 Middle JSON 统一

Pipeline、VLM、Office backend 当前存在 bbox、block type、span 粒度、page_info 和 page_size 等差异。P0 需要定义 canonical schema，并推动各 backend 对齐，最终收敛三套 `union_make` 逻辑。

详细设计见 [Middle JSON](middle-json.md)。

### 5.3 模型与插件生态

MinerU 的核心竞争力不只在单个模型，而在：

- 中间结构规范。
- pipeline 编排与多模型融合。
- CLI、SDK、Skill 和 Agent-native 体验。
- 中文与学术场景的持续优化。
- 一站式安装与运行体验。

模型策略采用混合路线：官方维护核心模型 adapter 和精选第三方 adapter，社区 adapter 独立演进。官方不提供商业模型 plugin，但允许用户自带 API key 实现 adapter。

## 6. 阶段优先级

### P0: 本地能力中心 MVP + 远程入口

P0 目标是让 Agent 和开发者可以稳定地把 MinerU 当作文档入口使用。

交付物：

- `mineru` / `mineru-kit` 双命令。
- Skill 配套发布。
- Agent-native 输出能力。
- `--remote` 一键切换远程高质量解析。
- 统一 REST API 在 `mineru.net/api` 服务端落地。
- Pipeline / VLM / Office middle_json 对齐。
- SQLite + SHA256 文档库缓存。
- 完整 watch、rules、search 主链路。
- Telemetry 设计，包括字段、开关、默认值、隐私边界和上报时机。
- Privacy First 首次选择与匿名统计机制。

验收方向：

- Agent 真实任务成功率。
- middle_json schema validation。
- page/block locator 跨进程稳定性。
- 相同输入二次 parse 缓存命中响应。
- watch 能发现文件、rules 能触发解析策略、search 能返回可用结果。
- telemetry 字段不泄露文档内容。
- telemetry 可以不作为公开 `docs/telemetry.md` 发布，但 P0 内部设计必须完成。

### P1: Server、SaaS 提速、桌面端

P1 在 P0 的本地能力中心上扩展完整体验：

- Server 扩展 Web UI 和 MCP Server。
- SaaS 推理速度优化，包括小模型替换、分辨率策略和端到端延迟优化。
- `MinerU.app/MinerU.exe` 桌面客户端。

### P2: 生态完善

P2 聚焦生态适配：

- LangChain document loader。
- LlamaIndex reader。
- 更多社区 adapter 和集成示例。

## 与其他文档的关系

- 架构落地见 [系统架构](architecture.md)。
- CLI 入口见 [CLI 规格](cli.md)。
- 云端和自部署 API 见 [Unified API](api.md)。
- SDK 分层见 [SDK 设计](sdk.md)。
- 解析档位见 [解析 Tier](tiers.md)。
- middle_json 统一方向见 [Middle JSON](middle-json.md)。
- 错误码与修复建议见 [错误码体系](errors.md)。
- 配置优先级见 [配置体系](config.md)。

## 未决问题

Roadmap、评测、其他公开范围和 P0/P1/P2 边界相关问题，集中维护在 [开放问题清单](open-questions.md)。
