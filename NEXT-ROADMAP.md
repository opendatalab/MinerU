# MinerU 产品路线图

## 1. 产品定位

MinerU = **Agent 时代的文档入口**

MinerU 提供四层产品界面：

| 界面 | 形态 | 面向 |
|------|------|------|
| mineru.net | Web | 路人体验、用户注册与 API Key 管理 |
| mineru.net/api | SaaS API | 开发者，作为 `mineru CLI`/MCP 的远程后端 |
| `MinerU.app/MinerU.exe` | 桌面客户端 | 普通用户，GUI 操作文档 |
| `mineru` | PyPI 包 | 开发者/Agent，本地文档能力中心 |

`mineru`（`pip install mineru`）是核心，集 CLI、Python SDK、Server、UI、MCP Server 于一体。`MinerU.app/MinerU.exe` 是其 GUI 壳。Agent 不直接调用 mineru.net/api，通过 `mineru CLI` 或 MCP Server 间接使用。

```
路人              普通用户                       开发者 / Agent
  │                  │                              │
  │                  ├──► Agent + Skill ────────────┤
  │                  │   (Skill 引导 uv 安装)        │
  │                  │                              │
  │                  └──► MinerU.app/MinerU.exe ────┤
  │                       (桌面 GUI 壳)             │
  ▼                                                 ▼
mineru.net                              mineru (pip / uv install)
                                                    │
                                                    ├── 本地引擎（默认，Privacy First）
                                                    │     CLI / Python SDK / Server / UI / MCP Server
                                                    │
                                                    └── mineru.net/api（--remote 显式指定）
                                                            远程高质量解析
```

### 1.1 命名规则

| 类别 | 规则 | 示例 |
|------|------|------|
| 品牌 / 产品名 | CamelCase | MinerU、`MinerU.app/MinerU.exe` |
| 技术标识符 | lowercase | 包名 `mineru`、命令 `mineru`、`mineru SDK`、域名 `mineru.net` |
| 子产品命名 | MinerU + 形态词 | `MinerU.app/MinerU.exe`（桌面端） |

### 1.2 边界（不做）

为坚守"做入口"原则，以下能力 MinerU 不内建。

#### 功能边界（交给生态层）

| 不做 | 交给谁 |
|------|--------|
| 文档 chunking 策略 | LangChain / LlamaIndex |
| embedding / 向量化 | 向量库 |
| chat with PDF | Agent |
| RAG pipeline | 生态层 |
| 内容 summary | 下游 LLM |
| 向量库 | 生态层 |
| Agent 框架 | 生态层 |
| prompt 模板 | 生态层 |

#### 商业边界（不做交付）

| 不做 | 说明 |
|------|------|
| 企业版商业产品（Enterprise tier） | — |
| 带 SLA 的私有化 / on-premise 商业部署 | — |
| 合规认证（SOC2 / HIPAA / GDPR / 等保 等） | — |
| 企业专属技术支持与定制 | — |

注：开源 MinerU 可本地部署，用户可在自有基础设施运行；此处"不做"指官方不提供商业化的企业交付通道。

## 2. 设计目标

### 2.1 Privacy First（隐私优先）

Privacy First 收窄到两件事，分别处理：

- **文档隐私（绝对）**：文档内容绝不离开本地；仅用户显式指定 `--remote` 时上传，且有明确提示（服务端数据政策见 6.4 节）
- **使用统计（透明可控）**：分级匿名遥测，不含文档内容 / 文件名 / 路径；首次启动强制用户选择不预选；用户可随时通过 `mineru telemetry` 命令审阅与切换

#### 遥测分级

| Tier | 内容 | 默认 |
|------|------|------|
| 关闭 | 无 | — |
| 基础 | install 数、版本、OS、backend 选择、错误码、文件类型、页数分桶、调用源（CLI / Skill / MCP / SDK） | CLI / GUI 强制选择；SDK / Server / CI 默认关 |
| 扩展 | + 解析耗时分桶、模型版本、命令组合模式 | opt-in |
| 反馈 | 用户主动发送错误样例（明确剥离内容） | 主动触发 |

#### 场景覆盖

| 场景 | 默认 |
|------|------|
| CLI / GUI 首次启动 | 强制选择对话框（不预选） |
| SDK / Server / CI（无 tty）| 默认关闭，env 显式开启 |
| Agent 调用 | 跟随宿主 CLI 设置 |

完整字段清单与上报示例待 P0 阶段随改进计划同步发布于 `docs/telemetry.md`。

### 2.2 For Agent

- 一切界面以 Agent 使用场景为设计原点
- 结构化输出、确定性行为、无歧义的错误码
- Agent 通过 Skill（CLI 通道）与 MCP Server（协议通道）使用 MinerU，详见第 4 章；具体交付节奏见第 8 章

### 2.3 易于使用

- 开发者一键安装：`pip install mineru`
- 普通用户零配置：通过 Agent + Skill 使用，Skill 内嵌 uv 安装引导，无需自己装 pip
- 包体小、依赖精简
- 本地模型数量少、Size 小
- `--remote` 一键访问远程高质量解析

### 2.4 SaaS 极致质量

- SaaS 只提供最高质量、最快速度的解析
- 具体改进方向与措施见 6.1 节

### 2.5 数据驱动迭代

基于 2.1 节遥测数据，建立三层指标体系：北极星 + 健康度面板 + 项目验收。

#### 北极星指标

**WAI（Weekly Active Installs）** — 每周至少完成 1 次成功 parse 的 install 数。直接反映"Agent 时代的文档入口"的规模。

#### 健康度面板

| 维度 | 指标 | 数据来源 |
|------|------|---------|
| 采纳 | 累计 install + WAI | install_uuid + startup 事件 |
| 使用质量 | 解析成功率（按 backend / 文件类型分组） | parse.result + backend + file_format |
| 能力分布 | backend 选择分布（pipeline / vlm / hybrid 占比） | parse.backend |
| 错误信号 | Top 5 错误码周分布 | parse.error_code |
| 留存 | install 4 周后回访率 | install_uuid 跨周存在 |
| 渠道 | Agent 调用占比（Skill / MCP / SDK / CLI） | parse.source |

#### Review 节奏

- P0 攻坚期：双周
- 产品稳定期：月度

#### 公开策略

- **公开**：累计 install、累计 parse 数（入口叙事载体）
- **不公开**：留存率、错误率等内部指标（避免早期数据被误读）

#### 不指标化（避免"为指标而做"）

下列重要但不进 dashboard：

- 单次复杂文档的解析质量 — 靠人工评测
- 社区情绪 / PR review 风评
- 内部代码质量 / 技术债

#### 项目验收指标

每个 P0 项目上线前必须定义成功标准。当前 P0 项目验收指标基线：

| 项目 | 验收指标 |
|------|---------|
| Skill 决策树 | eval 集 100 题 Agent 首选正确率 ≥85% |
| 中间结构统一 | 三 backend 通过统一 schema validation；同输入 page/block locator 跨进程一致 |
| 文档库 SHA256 缓存 | 二次 parse 相同输入 P95 响应 <100ms；缓存命中率 ≥60% |
| Agent-native（4.2 七条）| 端到端 5 个 Agent 真实任务场景成功率 ≥80% |
| 改进计划 | CLI / GUI 首次启动 Tier 1 opt-in 率 ≥40% |
| Privacy First | 遥测字段经第三方审阅无文档内容泄漏 |

基础设施类项目（`mineru` CLI 主命令、`--remote` 远程切换、统一 REST API 等）以**功能完成 + 集成测试通过**为验收，不单列定量指标。

## 3. 团队工作原则

### 3.1 拥抱 Vibe Coding

代码组织架构利于 AI 理解与修改，使用 AI 加速产品开发。

### 3.2 专注极致

一段时间内只做一件事，并把它做到极致。

## 4. Agent 通道设计

### 4.1 双通道

Agent 不直接调用 mineru.net/api。双通道设计（Skill 指 Claude Code / Codex 等 Agent 的扩展机制，下同）：

```
Agent
  │
  ├── Skill（推荐）► mineru CLI ─┬── 本地引擎（默认）
  │                              └── mineru.net/api（--remote）
  │
  └── MCP Server（高级）► mineru SDK ─┬── 本地引擎（默认）
                                      └── mineru.net/api（--remote）
```

| 通道 | 协议 | 定位 | 适用场景 |
|------|------|------|---------|
| CLI + Skill | shell 命令 | **推荐** | Agent 直接操作本地文件系统，如 Claude Code、Copilot CLI |
| MCP Server | MCP 协议 | 备选通道 | 需要跨平台标准协议或远程 Agent 调用的场景 |

- **Skill + CLI 是推荐方案**：Agent 读取 Skill 中内嵌的 CLI 使用说明，理解参数语义后自主组合命令
- **MCP Server 作为备选通道**：需要标准工具发现机制或跨网络调用时启用
- 两条通道底层共享同一 `mineru SDK`，本地/远程逻辑完全一致

### 4.2 Agent-native 特性

针对 Agent（Claude Code、Codex 等）使用场景重新设计的解析能力，与面向人类用户的传统 CLI/SDK 体验区分开。P0 优先通过 CLI 暴露，P1 通过 MCP Server 补齐标准协议入口。

| 特性 | 说明 |
|------|------|
| 结构化 message contract | 所有 CLI 与 MCP 出口遵循统一结构化 schema，替代 free-text 文本输出，Agent 无需文本解析即可消费 |
| 确定性错误码 | typed error code 替代 free-text 错误信息，每个错误自带修复建议（如 `E_OCR_REQUIRED` 提示加 `--ocr`），Agent 可基于错误码做分支处理与自动重试 |
| token-budgeted 输出 | 解析支持指定 token 上限，自动按页边界截断并返回 continuation 引导，避免长文档撑爆 Agent context；内置 tokenizer 开箱即用 |
| 可寻址 page/block locator | 每个可引用 block 输出稳定 locator（含 tier / page / block），支持按 locator 反查，Agent 答案天然可 citation，用户可回溯原文 |
| 区域/页面查询 | 解析阶段支持按页码与 block/content 类型过滤，只取所需内容，避免全文解析的 token 与时延浪费 |
| Skill 决策树 | Skill 内嵌三类映射："环境检测 → 引导安装（uv install mineru）"、"用户场景 → 命令路由"、"错误码 → 修复动作"。支持普通用户零配置首次使用，降低 Agent 参数选择错误率，配 eval 集回归 |
| SHA256 内容寻址缓存 | `mineru` 内置文档库提供 SHA256 去重缓存，Agent 相同输入二次调用毫秒级返回 |

#### 依赖

本节多数特性依赖 5.3 节中间结构统一的最小子集（稳定 page/block locator + page + bbox + type）。

不做的能力（chunking / embedding / chat / RAG / summary 等）见 1.2 节。

## 5. 本地能力中心

`pip install mineru` 安装的是本地文档解析与管理的能力中心，包含：

```
mineru
├── CLI         — mineru / mineru-kit 命令行工具
├── Python SDK  — 编程调用接口
└── Server      — 本地后台服务（含 Web UI 和 MCP Server，P1 阶段交付）
```

> 注：NEXT-CLI.md 覆盖 CLI + Server 的详细设计；Python SDK 设计见 mineru/api 模块；Web UI 和 MCP Server 为 P1 交付，作为 Server 的扩展能力集成。

### 5.1 两个 CLI 命令分工

| 命令 | 面向 | 定位 |
|------|------|------|
| `mineru` | 普通用户 / Agent | 文档管理中心 — 在 `mineru-kit` 之上叠加本地数据库、去重、搜索、与桌面端互通。**Agent 默认通道，Skill 仅介绍此命令** |
| `mineru-kit` | 解析内核 / 批处理开发者 | 纯解析工具 — 无状态、不建索引。子命令：`parse`（文件/目录解析）、`api-server`（端到端解析服务，兼容 SaaS API）、`vlm-server`（本地 VLM 服务，兼容 OpenAI API）。**Skill 不暴露此命令** |

### 5.2 本地能力矩阵

| 能力 | 优先级 | 说明 |
|------|--------|------|
| 解析引擎 | P0 | 三档 tier：`flash`（轻量 CPU）/ `medium`（Pipeline）/ `high`（VLM）。用户通过 `--tier` 选择档位，引擎名不暴露 |
| 远程切换 | P0 | `--remote` 一键访问 mineru.net/api |
| 文档库 | P0 | SQLite + SHA256 去重，Agent-native 缓存的底层依赖 |
| Server | P0(基础) / P1(完整) | 本地后台进程，提供 Watch / 解析队列 / 搜索索引。P1 阶段扩展 Web UI 和 MCP Server |
| 搜索 | P1 | 按文件名/内容检索 |
| 桌面端 | P1 | `MinerU.app/MinerU.exe` — CLI 的 GUI 壳 |

### 5.3 中间结构统一

将 Pipeline / VLM / Office 三个 backend 的 `pdf_info` 中间结构收敛到统一规范，消除三套 `union_make` 的重复代码。

#### 当前差异

| 差异项 | 现状 |
|--------|------|
| BBox 缺失 | Office backend 无 bbox |
| Block Type 差异 | VLM 和 Pipeline 各有独有类型 |
| Span 粒度不同 | VLM 单 span 整块，Pipeline 多 span 逐字 |
| page_info 结构不统一 | 各有各的构建流程 |
| page_size 缺失 | Office 无此字段 |

#### 统一路径

1. 定义规范 middle_json schema
2. 各 backend 输出对齐到统一结构
3. 合并为单套 `union_make` 逻辑

## 6. SaaS 能力

定位：只提供最高质量、最快速度的远程解析。

### 6.1 核心改进方向

| 方向 | 具体措施 |
|------|---------|
| 推理速度 | 端到端延迟优化 |
| 小模型替换 | Layout/OCR 步骤使用更小模型，质量不降 |
| 图像分辨率 | 尝试降低输入分辨率，减少处理时间 |

### 6.2 免费配额分级

当前 SaaS 解析对所有用户免费，按用户是否在 mineru.net 官网注册并提供 API Key 区分配额，不做付费版能力分层。

| 维度 | 匿名用户 | 注册用户 |
|------|---------|---------|
| 身份识别 | IP | API Key |
| 配额机制 | 按 IP 限速，较低基础配额 | 按 API Key 限速，更高配额 |

费用、解析质量、文件大小、导出格式（Markdown / JSON / Docx / LaTeX / HTML / ZIP 含 images）匿名用户与注册用户完全一致，仅配额机制不同。

### 6.3 访问方式

SaaS 通过统一 REST API 暴露能力，API 面向开发者和 `mineru CLI`/MCP 的远程后端；Agent 推荐经 CLI/MCP 调用。

### 6.4 服务端数据政策

`--remote` 上传到 mineru.net/api 的数据按以下策略处理。

#### 默认行为

| 数据类型 | 留存期 | 用户访问 | 用于训练 |
|---------|--------|---------|---------|
| 上传文件 | 30 天 | — | 否 |
| 解析结果 | 30 天 | 30 天内可下载 | 否 |
| 匿名 metadata | 长期 | — | 否（仅用于产品分析与聚合统计，完全脱敏不关联账号）|

留存期满后用户文件与解析结果自动删除。

#### 用户可选 opt-in

为不堵死数据飞轮，提供两条独立可控通道，默认关闭：

- **训练授权**：用户在 mineru.net 用户中心主动开启，允许 MinerU 使用其解析数据改进模型，换取额外配额 / 优先级队列等激励
- **质量样本捐赠**：解析时单次主动触发"该次效果不佳，捐赠用于改进"，与训练授权独立可控

## 7. 模型集成原则

定义"做入口"具体如何落地——哪些自研、哪些集成第三方——以及模型接入的开放度。

### 7.1 边界三栏

| 类别 | 范围 |
|------|------|
| **自研（核心竞争力）** | **架构层**：① 中间结构规范（见 5.3 节）、② pipeline 编排与多模型融合策略；**产品层**：③ CLI / SDK / Skill、④ Agent-native 特性（见章节 4.2）；**模型层**：⑤ 中文/学术场景的模型 fine-tune、⑥ Layout / OCR 自研模型、⑦ 自研 VLM |
| **集成（不拒绝第三方）** | layout / OCR / 表格 / 公式 / 端到端 VLM 等各类第三方模型，与自研模型并存可切换 |
| **不做（交给生态）** | 见 1.2 节 |

### 7.2 自研模型策略

- **Layout / OCR**：自研持续投入，发挥中文/学术场景数据优势；不拒绝同时集成 PaddleOCR、Surya 等第三方模型作为可选 backend
- **VLM**：自研持续投入。更新频率不强求高频，允许在部分时间窗口被竞品超越，但必须始终保持在第一梯队的桌面上。本地默认与 SaaS 高端可分别选用不同模型

### 7.3 Plugin 开放度

混合策略：

- **核心模型 adapter** 由官方维护，包含自研模型与精选第三方模型（用于补齐自研模型的短板领域，例如多语言 OCR、轻量端到端 VLM 等），保证稳定性与版本兼容
- **第三方模型 adapter** 走社区贡献目录，独立于主仓库维护
- Plugin 接口定义、license 隔离、版本兼容规范由官方制定

### 7.4 商业模型集成

- 官方不提供任何商业模型 plugin
- 允许用户自行实现 adapter，自带 API key 调用
- MinerU 保持模型中立，不与任何商业模型方做绑定或分成

### 7.5 防"套壳"价值锚点

集成第三方模型不等于套壳。MinerU 不可替代的价值在于：

1. **模型编排与融合策略**：多 backend hybrid、多模型互补
2. **中间结构规范**：统一 schema 是生态层的对接标准
3. **Agent-native 体验**：4.2 节列出的 7 项能力
4. **中文 / 学术场景 fine-tune**：OpenDataLab 数据优势
5. **一站式安装与运行**：解决用户实际部署痛点

## 8. 优先级与节奏

```
P0 — 本地能力中心 MVP + 远程入口

交付物
├── mineru CLI（mineru / mineru-kit 双命令）
├── Skill 配套发布
├── Agent-native 7 条特性（见 4.2 节）
├── --remote 一键切换远程高质量解析
├── 统一 REST API 在 mineru-be（mineru.net/api 服务端实现）落地
├── 中间结构统一：Pipeline / VLM / Office middle_json 对齐（Agent-native 特性的前置依赖）
├── 文档库：SQLite + SHA256 去重（Agent-native 缓存的前置依赖）
└── 改进计划（首次启动选择，匿名统计）

贯穿原则
└── Privacy First 默认本地（见 2.1 节）

P1 — 并行推进
├── Server 扩展：Web UI / MCP Server / 搜索
├── SaaS 提速：小模型替换 Layout+OCR / 降分辨率 / 推理加速
└── MinerU.app/MinerU.exe 桌面客户端

P2
└── 生态完善：LangChain document loader / LlamaIndex reader
```

## 9. 后续待办

以下议题在本版路线图中尚未展开，留作后续迭代补充。

| 议题 | 待回答的核心问题 |
|------|----------------|
| 时间维度 | P0 / P1 / P2 的粗粒度交付承诺（如"2026 上半年 P0 完成"） |
| 整体成功标准 | 北极星 WAI 在 1 年 / 2 年 内的目标值 |
| 测试与质量保障 | 解析质量回归框架（候选：OmniDocBench）、CI 策略、用户反馈循环 |
| 非功能性需求 | 性能基线（如 P95 解析 N 页 PDF 时延）、并发上限、可靠性 SLO |
| 国际化策略 | 中 / 英之外的语言支持优先级与排期 |
| 文档与教程投入 | "做入口"产品文档质量是命根，如何系统性投入 |
| 风险与对策 | License（AGPL）、竞品（Docling / Marker / unstructured）、模型 license 兼容性 |
| 竞品对位 | 与 Docling / Marker / unstructured / PyMuPDF4LLM 的差异化定位说明 |
| VLM "第一梯队" 量化 | 7.2 节"保持在第一梯队的桌面上"如何用 benchmark 与 gap 阈值落地 |
