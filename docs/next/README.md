# Next MinerU 文档体系

状态: Draft
读者: 项目核心开发者、SDK/API 设计参与者、需要理解下一代能力边界的外部开发者
范围: 组织下一阶段 MinerU 的产品方向、系统架构、公开接口和关键内部规范
非目标: 替代根目录 `NEXT-*.md` 底稿；立即改写现有正式用户文档

## 文档定位

`docs/next/` 是下一代 MinerU 设计文档的体系化入口。根目录 `NEXT-*.md` 保持原样，作为历史底稿和细节来源；本目录先负责建立导航、边界和读者路径。

当前阶段的事实来源仍是根目录底稿。后续当某个主题完成审阅后，可以把对应内容从底稿提升到本目录，成为稳定设计文档。

## 推荐阅读顺序

1. [产品路线图](roadmap.md): 先理解产品定位、边界和优先级。
2. [术语表](glossary.md): 统一 tier、backend、parser、doclib 和 parse-server 等核心词。
3. [端到端工作流](workflows.md): 串起 watch、ingest、parse、cache、search、API、SDK 和 Agent 读取流程。
4. [实现计划](implementation-plan.md): 面向编程 Agent 的细粒度任务拆分、边界和验收。
5. [系统架构](architecture.md): 再理解本地文档库、服务、任务和存储模型。
6. [解析 Tier](tiers.md): 理解默认选择、`flash`、`standard`、`pro` 的质量、速度、硬件和隐私边界。
7. [CLI 规格](cli.md): 理解本地入口和批处理体验。
8. [Unified API](api.md): 理解云端和自部署 API 的统一模型。
9. [SDK 设计](sdk.md): 理解 Tool SDK 与 Product SDK 的分层。
10. [Middle JSON](middle-json.md): 理解跨 backend 的中间结构统一方向。
11. [错误码体系](errors.md): 理解错误响应和本地错误分类。
12. [配置体系](config.md): 收敛本地、远端和解析参数的配置规则。
13. [Telemetry 设计](telemetry.md): 理解匿名使用统计、稳定性指标和粗粒度环境画像。
14. [开放问题清单](open-questions.md): 查看仍未定稿、需要进一步决策的问题。
15. [设计决策记录](decisions/README.md): 查看已定稿的关键决策。

## 文档地图

| 主题 | 新文档 | 底稿 | 主要读者 |
|------|--------|------|----------|
| 产品路线 | [roadmap.md](roadmap.md) | `../../NEXT-ROADMAP.md` | 产品、核心开发 |
| 术语统一 | [glossary.md](glossary.md) | 用户讨论 / 各底稿 | 核心开发、文档作者、SDK/API/CLI 设计参与者 |
| 端到端工作流 | [workflows.md](workflows.md) | 用户讨论 / 各专题文档 | 核心开发、Agent 能力开发、SDK/API/CLI 设计参与者 |
| 实现计划 | [implementation-plan.md](implementation-plan.md) | `docs/next/` 全部专题文档 | 编程 Agent、核心开发、代码审查者 |
| 系统架构 | [architecture.md](architecture.md) | `../../NEXT-DESIGN.md` | 核心开发 |
| 解析 Tier | [tiers.md](tiers.md) | 用户讨论 / `../../NEXT-ROADMAP.md` | CLI/API/SDK 使用者、核心开发 |
| Unified API | [api.md](api.md)、[api/](api/README.md) | `../../NEXT-API.md` | API 使用者、服务端开发 |
| CLI | [cli.md](cli.md) | `../../NEXT-CLI.md` | CLI 用户、核心开发 |
| SDK | [sdk.md](sdk.md)、[sdk/](sdk/README.md) | `../../NEXT-SDK.md` | SDK 使用者、核心开发 |
| Middle JSON | [middle-json.md](middle-json.md)、[middle-json/](middle-json/README.md) | `../../NEXT-JSON.md` | backend 开发、内容输出开发、Agent 能力开发 |
| 错误码 | [errors.md](errors.md) | `../../NEXT-ERROR.md` | API/CLI/SDK 开发 |
| 配置 | [config.md](config.md) | `../../NEXT-CFG.md` | CLI、服务端、SDK 开发 |
| Telemetry | [telemetry.md](telemetry.md) | 用户讨论 / `../../NEXT-ROADMAP.md` | 产品、核心开发、数据分析、合规参与者 |
| 开放问题 | [open-questions.md](open-questions.md) | 各专题文档、用户讨论 | 核心开发、编程 Agent、文档维护者 |

## 整理原则

- 先保留底稿，再逐步提升为稳定文档。
- 每篇文档只回答一个层级的问题，避免路线图、规格和实现细节互相混杂。
- 对外接口文档优先描述可观察行为；内部设计文档优先描述模块职责、数据流和约束。
- 已决定的内容进入正文；未决定的内容集中放在 [开放问题清单](open-questions.md)。
- 重大选择进入 `decisions/`，避免在长文中丢失决策背景。
