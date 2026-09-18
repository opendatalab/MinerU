# 开发陷阱清单

状态: Draft
读者: 核心开发、编程 Agent、代码审查者
范围: 4.x 开发中已知的结构性陷阱与兼容面联动要求
来源: MINERU-1 代码结构分析报告（SHANNON，2026-09-01），逐条对照代码复核后固化。行号以 commit `317428b8` 为基准。

本清单收录的是"改动前必须知道，否则容易做错"的事项。每条给出代码锚点与正确做法；纯推测、无法操作的观察不收录。

## 1. 双轨兼容面的联动要求

### 1.1 backend 不再是入参

历史过渡层 `resolve_tier_and_backend(tier, backend)` 与 `mineru-kit parse --backend` 专家参数（含 `pipeline`/`vlm-*` 等别名）已移除：backend 只用于推断 tier，推断后即被丢弃，没有独立作用。`backend_for_tier` 派生链与 `ParserRuntimeOptions.backend` 内部字段也已删除。

- 新代码只走 `tier`；basic/standard/advanced 共用同一个 hybrid-engine，仅 effort 不同（`HYBRID_EFFORT_BY_TIER`，`parser/tier.py` 的 `ParserRuntimeOptions(tier, effort)`）。
- 不要在任何 CLI / SDK / API 入口重新引入 `backend` 入参，也不要恢复 backend 派生函数或内部字段。

### 1.2 共享文档协议与缓存必须同步

改外层结构时同步 DocVortex schema/readers、MinerU ParseResult 和 Doclib 缓存。
当前使用 schema 身份 + 版本 2.0 + metadata，通用接口拒绝旧协议；不能仅按版本号识别。
Doclib 缓存命中、覆盖、默认档位与压缩均使用专属 read_cached_middle_json，兼容转换旧 3.4.5 和 MinerU 2.0；压缩只合并转换后元数据一致的批次。
详见 [envelope.md](middle-json/envelope.md)。

## 2. Alpha 高频迭代，公开 API 未稳定

4.x 处于 Alpha prerelease，迭代节奏约为日均 10 个提交，公开 API 未稳定（变更专门记录在 `docs/next/api/changes.md`）。

- 外部集成应 pin 具体 commit，不要跟浮动的 next HEAD。
- 内部改动涉及 NEXT v1 API 行为变化时，必须在 `docs/next/api/changes.md` 追加条目。

## 3. 错误协议：设计已定、实现缺位的部分

`docs/next/errors.md` 已设计 `user_action` / `retryable` / `docs_url` 扩展字段，但 `mineru/errors.py` 的 `error_response`（`errors.py:208`）目前只输出 OpenAI 兼容的 `{type, code, message, param}`，全仓无 Python 实现点。

- 依赖这些扩展字段的调用方代码不要提前编写。
- 补齐该缺口是自然切入点：改 `errors.py` 的 `error_response` 与 CLI JSON 输出，并同步 `docs/next/errors.md` 状态。

## 4. 仓库约定速查

两条来自 `AGENTS.md` / `CLAUDE.md` 的硬约定，最容易在自动生成提交时违反：

- mineru 子模块之间只使用 **relative import**（`from .base import X`），不用 `import mineru.xxx` 绝对导入。
- commit message / PR body 禁用 `fixes|closes|resolves #n`（会在 merge 后自动关闭 issue），用 `Refs #n`。
