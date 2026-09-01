# 开发陷阱清单

状态: Draft
读者: 核心开发、编程 Agent、代码审查者
范围: 4.x（next 分支）开发中已知的结构性陷阱与兼容面联动要求
来源: MINERU-1 代码结构分析报告（SHANNON，2026-09-01），逐条对照代码复核后固化。行号以 commit `317428b8` 为基准。

本清单收录的是"改动前必须知道，否则容易做错"的事项。每条给出代码锚点与正确做法；纯推测、无法操作的观察不收录。

## 1. 两个大文件不再堆功能

- `mineru/parser/api_server.py`（2495 行，parse-server，"MinerU v1 REST API"）
- `mineru/doclib/server.py`（2439 行，doclib 常驻服务端）

这两个文件已是仓库中最大的单文件。新增功能不要继续堆入，按现有分层放入对应模块：

- parse-server：路由 / 服务逻辑分层放置，与 `doclib/services/` → `doclib/core/` 的分层方式对齐。
- doclib：路由进 `server.py` 的路由层，业务进 `doclib/services/`（如 `parse_svc.py`、`search_svc.py`），存储与 IO 进 `doclib/core/`（`db.py`、`fts.py`、`file_io.py`）。

仓库 `tests/unittest/` 下约 2372 个测试函数是拆分与重构的安全网。

## 2. 双轨兼容面的联动要求

### 2.1 tier / backend 双入参过渡层

`parser/tier.py:230` 的 `resolve_tier_and_backend(tier, backend)` 是为旧专家用户保留的过渡层：同时接受公开 `tier` 和本地专家 `backend` 入参，并做兼容性校验（`_backend_supports_tier`）。

- 新代码只走 `tier`；`backend_for_tier`（`parser/tier.py:183`）已保证 basic/standard/advanced 共用同一个 hybrid-engine，仅 effort 不同（`HYBRID_EFFORT_BY_TIER` `parser/tier.py:25`）。
- 不要新增依赖 `backend` 入参的分支；`pipeline`/`vlm-*` 仅作为 legacy 别名存在。

### 2.2 Middle JSON legacy 读取三件套

改 Middle JSON 结构时必须同步检查三处，否则旧缓存文档读取会断：

1. legacy 读取：`mineru/parser/base.py:20-47` —— 识别 3.4.5 `pdf_info` 与 schema 1.0 `pages` 包装（`_legacy_raw_pages`），`MIDDLE_JSON_SCHEMA_VERSION = "2.0"`；
2. legacy 适配：`mineru/backend/postprocess/legacy_schema_adapter.py` —— 旧 payload 单向回推为 raw model-list；
3. 批次合并：`mineru/doclib/background/compaction.py:27-32` —— 仅合并同 schema 批次。

新增或调整 Middle JSON 字段时，先确认 ADR-0020（schema stability boundary）是否覆盖该字段，再同步上述三处与 `docs/next/middle-json.md`。

## 3. Alpha 高频迭代，公开 API 未稳定

4.x 处于 Alpha prerelease，迭代节奏约为日均 10 个提交，公开 API 未稳定（变更专门记录在 `docs/next/api/changes.md`）。

- 外部集成应 pin 具体 commit，不要跟浮动的 next HEAD。
- 内部改动涉及 NEXT v1 API 行为变化时，必须在 `docs/next/api/changes.md` 追加条目。

## 4. 错误协议：设计已定、实现缺位的部分

`docs/next/errors.md` 已设计 `user_action` / `retryable` / `docs_url` 扩展字段，但 `mineru/errors.py` 的 `error_response`（`errors.py:208`）目前只输出 OpenAI 兼容的 `{type, code, message, param}`，全仓无 Python 实现点。

- 依赖这些扩展字段的调用方代码不要提前编写。
- 补齐该缺口是自然切入点：改 `errors.py` 的 `error_response` 与 CLI JSON 输出，并同步 `docs/next/errors.md` 状态。

## 5. 仓库约定速查

两条来自 `AGENTS.md` / `CLAUDE.md` 的硬约定，最容易在自动生成提交时违反：

- mineru 子模块之间只使用 **relative import**（`from .base import X`），不用 `import mineru.xxx` 绝对导入。
- commit message / PR body 禁用 `fixes|closes|resolves #n`（会在 merge 后自动关闭 issue），用 `Refs #n`。
