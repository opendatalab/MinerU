# 不变量红线

状态: Draft
读者: 核心开发、编程 Agent、代码审查者
范围: Agent / 人类改动**永远不允许违反**的硬约束
来源: MINERU-1 代码结构分析报告对照代码复核后固化；行号以 commit `317428b8` 为基准。

违反任何一条都视为回归。改动触及这些位置时，PR 描述必须说明不破坏对应不变量。

## 1. 隐私边界：本地解析失败绝不静默回退 remote

文档内容默认全本地处理；只有显式 `--remote`（CLI）或 `privacy="remote"` 才把 PDF/图片发往远端 parse-server。

- `privacy` 二值赋值：`mineru/doclib/services/parse_svc.py:754`
- 门控集合：remote 仅支持 PDF/图片（`remote_unsupported_for_file_type` `parse_svc.py:772`）、remote 禁 flash（`tier_unsupported_for_remote` `:778`）、质量 tier 仅限 PDF/图片（`tier_unsupported_for_file_type` `:784`）

红线：解析失败、引擎不可用、超时等任何情况下，都不得把用户文档自动发往远端。失败路径必须原样报错（如 `no_engine`，`parse_svc.py:1239`），由用户显式改用 `--remote`。

## 2. 错误码语义稳定

`mineru/errors.py` 定义了结构化错误协议：code → type 映射（`_ERROR_TYPE_MAP` `errors.py:20`）、code → HTTP 覆盖表（`_ERROR_CODE_STATUS_MAP` `errors.py:119`）、响应构造（`error_response` `errors.py:208`）。

红线：

- 不得改变既有 error code 的语义或取值集合的成员含义；已发布的 code 视为对外契约。
- 不得随意改动 code → type 与 code → HTTP 的映射关系；新增 code 必须同时登记进两张表。
- 响应外层结构（OpenAI 兼容 `{type, code, message, param}`）不得变更；扩展字段见 `docs/next/errors.md` 与 `docs/next/pitfalls.md` 第 3 条。

## 3. Middle JSON schema 2.0 输出契约（ADR-0020）

Middle JSON 的对外 schema 固定为 `2.0`（`mineru/parser/base.py:20` `MIDDLE_JSON_SCHEMA_VERSION`）；稳定的是对外 JSON / Pydantic 模型层，不是 SQLite 表结构（ADR-0020：`docs/next/decisions/0020-doclib-schema-stability-boundary.md`）。

红线：

- 不输出 `_backend` / `_version_name` / `pdf_info` / `_ocr_enable` / `_vlm_ocr_enable` 等旧字段；旧字段只能作为受支持旧 payload 的迁移输入出现（legacy 读取路径见 `parser/base.py:20-47` 与 `backend/postprocess/legacy_schema_adapter.py`）。
- 改动输出结构即破坏性变更：必须走 ADR 流程，并同步 `docs/next/api/changes.md`、`docs/next/middle-json.md` 与 compaction（`doclib/background/compaction.py:27`）。

## 4. 依赖方向：基础层不得反向引用上层

稳定依赖方向为 `utils/types → model → backend → render → parser/kit/doclib`，禁止上层模块被基础层反向引用（事实源：`CLAUDE.md` "5.1 目录职责"，此处引用并保持单一事实源）。

配套规则：mineru 子模块之间只用 relative import。

## 5. 定位符协议语法向后兼容

内容定位符语法 `doc:{short_id}/tier:{tier}/page:{n}[/block:{n}[/char:{n}]]` 定义于 `mineru/doclib/locators.py:10-16` 的 `_CURSOR_RE`；构造器 `page_ref` / `block_ref` / `block_char_ref` 在 `locators.py:33-46`。

红线：

- 已发布格式的 locator（含持久化在缓存、续读标记 `<!-- Next: mineru read … -->` 中的）必须永久可解析；扩展语法只能以追加段的方式演进，不得改变既有段语义。
- `short_id` 取 sha256 十六进制前缀、冲突时延长（`doclib/services/parse_svc.py:247-282`），不得改换派生方式。
