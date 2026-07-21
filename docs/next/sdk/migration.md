# 迁移路径

状态: Draft
读者: 核心开发者、SDK 开发者
范围: 从当前 SDK 代码走向目标公开契约的步骤
来源: 由旧 SDK 底稿迁移整理而来；旧底稿已归档删除

## 目标

SDK 迁移的目标不是重写现有 parser，而是稳定已经成形的边界:

- `mineru.parser` 成为 Tool SDK 主入口。
- `MinerUApiParser` 成为 v1 API-backed parser。
- `DoclibClient` 成为本地 doclib SDK。
- `ParseResult` 成为跨 backend 和 API 的统一结果对象。

## 当前状态

| 主题 | 当前状态 |
|------|----------|
| Tool SDK 导出 | 已有 `mineru.parser.__all__`。 |
| `DocumentParser` | 已有 sync/async/batch/context manager 接口。 |
| `ParseResult` | 已有输出方法，`from_dict()` / `from_json()` 已实现；仍需稳定 envelope 兼容边界。 |
| Tier 参数 | `parse()` 当前主要使用 `backend`，tier 语义分散在 doclib/service 中。 |
| API-backed parser | 已有 `MinerUApiParser`，使用 v1 uploads/jobs/files。 |
| parse-server runtime | 已有 `mineru.parser.api_server`。 |
| Doclib SDK | 已有 `DoclibClient`，继承 `DoclibInterface` 并返回 typed Pydantic response。 |
| 错误模型 | 已有 `mineru.errors`；仍需确认 SDK exception 是否暴露 `user_action` 等扩展字段。 |

## Phase 1: 文档和命名稳定

1. 固化 SDK 分层文档。
2. 确认 public import path:
   - `from mineru.parser import parse, ParseResult`
   - `from mineru.parser import MinerUApiParser`
   - `from mineru.doclib.client import DoclibClient`
3. 决定是否增加:
   - `from mineru.client import DoclibClient`

不新增 `from mineru.parser.remote import MinerUApiParser`；API-backed parser 保持现有入口。

验收:

- 文档中每个公开入口都有用途、参数和错误说明。
- CLI 文档与 SDK 文档的入口名一致。

## Phase 2: `ParseResult` 补完

1. 锁定 `ParseResult.from_dict()` / `from_json()` 对历史 payload 的兼容范围。
2. 稳定 `to_dict()` 输出 envelope。
3. 定义 writer protocol。
4. 统一 `save()` 的文件命名。

验收:

- 本地 parser result 可以 round-trip: `to_json()` -> `from_json()`。
- API `middle_json` output 可以恢复为 `ParseResult`。
- Office/PDF/HTML 的 markdown 和 content_list 输出不回退。

## Phase 3: Tier 进入 Tool SDK

1. 在 `parse()` 增加 `tier` 参数。
2. 保留 `backend` 作为高级兼容参数。
3. 明确 `backend` 覆盖 `tier`。
4. 增加 `tier=None` 默认选择逻辑：PDF/image 永不回退到 `flash`，Office/HTML 这类仅支持 flash tier 的输入按实际能力归一为 `flash`。

验收:

- `parse(path, tier="basic")` 可用。
- `parse(path, tier="standard")` 可用或给出明确 engine error。
- PDF/image 的 `parse(path)` 或 `parse(path, tier=None)` 不会静默使用 flash；Office/HTML 这类仅支持 flash tier 的输入未指定 tier 时返回实际 `flash` 语义。
- `parse(path, tier="flash")` 只有显式请求时使用 flash。

## Phase 4: 错误模型统一

1. 将 `MinerUApiParser` 的内部 `_V1APIError` 映射到 `MineruError`。
2. 补齐 API/tier 错误码:
   - `unsupported_parameter`
   - `invalid_request`
   - `quality_tier_unavailable`
3. 为 doclib client 保留 error envelope -> exception 的一致行为。

验收:

- API-backed parser、doclib client、本地 parser 的错误都能被统一捕获。
- CLI 可以根据 `code` 给出 user action。

## Phase 5: Doclib Client 类型化

1. 保持 `mineru.doclib.types` 作为 client 返回类型来源。
2. 评估是否为 `DoclibClient` 提供高层 `parse(...)` convenience wrapper；当前稳定入口是 `ensure_parse(ParseRequest)`。
3. 增加 context manager。
4. 评估 `AsyncDoclibClient`。

验收:

- `client.ensure_parse(ParseRequest(...))` 返回稳定 `ParseResponse`。
- `client.search()` 返回稳定 `SearchResponse`。
- `client.get_file_by_path()` 返回稳定 `FileInfoResponse`。
- 现有 CLI typed model 调用不破坏。

## Phase 6: 测试约束

需要覆盖:

- Import 不加载 heavy backend。
- `parse()` 后缀分派。
- `ParseResult` round-trip。
- `MinerUApiParser` upload/job/files mock。
- `DoclibClient` UDS error 映射。
- 默认选择不回退 `flash`。

## 未决问题

共享 schema 模块、低层 v1 client 和 Product SDK 覆盖范围，集中维护在 [开放问题清单](../open-questions.md)。
