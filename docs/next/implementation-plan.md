# 实现计划与任务拆分

状态: Draft
读者: 编程 Agent、核心开发者、代码审查者
范围: 将 `docs/next/` 中的设计文档拆成可落地、可验证、可并行领取的工程任务
非目标: 替代专题设计文档；一次性重写现有实现；要求所有任务在同一个 PR 完成

## 1. 使用方式

本文件是给编程 Agent 使用的任务手册。Agent 领取任务时，应只做一个任务卡片内的内容，并以该任务的“完成边界”和“验证方式”为准。

任务状态约定:

| 状态 | 含义 |
|------|------|
| `ready` | 需求清楚，可以开始实现。 |
| `blocked` | 需要先完成依赖任务。 |
| `verify-first` | 代码可能已经部分实现；先补测试或确认行为，再决定是否改代码。 |
| `design-needed` | 仍需产品或架构确认，不应直接实现。 |
| `done` | 当前代码和测试已满足完成边界；后续只在回归或重构时触碰。 |

Agent 通用规则:

1. 先读本任务引用的专题文档和代码路径。
2. 如果现有代码已经满足任务，不重写；补测试或补文档锁定行为。
3. 任务只改“建议修改范围”内的文件；发现邻近问题时另开任务。
4. 不改变隐私边界：没有显式 remote，不得上传用户文档。
5. 不改变 tier 语义：默认选择不解析为 `flash`；结果记录实际使用的实体 tier。
6. doclib 的 `parsed/` 目录只持久化 JSON，不落盘 Markdown / Content List / HTML。
7. 新增公开函数必须有类型注解，项目内部 import 使用 relative import。
8. 每个任务结束前必须运行该任务列出的验证命令；不能运行时要说明原因。

## 2. 任务卡片模板

每个任务都按同一结构理解:

```text
ID:
状态:
目标:
依赖:
参考文档:
建议修改范围:
具体步骤:
完成边界:
验证方式:
禁止范围:
```

完成边界是最重要的部分。Agent 只有在完成边界全部满足后，才可以声称任务完成。

## 3. 里程碑总览

### 3.1 当前代码审计范围

本计划已经结合当前代码做过多轮快速审计，但不是完整逐行 review。最近一次对齐时间为 2026-06-16，已检查的关键路径:

| 代码路径 | 结论 |
|----------|------|
| `mineru/doclib/services/parse_svc.py` | 已有 ingest、parse request、页码覆盖判断、JSON 批次写入、FTS 临时 Markdown、tier/remote 路由的基础实现。 |
| `mineru/doclib/background/compaction.py` | 已有 done parse batch 合并和 JSON 按 `page_idx` 合并逻辑。 |
| `mineru/doclib/server.py` | 新 doclib server 已接管 Interface 路由；`GET /docs/{sha256}/content` 已支持 read-time Markdown、`limit`、`after`、`content_ranges` 和 `next_request`。 |
| `mineru/doclib/client.py` | 同步 Doclib client 已实现 Interface；route metadata 与 server 由测试约束一致。 |
| `mineru/doclib/core/fts.py`、`services/search_svc.py` | FTS 已存 tier；搜索结果已返回来源 tier，并支持 `tier`、`min_tier`、`file_type` 过滤。 |
| `mineru/doclib/locators.py` | 已有 `short_id` 相关 block/page/char cursor helper；P0 引用模型收敛为稳定 page/block locator。 |
| `mineru/parser/base.py` | `ParseResult.from_dict()` / `from_json()` 仍是 TODO。 |
| `mineru/parser/__init__.py` | Tool SDK `parse()` / `parse_async()` 已有 `tier` 参数，并保留 `backend` 高级参数。 |
| `mineru/types.py` | `Tier`、`TIER_ORDER` 与 Middle JSON typed dataclass 在此定义；尚无 Middle JSON normalize / validator。 |
| `mineru/errors.py` | 已有 `MineruError`、`engine_error` 映射和错误 envelope helper，但部分新 code 仍未补齐。 |
| `mineru/cli_next/main.py` | `mineru parse/scan/watch/search/find/list/show/server/config/invalidate/forget/cleanup` 已按 NEXT 入口组织；尚无 `mineru telemetry`。 |
| `mineru/parser/api_server.py`、`mineru/parser/api_client.py` | v1 parse API、uploads/files、tiers、usage 和 API-backed parser 已有基础；callback/webhook 本地差异和 chat/responses 仍需明确边界。 |

因此，任务分成三类:

| 类型 | 含义 |
|------|------|
| 需要实现 | 当前代码缺失目标能力。 |
| 部分实现 | 当前代码已有基础，但与文档目标还有缺口。 |
| 已实现需锁定 | 当前代码看起来已满足目标，任务重点是补测试、防回退、修小边界。 |

Agent 领取任务时，应先读本节。如果任务标记为“已实现需锁定”，不要重写实现，优先补测试。

### 3.2 里程碑

| 里程碑 | 目标 | 结果 |
|--------|------|------|
| M0 | 锁定当前行为与测试骨架 | 后续任务有稳定测试入口。 |
| M1 | Middle JSON 和 ParseResult 基础设施 | JSON 批次可恢复、校验、渲染。 |
| M2 | doclib 主链路 | watch/ingest/parse/cache/compaction/read-time render 可闭环。 |
| M3 | tier、privacy、parse-server 路由 | 默认选择、local/remote、medium/high 行为可验证。 |
| M4 | CLI / SDK / API 对齐 | 用户入口使用同一语义和错误模型。 |
| M5 | Agent-native 输出 | locator、marker、citation 基础可用。 |
| M6 | Telemetry | doclib server 聚合、flush、CLI 管理入口和隐私边界可验证。 |

建议顺序:

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5
M6 可独立并行推进，但不能影响 parse / watch / search 主流程。
```

M1 和 M2 的部分测试任务可以并行，但 `ParseResult.from_dict()`、JSON batch 读取和 read-time render 有依赖关系。

### 3.3 任务状态索引

| 任务 | 当前代码状态 | Agent 领取方式 |
|------|--------------|----------------|
| M0-001 | 已实现 | 已有 doclib / parser / CLI 单元测试骨架；新增任务直接复用。 |
| M0-002 | 已实现需锁定 | 已有 JSON-only 行为基础；继续补高风险边界测试。 |
| M0-003 | 需要实现 | 新增 Middle JSON fixtures。 |
| M1-001 | 需要实现 | `ParseResult.from_dict()` / `from_json()` 是 TODO。 |
| M1-002 | 需要实现 | 新增 normalize helper。 |
| M1-003 | 需要实现 | 新增 validator。 |
| M1-004 | 部分实现 | 先用 validator 跑 HTML 输出，再最小修。 |
| M1-005 | 需要实现 | 新增 `bbox_known()`。 |
| M2-001 | 已实现 | `parse_batch_json_path()` 已被 parse_svc / compaction 复用。 |
| M2-002 | 已实现 | cache coverage 与 compaction 均使用配置 data_dir；已有相关测试。 |
| M2-003 | 已实现需锁定 | 未覆盖页入队、active parse 复用和 priority bump 已有实现与测试；继续补失败边界。 |
| M2-004 | 已实现需锁定 | compaction JSON 合并已有实现与测试；继续锁定重复页/invalidated 边界。 |
| M2-005 | 部分实现 | `GET /docs/{sha256}/content` 已支持 Markdown / progressive reading；Content List / HTML read-time render 仍需确认。 |
| M2-006 | 已实现需锁定 | FTS 临时 Markdown、JSON-only、tier-gated 更新已有实现；继续补回归测试。 |
| M3-001 | 已实现需锁定 | `tier=None` 已在入队前解析为实体 tier；需补默认选择错误码边界。 |
| M3-002 | 已实现需锁定 | local 不自动 remote 已有实现；补调用边界测试。 |
| M3-003 | 已实现需锁定 | remote 失败 fallback local 已有实现；补 `via=local` 和 tier 能力测试。 |
| M3-004 | 已实现需锁定 | tier mismatch 不降级已有实现；补错误 code / message 边界测试。 |
| M4-001 | 部分实现 | CLI 已调用 doclib read-time render；JSON/HTML/Content List 输出边界仍需对齐。 |
| M4-002 | 已实现需锁定 | Tool SDK `parse()` / `parse_async()` 已有 `tier` 参数；补 backend 覆盖 tier 测试。 |
| M4-003 | 部分实现 | Doclib client 已有映射；API-backed parser 等仍需统一。 |
| M4-004 | 部分实现 | api_server/client 已有基础，需按文档补差异测试。 |
| M5-001a | 已实现需锁定 | doclib locator helper 已实现；补 Structured Content / marker 复用边界。 |
| M5-002 | ready | Markdown block marker 尚未完整实现。 |
| M5-003 | 已实现需锁定 | 搜索结果已返回 tier，并支持过滤；补 CLI/JSON 边界。 |
| M6-001 | 需要实现 | Telemetry DB schema、状态管理、聚合与 flush 尚未实现。 |
| M6-002 | 需要实现 | `mineru telemetry status|enable|disable|preview|flush` 尚未实现。 |

## 4. M0: 测试与基线

### M0-001 建立 doclib parse flow 测试入口

状态: `done`

目标:

为 doclib 的解析主链路建立最小测试入口，后续任务可以在同一测试目录中增量添加用例。

依赖:

- 无。

参考文档:

- [端到端工作流](workflows.md)
- [系统架构](architecture.md)

建议修改范围:

- `tests/` 或项目现有测试目录。
- `mineru/doclib/services/parse_svc.py` 仅在测试需要暴露小 helper 时修改。

具体步骤:

1. 查找项目现有测试布局。
2. 如果已有 doclib 测试目录，在其中新增 parse flow 测试文件。
3. 如果没有，创建最小测试目录和命名约定。
4. 提供可复用的临时 data_dir fixture。
5. 提供可复用的 fake db / sqlite test db fixture。
6. 提供可复用的 fake parser result fixture，能返回 `{"pages": [...]}`。

完成边界:

- 存在一个可运行的 doclib parse flow 测试文件。
- 测试能创建临时 data_dir，不污染 `~/.mineru`。
- 至少一个 smoke test 能启动必要服务对象或 helper，并通过。

验证方式:

```bash
.venv/bin/python -m pytest tests -q
```

禁止范围:

- 不改解析逻辑。
- 不改生产默认 data_dir。

### M0-002 锁定 doclib 只持久化 JSON 的行为

状态: `verify-first`

目标:

用测试固定 doclib 的持久化产物模型: `parsed/` 下只保存 JSON 批次文件，其他格式读取时生成。

依赖:

- M0-001。

参考文档:

- [端到端工作流: JSON 产物、页合并与缓存](workflows.md#10-json-产物页合并与缓存)

建议修改范围:

- `tests/...`
- `mineru/doclib/services/parse_svc.py`

具体步骤:

1. 构造一个 parse task，包含 `sha256`、`tier`、`page_range`。
2. mock 本地 parser，使它返回带 `pages` 的 ParseResult-like 对象。
3. 调用 `ParseService.process_doc()`。
4. 检查 `parsed/<sha-prefix>/<sha>/<tier>/` 中只有 `.json` 文件。
5. 检查 JSON 内容形态为 `{"pages": [...]}`。
6. 检查没有 `output.md`、`content_list.json`、`structured_content.json`、`output.html`。

完成边界:

- 测试证明 doclib parse 完成后只落 JSON 文件。
- 如果当前代码已经满足，只提交测试。
- 如果当前代码不满足，只做最小修改使其满足。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "json"
```

禁止范围:

- 不实现新 render 格式。
- 不改 Tool SDK 的 `ParseResult.save()` 行为；该行为不等同 doclib 持久化规则。

### M0-003 建立 Middle JSON fixture

状态: `ready`

目标:

为 Pipeline、VLM、Office、HTML 至少建立可用于 validator / render / round-trip 的最小 fixture。

依赖:

- 无。

参考文档:

- [Middle JSON 当前标准](middle-json/current-medium.md)
- [Middle JSON 后端差异](middle-json/backend-gaps.md)

建议修改范围:

- `tests/fixtures/middle_json/`
- `tests/...`

具体步骤:

1. 新增最小 Pipeline fixture，至少包含一个 page、一个 text block、line、span。
2. 新增最小 VLM fixture，体现一行一 span 的粒度。
3. 新增最小 Office fixture，允许 unknown bbox。
4. 新增最小 HTML fixture，覆盖 HTML parser 的目标结构。
5. 每个 fixture 包含 `pages` 顶层结构。

完成边界:

- 每个 fixture 都能被 JSON parser 读取。
- fixture 不包含真实用户文档内容。
- fixture 中的页码、block index、bbox 语义清楚。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "fixture or middle"
```

禁止范围:

- 不引入大型 PDF 或 Office 文件。
- 不把模型输出原始大对象放入 fixture。

## 5. M1: Middle JSON 与 ParseResult

### M1-001 实现 `ParseResult.from_dict()` / `from_json()`

状态: `ready`

目标:

让 API JSON、doclib JSON 批次和缓存结果能恢复为 `ParseResult`。

依赖:

- M0-003。

参考文档:

- [SDK ParseResult](sdk/parse-result.md)
- [Middle JSON Envelope](middle-json/envelope.md)

建议修改范围:

- `mineru/parser/base.py`
- `mineru/types.py`
- `tests/...`

具体步骤:

1. 阅读 `ParseResult.to_dict()` 当前输出。
2. 支持输入 `{"pages": [...]}`。
3. 支持输入 canonical envelope 的 `pages`。
4. 支持输入旧结构 `{"pdf_info": [...]}`，如 normalize 尚未实现，可先在本任务内只做最小兼容。
5. 将 page dict 转为 `PageInfo.from_dict()`。
6. 实现 `from_json()`，只负责 JSON decode 后调用 `from_dict()`。
7. 增加 round-trip 测试。

完成边界:

- `ParseResult.to_json()` -> `ParseResult.from_json()` round-trip 可用。
- `{"pages": [...]}` 可恢复。
- 至少 Pipeline/Office fixture 可恢复并 render Markdown。
- 无效输入给出明确异常。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "ParseResult or roundtrip"
```

禁止范围:

- 不重写 render。
- 不改变 `ParseResult.markdown()` / `content_list()` 对外行为。

### M1-002 实现 `normalize_middle_json()`

状态: `ready`

目标:

统一接收 canonical envelope、当前 `{"pages": [...]}`、旧 `{"pdf_info": [...]}` 三种结构，并输出标准 envelope。

依赖:

- M1-001 可并行，但最终应互相调用。

参考文档:

- [Middle JSON Envelope](middle-json/envelope.md)
- [Middle JSON Migration](middle-json/migration.md)

建议修改范围:

- `mineru/parser/base.py`
- `tests/...`

具体步骤:

1. 在 parser 层复用 `mineru.parser.MIDDLE_JSON_SCHEMA_VERSION`，并定义 `normalize_middle_json(payload, *, backend=None, tier=None, sha256=None)`。
2. 如果 payload 已有 `schema_version` 和 `pages`，直接接受；`_meta` 在当前 P0 写出结构中可缺省。
3. 如果 payload 是 `{"pages": [...]}`，包装成 canonical envelope。
4. 如果 payload 是 `{"pdf_info": [...]}`，迁移为 `pages`。
5. 将旧 `_backend` 放入 `_meta.backend`。
6. `tier` 只记录实际使用的 tier，不记录 `requested_tier`。
7. 保持未知字段向后兼容，不主动丢弃 `_meta` 扩展。

完成边界:

- 三种输入结构都可 normalize。
- 输出包含 `schema_version`、`pages`、`_meta`。
- 不产生 `requested_tier` / `resolved_tier` 字段。
- 旧 `_backend` 能迁移到 `_meta.backend`。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "normalize_middle_json"
```

禁止范围:

- 不改 backend 输出逻辑。
- 不强制所有调用点一次性迁移。

### M1-003 增加 Middle JSON P0 validator

状态: `ready`

目标:

提供轻量 validator，能发现 P0 结构错误，同时允许后续 schema 演进。

依赖:

- M1-002。

参考文档:

- [Middle JSON Envelope](middle-json/envelope.md)
- [Middle JSON Backend Gaps](middle-json/backend-gaps.md)

建议修改范围:

- validator 生产入口位置待定
- `tests/...`

具体步骤:

1. 定义 `ValidationIssue` 类型。
2. 实现 `validate_middle_json(payload) -> list[ValidationIssue]`。
3. validator 内部先调用 normalize 或要求输入已 normalize，二者择一并文档化。
4. 校验 `pages` 是 list。
5. 校验 page 至少有 `page_idx`。
6. 校验 block 至少有 `index`、`type`、`bbox`、`lines` 或 `blocks`。
7. 允许 Office unknown bbox，但 issue 中能提示 `bbox_known=false` 场景。
8. 对 `_backend` 旧字段给 warning，不给 fatal。

完成边界:

- fixtures 通过 validator。
- 人工破坏 page/block 必填字段时 validator 能报 issue。
- validator 不因 Office bbox unknown 直接失败。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "validator"
```

禁止范围:

- 不引入重型 JSON schema 依赖，除非项目已有。
- 不把所有 backend 私有字段列为公开必填。

### M1-004 修正 HTML parser 的 typed structure 输出

状态: `verify-first`

目标:

确保 HTML parser 输出满足 `PageInfo` / `Block` / `Line` / `Span` 的必填字段契约。

依赖:

- M0-003。
- M1-003。

参考文档:

- [Middle JSON Backend Gaps](middle-json/backend-gaps.md)

建议修改范围:

- `mineru/parser/html.py`
- `mineru/types.py` 仅在确有必要时修改。
- `tests/...`

具体步骤:

1. 运行 HTML parser 现有测试或新增最小 HTML 解析测试。
2. 用 validator 检查 HTML 输出。
3. 如果缺少 `bbox`、`index`、`page_size` 或 `_backend` 合法值，最小修正。
4. 对没有真实 bbox 的内容使用 unknown bbox 约定。
5. 确认 render Markdown 不回退。

完成边界:

- HTML fixture 能生成合法 pages。
- validator 无 fatal issue。
- `ParseResult.markdown()` 可输出。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "html and middle"
```

禁止范围:

- 不重写 HTML converter。
- 不引入浏览器运行时。

### M1-005 定义 `bbox_known()` helper

状态: `ready`

目标:

为 unknown bbox 提供统一判断，避免 Office/HTML 等 backend 用假 bbox 被误当真实坐标。

依赖:

- M1-003。

参考文档:

- [Agent Gaps](middle-json/agent-gaps.md)
- [Backend Gaps](middle-json/backend-gaps.md)

建议修改范围:

- `mineru/types.py` 或未来 validator 模块
- `tests/...`

具体步骤:

1. 确认当前 unknown bbox 表达。
2. 定义 `bbox_known(bbox) -> bool`。
3. 处理 `None`、空列表、全 0、负数或 sentinel 值。
4. 在 validator issue 中使用该 helper。
5. 添加 Office unknown bbox 测试。

完成边界:

- helper 有明确 docstring。
- validator 能区分 unknown bbox 和非法 bbox。
- 不改变已有 backend 坐标数值。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "bbox"
```

禁止范围:

- 不要求 Office 立即补真实 bbox。

## 6. M2: doclib JSON 批次与缓存

### M2-001 抽出 JSON 批次路径 helper

状态: `done`

目标:

把 doclib JSON 批次文件路径生成逻辑集中，避免 `process_doc()`、cache check、compaction 各自拼路径。

依赖:

- M0-002。

参考文档:

- [端到端工作流: JSON 产物](workflows.md#10-json-产物页合并与缓存)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/doclib/background/compaction.py`
- `tests/...`

具体步骤:

1. 找出 `_safe_filename()`、`_json_file_exists_by_batch()` 和 compaction 中的重复路径逻辑。
2. 抽出 helper，例如 `parse_batch_json_path(data_dir, sha256, tier, page_range, done_at)`。
3. 所有路径都使用传入的 `data_dir`，不能硬编码 `~/.mineru`。
4. 更新 process、cache check、compaction 调用。
5. 补测试覆盖 page range 文件名。

完成边界:

- 生产代码不再在多个地方手写 batch JSON 路径。
- 测试 data_dir 不会误读 `~/.mineru`。
- 现有 JSON 文件命名保持兼容。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "batch_json_path or parse"
```

禁止范围:

- 不迁移旧文件名。
- 不改变 JSON 内容结构。

### M2-002 修正 cache coverage 对 data_dir 的依赖

状态: `done`

目标:

确认 cache coverage 判断使用当前 ParseService 的 data_dir，而不是固定 `~/.mineru`。

依赖:

- M2-001。

参考文档:

- [端到端工作流: 缓存键与覆盖判断](workflows.md#102-缓存键与覆盖判断)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `tests/...`

具体步骤:

1. 编写测试: 临时 data_dir 中存在 done batch JSON。
2. 调用 `request_parse()` 请求同一页码。
3. 期望返回 cached done。
4. 删除 JSON 文件后再次请求，期望重新入队。
5. 如果测试失败，修正 data_dir 依赖。

完成边界:

- JSON 存在时命中缓存。
- JSON 丢失时不命中缓存。
- 测试不访问真实用户目录。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "cache and data_dir"
```

禁止范围:

- 不改变 `parses` 表 schema。

### M2-003 实现或锁定增量页码入队

状态: `verify-first`

目标:

确保请求页码只对未覆盖页面创建新 parse record，并返回本次请求需要等待的 parse ids。

依赖:

- M2-002。

参考文档:

- [端到端工作流: 增量页码解析与合并](workflows.md#136-增量页码解析与合并)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `tests/...`

具体步骤:

1. 准备已有 done batch: `page_range=1~5`。
2. 请求 `page_range=1~10`。
3. 期望只创建 `6~10` 的 pending batch。
4. 准备已有 pending batch: `page_range=6~10`。
5. 再请求 `1~10`，期望不重复创建，而是提升 priority。
6. 测试 `force=True` 时跳过 done cache，但复用 active batch。
7. 测试 `force=True` 只为 active 未覆盖页创建新 parse。
8. 测试返回 `wait_parse_ids`、`created_parse_ids`、`reused_parse_ids`。
9. 测试 force wait parse 失败后，本次请求失败，但旧 done batch 仍然有效。
10. 测试 invalidated batch 不参与覆盖判断。

完成边界:

- 覆盖判断按页集合工作。
- pending/parsing 覆盖时只提升优先级。
- force 跳过 done cache，但可复用 active batch，只为未覆盖页创建新任务。
- response 能表达 wait / created / reused parse ids。
- invalidated batch 不参与缓存命中。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "pages_uncovered or coverage"
```

禁止范围:

- 不改默认页码策略。

### M2-004 补齐 compaction JSON 合并测试

状态: `verify-first`

目标:

用测试锁定 compaction 按 `page_idx` 合并 JSON 的行为。

依赖:

- M2-001。

参考文档:

- [端到端工作流: 页合并与 compaction](workflows.md#103-页合并与-compaction)

建议修改范围:

- `mineru/doclib/background/compaction.py`
- `tests/...`

具体步骤:

1. 创建同一 `sha256 + tier` 下两个有效 done row。
2. 创建对应 JSON 文件。
3. 调用 compaction。
4. 检查旧 JSON 被删除。
5. 检查新 JSON 覆盖合并后的页。
6. 添加重复页测试，确认 `done_at` 较新的有效批次覆盖较旧批次。
7. 添加 invalidated batch 测试，确认它不会被合入新 JSON。

完成边界:

- compaction 合并 parse row。
- compaction 合并 JSON 文件。
- compaction 不生成 Markdown / Content List。
- compaction 忽略 invalidated batch。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "compaction"
```

禁止范围:

- 不改变 compaction 定时策略。

### M2-005 实现 read-time render from JSON batches

状态: `verify-first`

目标:

确保 doclib 对外返回 Markdown / Content List / HTML 时，从已保存 JSON 批次读取并转换，而不是读取持久化派生文件。

依赖:

- M1-001。
- M2-003。

参考文档:

- [端到端工作流: 读取时转换](workflows.md#104-读取时转换)
- [SDK ParseResult](sdk/parse-result.md)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/doclib/server.py`
- `mineru/doclib/client.py`
- `tests/...`

具体步骤:

1. 定位当前 `parse_content()` 或等价读取入口，并迁移为 `GET /docs/{sha256}/content`。
2. 实现按 `sha256 + tier + page_range` 收集已完成 JSON 批次。
3. 使用 `ParseResult.from_dict()` 恢复 pages。
4. 按请求格式调用 `markdown()`、`content_list()`、`structured_content()` 或 HTML render。
5. 如果请求页未被覆盖，返回明确状态或错误，不生成空内容。
6. 保持输出文件写入只发生在用户指定 `output` 时。

完成边界:

- doclib 不需要 `output.md` 也能返回 Markdown。
- JSON 输出返回 Middle JSON pages。
- Content List 输出从同一 JSON 转换。
- 缺页时返回可操作错误或 pending 状态。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "parse_content or read_time_render"
```

禁止范围:

- 不把 Markdown/Content List 写入 `parsed/`。
- 不改 Tool SDK `save()`。

### M2-006 FTS 只消费临时 Markdown，不持久化

状态: `verify-first`

目标:

确认搜索索引更新可以使用临时 Markdown 文本，但不会在 doclib 产物目录落盘。

依赖:

- M0-002。

参考文档:

- [端到端工作流: 搜索索引](workflows.md#106-搜索索引)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/doclib/core/fts.py`
- `tests/...`

具体步骤:

1. mock ParseResult.markdown() 返回可搜索文本。
2. 执行 process_doc。
3. 检查 FTS 收到文本。
4. 检查 parsed 目录无 Markdown 文件。
5. 检查低 tier 不覆盖高 tier FTS。

完成边界:

- FTS 有内容。
- Markdown 不落盘。
- tier-gated 更新符合 `flash < medium < high`。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "fts"
```

禁止范围:

- 不改搜索 query 语义。

## 7. M3: tier、privacy 与 parse-server 路由

### M3-001 默认 tier 在入队前解析为实际 tier

状态: `verify-first`

目标:

确保默认选择不作为任务 tier、缓存目录或产物目录持久化；最终只记录实际使用的实体 tier。

依赖:

- M2-003。

参考文档:

- [解析 Tier](tiers.md)
- [端到端工作流: Tier 决策](workflows.md#64-tier-决策)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/doclib/background/parse_server_health.py`
- `tests/...`

具体步骤:

1. 构造 health: local supported tiers = `["medium", "high"]`。
2. 请求 `tier=None`。
3. 期望入队 tier 为 `high`。
4. 构造 health: local supported tiers = `["medium"]`。
5. 期望入队 tier 为 `medium`。
6. 构造 health: only `flash` 或空。
7. 期望返回 `quality_tier_unavailable` 或等价错误。

完成边界:

- 不创建默认选择专用 parse row。
- 不创建默认选择专用缓存目录。
- 不记录 `requested_tier` 字段。
- 默认选择永不选择 `flash`。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "default and tier"
```

禁止范围:

- 不改变显式 `tier=flash` 行为。

### M3-002 隐私边界: local 失败不自动 remote

状态: `verify-first`

目标:

用测试和最小实现保证没有显式 remote 时，不会调用 Remote Parse Server。

依赖:

- M3-001。

参考文档:

- [端到端工作流: 隐私决策](workflows.md#65-隐私决策)
- [解析 Tier: 隐私优先](tiers.md#71-隐私优先)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/parser/api_client.py` 仅用于 mock 调用边界。
- `tests/...`

具体步骤:

1. mock local parse-server unavailable。
2. mock remote parse-server healthy。
3. 请求 `remote=False`。
4. 期望返回本地能力不足错误。
5. 确认没有构造指向 remote URL 的 `MinerUApiParser`。
6. 请求 `remote=True`。
7. 期望允许 remote。

完成边界:

- local 请求不上传。
- 错误包含可操作 suggestion。
- remote 请求才允许使用 remote URL。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "privacy or remote"
```

禁止范围:

- 不新增全局配置来默认允许 remote。

### M3-003 remote 失败可以 fallback 到 local

状态: `verify-first`

目标:

用户已显式允许 remote 时，如果 remote 不可达，可以 fallback 到 local，但不能改变结果审计。

依赖:

- M3-002。

参考文档:

- [端到端工作流: fallback 不扩大隐私边界](workflows.md)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `tests/...`

具体步骤:

1. 设置 `privacy=remote`。
2. mock remote unhealthy。
3. mock local healthy 且支持请求 tier。
4. 执行解析。
5. 期望使用 local URL。
6. 完成后 `via=local`。

完成边界:

- fallback 只发生在用户已允许 remote 时。
- `via` 记录实际执行路径。
- tier 能力仍需匹配。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "fallback"
```

禁止范围:

- 不对文件损坏、加密等非网络错误做 fallback。

### M3-004 parse-server tier mismatch 错误

状态: `verify-first`

目标:

当 parse-server 不支持请求 tier 时，返回稳定错误，不自动降级。

依赖:

- M3-001。

参考文档:

- [错误码体系](errors.md)
- [SDK Tier 与错误](sdk/tiers-errors.md)

建议修改范围:

- `mineru/doclib/services/parse_svc.py`
- `mineru/errors.py`
- `tests/...`

具体步骤:

1. mock parse-server supported tiers = `["medium"]`。
2. 请求 `tier="high"`。
3. 期望 `tier_mismatch`。
4. 确认不会改成 `medium`。
5. 确认不会改成 `flash`。

完成边界:

- 错误 code 稳定。
- 错误 message 包含 available tiers。
- 无降级 parse row。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "tier_mismatch"
```

禁止范围:

- 不改变默认选择逻辑。

## 8. M4: CLI / SDK / API 对齐

### M4-001 `mineru parse` 使用 doclib read-time render

状态: `verify-first`

目标:

确保 `mineru parse` 的 Markdown/Text/JSON 输出来自 doclib JSON 批次，而不是派生产物文件。

依赖:

- M2-005。

参考文档:

- [mineru parse](cli/mineru-parse.md)
- [端到端工作流](workflows.md)

建议修改范围:

- `mineru/cli_next/commands/parse.py`
- `mineru/cli_next/output.py`
- `mineru/doclib/client.py`
- `tests/...`

具体步骤:

1. 检查 CLI parse 当前调用链。
2. JSON 格式输出 Middle JSON 或 response envelope，按 CLI 文档确定。
3. Markdown/Text/HTML 格式通过 doclib client 请求 read-time render。
4. `--output` 写用户指定路径，不写 doclib `parsed/`。
5. `--no-wait` 未完成时返回任务状态。

完成边界:

- CLI 不依赖 `output.md`。
- CLI 输出格式符合文档。
- 缓存命中时可直接输出。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "cli and parse"
```

禁止范围:

- 不在普通 `mineru parse` 暴露 backend 参数。

### M4-002 Tool SDK `parse()` 增加 tier 参数

状态: `verify-first`

目标:

让 Tool SDK 支持 `tier` 参数，同时保留 `backend` 作为高级兼容参数。

依赖:

- M1-001。

参考文档:

- [SDK Parser](sdk/parser.md)
- [SDK Migration](sdk/migration.md)

建议修改范围:

- `mineru/parser/__init__.py`
- `mineru/parser/pdf.py`
- `mineru/parser/base.py`
- `tests/...`

具体步骤:

1. 查清 `parse()` 当前签名。
2. 增加 `tier: str | None = None`。
3. `backend` 显式传入时覆盖 `tier`。
4. `tier=flash` 映射到 `flash` backend。
5. `tier=medium` 映射到默认 medium backend。
6. `tier=high` 映射到默认 high backend。
7. `tier=None` 在 Tool SDK 场景中如果不能发现 medium/high，应报错，不静默 flash。

完成边界:

- `parse(path, tier="flash")` 可用。
- `parse(path, tier="medium")` 可用或返回明确能力错误。
- `parse(path, backend="pipeline")` 兼容。
- `backend` 覆盖 `tier` 有测试。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "parser and tier"
```

禁止范围:

- 不让 Tool SDK 隐式启动 doclib server。
- 不让 Tool SDK 隐式 remote 上传。

### M4-003 统一错误 envelope 到 CLI/SDK 异常

状态: `ready`

目标:

API-backed parser、Doclib SDK、本地 parser 的错误都能被统一捕获，并保留 code。

依赖:

- M3-004。

参考文档:

- [错误码体系](errors.md)
- [API Responses](api/responses.md)
- [SDK Tier 与错误](sdk/tiers-errors.md)

建议修改范围:

- `mineru/errors.py`
- `mineru/parser/api_client.py`
- `mineru/doclib/client.py`
- `tests/...`

具体步骤:

1. 定义或补齐 `MineruError` 子类和 code 属性。
2. 将 `_V1APIError` 映射到 `MineruError`。
3. Doclib client 收到 error envelope 时抛出统一异常。
4. CLI 根据 code 输出修复建议。
5. 增加 `quality_tier_unavailable` 等错误测试。

完成边界:

- 三类入口都能捕获 `MineruError`。
- 异常保留 `type`、`code`、`param` 或等价信息。
- CLI 不吞掉机器可读 code。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "error"
```

禁止范围:

- 不改所有错误文案，只保证结构和核心 code。

### M4-004 Local Parse Server API 与官方 API 差异测试

状态: `ready`

目标:

确保 Local Parse Server 可以复用 v1 API 客户端，同时允许本地差异。

依赖:

- M4-003。

参考文档:

- [API 总览](api/README.md)
- [Parse Jobs](api/parse-jobs.md)

建议修改范围:

- `mineru/parser/api_server.py`
- `mineru/parser/api_client.py`
- `tests/...`

具体步骤:

1. 测试 `GET /v1/health`。
2. 测试 `GET /v1/tiers`。
3. 测试 `POST /v1/parse/jobs` 的最小本地路径 source。
4. local source 必须校验 allowlist。
5. 未实现 webhook 时 health features 应标记 false。
6. 使用同一个 `MinerUApiParser` 调用本地 server。

完成边界:

- Local Parse Server 不需要实现官方全部能力，但 endpoint 行为和差异符合文档。
- 本地路径越权被拒绝。
- 客户端可复用。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "api_server or local_source"
```

禁止范围:

- 不实现公网计费、CDN、Webhook 完整能力。

## 9. M5: Agent-native 输出

### M5-001a 锁定 doclib locator helper

状态: `verify-first`

目标:

锁定当前 doclib locator / cursor helper 行为，确保 public locator 使用 1-based 页号和块号，且不包含本地路径。

依赖:

- ADR-0011。
- ADR-0012。

参考文档:

- [Agent Gaps](middle-json/agent-gaps.md)
- [ADR-0011: Doc Short ID](decisions/0011-doclib-doc-short-id.md)
- [ADR-0012: Doclib Block Locator](decisions/0012-doclib-block-locator.md)

建议修改范围:

- `mineru/doclib/locators.py`
- `tests/...`

具体步骤:

1. 确认 `locator_for_block(page_no, block_no)` 使用 1-based public number。
2. 确认 `page_ref()`、`block_ref()`、`block_char_ref()` 使用 `short_id`、实体 tier、1-based page/block 和 0-based char offset。
3. 确认 `parse_content_cursor()` 能解析 page / block / char 三种 cursor。
4. 补充或保留测试，避免 0-based page/block 泄漏到 public cursor。
5. 确认 marker / continuation 不包含本地绝对路径。

完成边界:

- locator helper 行为由测试锁定。
- page/block public number 均为 1-based。
- char offset 为 block 渲染文本内 0-based offset。
- cursor 中不包含 path 或 filename。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "locator"
```

禁止范围:

- 不实现独立 hash 型引用 ID。
- 不实现 span/cell/list-item locator。

### M5-002 Markdown locator marker

状态: `ready`

目标:

Markdown 输出可选择携带 Agent locator marker。

依赖:

- M5-001a。

参考文档:

- [Middle JSON Rendering](middle-json/rendering.md)
- [端到端工作流: Agent marker](workflows.md#74-agent-marker)

建议修改范围:

- `mineru/render/union_make.py`
- `mineru/render/markdown.py`
- `tests/...`

具体步骤:

1. 找到 `add_markers` 参数当前行为。
2. 为 block 输出前增加可选 marker。
3. marker 包含稳定 page/block locator。
4. 默认不输出 marker。
5. CLI/SDK 需要显式 opt-in。

完成边界:

- `markdown(add_markers=False)` 输出不变。
- `markdown(add_markers=True)` 输出 marker。
- marker 可被正则稳定解析。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "marker"
```

禁止范围:

- 不改变 Markdown 默认视觉格式。

### M5-003 搜索结果返回来源 tier

状态: `verify-first`

目标:

让 Agent 知道搜索 snippet 来自哪个 tier，以便决定是否升级解析。

依赖:

- M2-006。

参考文档:

- [端到端工作流: Agent 从搜索结果继续阅读](workflows.md#7-流程-c-agent-从搜索结果继续阅读)

建议修改范围:

- `mineru/doclib/services/search_svc.py`
- `mineru/doclib/core/fts.py`
- `mineru/doclib/client.py`
- `tests/...`

具体步骤:

1. 确认 FTS 表是否已有 tier 字段。
2. search response 中暴露来源 tier。
3. CLI search 可显示或 JSON 输出该字段。
4. 如果没有 tier，返回 null 或 omit，并补 TODO issue。

完成边界:

- 搜索结果机器可读地包含来源 tier。
- `flash` snippet 能被识别。
- 高 tier 更新 FTS 后来源 tier 更新。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "search and tier"
```

禁止范围:

- 不实现搜索排序重构。

## 10. M6: Telemetry

### M6-001 实现 telemetry DB 状态、聚合与 flush

状态: `ready`

目标:

实现 doclib server 侧 telemetry 状态管理、本地聚合存储和周期 flush。纯工具层不得具备 telemetry 能力。

依赖:

- doclib DB migration。
- [Telemetry 设计](telemetry.md)。

参考文档:

- [Telemetry 设计](telemetry.md)
- [配置体系: Telemetry](config.md#telemetry)

建议修改范围:

- `mineru/doclib/migrations/001_init.sql`
- `mineru/doclib/services/`
- `mineru/doclib/background/`
- `mineru/doclib/app.py`
- `tests/...`

具体步骤:

1. 增加 telemetry 状态表或配置记录，保存 `consent_state`、内部 `installation_id`、flush lock 和 last flush 状态。
2. 增加 telemetry 聚合表，按 period、metric name、dimensions hash upsert 累加。
3. 实现 metric / dimension 白名单，不允许透传请求体、异常对象、配置对象或任意 dict。
4. 实现 doclib API / CLI source / caller 的 best-effort 透传入口。
5. 实现 flush 任务，成功后删除已确认上报的聚合记录。
6. flush 失败不得影响 parse、watch、search 或任意 doclib API。
7. 用户关闭 telemetry 时停止写入新 metric，并清除未上报聚合数据。

完成边界:

- `consent_state=disabled` 时不写入新 telemetry 聚合。
- `consent_state=enabled` 时 metrics 按小时级 period 聚合。
- flush 成功后本地聚合记录被清理。
- flush 失败保留记录等待下次重试。
- telemetry 不记录路径、文件名、query、snippet、文档内容、traceback 或 API Key。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "telemetry"
```

禁止范围:

- 不在 parser SDK、`mineru-kit parse` 或 `mineru-kit api-server` 中加入 telemetry 能力。
- 不实现每事件实时上报。
- 不阻塞主业务链路等待 telemetry。

### M6-002 实现 `mineru telemetry` CLI

状态: `ready`

目标:

提供用户可见的 telemetry 状态查看和开关命令。

依赖:

- M6-001。

参考文档:

- [Telemetry 设计](telemetry.md)
- [CLI 总览](cli.md)

建议修改范围:

- `mineru/cli_next/main.py`
- `mineru/cli_next/commands/`
- `mineru/doclib/base.py`
- `mineru/doclib/client.py`
- `mineru/doclib/server.py`
- `mineru/doclib/types.py`
- `tests/...`

具体步骤:

1. 增加 `mineru telemetry status`。
2. 增加 `mineru telemetry enable`。
3. 增加 `mineru telemetry disable`。
4. 增加 `mineru telemetry preview`，只展示将上报的聚合摘要，不展示敏感字段。
5. 增加 `mineru telemetry flush`，触发一次手动 flush。
6. 首次启动选择流程如暂不实现 UI，应至少让 `consent_state=unset` 可观测，并在 CLI/server status 中提示。

完成边界:

- 用户可以查看当前 consent 状态。
- 用户可以开启和关闭 telemetry。
- 关闭时清理未上报聚合数据。
- preview 不包含路径、文件名、query、snippet、文档内容、traceback 或 API Key。
- 命令失败时返回稳定错误 code。

验证方式:

```bash
.venv/bin/python -m pytest tests -q -k "telemetry or cli"
```

禁止范围:

- 不要求实现图形化首次启动选择。
- 不向 parser/tool 层传递 telemetry 配置。

## 11. 可并行性

| 可并行组 | 任务 |
|----------|------|
| Fixtures / validator | M0-003、M1-003、M1-005 |
| doclib JSON behavior | M0-002、M2-003、M2-004 |
| privacy / tier tests | M3-001、M3-002、M3-004 |
| CLI / SDK | M4-001、M4-002，需避免同文件冲突 |
| Agent locator | M5-001a 可独立锁定；M5-002 依赖 locator helper |
| Telemetry | M6-001 可独立推进；M6-002 依赖 M6-001 |

不建议并行:

- M3-001 与 M3-004 同时改 `_resolve_tier()`。
- M1-001 与 M2-005 同时改 `ParseResult.from_dict()` 调用边界。

## 12. PR 切分建议

| PR | 包含任务 | 目标 |
|----|----------|------|
| PR-1 | M0-003 | Middle JSON fixture。 |
| PR-2 | M1-001, M1-002, M1-003 | Middle JSON normalize / round-trip / validate。 |
| PR-3 | M2-003, M2-004 | doclib page coverage 和 compaction 边界锁定。 |
| PR-4 | M2-005, M2-006 | read-time render 和 FTS。 |
| PR-5 | M3-001, M3-002, M3-003, M3-004 | tier / privacy / parse-server 路由。 |
| PR-6 | M4-001, M4-002, M4-003 | CLI / SDK / 错误模型。 |
| PR-7 | M5-001a, M5-002, M5-003 | Agent-native locator / marker / search tier。 |
| PR-8 | M6-001, M6-002 | Telemetry DB、flush 和 CLI。 |

每个 PR 都应能独立通过测试，并保留向后兼容。

## 13. 完成定义

P0 完成需要满足:

1. `mineru parse` 能在本地完成 ingest、cache check、parse task、JSON batch 写入、read-time Markdown 输出。
2. watch 能创建 `flash` 任务并更新搜索索引。
3. doclib `parsed/` 目录只保存 JSON。
4. 默认选择不产生独立缓存目录，不解析为 `flash`。
5. 未显式 remote 时不会上传。
6. Local Parse Server 支持 tier discovery 并可被 doclib 调用。
7. `ParseResult.from_json()` 可恢复 JSON 批次。
8. Middle JSON validator 覆盖 P0 必填字段。
9. 错误 code 对 CLI/SDK/API 可见。
10. Telemetry consent、聚合、flush 和 CLI 管理入口可验证。
11. Agent-native 输出具备稳定 page/block locator、`next_request`、`truncated` 和 CLI next marker。
12. workflows 中 13.1 到 13.7 的端到端验收可以通过测试或手工脚本验证。

P1 完成需要满足:

1. Agent 主动读取默认升级到 `medium` / `high`。
2. Markdown 可选输出 locator marker。
3. 高质量解析完成后刷新搜索索引。

## 14. Agent 领取任务时的最终检查

每个任务完成前，Agent 必须回答:

1. 我改了哪些文件？
2. 哪些完成边界已经满足？
3. 哪些验证命令已经运行，结果是什么？
4. 是否有未完成但相关的后续任务？
5. 是否触碰了任务禁止范围？

如果第 5 项答案是“是”，该任务不应合并，除非用户或维护者明确批准。
