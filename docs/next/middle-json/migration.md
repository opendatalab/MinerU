# 迁移计划

状态: Draft
读者: 核心开发者、backend 开发者、SDK 开发者
范围: Middle JSON 统一工作的阶段、任务、验收和风险
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 目标

把当前 typed document model 推进为可验证、可迁移、可供 Agent 引用的 Middle JSON 标准。

## Phase 0: 确认事实标准

任务:

1. 明确 `mineru/types.py` 是当前 Middle JSON 基础类型来源。
2. 将 `PageInfo` / 严格 Pydantic block tree / `InlineSpan` 字段写入文档。
3. 标注内部字段和 public 字段。

验收:

- 新增文档能回答“当前标准是什么”。
- backend 开发者知道新增字段应该加在哪里。

## Phase 1: Envelope 与恢复

任务:

1. 定义 canonical envelope。
2. 已实现 `ParseResult.from_dict()`。
3. 已实现 `ParseResult.from_json()`。
4. 已实现 `ParseResult.to_dict()` 输出 `schema_version + pages`。
5. 待实现 `normalize_middle_json()`。
6. 当前运行时兼容的输入（`ParseResult.from_dict()`）:
   - `{"schema_version": "2.0", "pages": [...], ...}` 直接读取。
   - `{"schema_version": "1.0", "pages": [...]}` 经 legacy 分支单向迁移。
   - `{"pdf_info": [...], "_backend": ..., ...}`（MinerU 3.4.5 产物）按 envelope 识别后经同一 legacy 分支单向迁移。
7. 无版本号的裸 `{"pages": [...]}` 与其它未知旧 payload 被明确拒绝，要求重新解析，不做静默迁移。

验收:

- API `middle_json` output 可以恢复为 `ParseResult`。
- doclib 当前缓存 JSON 可以恢复为 `ParseResult`。
- 历史旧 CLI `pdf_info` 产物经运行时 legacy 分支单向恢复；无法识别的旧 payload 按 stale 处理并要求重新解析。

## Phase 2: Validator

任务:

1. 生产代码暂不提供 validator API。
2. 单测中保留 `validate_pages()` 与 `ValidationIssue` 作为 test-local helper。
3. 单测已覆盖 P0 页面树校验:
   - page_idx
   - block index/type/bbox（固定版式顶层 block 强制 bbox）
   - 自然语言 block 的行内 `content` 为合法 InlineSpan 列表
4. 待补 envelope-level 校验:
   - schema_version
   - pages list
   - page_count
5. 增加 fixtures。

验收:

- Pipeline/VLM/Hybrid/Office 至少各有一个 fixture。
- validator 能区分 error / warning。
- 当前已知 Office unknown bbox 以 warning 处理，不阻塞。

## Phase 3: Agent Locator

任务:

1. （已完成）定义 locator（`mineru/doclib/locators.py`）。
2. （已完成）实现 block locator（`locator_for_block()` / `block_ref()` / `block_char_ref()`，doclib server 与 CLI 已消费）。
3. 待实现 citation record helper。
4. doclib 已持久化 `short_id` 与 `sha256`；`parses` 表暂无 schema_version 列，仍待补。

验收:

- 同一文件重复解析 locator 稳定。
- 不同 tier 的 block reference 可区分。
- unknown bbox 输出 `bbox_known=false`。
- Agent 可以从 block reference 回查 page/block。

## Phase 4: Backend Normalization

任务:

1. Pipeline: 检查 index 全页稳定性。
2. VLM: 确认 bbox 为 schema 约定的归一化坐标（0-1）且不含内部 metadata。
3. Hybrid: 将 features/models 进入 `_meta`。
4. Office: 明确 unknown bbox（`bbox: null`）语义。
5. HTML: 修正必填字段和 DOM order index。
6. 所有 backend 输出通过 validator。

验收:

- 每个 backend 生成 canonical envelope。
- 每个 backend 可生成 locator。
- `PageInfo` 不携带 `_backend` 等 backend metadata。

## Phase 5: Rendering 收敛

任务:

1. render facade 接受 envelope 或 pages + meta。
2. 为 markdown 增加可选 Agent marker。
3. 明确 Structured Content schema。
4. 把 Office-specific structured_content 逐步收敛到 type-specific helper。

验收:

- Markdown / Content List / Structured Content 均可从 canonical envelope 生成。
- renderer 不依赖 `pdf_info`。
- backend-specific dispatch 数量下降。

## Phase 6: 历史数据迁移

任务:

1. 为 doclib 缓存结果提供 lazy migration。
2. 评估历史 `pdf_info` CLI 输出是否仍需离线批量迁移命令（单文档恢复已由运行时 legacy 分支覆盖），或仅保留重新生成说明。
3. 对缺少 sha256 的历史数据给出明确错误或要求调用方提供文件 hash。

验收:

- 老 `*_middle.json` 能被新版 SDK 读取。
- 无 sha256 的数据不能生成可严格校验的跨文档引用，但仍可 render。
- 迁移不会改变用户可见 markdown 输出。

## 风险

| 风险 | 缓解 |
|------|------|
| 强行要求 Office bbox 导致大量无效框 | 使用 `bbox_known=false`，先承认 unknown。 |
| 改 envelope 破坏旧 CLI | 运行时 legacy 分支单向迁移可识别旧 payload；不可识别的按 stale 处理并重新解析，不静默降级。 |
| locator 因 index 不稳定而漂移 | 先做 normalization，再生成 locator。 |
| render 收敛过大 | 分阶段，先 facade，后 type-specific helper。 |
| filename 泄露隐私 | `_meta.file.filename` 默认可为空。 |

## 首批可执行任务

1. 实现 `normalize_middle_json()`。
2. 增加 envelope-level validator。
3. 修正 HTML 解析必填字段。
4. 定义 `locator_for_block()`。
5. 给 Pipeline/VLM/Office 各加一个 fixture。
6. 复核历史 `pdf_info` 迁移路径（运行时 legacy 分支已覆盖单文档恢复，确认离线批量迁移是否仍有必要）。

完成这些任务后，Middle JSON 就可以支撑 API/SDK 的 `middle_json` output、doclib 缓存恢复和 Agent citation 的第一版闭环。
