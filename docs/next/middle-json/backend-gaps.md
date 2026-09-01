# Backend 差异

状态: Draft
读者: backend 开发者、render 开发者、SDK 开发者
范围: Pipeline / VLM / Hybrid / Office / HTML 与当前事实标准的差异、影响和工作项
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 状态分级

| 状态 | 含义 |
|------|------|
| 已解决 | 当前代码已基本满足统一结构。 |
| 部分解决 | 已有统一类型或 facade，但仍有 backend-specific 差异。 |
| 未解决 | 仍缺正式 schema、normalization 或实现。 |

## 总览

| 问题 | 当前状态 | 影响 | 下一步 |
|------|----------|------|--------|
| typed schema | 已解决 | `PageInfo` 与 Pydantic block tree 已是事实标准。 | 保持严格 validation。 |
| 顶层 envelope | 部分解决 | 当前运行时使用 `schema_version + pages`；历史 `pdf_info/_backend` 仍是离线迁移对象。 | 补 canonical `_meta` 与 envelope validator。 |
| bbox 缺失 | 已解决 | 非 PDF Flash 文档允许无 bbox；PDF 顶层 block 仍强制 bbox。 | renderer 与 locator 不得把缺失值当真实坐标。 |
| page_size 缺失 | 已解决 | schema 3.0 的 `PageInfo` 不再包含 page_size。 | 无。 |
| index 稳定性 | 部分解决 | reading order 可用，但 Agent locator 不够稳定。 | normalization 阶段重编号。 |
| `preproc_blocks` | 已解决 | schema 3.0 只公开统一 `blocks`。 | 无。 |
| render 统一 | 已解决 | 四种输出通过统一 render facade 消费严格 MiddleJson。 | 保持格式实现单向依赖共享后处理。 |
| `_backend` | 部分解决 | render 依赖临时字段。 | 迁移到 envelope `_meta.backend`。 |
| locator | 部分解决 | Agent 需要稳定 page/block 引用。 | 锁定 locator helper 并补齐输出契约。 |

## Pipeline

现状:

- PDF 分析走统一 `doc_analyze()` 入口（`mineru/backend/analyze.py`），产出严格 `ModelJson` 后经 `model_json_to_middle_json()` 统一转换为 schema 2.0 `MiddleJson`，与 VLM/Hybrid/Office/HTML 共用同一 ModelJson → MiddleJson 路径。
- 输出只有 `PageInfo` / 严格 Pydantic block tree；不再有 `Line` / 几何 `Span` / `preproc_blocks` / `para_blocks` 等中间结构。
- 固定版式（`pdf`/`ofd`）顶层 block 强制携带 `bbox`（归一化坐标），并校验 page_idx 与顶层 index 唯一有序。
- `doc_title` / `paragraph_title` 等类型直接进入 block tree，不再后处理转为 `title`。
- structured content 由统一 renderer 直接消费 `MiddleJson` 生成，不再存在 PDF/Office backend converter 分发。

已解决:

- typed structure 已接入并通过严格 validation。
- PDF page index 修正已有处理；显式抽页映射由 `ModelJson.page_index_map` 承载。

仍需工作:

1. 持续校准 PDF 解析质量并保持 strict validation 不回退。
2. 确认 block index 在含嵌套 children 的 block tree 中满足全页稳定排序。
3. 确认 `doc_title` / `paragraph_title` / `vertical_text` 等类型进入 render 前的归一化规则。

验收:

- Pipeline 输出可以通过 validator。
- 不依赖 backend-specific `_backend` 也能 render markdown。
- 同一文件同一版本重复解析，locator 稳定。

## VLM

现状:

- 使用 typed structure。
- block 统一带 `bbox` 字段，schema 内为归一化坐标（0-1），`null` 表示 unknown。
- text block 粒度通常是 1 行 1 span。
- 有 VLM 2.5 独有类型，如 `code`、`algorithm`、`ref_text`、`phonetic`、`header`、`footer` 等。

已解决:

- VLM 输出已进入 `PageInfo`。
- 内部 metadata 已在统一 ModelJson → MiddleJson 转换中剥离，不进入 public Middle JSON。
- markdown render 已走统一 facade。

仍需工作:

1. 明确 VLM block 粒度与统一 PDF 分析文本粒度的兼容语义。
2. 统一 VLM-specific type 在 structured_content / Agent citation 中的表达。
3. 确认进入 public Middle JSON 的 bbox 均为 schema 约定的归一化坐标（0-1）。
4. 确认统一 structured_content renderer 下各来源字段的一致性。

验收:

- VLM 输出的 bbox 均为 schema 约定的归一化坐标（0-1）或 `null`，不出现页面坐标。
- VLM 特有 block type 能被 renderer 和 Agent locator 识别。
- 默认选择得到的 `standard` 结果可以恢复为 `ParseResult`。

## Hybrid

现状:

- 使用 typed structure。
- 同时融合 Pipeline 和 VLM 信息。
- `_ocr_enable` / `_vlm_ocr_enable` 等信息在旧结构里曾作为 backend metadata 出现。
- Hybrid 有自己的 analyze 和 middle_json conversion。

已解决:

- Hybrid 输出已是 `PageInfo` list。
- 共享了部分 PDF backend 构建和 post OCR 逻辑。

仍需工作:

1. 将 hybrid 特有 feature 进入 `_meta.features`。
2. 明确 hybrid 中 `model_used` / `models` 的记录粒度。
3. 检查 Hybrid 的 block type 与 Pipeline/VLM 是否有同义项重复。
4. 确认跨页表格合并后的 locator 稳定性。

验收:

- Hybrid 输出带可追踪的 `_meta.features`。
- Hybrid 中每个 Agent citation 可追溯到原 page 和 bbox。

## Office

现状:

- Office 已通过统一 `doc_analyze()` 转换为 typed `PageInfo`。
- 与其它 backend 共用统一 `blocks` block tree，不再有独立的 `para_blocks` / `preproc_blocks`。
- 大量 block 的 `bbox` 为 `null`（unknown），由 strict model 直接表达。
- `PageInfo` 不携带 `page_size`。
- 保留 Office 特有字段，如 `section_number`、`anchor`、`is_numbered_style`。
- Office render 仍有专门逻辑。

已解决:

- 已从“无 bbox 字段”推进到 typed bbox 字段，但值常为 unknown。
- 标题编号、目录 anchor 已有一定结构化处理。
- image/table/chart/list/index 已向统一 block tree 靠拢。

仍需工作:

1. 定义 Office unknown bbox（`bbox: null`）在 citation 与 UI 中的展示语义。
2. 将 Office `anchor` / `_style` / `_children` 等字段决定是否公开。
3. 收敛 Office render 到通用 render 能消费的结构。
4. 检查 Office list/index child block 的 index 是否稳定。

验收:

- Office 输出可以通过 validator。
- Office 标题、目录、列表可以生成稳定 locator。
- 对没有真实 bbox 的 block，Agent 引用能明确标记 `bbox_known=false`。

## HTML

现状:

- HTML 经统一 `doc_analyze()` 和 ModelJson → MiddleJson 3.0 路径映射到单页 `PageInfo(page_idx=0)`。
- 顶层 block 使用 DOM 顺序生成稳定 index，HTML 不生成 bbox，也不再构造历史 `Line`/`Span`。
- 标题、正文、列表、表格、图片、代码、公式和页面脚注复用统一 block 与 renderer。
- 本地/data 图片进入 `image_base64`，远程图片进入受限 `image_url`，解析阶段不下载远程资源。

仍需工作:

1. 根据真实文章、论坛和文档站语料持续校准 `auto` 正文选择阈值。
2. 如需动态页面，再单独定义浏览器执行、隔离、超时和坐标契约。

验收:

- HTML 单测覆盖正文回退、常见标签、资源限制、路径逃逸和危险 URL。
- HTML 输出可稳定 render Markdown、HTML、DOCX 与 Structured Content，并通过严格 validator。

## 跨 backend 工作项

P0:

1. 定义 canonical envelope。
2. 实现 validator。
3. 设计历史 migration: `pdf_info/_backend` -> envelope。
4. 实现并锁定 locator。
5. （已完成）修正 `ParseResult.from_dict()` / `from_json()`，兼容 schema 2.0、legacy `pdf_info` 与 schema 1.0 `pages` 包装。

P1:

1. 收敛 Office/HTML 与 PDF structured_content 字段差异。
2. 统一 Office/HTML unknown bbox 语义。
3. 公开或隐藏 Office style/hyperlink 内部字段。
4. 明确 backend 信息在 envelope `_meta` 中的记录方式（当前 `PageInfo` 已不携带 `_backend`）。

P2:

1. 增加 schema fixtures。
2. 增加跨 backend regression。
3. 对历史 middle_json 做批量 migration 工具。
