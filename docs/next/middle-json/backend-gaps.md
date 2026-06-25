# Backend 差异

状态: Draft
读者: backend 开发者、render 开发者、SDK 开发者
范围: Pipeline / VLM / Hybrid / Office / HTML 与当前事实标准的差异、影响和工作项
底稿: `../../../NEXT-JSON.md`

## 状态分级

| 状态 | 含义 |
|------|------|
| 已解决 | 当前代码已基本满足统一结构。 |
| 部分解决 | 已有统一类型或 facade，但仍有 backend-specific 差异。 |
| 未解决 | 仍缺正式 schema、normalization 或实现。 |

## 总览

| 问题 | 当前状态 | 影响 | 下一步 |
|------|----------|------|--------|
| typed schema | 已解决 | `PageInfo/Block/Line/Span` 已是事实标准。 | 文档化并增加 validation。 |
| 顶层 envelope | 未解决 | `pdf_info/_backend`、`pages`、`ParseResult.pages` 并存。 | 定义 canonical envelope。 |
| bbox 缺失 | 部分解决 | Office/HTML 使用 `EMPTY_BBOX`，Agent 定位不足。 | 标准化 unknown bbox 语义和补齐策略。 |
| page_size 缺失 | 部分解决 | Office/HTML 可能为空，structured_content bbox 归一化受影响。 | Office/HTML 明确 page_size 策略。 |
| index 稳定性 | 部分解决 | reading order 可用，但 Agent locator 不够稳定。 | normalization 阶段重编号。 |
| `preproc_blocks` | 部分解决 | PDF backend 有，Office/HTML 主要直接产出 `para_blocks`。 | 明确 public schema 是否保留。 |
| render 统一 | 部分解决 | 有统一 facade，内部仍按 backend dispatch。 | 收敛 backend-specific content list。 |
| `_backend` | 部分解决 | render 依赖临时字段。 | 迁移到 envelope `_meta.backend`。 |
| locator | 部分解决 | Agent 需要稳定 page/block 引用。 | 锁定 locator helper 并补齐输出契约。 |

## Pipeline

现状:

- 使用 `PageInfo` / `Block` / `Line` / `Span`。
- 通常有 `page_size` 和 bbox。
- 有 `preproc_blocks`，并经过 para split 生成 `para_blocks`。
- `doc_title` / `paragraph_title` 等类型可能经后处理转为 `title`。
- structured_content 的 PDF 后端实现已收敛到 `render/structured_content.py`；Office 仍保留 office-specific converter。

已解决:

- typed structure 已接入。
- PDF page index 修正已有处理。
- `middle_json_utils.append_pages()` 抽出了部分共享构建逻辑。

仍需工作:

1. 明确 `preproc_blocks` 是否写入 public envelope。
2. 继续确认 PDF 通用 structured_content converter 与 Office converter 的字段差异。
3. 确认 block index 在 `para_blocks + discarded_blocks + children` 中是否满足全页稳定排序。
4. 清理或规范 `doc_title` / `paragraph_title` / `vertical_text` 等类型进入 render 前的归一化规则。

验收:

- Pipeline 输出可以通过 validator。
- 不依赖 backend-specific `_backend` 也能 render markdown。
- 同一文件同一版本重复解析，locator 稳定。

## VLM

现状:

- 使用 typed structure。
- 通常有 `page_size` 和 bbox。
- VLM 归一化坐标已在转换阶段转为页面坐标。
- text block 粒度通常是 1 行 1 span。
- 有 VLM 2.5 独有类型，如 `code`、`algorithm`、`ref_text`、`phonetic`、`header`、`footer` 等。

已解决:

- VLM 输出已进入 `PageInfo`。
- `cleanup_internal_para_block_metadata()` 已用于清理内部字段。
- markdown render 已走统一 facade。

仍需工作:

1. 明确 VLM block 粒度与 Pipeline OCR 粒度的兼容语义。
2. 统一 VLM-specific type 在 content_list / Agent citation 中的表达。
3. 确认归一化坐标绝不进入 public Middle JSON。
4. 确认 VLM/Hybrid 与 Pipeline 在统一 structured_content converter 下的字段一致性。

验收:

- VLM 输出不出现归一化 bbox。
- VLM 特有 block type 能被 renderer 和 Agent locator 识别。
- 默认选择得到的 `pro` 结果可以恢复为 `ParseResult`。

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

- Office 已转换为 typed `PageInfo`。
- 直接写入 `para_blocks`，通常没有 `preproc_blocks`。
- 大量 bbox 使用 `EMPTY_BBOX`。
- `page_size` 当前通常为空。
- 保留 Office 特有字段，如 `section_number`、`anchor`、`is_numbered_style`。
- Office render 仍有专门逻辑。

已解决:

- 已从“无 bbox 字段”推进到 typed bbox 字段，但值常为 unknown。
- 标题编号、目录 anchor 已有一定结构化处理。
- image/table/chart/list/index 已向统一 block tree 靠拢。

仍需工作:

1. 定义 Office 的 page_size 策略。
2. 定义 `EMPTY_BBOX` 在 Office 中的正式含义。
3. 将 Office `anchor` / `_style` / `_children` 等字段决定是否公开。
4. 收敛 Office render 到通用 render 能消费的结构。
5. 检查 Office list/index child block 的 index 是否稳定。

验收:

- Office 输出可以通过 validator。
- Office 标题、目录、列表可以生成稳定 locator。
- 对没有真实 bbox 的 block，Agent 引用能明确标记 `bbox_known=false`。

## HTML

现状:

- `HtmlParser` 已存在，但当前构造 `Span` / `Block` / `Line` 时存在与 dataclass 必填字段不一致的风险。
- HTML 输出映射到单页 `PageInfo(page_idx=0)`。
- page_size、bbox、index 策略尚未稳定。

仍需工作:

1. 修正 `HtmlParser` 所有 `Span` / `Line` / `Block` 构造，补齐 `bbox` 和 `index`。
2. 定义 HTML 的 page_size。可以为 `None`，但要明确。
3. 为 DOM order 生成稳定 block index。
4. 明确 HTML `img src`、table html、code block 的 locator 和隐私策略。

验收:

- HTML parser 单测覆盖常见标签。
- HTML 输出可 render markdown/content_list。
- HTML 输出可通过 validator。

## 跨 backend 工作项

P0:

1. 定义 canonical envelope。
2. 实现 validator。
3. 实现 migration: `pdf_info/_backend` -> envelope。
4. 实现并锁定 locator。
5. 修正 `ParseResult.from_dict()` / `from_json()`。

P1:

1. 收敛 Office/HTML 与 PDF structured_content 字段差异。
2. 统一 Office/HTML unknown bbox 语义。
3. 公开或隐藏 Office style/hyperlink 内部字段。
4. 将 `_backend` 从 `PageInfo` 迁移到 `_meta.backend`。

P2:

1. 增加 schema fixtures。
2. 增加跨 backend regression。
3. 对历史 middle_json 做批量 migration 工具。
