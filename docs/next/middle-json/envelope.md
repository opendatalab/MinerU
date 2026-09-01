# Canonical Envelope

状态: Draft
读者: SDK 开发者、API 开发者、doclib 开发者
范围: Middle JSON 顶层结构、metadata、版本和兼容输入
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 目标结构

当前 P0 写出的 `pages` 结构先包含顶层 `schema_version` 和 `pages`，暂不写 `_meta`:

```json
{
  "schema_version": "2.0",
  "pages": []
}
```

后续 canonical envelope 目标结构:

```json
{
  "schema_version": "2.0",
  "pages": [],
  "_meta": {
    "mineru_version": "2.x",
    "backend": "hybrid",
    "tier": "basic",
    "file": {
      "sha256": "...",
      "page_count": 12,
      "filename": null
    },
    "features": {
      "ocr_enabled": true,
      "vlm_ocr_enabled": false
    },
    "models": {}
  }
}
```

设计选择:

- 新结构使用 `pages`，对应 `ParseResult.pages`。
- 运行时只读取当前结构 `pages`；历史 `pdf_info` 文件需要离线迁移或重新生成。
- 当前 `ParseResult.to_dict()` 输出 `schema_version` 加 `MiddleJson` 顶层字段（`pages`、`is_full_document`、`file_suffix`、`effort`、`parse_mode`、`mineru_version`），不携带 `_backend` 等 backend metadata。
- `_meta.backend` 设计为替代旧 payload 中的 `_backend` 字段；当前 schema 2.0 已不再有该字段。
- `schema_version` 放在顶层，便于快速判断 migration。
- 代码常量定义为 `mineru.parser.MIDDLE_JSON_SCHEMA_VERSION`，由 normalize、validate、writer 和 exporter 统一引用。
- 当前 P0 写出路径只增加 `schema_version`，不新增 `_meta`；`_meta` 由后续 canonical envelope migration / writer 引入。

## 字段

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `schema_version` | string | 是 | 当前 `"2.0"`；代码常量为 `MIDDLE_JSON_SCHEMA_VERSION`。 |
| `pages` | list[PageInfo] | 是 | typed pages 的 JSON 表达。 |
| `is_full_document` | bool | 是 | 是否整本文档解析（空 `page_index_map` 时为 `true`）。 |
| `file_suffix` | string | 是 | 输入文件类型（`pdf`、`docx`、`pptx`、`epub`、`html`、`ofd` 等）。 |
| `effort` | string | 是 | 分析强度：`flash`、`medium`、`high`、`xhigh`。 |
| `parse_mode` | string | 是 | `txt` 或 `ocr`。 |
| `mineru_version` | string | 是 | 生成该结果的 MinerU 版本。 |
| `_meta` | object | 后续 | 元数据；当前 P0 写出路径暂不增加。 |

## `_meta`

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `mineru_version` | string | 是 | 生成该结果的 MinerU 版本。 |
| `backend` | string | 是 | `hybrid`、`office`、`html`、`flash`。 |
| `tier` | string 或 null | 是 | `flash`、`basic`、`standard`、`advanced` 的解析结果语义；未经过 tier 解析的工具层结果可为 `null`。 |
| `file` | object | 是 | 文件级信息。 |
| `features` | object | 是 | 本次解析启用的能力。 |
| `models` | object | 是 | 实际模型信息。 |
| `parsed_at` | string 或 null | 否 | 解析时间。默认不参与 locator。 |

## `_meta.file`

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `sha256` | string | 是 | 原文件 SHA-256。Agent locator 严格校验时需要。 |
| `page_count` | integer | 是 | `len(pages)`。 |
| `filename` | string 或 null | 否 | 原文件名。存在隐私争议，默认可为 null。 |

如果缺少 `sha256`，可以生成局部 locator，但不能生成可严格校验的跨文档 block reference。migration 可以允许临时为空，但 Agent citation 功能必须明确降级或报错。

## `_meta.features`

建议字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `ocr_enabled` | bool | 是否启用 OCR。 |
| `vlm_ocr_enabled` | bool | Hybrid 中 VLM OCR 是否启用。 |
| `image_analysis` | bool | 是否启用图片分析。 |

features 是开放字典。新增字段不破坏 schema。

## `_meta.models`

建议字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `layout` | string | layout 模型。 |
| `ocr` | string | OCR 模型。 |
| `formula` | string | 公式模型。 |
| `table` | string | 表格模型。 |
| `vlm` | string | VLM 模型。 |

models 也是开放字典。字段粒度可以随 backend 增加。

## 兼容输入

当前 `ParseResult.from_dict()` 接受 schema 2.0 `pages` envelope，并兼容历史 `pdf_info` 与 schema 1.0 `pages` 包装；无版本标识的裸 `pages` 会被拒绝。目标读入函数可支持以下输入:

### 1. Canonical envelope

```json
{
  "schema_version": "2.0",
  "pages": [],
  "_meta": {}
}
```

直接读取。

### 2. 当前 SDK envelope

```json
{
  "pages": []
}
```

视为无 metadata 的当前结构。migration 应补:

- `schema_version`
- `_meta.backend`，如果调用方提供。
- `_meta.file.sha256`，如果调用方提供。

### 3. 历史旧 CLI middle_json

```json
{
  "pdf_info": [],
  "_backend": "hybrid",
  "_version_name": "2.x"
}
```

当前运行时 `from_dict()` 已兼容 `pdf_info`（以及 schema 1.0 `pages` 包装）作为 legacy 分支；无版本标识的裸 `pages` 会被拒绝。旧产物中的 `_backend: "pipeline"` 不属于当前 schema 字段，legacy 读取时仅被忽略。

历史文件的离线 migration 可转换为:

- `pages = pdf_info`
- `_meta.backend = _backend`
- `_meta.mineru_version = _version_name`
- `schema_version = "1.0"`

## Migration 函数

目标 API:

```python
def normalize_middle_json(
    payload: dict | list,
    *,
    sha256: str | None = None,
    backend: str | None = None,
    tier: str | None = None,
    filename: str | None = None,
) -> dict: ...
```

规则:

1. 运行时 payload 必须是 dict，且 `pages` 必须是 list。
2. 运行时 `from_dict()` 已兼容历史 `pdf_info` 与 schema 1.0 `pages` 包装；无版本标识的裸 `pages` 会被拒绝。
3. 如果没有 sha256，则保留 null，但禁用需要严格校验 source identity 的 citation 能力。
4. 输出必须是 canonical envelope。

## Validator

当前生产代码不提供 envelope validator API；页面树校验逻辑仅保留在单测中作为 test-local helper。

```python
from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
```

目标 envelope API:

```python
def validate_middle_json(payload: dict) -> list[ValidationIssue]: ...
```

P0 校验:

- 有 `schema_version`。
- 有 `pages` list。
- 每个 page 有 `page_idx`。
- 每个 block 有 `index`、`type`；固定版式（`pdf`/`ofd`）顶层 block 必须有 `bbox`。
- 自然语言 block 的 `content` 是合法的 `InlineSpan` 列表（`TextSpan` / `EquationInlineSpan` / `CodeInlineSpan` / `HyperlinkSpan`），span 只表达行内语义，不携带 bbox。
- `page_count == len(pages)`。

P1 校验:

- block index 页内唯一。
- locator 可生成。
- bbox 在 page_size 范围内，unknown bbox 除外。
- 内部字段不出现在 public output。

## 与 `ParseResult`

`ParseResult.to_dict()` 当前输出 `schema_version` 加 `MiddleJson` 顶层字段（`pages`、`is_full_document`、`file_suffix`、`effort`、`parse_mode`、`mineru_version`）。`ParseResult.from_dict()` 当前要求 payload 为 dict，接受 schema 2.0 `pages` envelope、历史 `pdf_info` 与 schema 1.0 `pages` 包装。

建议:

- `ParseResult.to_dict()` 继续输出 `schema_version + pages`。
- 新增 `ParseResult.to_envelope(meta=...)` 输出 canonical envelope。
- `ParseResult.from_dict()` 调用 `normalize_middle_json()`。

## 未决问题

`filename` 是否默认写入、`parsed_at` 是否默认写入，集中维护在 [开放问题清单](../open-questions.md)。
