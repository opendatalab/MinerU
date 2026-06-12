# Canonical Envelope

状态: Draft
读者: SDK 开发者、API 开发者、doclib 开发者
范围: Middle JSON 顶层结构、metadata、版本和兼容输入
底稿: `../../../NEXT-JSON.md`

## 目标结构

下一版 canonical envelope:

```json
{
  "schema_version": "1.0",
  "pages": [],
  "_meta": {
    "mineru_version": "2.x",
    "backend": "pipeline",
    "tier": "standard",
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
- 兼容读取旧结构 `pdf_info`。
- `_meta.backend` 取代长期依赖 `PageInfo._backend`。
- `schema_version` 放在顶层，便于快速判断 migration。

## 字段

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `schema_version` | string | 是 | 当前 `"1.0"`。 |
| `pages` | list[PageInfo] | 是 | typed pages 的 JSON 表达。 |
| `_meta` | object | 是 | 元数据。 |

## `_meta`

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `mineru_version` | string | 是 | 生成该结果的 MinerU 版本。 |
| `backend` | string | 是 | `pipeline`、`vlm`、`hybrid`、`office`、`html`、`flash`。 |
| `tier` | string 或 null | 是 | `flash`、`standard`、`pro` 的解析结果语义；未经过 tier 解析的工具层结果可为 `null`。 |
| `file` | object | 是 | 文件级信息。 |
| `features` | object | 是 | 本次解析启用的能力。 |
| `models` | object | 是 | 实际模型信息。 |
| `parsed_at` | string 或 null | 否 | 解析时间。默认不参与 chunk id。 |

## `_meta.file`

| 字段 | 类型 | 必带 | 说明 |
|------|------|:--:|------|
| `sha256` | string | 是 | 原文件 SHA-256。Agent chunk id 必需。 |
| `page_count` | integer | 是 | `len(pages)`。 |
| `filename` | string 或 null | 否 | 原文件名。存在隐私争议，默认可为 null。 |

如果缺少 `sha256`，不能生成稳定 chunk id。migration 可以允许临时为空，但 Agent locator 功能必须报错。

## `_meta.features`

建议字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `ocr_enabled` | bool | 是否启用 OCR。 |
| `vlm_ocr_enabled` | bool | Hybrid 中 VLM OCR 是否启用。 |
| `formula_enabled` | bool | 是否启用公式识别。 |
| `table_enabled` | bool | 是否启用表格识别。 |
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

读入函数必须支持三种输入:

### 1. Canonical envelope

```json
{
  "schema_version": "1.0",
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

### 3. 旧 CLI middle_json

```json
{
  "pdf_info": [],
  "_backend": "pipeline",
  "_version_name": "2.x"
}
```

migration 应转换为:

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

1. 如果 payload 是 list，视为 pages。
2. 如果 payload 有 `pages`，读取 pages。
3. 如果 payload 有 `pdf_info`，读取 `pdf_info`。
4. 如果 payload 有 `_backend`，作为 backend fallback。
5. 如果没有 sha256，则保留 null，但禁用 chunk id。
6. 输出必须是 canonical envelope。

## Validator

目标 API:

```python
def validate_middle_json(payload: dict) -> list[ValidationIssue]: ...
```

P0 校验:

- 有 `schema_version`。
- 有 `pages` list。
- 每个 page 有 `page_idx`。
- 每个 block 有 `index`、`type`、`bbox`。
- 每个 line 有 `bbox`、`spans`。
- 每个 span 有 `type`、`bbox`。
- `page_count == len(pages)`。

P1 校验:

- block index 页内唯一。
- locator 可生成。
- bbox 在 page_size 范围内，unknown bbox 除外。
- 内部字段不出现在 public output。

## 与 `ParseResult`

`ParseResult.to_dict()` 目标应输出 canonical envelope 或至少输出可被 `normalize_middle_json()` 接受的结构。

建议:

- `ParseResult.to_dict()` 输出 `{"pages": ...}` 保持轻量。
- 新增 `ParseResult.to_envelope(meta=...)` 输出 canonical envelope。
- `ParseResult.from_dict()` 调用 `normalize_middle_json()`。

## 未决问题

`schema_version` 位置、`filename` 是否默认写入、`parsed_at` 是否默认写入，集中维护在 [开放问题清单](../open-questions.md)。
