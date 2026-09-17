# Remote 3.x Middle JSON 兼容 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 doclib remote 解析链路（HTTP/ZIP）临时兼容 3.x remote server 返回的历史 middle_json，同时保持 4.0 新协议行为不变。

**Architecture:** 把 doclib 缓存里的 legacy 读取逻辑抽到 `backend/postprocess/legacy_middle_json.py` 共享层，删除 `doclib/core/middle_json.py`；`parser/api_client.py` 的 remote 结果入口先严格读新协议、失败后走共享 legacy 读取器。

**Tech Stack:** Python 3.10+ / pydantic (DocVortex schema) / pytest

---

## File Structure

- **Create** `mineru/backend/postprocess/legacy_middle_json.py` — legacy middle_json 读取器（唯一实现）。
- **Delete** `mineru/doclib/core/middle_json.py` — 逻辑全部上移，无残留。
- **Modify** `mineru/parser/api_client.py:1161` — remote 入口加 legacy fallback。
- **Modify** `mineru/doclib/background/compaction.py`、`mineru/doclib/services/parse_svc.py` — 改 import 与函数名。
- **Modify** `tests/unittest/test_doclib_middle_json.py` — 改 import 与函数名。
- **Modify** `tests/unittest/test_parser_api_contract.py:814,1055,1078` — 3 个「拒绝」测试翻转为「接受」。
- **Modify** `docs/next/middle-json/doclib-compatibility.md`、`docs/doclib-middle-json-compat-validation.md` — 更新边界描述。

---

## Task 1: 提取共享 legacy 读取模块（重构，无行为变化）

**Files:**
- Create: `mineru/backend/postprocess/legacy_middle_json.py`
- Modify: `mineru/doclib/background/compaction.py:17`
- Modify: `mineru/doclib/services/parse_svc.py:31`
- Modify: `tests/unittest/test_doclib_middle_json.py:10`
- Delete: `mineru/doclib/core/middle_json.py`

- [ ] **Step 1: 新建共享模块并完整移入 legacy 逻辑**

创建 `mineru/backend/postprocess/legacy_middle_json.py`：

```python
# Copyright (c) Opendatalab. All rights reserved.
"""历史/旧协议 Middle JSON 读取与转换边界。

两个历史分支集中在此，未来可一并移除。读取只构造内存对象，不重写文件、不重新推理。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from docvortex.schema import DocumentMetadata, MiddleJson, ModelJson, Producer

from ...integrations.docvortex import MinerUMetadata, build_metadata, validate_mineru_metadata

_LEGACY_PAGE_FIELDS = frozenset({"page_size", "preproc_blocks", "para_blocks", "discarded_blocks"})
_LEGACY_V2_FIELDS = frozenset(
    {"schema_version", "pages", "is_full_document", "file_suffix", "effort", "parse_mode", "mineru_version", "extensions"}
)


def _legacy_metadata(payload: dict[str, Any], *, version_keys: tuple[str, ...]) -> DocumentMetadata:
    """保留可辨识的历史生产版本，缺失时标记未知而不使用当前程序版本。"""
    versions = [payload[key].strip() for key in version_keys if isinstance(payload.get(key), str) and payload[key].strip()]
    if len(set(versions)) > 1:
        raise ValueError("Conflicting legacy producer versions")
    version = versions[0] if versions else "unknown"
    return DocumentMetadata(
        file_suffix=payload.get("file_suffix", "pdf"),
        producer=Producer(name="mineru", version=version),
    )


def _legacy_extensions(payload: dict[str, Any]) -> dict[str, Any]:
    """只迁移可靠的档位及模式，保留应用扩展并拒绝保留字段冲突。"""
    extensions = deepcopy(payload.get("extensions", {}))
    if not isinstance(extensions, dict):
        raise ValueError("legacy extensions must be a JSON object")
    efforts = {
        payload[key]
        for key in ("_effort", "effort")
        if isinstance(payload.get(key), str) and payload[key] in {"flash", "medium", "high", "xhigh"}
    }
    if len(efforts) > 1:
        raise ValueError("Conflicting legacy effort records")
    effort = next(iter(efforts), None)
    mode = payload.get("parse_mode")
    ocr_enabled = payload.get("_ocr_enable")
    if mode in ("txt", "ocr") and type(ocr_enabled) is bool and (mode == "ocr") != ocr_enabled:
        raise ValueError("Conflicting legacy parse_mode and _ocr_enable")
    if mode not in ("txt", "ocr"):
        mode = ("ocr" if ocr_enabled else "txt") if type(ocr_enabled) is bool else None

    if "mineru" in extensions:
        existing = MinerUMetadata.model_validate(extensions["mineru"])
        if mode is not None and existing.parse_mode != mode:
            raise ValueError("Conflicting legacy parse_mode and extensions.mineru")
        if effort is not None and existing.tier != build_metadata(effort=effort, parse_mode="txt")["mineru"]["tier"]:
            raise ValueError("Conflicting legacy effort and extensions.mineru")
    elif effort is not None and mode is not None:
        extensions.update(build_metadata(effort=effort, parse_mode=mode))
    return extensions


def _read_legacy_345(payload: dict[str, Any], *, page_field: str) -> MiddleJson:
    """把可识别的 3.4.5 页面转为当前 raw ModelJson，再做确定性后处理。"""
    from .legacy_schema_adapter import legacy_page_to_model_list
    from docvortex.postprocess.document import model_json_to_middle_json

    pages = payload[page_field]
    if not isinstance(pages, list) or any(not isinstance(page, dict) for page in pages):
        raise ValueError("legacy pages must be a list of page objects")
    for page in pages:
        if "blocks" in page or (page_field == "pages" and not _LEGACY_PAGE_FIELDS.intersection(page)):
            raise ValueError("Mixed or unrecognized legacy page structure")
    indices = [page.get("page_idx", index) for index, page in enumerate(pages)]
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("legacy page indices must be non-negative integers")
    model = ModelJson(
        pages=[legacy_page_to_model_list(page) for page in pages],
        page_index_map=[] if indices == list(range(len(pages))) else indices,
        metadata=_legacy_metadata(payload, version_keys=("_version_name", "mineru_version")),
        extensions=_legacy_extensions(payload),
    )
    if "is_full_document" in payload:
        full_document = payload["is_full_document"]
        if type(full_document) is not bool or full_document != model.is_full_document:
            raise ValueError("Conflicting legacy is_full_document and page indices")
    return model_json_to_middle_json(model)


def _read_legacy_v2(payload: dict[str, Any]) -> MiddleJson:
    """只移动旧 Schema 2.0 外层字段，页面树直接经过当前类型校验。"""
    unexpected = payload.keys() - _LEGACY_V2_FIELDS
    if unexpected:
        raise ValueError(f"Conflicting or unsupported legacy Middle JSON fields: {sorted(unexpected)}")
    return MiddleJson(
        pages=payload.get("pages"),
        is_full_document=payload.get("is_full_document"),
        metadata=_legacy_metadata(payload, version_keys=("mineru_version",)),
        extensions=_legacy_extensions(payload),
    )


def read_legacy_middle_json(payload: dict[str, Any]) -> MiddleJson:
    """先识别显式协议，再转换两个历史分支；失败时绝不跨协议兜底。"""
    if not isinstance(payload, dict):
        raise ValueError("Middle JSON must be an object; source reparse required")
    if "schema" in payload:
        document = MiddleJson.from_dict(payload)
    else:
        if {"metadata", "producer", "page_index_map"}.intersection(payload):
            raise ValueError("Mixed document envelope; source reparse required")
        version = payload.get("schema_version")
        source = deepcopy(payload)
        if "pdf_info" in source:
            if version is not None or "pages" in source:
                raise ValueError("Mixed legacy pdf_info envelope; source reparse required")
            document = _read_legacy_345(source, page_field="pdf_info")
        elif version == "1.0" and "pages" in source:
            document = _read_legacy_345(source, page_field="pages")
        elif version == "2.0" and "pages" in source:
            document = _read_legacy_v2(source)
        else:
            raise ValueError("Unsupported Middle JSON format; source reparse required")
    validate_mineru_metadata(document)
    return document


__all__ = ["read_legacy_middle_json"]
```

- [ ] **Step 2: 更新 doclib 两个调用方**

`mineru/doclib/background/compaction.py` 把第 17 行：

```python
from ..core.middle_json import read_cached_middle_json
```

改为：

```python
from ...backend.postprocess.legacy_middle_json import read_legacy_middle_json
```

并把文件内 2 处 `read_cached_middle_json(` 调用改为 `read_legacy_middle_json(`（第 27、153 行）。

`mineru/doclib/services/parse_svc.py` 把第 31 行同样的 import 改为：

```python
from ...backend.postprocess.legacy_middle_json import read_legacy_middle_json
```

并把文件内 2 处 `read_cached_middle_json(` 调用改为 `read_legacy_middle_json(`（第 1541、1572 行）。

- [ ] **Step 3: 更新测试 import 与调用**

`tests/unittest/test_doclib_middle_json.py` 第 10 行改为：

```python
from mineru.backend.postprocess.legacy_middle_json import read_legacy_middle_json
```

并把文件内所有 `read_cached_middle_json(` 调用替换为 `read_legacy_middle_json(`。

- [ ] **Step 4: 删除旧模块**

```bash
git rm mineru/doclib/core/middle_json.py
```

- [ ] **Step 5: 跑重构回归测试**

Run: `.venv/bin/python -m pytest tests/unittest/test_doclib_middle_json.py tests/unittest/test_doclib_legacy_schema_adapter.py tests/unittest/test_doclib_legacy_page_ranges.py tests/unittest/test_doclib_cache_semantics.py -q`
Expected: PASS（纯重构，行为不变）

- [ ] **Step 6: Commit**

```bash
git add mineru/backend/postprocess/legacy_middle_json.py mineru/doclib/background/compaction.py mineru/doclib/services/parse_svc.py tests/unittest/test_doclib_middle_json.py mineru/doclib/core/middle_json.py
git commit -m "refactor: extract shared legacy middle JSON reader to backend postprocess"
```

---

## Task 2: remote 路径加 legacy fallback（TDD）

**Files:**
- Modify: `tests/unittest/test_parser_api_contract.py:814,1055,1078`
- Modify: `mineru/parser/api_client.py:1161`

- [ ] **Step 1: 翻转三个「拒绝」测试为「接受」**

在 `tests/unittest/test_parser_api_contract.py` 中做以下三处替换。

（1）第 814 行 `test_api_client_rejects_legacy_official_layout_json` 整体替换为：

```python
def test_api_client_accepts_legacy_official_layout_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """官方 API ZIP 内的历史 pdf_info 文档迁移为当前 MiddleJson，不再拒绝。"""
    parser = MinerUApiParser(
        api_url="https://mineru.net/api",
        tier="standard",
        include_images=True,
        include_model_output=True,
    )
    zip_ref = {"file_id": "file-zip", "bytes": 10}
    middle_json = {
        "_backend": "hybrid",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 200],
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "image",
                        "bbox": [0, 0, 10, 10],
                        "lines": [
                            {
                                "bbox": [0, 0, 10, 10],
                                "spans": [{"type": "image", "bbox": [0, 0, 10, 10], "image_path": "chart.png"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("layout.json", json.dumps(middle_json, ensure_ascii=False))
        archive.writestr("images/chart.png", b"chart-bytes")
        archive.writestr("ab3a55b0-2017-4e35-8507-ab8e2c012160_model.json", json.dumps(_model_payload()))

    monkeypatch.setattr(api_client, "_download_bytes", lambda _parser, ref: zip_buffer.getvalue() if ref is zip_ref else b"")

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"zip": zip_ref}}],
        },
        "demo.pdf",
        parser,
    )

    assert isinstance(result, ParseResult)
    assert isinstance(result.pages, list)
```

（2）第 1055 行 `test_api_client_rejects_remote_pdf_info_middle_json` 整体替换为：

```python
def test_api_client_accepts_remote_pdf_info_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """远程返回历史 pdf_info 协议时迁移为当前 MiddleJson，不猜测但也不拒绝。"""
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    middle_json = {
        "_backend": "hybrid",
        "_version_name": "remote",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    monkeypatch.setattr(api_client, "_download_json", lambda _parser, _outputs: middle_json)

    result = _parse_result_from_job(
        {
            "job_id": "job_1",
            "status": "completed",
            "files": [{"output_files": {"middle_json": {"file_id": "file-middle-json", "bytes": 10}}}],
        },
        "demo.pdf",
        parser,
    )

    assert isinstance(result, ParseResult)
    assert isinstance(result.pages, list)
```

（3）第 1078 行 `test_async_api_client_rejects_remote_pdf_info_middle_json` 整体替换为：

```python
def test_async_api_client_accepts_remote_pdf_info_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """异步客户端同样迁移历史 pdf_info 协议。"""
    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")
    middle_json = {
        "_backend": "hybrid",
        "pdf_info": [{"page_idx": 0, "page_size": [100, 200]}],
    }

    async def _download_json(*_args: object, **_kwargs: object) -> dict[str, object]:
        return middle_json

    monkeypatch.setattr(api_client, "_async_download_json", _download_json)

    result = asyncio.run(
        api_client._async_parse_result_from_job(
            {
                "job_id": "job_1",
                "status": "completed",
                "files": [{"output_files": {"middle_json": {"file_id": "file-middle-json", "bytes": 10}}}],
            },
            "demo.pdf",
            parser,
        )
    )

    assert isinstance(result, ParseResult)
    assert isinstance(result.pages, list)
```

- [ ] **Step 2: 跑测试验证失败**

Run: `.venv/bin/python -m pytest tests/unittest/test_parser_api_contract.py -k "legacy_official_layout_json or remote_pdf_info_middle_json" -q`
Expected: FAIL（旧实现仍拒绝 legacy，断言 `isinstance(result, ParseResult)` 不会到达，而是抛 `_V1APIError`）

- [ ] **Step 3: 实现 fallback**

`mineru/parser/api_client.py:1161` 的 `_parse_result_from_middle_json` 替换为：

```python
def _parse_result_from_middle_json(mid_json: dict[str, Any]) -> ParseResult:
    """把远端 middle_json 恢复为 ParseResult；优先当前协议，临时兼容 3.x 历史协议。"""
    if not isinstance(mid_json, dict):
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object")
    try:
        return ParseResult.from_dict(mid_json)
    except ValueError as exc:
        strict_error = exc
    # 临时：兼容 3.x remote server 的历史 middle_json（pdf_info / 旧 1.0 / 旧 2.0），待其升级到 4.0 后移除。
    from ..backend.postprocess.legacy_middle_json import read_legacy_middle_json

    try:
        return ParseResult(middle_json=read_legacy_middle_json(mid_json))
    except ValueError:
        raise _V1APIError("invalid_middle_json_output", str(strict_error)) from strict_error
```

- [ ] **Step 4: 跑测试验证通过**

Run: `.venv/bin/python -m pytest tests/unittest/test_parser_api_contract.py -k "legacy_official_layout_json or remote_pdf_info_middle_json" -q`
Expected: PASS

- [ ] **Step 5: 全量相关回归**

Run: `.venv/bin/python -m pytest tests/unittest/test_parser_api_contract.py tests/unittest/test_doclib_middle_json.py -q`
Expected: PASS（新协议路径不受影响，doclib 兼容路径不受影响）

- [ ] **Step 6: Commit**

```bash
git add mineru/parser/api_client.py tests/unittest/test_parser_api_contract.py
git commit -m "feat: accept legacy 3.x middle JSON in remote API results"
```

---

## Task 3: 更新兼容边界文档

**Files:**
- Modify: `docs/next/middle-json/doclib-compatibility.md`
- Modify: `docs/doclib-middle-json-compat-validation.md`

- [ ] **Step 1: 修订协议边界说明**

`docs/next/middle-json/doclib-compatibility.md` 第 5 行：

原文：

```
通用 ParseResult、DocVortex codec、HTTP/ZIP 与 Gradio 继续严格读取新协议。
```

改为：

```
通用 ParseResult 与 DocVortex codec 继续严格读取新协议；HTTP/ZIP 临时兼容
3.x remote server 的历史 middle_json，待 remote server 升级到 4.0 后移除该兼容。
```

- [ ] **Step 2: 修订验证记录**

`docs/doclib-middle-json-compat-validation.md` 第 14 行：

原文：

```
- 通用 ParseResult、DocVortex codec、HTTP/ZIP 和 Gradio 没有增加旧格式读取能力。
```

改为：

```
- 通用 ParseResult 与 DocVortex codec 未增加旧格式读取能力；HTTP/ZIP 已临时
  兼容 3.x remote server，待其升级到 4.0 后移除。
```

- [ ] **Step 3: 跑全量相关测试**

Run: `.venv/bin/python -m pytest tests/unittest/test_parser_api_contract.py tests/unittest/test_doclib_middle_json.py tests/unittest/test_doclib_legacy_schema_adapter.py tests/unittest/test_doclib_legacy_page_ranges.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/next/middle-json/doclib-compatibility.md docs/doclib-middle-json-compat-validation.md
git commit -m "docs: update middle JSON compatibility boundary for 3.x remote"
```

---

## Self-Review

- **Spec coverage:** 设计文档的模块布局（Task 1）、remote fallback（Task 2）、文档更新（Task 3）均有对应任务；测试翻转（Task 2 Step 1）覆盖三个锁定点。
- **Placeholder scan:** 无 TBD/占位符；所有代码步骤含完整代码。
- **Type consistency:** 函数名统一为 `read_legacy_middle_json`，import 路径在三处调用方与实现中一致（`...backend.postprocess.legacy_middle_json`）。