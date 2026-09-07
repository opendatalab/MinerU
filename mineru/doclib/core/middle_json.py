"""仅供 Doclib 持久化结果使用的 Middle JSON 读取与历史转换边界。

两个历史分支集中在此，未来可一并移除；通用 ParseResult 与 DocVortex codec
不调用本模块。读取只构造内存对象，不重写文件、不重新推理。
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
    from ...backend.postprocess.legacy_schema_adapter import legacy_page_to_model_list
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


def read_cached_middle_json(payload: dict[str, Any]) -> MiddleJson:
    """先识别显式协议，再转换两个历史缓存族；失败时绝不跨协议兜底。"""
    if not isinstance(payload, dict):
        raise ValueError("Cached Middle JSON must be an object; source reparse required")
    if "schema" in payload:
        document = MiddleJson.from_dict(payload)
    else:
        if {"metadata", "producer", "page_index_map"}.intersection(payload):
            raise ValueError("Mixed cached document envelope; source reparse required")
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
            raise ValueError("Unsupported cached Middle JSON format; source reparse required")
    validate_mineru_metadata(document)
    return document


__all__ = ["read_cached_middle_json"]
