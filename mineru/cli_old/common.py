# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from loguru import logger

from mineru.backend.office.docx_analyze import office_docx_analyze
from mineru.backend.office.pptx_analyze import office_pptx_analyze
from mineru.backend.office.xlsx_analyze import office_xlsx_analyze
from mineru.cli_old.visualization import select_pages_for_pdf_visualization
from mineru.parser.base import ParseResult
from mineru.render import render_content_list, render_markdown, render_structured_content
from mineru.render.writer import FileBasedDataWriter
from mineru.utils.backend_options import DEFAULT_BACKEND, DEFAULT_HYBRID_EFFORT, LOCAL_HYBRID_EFFORT, resolve_backend_and_effort
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from mineru.utils.image_payload import ImagePayloadCache
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.utils.pdfium_guard import safe_rewrite_pdf_bytes_with_pdfium, safe_rewrite_pdf_bytes_with_pdfium_result

from ..types import PageInfo
from ..version import __version__

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
if os.getenv("MINERU_LMDEPLOY_DEVICE", "") == "maca":
    import torch

    torch.backends.cudnn.enabled = False


pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]
docx_suffixes = ["docx"]
pptx_suffixes = ["pptx"]
xlsx_suffixes = ["xlsx"]
office_suffixes = docx_suffixes + pptx_suffixes + xlsx_suffixes

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Maximum UTF-8 byte length allowed for task stems used in filenames.
# 200 bytes is chosen to stay well below common filesystem limits (e.g. 255 bytes)
# and to prevent generating excessively long or incompatible filenames.
MAX_TASK_STEM_BYTES = 200


@dataclass(frozen=True)
class PreparedPdfBytes:
    """记录旧 CLI PDF 预处理结果，避免页号映射在 backend 前丢失。"""

    pdf_bytes: bytes
    retained_page_indices: list[int] | None = None
    broken_page_indices: list[int] | None = None


class HybridDependencyError(RuntimeError):
    pass


def build_hybrid_dependency_error_message(backend: str) -> str:
    return (
        f"`{backend}` requires local pipeline dependencies (`mineru[pipeline]`, "
        "including `torch`). Install `mineru[pipeline]` or `mineru[core]`. "
        "The legacy `vlm-http-client` option is now mapped to `hybrid-http-client` "
        "and needs the same local dependencies."
    )


def ensure_backend_dependencies(backend: str) -> None:
    if not backend.startswith("hybrid-"):
        return
    if importlib.util.find_spec("torch") is None:
        raise HybridDependencyError(build_hybrid_dependency_error_message(backend))


def _load_hybrid_analyze_entrypoint(entrypoint_name: str, backend: str) -> Callable[..., Any]:
    ensure_backend_dependencies(backend)
    try:
        hybrid_analyze = importlib.import_module("mineru.backend.hybrid.hybrid_analyze")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HybridDependencyError(build_hybrid_dependency_error_message(backend)) from exc
    return getattr(hybrid_analyze, entrypoint_name)


def utf8_byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_to_utf8_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            truncated = truncated[: exc.start]
    return ""


def normalize_task_stem(stem: str, max_bytes: int = MAX_TASK_STEM_BYTES) -> str:
    return truncate_to_utf8_bytes(stem, max_bytes)


def build_task_stem_candidate(
    stem: str,
    suffix: str = "",
    max_bytes: int = MAX_TASK_STEM_BYTES,
) -> str:
    if utf8_byte_length(f"{stem}{suffix}") <= max_bytes:
        return f"{stem}{suffix}"
    suffix_bytes = utf8_byte_length(suffix)
    if suffix_bytes >= max_bytes:
        return truncate_to_utf8_bytes(suffix, max_bytes)
    return f"{truncate_to_utf8_bytes(stem, max_bytes - suffix_bytes)}{suffix}"


def uniquify_task_stems(
    stems: Sequence[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Assign task-local unique stems while preserving input order."""
    normalized_inputs = [normalize_task_stem(stem) for stem in stems]
    raw_keys = {stem.casefold() for stem in normalized_inputs}
    occurrence_counts: dict[str, int] = {}
    assigned_keys: set[str] = set()
    unique_stems: list[str] = []
    renamed: list[tuple[str, str]] = []

    for stem, normalized_stem in zip(stems, normalized_inputs):
        stem_base = normalized_stem or stem
        stem_key = stem_base.casefold()
        seen_count = occurrence_counts.get(stem_key, 0)
        occurrence_counts[stem_key] = seen_count + 1

        if seen_count == 0 and stem_key not in assigned_keys:
            effective_stem = stem_base
        else:
            suffix = seen_count + 1
            while True:
                candidate = build_task_stem_candidate(stem_base, f"_{suffix}")
                candidate_key = candidate.casefold()
                if candidate_key not in raw_keys and candidate_key not in assigned_keys:
                    effective_stem = candidate
                    break
                suffix += 1

        assigned_keys.add(effective_stem.casefold())
        unique_stems.append(effective_stem)
        if effective_stem != stem:
            renamed.append((stem, effective_stem))

    return unique_stems, renamed


def read_fn(path: str | Path, file_suffix: str | None = None) -> bytes:
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if file_suffix is None:
            file_suffix = guess_suffix_by_bytes(file_bytes, path)
        if file_suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes + office_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {file_suffix}")


def prepare_env(output_dir: str, pdf_file_name: str, parse_method: str) -> tuple[str, str]:
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes(pdf_bytes: bytes, start_page_id: int = 0, end_page_id: int | None = None) -> bytes:
    return safe_rewrite_pdf_bytes_with_pdfium(
        pdf_bytes,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )


def prepare_pdf_bytes_with_page_map(
    pdf_bytes: bytes,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> PreparedPdfBytes:
    """重写 PDF 字节并保留小 PDF 物理页到原始页号的映射。"""
    rewrite_result = safe_rewrite_pdf_bytes_with_pdfium_result(
        pdf_bytes,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
    if rewrite_result.used_original:
        return PreparedPdfBytes(
            pdf_bytes=rewrite_result.pdf_bytes or pdf_bytes,
            retained_page_indices=None,
            broken_page_indices=rewrite_result.broken_page_indices,
        )
    return PreparedPdfBytes(
        pdf_bytes=rewrite_result.pdf_bytes or pdf_bytes,
        retained_page_indices=rewrite_result.retained_page_indices,
        broken_page_indices=rewrite_result.broken_page_indices,
    )


def _prepare_pdf_bytes(pdf_bytes_list: list[bytes], start_page_id: int, end_page_id: int | None) -> list[bytes]:
    """准备处理PDF字节数据"""
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result


def _prepare_pdf_inputs(
    pdf_bytes_list: list[bytes],
    start_page_id: int,
    end_page_id: int | None,
) -> list[PreparedPdfBytes]:
    """批量准备 PDF 输入，并为 backend 保留逐文档页号映射。"""
    return [prepare_pdf_bytes_with_page_map(pdf_bytes, start_page_id, end_page_id) for pdf_bytes in pdf_bytes_list]


def _process_output(
    middle_json: list[PageInfo],
    pdf_bytes: bytes,
    pdf_file_name: str,
    local_md_dir: str,
    local_image_dir: str,
    md_writer: Any,
    f_draw_layout_bbox: bool,
    f_draw_span_bbox: bool,
    f_dump_orig_pdf: bool,
    f_dump_md: bool,
    f_dump_content_list: bool,
    f_dump_middle_json: bool,
    f_dump_model_output: bool,
    f_make_md_mode: str,
    model_output: list[list[dict[str, Any]]] | None = None,
    *,
    process_mode: str,
    backend: str,
    retained_page_indices: list[int] | None = None,
    broken_page_indices: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
) -> None:
    """处理输出文件"""
    visualization_pages = select_pages_for_pdf_visualization(middle_json, retained_page_indices)
    if f_draw_layout_bbox:
        try:
            draw_layout_bbox(visualization_pages, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
        except Exception as exc:
            logger.warning(f"Skipping layout bbox visualization for {pdf_file_name}: {exc}")

    if f_draw_span_bbox:
        try:
            draw_span_bbox(visualization_pages, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
        except Exception as exc:
            logger.warning(f"Skipping span bbox visualization for {pdf_file_name}: {exc}")

    if f_dump_orig_pdf:
        if process_mode in ["hybrid", "vlm"]:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )
        elif process_mode in office_suffixes:
            md_writer.write(
                f"{pdf_file_name}_origin.{process_mode}",
                pdf_bytes,
            )

    image_dir = str(os.path.basename(local_image_dir))
    export_result = ParseResult(
        pages=middle_json,
        _image_cache=image_cache,
        _retained_page_indices=retained_page_indices,
        _broken_page_indices=broken_page_indices,
    )
    final_writer = FileBasedDataWriter(local_image_dir)
    for img_path, img_bytes in export_result.images().items():
        final_writer.write(img_path, img_bytes)
    public_render_pages: list[PageInfo] | None = None

    def get_public_render_pages() -> list[PageInfo]:
        """按需生成 public 渲染页副本，避免 staged base64 载荷进入 markdown/content 输出。"""
        nonlocal public_render_pages
        if public_render_pages is None:
            public_render_pages = export_result.export_pages()
        return public_render_pages

    if f_dump_md:
        md_content_str = render_markdown(
            get_public_render_pages(),
            image_dir,
            no_rich_content=(f_make_md_mode != "mm_markdown"),
        )
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        content_list = render_content_list(get_public_render_pages(), image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        structured_content = render_structured_content(get_public_render_pages(), image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_structured_content.json",
            json.dumps(structured_content, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        dump_dict = export_result.to_dict()
        dump_dict["_backend"] = backend
        dump_dict["_version_name"] = __version__
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(dump_dict, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.debug(f"local output dir is {local_md_dir}")


def _process_hybrid(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    h_lang_list: list[str],
    parse_method: str,
    backend: str,
    f_draw_layout_bbox: bool,
    f_draw_span_bbox: bool,
    f_dump_md: bool,
    f_dump_middle_json: bool,
    f_dump_model_output: bool,
    f_dump_orig_pdf: bool,
    f_dump_content_list: bool,
    f_make_md_mode: str,
    server_url: str | None = None,
    page_index_map_list: list[list[int] | None] | None = None,
    broken_page_indices_list: list[list[int] | None] | None = None,
    **kwargs: Any,
) -> None:
    hybrid_doc_analyze = _load_hybrid_analyze_entrypoint(
        "doc_analyze",
        f"hybrid-{backend}",
    )
    """同步处理hybrid后端逻辑"""
    if not backend.endswith("client"):
        server_url = None

    for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
        md_writer = FileBasedDataWriter(local_md_dir)
        image_cache = ImagePayloadCache()

        middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
            pdf_bytes,
            backend=backend,
            parse_method=parse_method,
            language=lang,
            server_url=server_url,
            page_index_map=page_index_map_list[idx] if page_index_map_list is not None else None,
            image_cache=image_cache,
            **kwargs,
        )

        # f_draw_span_bbox = not _vlm_ocr_enable
        f_draw_span_bbox = False

        _process_output(
            middle_json,
            pdf_bytes,
            pdf_file_name,
            local_md_dir,
            local_image_dir,
            md_writer,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_orig_pdf,
            f_dump_md,
            f_dump_content_list,
            f_dump_middle_json,
            f_dump_model_output,
            f_make_md_mode,
            infer_result,
            process_mode="hybrid",
            backend="hybrid",
            retained_page_indices=page_index_map_list[idx] if page_index_map_list is not None else None,
            broken_page_indices=broken_page_indices_list[idx] if broken_page_indices_list is not None else None,
            image_cache=image_cache,
        )


async def _async_process_hybrid(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    h_lang_list: list[str],
    parse_method: str,
    backend: str,
    f_draw_layout_bbox: bool,
    f_draw_span_bbox: bool,
    f_dump_md: bool,
    f_dump_middle_json: bool,
    f_dump_model_output: bool,
    f_dump_orig_pdf: bool,
    f_dump_content_list: bool,
    f_make_md_mode: str,
    server_url: str | None = None,
    page_index_map_list: list[list[int] | None] | None = None,
    broken_page_indices_list: list[list[int] | None] | None = None,
    **kwargs: Any,
) -> None:
    aio_hybrid_doc_analyze = _load_hybrid_analyze_entrypoint(
        "aio_doc_analyze",
        f"hybrid-{backend}",
    )
    """异步处理hybrid后端逻辑"""
    if not backend.endswith("client"):
        server_url = None

    for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
        md_writer = FileBasedDataWriter(local_md_dir)
        image_cache = ImagePayloadCache()

        middle_json, infer_result, _vlm_ocr_enable = await aio_hybrid_doc_analyze(
            pdf_bytes,
            backend=backend,
            parse_method=parse_method,
            language=lang,
            server_url=server_url,
            page_index_map=page_index_map_list[idx] if page_index_map_list is not None else None,
            image_cache=image_cache,
            **kwargs,
        )

        # f_draw_span_bbox = not _vlm_ocr_enable
        f_draw_span_bbox = False

        _process_output(
            middle_json,
            pdf_bytes,
            pdf_file_name,
            local_md_dir,
            local_image_dir,
            md_writer,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_orig_pdf,
            f_dump_md,
            f_dump_content_list,
            f_dump_middle_json,
            f_dump_model_output,
            f_make_md_mode,
            infer_result,
            process_mode="hybrid",
            backend="hybrid",
            retained_page_indices=page_index_map_list[idx] if page_index_map_list is not None else None,
            broken_page_indices=broken_page_indices_list[idx] if broken_page_indices_list is not None else None,
            image_cache=image_cache,
        )


def _process_office_doc(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    f_dump_md: bool = True,
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_file: bool = True,
    f_dump_content_list: bool = True,
    f_make_md_mode: str = "mm_markdown",
) -> list[int]:
    need_remove_index = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[i]
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix in office_suffixes:
            need_remove_index.append(i)

            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "office")
            md_writer = FileBasedDataWriter(local_md_dir)

            if file_suffix in docx_suffixes:
                office_analyze = office_docx_analyze
            elif file_suffix in pptx_suffixes:
                office_analyze = office_pptx_analyze
            elif file_suffix in xlsx_suffixes:
                office_analyze = office_xlsx_analyze
            else:
                raise ValueError(f"Unsupported office suffix: {file_suffix}")

            image_cache = ImagePayloadCache()
            middle_json, infer_result = office_analyze(file_bytes, image_cache=image_cache)

            f_draw_layout_bbox = False
            f_draw_span_bbox = False

            _process_output(
                middle_json,
                file_bytes,
                pdf_file_name,
                local_md_dir,
                local_image_dir,
                md_writer,
                f_draw_layout_bbox,
                f_draw_span_bbox,
                f_dump_orig_file,
                f_dump_md,
                f_dump_content_list,
                f_dump_middle_json,
                f_dump_model_output,
                f_make_md_mode,
                infer_result,
                process_mode=file_suffix,
                backend="office",
                image_cache=image_cache,
            )

    return need_remove_index


def do_parse(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    p_lang_list: list[str],
    backend: str = DEFAULT_BACKEND,
    parse_method: str = "auto",
    server_url: str | None = None,
    f_draw_layout_bbox: bool = True,
    f_draw_span_bbox: bool = True,
    f_dump_md: bool = True,
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_pdf: bool = True,
    f_dump_content_list: bool = True,
    f_make_md_mode: str = "mm_markdown",
    start_page_id: int = 0,
    end_page_id: int | None = None,
    image_analysis: bool = True,
    effort: str = DEFAULT_HYBRID_EFFORT,
    client_side_output_generation: bool = False,
    **kwargs: Any,
) -> None:
    backend, effort = resolve_backend_and_effort(backend, effort)
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    # 预处理PDF字节数据，同时保留裁剪后 PDF 页序到原始页号的映射。
    prepared_pdf_inputs = _prepare_pdf_inputs(pdf_bytes_list, start_page_id, end_page_id)
    pdf_bytes_list = [prepared.pdf_bytes for prepared in prepared_pdf_inputs]
    page_index_map_list = [prepared.retained_page_indices for prepared in prepared_pdf_inputs]
    broken_page_indices_list = [prepared.broken_page_indices for prepared in prepared_pdf_inputs]

    if backend.startswith("hybrid-"):
        ensure_backend_dependencies(backend)
        backend = backend[7:]

        if backend == "engine" and effort != LOCAL_HYBRID_EFFORT:
            backend = get_vlm_engine(inference_engine="auto", is_async=False)

        _process_hybrid(
            output_dir,
            pdf_file_names,
            pdf_bytes_list,
            p_lang_list,
            parse_method,
            backend,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_md,
            f_dump_middle_json,
            f_dump_model_output,
            f_dump_orig_pdf,
            f_dump_content_list,
            f_make_md_mode,
            server_url,
            page_index_map_list=page_index_map_list,
            broken_page_indices_list=broken_page_indices_list,
            image_analysis=image_analysis,
            effort=effort,
            client_side_output_generation=client_side_output_generation,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'")


async def aio_do_parse(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    p_lang_list: list[str],
    backend: str = DEFAULT_BACKEND,
    parse_method: str = "auto",
    server_url: str | None = None,
    f_draw_layout_bbox: bool = True,
    f_draw_span_bbox: bool = True,
    f_dump_md: bool = True,
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_pdf: bool = True,
    f_dump_content_list: bool = True,
    f_make_md_mode: str = "mm_markdown",
    start_page_id: int = 0,
    end_page_id: int | None = None,
    image_analysis: bool = True,
    effort: str = DEFAULT_HYBRID_EFFORT,
    client_side_output_generation: bool = False,
    **kwargs: Any,
) -> None:
    backend, effort = resolve_backend_and_effort(backend, effort)
    # Office 解析是同步且可能耗时的操作，异步入口需要放到线程中避免阻塞事件循环。
    need_remove_index = await asyncio.to_thread(
        _process_office_doc,
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    # 预处理PDF字节数据，同时保留裁剪后 PDF 页序到原始页号的映射。
    prepared_pdf_inputs = _prepare_pdf_inputs(pdf_bytes_list, start_page_id, end_page_id)
    pdf_bytes_list = [prepared.pdf_bytes for prepared in prepared_pdf_inputs]
    page_index_map_list = [prepared.retained_page_indices for prepared in prepared_pdf_inputs]
    broken_page_indices_list = [prepared.broken_page_indices for prepared in prepared_pdf_inputs]

    if backend.startswith("hybrid-"):
        ensure_backend_dependencies(backend)
        backend = backend[7:]

        if backend == "engine" and effort != LOCAL_HYBRID_EFFORT:
            backend = get_vlm_engine(inference_engine="auto", is_async=True)

        await _async_process_hybrid(
            output_dir,
            pdf_file_names,
            pdf_bytes_list,
            p_lang_list,
            parse_method,
            backend,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_md,
            f_dump_middle_json,
            f_dump_model_output,
            f_dump_orig_pdf,
            f_dump_content_list,
            f_make_md_mode,
            server_url,
            page_index_map_list=page_index_map_list,
            broken_page_indices_list=broken_page_indices_list,
            image_analysis=image_analysis,
            effort=effort,
            client_side_output_generation=client_side_output_generation,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'")


if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
        do_parse(
            "./output",
            [Path(pdf_path).stem],
            [read_fn(Path(pdf_path))],
            ["ch"],
            end_page_id=10,
            backend="hybrid-engine",
        )
    except Exception as e:
        logger.exception(e)
