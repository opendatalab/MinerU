# Copyright (c) Opendatalab. All rights reserved.
import os
import time

import pypdfium2 as pdfium
from loguru import logger
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import ContentBlock
from tqdm import tqdm

from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.backend.vlm.vlm_analyze import (
    ModelSingleton,
    _get_model_async,
    _maybe_enable_serial_execution,
    aio_predictor_execution_guard,
    predictor_execution_guard,
)
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.config_reader import get_device, get_processing_window_size
from mineru.utils.enum_class import ImageType
from mineru.utils.model_utils import clean_memory
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)
from mineru.version import __version__


VLM_VISUAL_LABELS = {"image", "chart", "seal"}
VLM_TEXT_LABEL_TO_TYPE = {
    "abstract": "text",
    "algorithm": "code",
    "aside_text": "aside_text",
    "content": "text",
    "doc_title": "title",
    "footer": "footer",
    "footer_image": "footer",
    "footnote": "page_footnote",
    "formula_number": "text",
    "header": "header",
    "header_image": "header",
    "number": "page_number",
    "paragraph_title": "title",
    "reference_content": "ref_text",
    "text": "text",
    "vertical_text": "text",
    "figure_title": "image_caption",
    "vision_footnote": "image_footnote",
}


def batch_image_analyze(*args, **kwargs):
    """懒加载pipeline analyze，避免导入hybrid-flash模块时提前要求torch环境。"""
    from mineru.backend.pipeline.pipeline_analyze import batch_image_analyze as pipeline_batch_image_analyze

    return pipeline_batch_image_analyze(*args, **kwargs)


def _get_ocr_enable(pdf_bytes, parse_method: str) -> bool:
    """根据parse_method解析OCR开关，保持和pipeline/hybrid入口一致。"""
    if parse_method == "auto":
        return classify(pdf_bytes) == "ocr"
    if parse_method == "ocr":
        return True
    return False


def _should_enable_vlm_ocr(ocr_enable: bool, language: str, inline_formula_enable: bool) -> bool:
    """判断是否让VLM抽取文本内容，OCR-det本身不受该开关影响。"""
    force_enable = os.getenv("MINERU_FORCE_VLM_OCR_ENABLE", "0").lower() in ("1", "true", "yes")
    if force_enable:
        return True

    force_pipeline = os.getenv("MINERU_HYBRID_FORCE_PIPELINE_ENABLE", "0").lower() in ("1", "true", "yes")
    return (
        ocr_enable
        and language in ["ch", "en"]
        and inline_formula_enable
        and not force_pipeline
    )


def _close_images(images_list):
    """关闭窗口内PIL图片，避免长文档处理时文件句柄或内存累积。"""
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _normalize_bbox_to_unit(bbox, page_width: int, page_height: int) -> list[float] | None:
    """把pipeline像素bbox转换为mineru-vl-utils需要的归一化bbox。"""
    if bbox is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
        normalized_bbox = [x0, y0, x1, y1]
    else:
        normalized_bbox = [
            x0 / page_width,
            y0 / page_height,
            x1 / page_width,
            y1 / page_height,
        ]
    normalized_bbox = [round(min(max(v, 0.0), 1.0), 6) for v in normalized_bbox]
    if normalized_bbox[0] >= normalized_bbox[2] or normalized_bbox[1] >= normalized_bbox[3]:
        return None
    return normalized_bbox


def _vlm_type_for_layout_det(layout_det: dict, vlm_ocr_enable: bool, table_enable: bool, image_analysis: bool) -> str | None:
    """把pipeline layout label映射为VLM内容抽取类型，未命中则跳过。"""
    label = layout_det.get("label")
    if label is None:
        logger.warning("Layout detection result missing label: %s", layout_det)
        return None
    if label == "table":
        return "table" if table_enable else None
    if label == "display_formula":
        return "equation"
    if label in VLM_VISUAL_LABELS:
        return "image" if image_analysis else None
    if vlm_ocr_enable:
        return VLM_TEXT_LABEL_TO_TYPE.get(label)
    return None


def _build_vlm_layout_blocks(
    layout_dets: list[dict],
    page_width: int,
    page_height: int,
    vlm_ocr_enable: bool,
    table_enable: bool,
    image_analysis: bool,
) -> list[ContentBlock]:
    """从pipeline layout结果构造VLM sidecar输入，并保留回填用索引。"""
    blocks = []
    for position, layout_det in enumerate(layout_dets):
        vlm_type = _vlm_type_for_layout_det(
            layout_det,
            vlm_ocr_enable=vlm_ocr_enable,
            table_enable=table_enable,
            image_analysis=image_analysis,
        )
        if vlm_type is None:
            continue
        bbox = _normalize_bbox_to_unit(layout_det.get("bbox"), page_width, page_height)
        if bbox is None:
            continue
        try:
            block = ContentBlock(
                vlm_type,
                bbox,
                angle=layout_det.get("angle", 0),
                content=layout_det.get("content"),
            )
        except AssertionError as exc:
            logger.warning(f"Skip invalid hybrid-flash VLM block: {layout_det}, error: {exc}")
            continue
        block["_layout_det_index"] = layout_det.get("index", position)
        block["_layout_det_position"] = position
        block["_layout_det_label"] = layout_det.get("label")
        blocks.append(block)
    return blocks


def _build_not_extract_list(vlm_ocr_enable: bool) -> list[str] | None:
    """非VLM-OCR模式下显式跳过文本抽取，保持hybrid文本边界。"""
    if vlm_ocr_enable:
        return None
    return ["text"]


def _strip_display_formula_delimiters(content: str) -> str:
    """去除VLM公式结果外层display delimiters，便于回填pipeline latex字段。"""
    stripped = content.strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        stripped = stripped[2:-2].strip()
    return stripped


def _merge_vlm_sidecar_result(layout_dets: list[dict], sidecar_blocks) -> None:
    """把VLM sidecar结果回填到pipeline layout_dets，保持原始label不变。"""
    by_index = {
        layout_det.get("index", position): layout_det
        for position, layout_det in enumerate(layout_dets)
    }
    for block in sidecar_blocks or []:
        layout_det = by_index.get(block.get("_layout_det_index"))
        if layout_det is None:
            position = block.get("_layout_det_position")
            if isinstance(position, int) and 0 <= position < len(layout_dets):
                layout_det = layout_dets[position]
        if layout_det is None:
            continue

        block_type = block.get("type")
        block_content = block.get("content")
        label = layout_det.get("label")

        if label == "seal":
            if block_content is not None:
                layout_det["text"] = block_content
                layout_det["content"] = block_content
            layout_det["sub_type"] = "seal"
            continue

        if label in VLM_VISUAL_LABELS:
            if block_content is not None:
                layout_det["content"] = block_content
            if "sub_type" in block:
                layout_det["sub_type"] = block["sub_type"]
            continue

        if label == "table" or block_type == "table":
            if block_content is not None:
                layout_det["html"] = block_content
            if "cell_merge" in block:
                layout_det["cell_merge"] = block["cell_merge"]
            continue

        if label == "display_formula" or block_type == "equation":
            if block_content:
                layout_det["latex"] = _strip_display_formula_delimiters(block_content)
            continue

        if block_type in VLM_TEXT_LABEL_TO_TYPE.values() and block_content is not None:
            layout_det["text"] = block_content
            layout_det["content"] = block_content


def _build_page_model_info(page_layout_dets: list[dict], page_index: int, pil_img) -> dict:
    """按pipeline model_list格式包装单页analyze结果。"""
    return {
        "layout_dets": page_layout_dets,
        "page_info": {
            "page_no": page_index,
            "width": pil_img.width,
            "height": pil_img.height,
        },
    }


def _build_analyze_meta(ocr_enable: bool, vlm_ocr_enable: bool) -> dict:
    """构造第一阶段analyze元信息，供后续阶段判断backend分支和OCR策略。"""
    return {
        "_backend": "hybrid-flash",
        "_ocr_enable": ocr_enable,
        "_vlm_ocr_enable": vlm_ocr_enable,
        "_version_name": __version__,
    }


def _build_pipeline_batch_options(vlm_ocr_enable: bool) -> dict:
    """构造 hybrid-flash 调用 pipeline batch 时的识别策略。"""
    if vlm_ocr_enable:
        return {
            "formula_recognition_scope": "none",
            "ocr_rec_enable": False,
        }
    return {
        "formula_recognition_scope": "inline_only",
        "ocr_rec_enable": True,
    }


def _get_device_for_cleanup():
    """获取清理显存用device；测试或轻量环境缺少torch时退回CPU。"""
    try:
        return get_device()
    except NameError:
        return "cpu"


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None = None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = "auto",
    language: str = "ch",
    inline_formula_enable: bool = True,
    table_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    """hybrid-flash第一阶段analyze：返回pipeline形态model_list和backend元信息。"""
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    device = _get_device_for_cleanup()
    ocr_enable = _get_ocr_enable(pdf_bytes, parse_method=parse_method)
    vlm_ocr_enable = _should_enable_vlm_ocr(ocr_enable, language, inline_formula_enable)
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    doc_closed = False
    model_list = []
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f"Hybrid-flash analyze window run. page_count={page_count}, "
            f"window_size={configured_window_size}, total_windows={total_windows}"
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = load_images_from_pdf_doc(
                    pdf_doc,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                    pdf_bytes=pdf_bytes,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f"Hybrid-flash analyze window {window_index + 1}/{total_windows}: "
                        f"pages {window_start + 1}-{window_end + 1}/{page_count} "
                        f"({len(images_pil_list)} pages)"
                    )
                    pipeline_inputs = [
                        (pil_img, ocr_enable, language)
                        for pil_img in images_pil_list
                    ]
                    page_layout_dets_list = batch_image_analyze(
                        pipeline_inputs,
                        formula_enable=inline_formula_enable,
                        table_enable=False,
                        seal_ocr_rec_enable=False,
                        **_build_pipeline_batch_options(vlm_ocr_enable),
                    )
                    vlm_blocks_list = [
                        _build_vlm_layout_blocks(
                            page_layout_dets,
                            pil_img.width,
                            pil_img.height,
                            vlm_ocr_enable=vlm_ocr_enable,
                            table_enable=table_enable,
                            image_analysis=image_analysis,
                        )
                        for page_layout_dets, pil_img in zip(page_layout_dets_list, images_pil_list)
                    ]
                    if any(vlm_blocks_list):
                        with predictor_execution_guard(predictor):
                            sidecar_results = predictor.batch_extract_with_layout(
                                images_pil_list,
                                vlm_blocks_list,
                                not_extract_list=_build_not_extract_list(vlm_ocr_enable),
                                image_analysis=image_analysis,
                            )
                        for page_layout_dets, sidecar_blocks in zip(page_layout_dets_list, sidecar_results):
                            _merge_vlm_sidecar_result(page_layout_dets, sidecar_blocks)

                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )

                    for offset, (page_layout_dets, pil_img) in enumerate(zip(page_layout_dets_list, images_pil_list)):
                        page_index = window_start + offset
                        model_list.append(_build_page_model_info(page_layout_dets, page_index, pil_img))
                        if progress_bar is not None:
                            progress_bar.update(1)
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"hybrid-flash analyze finished, cost: {infer_time}, "
                f"speed: {round(len(model_list) / infer_time, 3)} page/s"
            )
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return model_list, _build_analyze_meta(ocr_enable, vlm_ocr_enable)
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None = None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = "auto",
    language: str = "ch",
    inline_formula_enable: bool = True,
    table_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    """异步hybrid-flash analyze入口，返回pipeline形态model_list和backend元信息。"""
    if predictor is None:
        predictor = await _get_model_async(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    device = _get_device_for_cleanup()
    ocr_enable = _get_ocr_enable(pdf_bytes, parse_method=parse_method)
    vlm_ocr_enable = _should_enable_vlm_ocr(ocr_enable, language, inline_formula_enable)
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    doc_closed = False
    model_list = []
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0

        for window_start in range(0, page_count, effective_window_size or 1):
            window_end = min(page_count - 1, window_start + effective_window_size - 1)
            images_list = await aio_load_images_from_pdf_bytes_range(
                pdf_bytes,
                start_page_id=window_start,
                end_page_id=window_end,
                image_type=ImageType.PIL,
            )
            try:
                images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                pipeline_inputs = [
                    (pil_img, ocr_enable, language)
                    for pil_img in images_pil_list
                ]
                page_layout_dets_list = batch_image_analyze(
                    pipeline_inputs,
                    formula_enable=inline_formula_enable,
                    table_enable=False,
                    seal_ocr_rec_enable=False,
                    **_build_pipeline_batch_options(vlm_ocr_enable),
                )
                vlm_blocks_list = [
                    _build_vlm_layout_blocks(
                        page_layout_dets,
                        pil_img.width,
                        pil_img.height,
                        vlm_ocr_enable=vlm_ocr_enable,
                        table_enable=table_enable,
                        image_analysis=image_analysis,
                    )
                    for page_layout_dets, pil_img in zip(page_layout_dets_list, images_pil_list)
                ]
                if any(vlm_blocks_list):
                    async with aio_predictor_execution_guard(predictor):
                        sidecar_results = await predictor.aio_batch_extract_with_layout(
                            images_pil_list,
                            vlm_blocks_list,
                            not_extract_list=_build_not_extract_list(vlm_ocr_enable),
                            image_analysis=image_analysis,
                        )
                    for page_layout_dets, sidecar_blocks in zip(page_layout_dets_list, sidecar_results):
                        _merge_vlm_sidecar_result(page_layout_dets, sidecar_blocks)
                for offset, (page_layout_dets, pil_img) in enumerate(zip(page_layout_dets_list, images_pil_list)):
                    model_list.append(_build_page_model_info(page_layout_dets, window_start + offset, pil_img))
            finally:
                _close_images(images_list)
        close_pdfium_document(pdf_doc)
        doc_closed = True
        clean_memory(device)
        return model_list, _build_analyze_meta(ocr_enable, vlm_ocr_enable)
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
