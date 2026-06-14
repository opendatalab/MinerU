# Copyright (c) Opendatalab. All rights reserved.

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Tuple

import pypdfium2 as pdfium
from loguru import logger
from PIL import Image
from tqdm import tqdm

from .model_init import MineruPipelineModel, PIPELINE_MODEL_INIT_LOCK
from .model_json_to_middle_json import (
    apply_server_side_postprocess,
    append_batch_results_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from ..utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.utils.config_reader import get_device, get_processing_window_size
from ...utils.enum_class import ImageType
from ...utils.model_utils import clean_memory, get_vram
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import load_images_from_pdf_bytes_range
from ...utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让 mps 可以 fallback


class ModelSingleton:
    _instance = None
    _models = {}
    _lock = PIPELINE_MODEL_INIT_LOCK

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def get_model(
        self,
        lang=None,
        formula_enable=None,
        table_enable=None,
    ):
        key = (lang, formula_enable, table_enable)
        with self._lock:
            if key not in self._models:
                self._models[key] = custom_model_init(
                    lang=lang,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
            return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
):
    model_init_start = time.time()
    device = get_device()
    formula_config = {"enable": formula_enable}
    table_config = {"enable": table_enable}
    model_input = {
        "device": device,
        "table_config": table_config,
        "formula_config": formula_config,
        "lang": lang,
    }
    custom_model = MineruPipelineModel(**model_input)
    model_init_cost = time.time() - model_init_start
    logger.info(f"model init cost: {model_init_cost}")
    return custom_model


def _get_ocr_enable(pdf_bytes, parse_method: str) -> bool:
    if parse_method == "auto":
        return classify(pdf_bytes) == "ocr"
    if parse_method == "ocr":
        return True
    return False


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _format_doc_slices(batch_slices):
    return ",".join(
        f"doc{item['doc_index']}:{item['page_start'] + 1}-{item['page_end'] + 1}"
        for item in batch_slices
    )


@dataclass
class _PipelineRenderedBatch:
    batch_index: int
    batch_slices: list[dict[str, Any]]
    batch_images: list[Tuple[Image.Image, bool, str]]
    batch_payloads: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]]
    render_cost: float


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_pipeline_pdf_render_prefetch_enabled() -> bool:
    """
    Enable pipeline PDF render prefetch.

    Recommended env:
        MINERU_PIPELINE_PDF_RENDER_PREFETCH=1

    Compatible env:
        MINERU_ENABLE_PDF_RENDER_PREFETCH=1
    """
    return (
        _get_bool_env("MINERU_PIPELINE_PDF_RENDER_PREFETCH", False)
        or _get_bool_env("MINERU_ENABLE_PDF_RENDER_PREFETCH", False)
    )


def _build_next_pipeline_batch_slices(
    doc_contexts: list[dict[str, Any]],
    window_size: int,
) -> list[dict[str, Any]]:
    """
    Reserve the next processing window.

    Important:
    - This function updates context['next_page_idx'] when scheduling a batch.
    - The scheduled page range must be processed in order later.
    - If rendering or inference fails, the whole task fails and outer cleanup closes pdf_doc.
    """
    batch_capacity = window_size
    batch_slices: list[dict[str, Any]] = []

    for context in doc_contexts:
        if batch_capacity == 0:
            break

        page_start = context["next_page_idx"]
        if page_start >= context["page_count"]:
            continue

        take_count = min(batch_capacity, context["page_count"] - page_start)
        page_end = page_start + take_count - 1

        batch_slices.append(
            {
                "context": context,
                "doc_index": context["doc_index"],
                "page_start": page_start,
                "page_end": page_end,
                "count": take_count,
            }
        )

        context["next_page_idx"] = page_end + 1
        batch_capacity -= take_count

    return batch_slices


def _render_pipeline_batch(
    batch_index: int,
    batch_slices: list[dict[str, Any]],
) -> _PipelineRenderedBatch:
    """
    Render one pipeline processing window.

    This function is safe to run in a background thread because it renders from
    pdf_bytes instead of sharing pdfium.PdfDocument across threads.
    """
    render_start = time.time()

    batch_images: list[Tuple[Image.Image, bool, str]] = []
    batch_payloads: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]] = []

    try:
        for batch_slice in batch_slices:
            context = batch_slice["context"]
            page_start = batch_slice["page_start"]
            page_end = batch_slice["page_end"]
            take_count = batch_slice["count"]

            images_list = load_images_from_pdf_bytes_range(
                context["pdf_bytes"],
                start_page_id=page_start,
                end_page_id=page_end,
                image_type=ImageType.PIL,
            )

            images_with_extra_info = [
                (image_dict["img_pil"], context["ocr_enable"], context["lang"])
                for image_dict in images_list
            ]

            batch_images.extend(images_with_extra_info)
            batch_payloads.append((context, images_list, page_start, take_count))

        render_cost = time.time() - render_start

        logger.debug(
            f"Pipeline render batch {batch_index}: "
            f"render_cost={render_cost:.3f}s, "
            f"batch_pages={len(batch_images)}, "
            f"doc_slices={_format_doc_slices(batch_slices)}"
        )

        return _PipelineRenderedBatch(
            batch_index=batch_index,
            batch_slices=batch_slices,
            batch_images=batch_images,
            batch_payloads=batch_payloads,
            render_cost=render_cost,
        )

    except Exception:
        for _context, images_list, _page_start, _take_count in batch_payloads:
            _close_images(images_list)
            images_list.clear()
        raise


def _close_rendered_pipeline_batch(rendered_batch: _PipelineRenderedBatch | None) -> None:
    if rendered_batch is None:
        return

    for _context, images_list, _page_start, _take_count in rendered_batch.batch_payloads:
        _close_images(images_list)
        images_list.clear()


def _cleanup_prefetch_future(future: Future | None) -> None:
    """
    If inference fails while the next render batch is already running, clean
    images produced by the prefetch future to avoid PIL image leakage.
    """
    if future is None:
        return

    if future.cancel():
        return

    try:
        rendered_batch = future.result()
    except Exception:
        return

    _close_rendered_pipeline_batch(rendered_batch)


def _finalize_processing_window_context(
    context,
    on_doc_ready,
    client_side_output_generation=False,
):
    if context["closed"]:
        return

    if client_side_output_generation:
        apply_server_side_postprocess(
            context["middle_json"]["pdf_info"],
            lang=context["lang"],
        )
    else:
        finalize_middle_json(
            context["middle_json"]["pdf_info"],
            lang=context["lang"],
        )

    logger.debug(
        f"Pipeline doc ready: doc{context['doc_index']} pages={context['page_count']}"
    )

    on_doc_ready(
        context["doc_index"],
        context["model_list"],
        context["middle_json"],
        context["ocr_enable"],
    )

    close_pdfium_document(context["pdf_doc"])
    context["closed"] = True


def _emit_zero_page_contexts(
    doc_contexts,
    on_doc_ready,
    client_side_output_generation=False,
):
    for context in doc_contexts:
        if context["page_count"] == 0 and not context["closed"]:
            _finalize_processing_window_context(
                context,
                on_doc_ready,
                client_side_output_generation=client_side_output_generation,
            )


def _analyze_rendered_pipeline_batch(
    rendered_batch: _PipelineRenderedBatch,
    *,
    total_batches: int,
    total_pages: int,
    processed_pages: int,
    progress_bar,
    last_append_end_time,
    formula_enable: bool,
    table_enable: bool,
    on_doc_ready,
    client_side_output_generation: bool,
):
    """
    Run pipeline model inference and append the rendered batch result to middle_json.

    Returns:
        progress_bar, last_append_end_time, processed_page_count
    """
    batch_images = rendered_batch.batch_images
    batch_slices = rendered_batch.batch_slices
    batch_payloads = rendered_batch.batch_payloads

    logger.info(
        f"Pipeline processing window batch "
        f"{rendered_batch.batch_index}/{total_batches}: "
        f"{processed_pages + len(batch_images)}/{total_pages} pages, "
        f"batch_pages={len(batch_images)}, "
        f"render_cost={rendered_batch.render_cost:.3f}s, "
        f"doc_slices={_format_doc_slices(batch_slices)}"
    )

    try:
        infer_start = time.time()

        batch_results = batch_image_analyze(
            batch_images,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )

        infer_cost = time.time() - infer_start

        logger.debug(
            f"Pipeline infer batch {rendered_batch.batch_index}: "
            f"infer_cost={infer_cost:.3f}s, "
            f"batch_pages={len(batch_images)}"
        )

        if progress_bar is None:
            progress_bar = tqdm(total=total_pages, desc="Processing pages")
        else:
            exclude_progress_bar_idle_time(
                progress_bar,
                last_append_end_time,
                now=time.time(),
            )

        result_offset = 0

        for context, images_list, page_start, take_count in batch_payloads:
            result_slice = batch_results[result_offset: result_offset + take_count]

            append_batch_results_to_middle_json(
                context["middle_json"],
                result_slice,
                images_list,
                context["pdf_doc"],
                context["image_writer"],
                page_start_index=page_start,
                ocr_enable=context["ocr_enable"],
                model_list=context["model_list"],
                progress_bar=progress_bar,
            )

            result_offset += take_count

            # Mark pages that have actually been appended to middle_json.
            # Do not use next_page_idx here: next_page_idx is advanced when a window is
            # scheduled for render prefetch, so it can reach page_count before the last
            # scheduled window is really appended. Closing pdf_doc too early causes
            # pypdfium2 to see a None raw document pointer in later append steps.
            context["appended_page_idx"] = max(
                context.get("appended_page_idx", 0),
                page_start + take_count,
            )

            if context["appended_page_idx"] >= context["page_count"] and not context["closed"]:
                _finalize_processing_window_context(
                    context,
                    on_doc_ready,
                    client_side_output_generation=client_side_output_generation,
                )

        last_append_end_time = time.time()

        return progress_bar, last_append_end_time, len(batch_images)

    finally:
        _close_rendered_pipeline_batch(rendered_batch)


def doc_analyze_streaming(
    pdf_bytes_list,
    image_writer_list,
    lang_list,
    on_doc_ready,
    parse_method: str = "auto",
    formula_enable=True,
    table_enable=True,
    client_side_output_generation=False,
):
    if not (len(pdf_bytes_list) == len(image_writer_list) == len(lang_list)):
        raise ValueError(
            "pdf_bytes_list, image_writer_list, and lang_list must have the same length"
        )

    doc_contexts = []

    try:
        total_pages = 0

        for doc_index, (pdf_bytes, image_writer, lang) in enumerate(
            zip(pdf_bytes_list, image_writer_list, lang_list)
        ):
            _ocr_enable = _get_ocr_enable(pdf_bytes, parse_method)
            pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)

            try:
                page_count = get_pdfium_document_page_count(pdf_doc)

                context = {
                    "doc_index": doc_index,
                    "pdf_bytes": pdf_bytes,
                    "pdf_doc": pdf_doc,
                    "page_count": page_count,
                    "next_page_idx": 0,
                    # next_page_idx means scheduled page position, not appended/finished page position.
                    # Render prefetch may schedule the last window before the previous window is appended,
                    # so finalization must be based on appended_page_idx instead of next_page_idx.
                    "appended_page_idx": 0,
                    "middle_json": init_middle_json(),
                    "model_list": [],
                    "image_writer": image_writer,
                    "lang": lang,
                    "ocr_enable": _ocr_enable,
                    "closed": False,
                }

            except Exception:
                close_pdfium_document(pdf_doc)
                raise

            total_pages += page_count
            doc_contexts.append(context)

        if total_pages == 0:
            _emit_zero_page_contexts(
                doc_contexts,
                on_doc_ready,
                client_side_output_generation=client_side_output_generation,
            )
            return

        window_size = get_processing_window_size(default=64)
        total_batches = (total_pages + window_size - 1) // window_size
        prefetch_enabled = _get_pipeline_pdf_render_prefetch_enabled()

        logger.info(
            f"Pipeline processing-window multi-file run. "
            f"doc_count={len(doc_contexts)}, "
            f"total_pages={total_pages}, "
            f"window_size={window_size}, "
            f"total_batches={total_batches}, "
            f"render_prefetch={prefetch_enabled}"
        )

        _emit_zero_page_contexts(
            doc_contexts,
            on_doc_ready,
            client_side_output_generation=client_side_output_generation,
        )

        processed_pages = 0
        total_start = time.time()
        progress_bar = None
        last_append_end_time = None
        batch_index = 0

        prefetch_executor: ThreadPoolExecutor | None = None
        next_future: Future | None = None

        def submit_next_batch() -> Future | None:
            nonlocal batch_index

            batch_slices = _build_next_pipeline_batch_slices(
                doc_contexts,
                window_size,
            )

            if not batch_slices:
                return None

            batch_index += 1

            if prefetch_executor is not None:
                return prefetch_executor.submit(
                    _render_pipeline_batch,
                    batch_index,
                    batch_slices,
                )

            future = Future()

            try:
                future.set_result(
                    _render_pipeline_batch(
                        batch_index,
                        batch_slices,
                    )
                )
            except Exception as exc:
                future.set_exception(exc)

            return future

        try:
            if prefetch_enabled:
                prefetch_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="mineru-pipeline-render-prefetch",
                )

                next_future = submit_next_batch()

                while next_future is not None:
                    current_future = next_future
                    rendered_batch = current_future.result()

                    # Submit the next render immediately before current batch inference.
                    # This is the key overlap:
                    #   current batch infer/postprocess + next batch render
                    next_future = submit_next_batch()

                    try:
                        (
                            progress_bar,
                            last_append_end_time,
                            processed_count,
                        ) = _analyze_rendered_pipeline_batch(
                            rendered_batch,
                            total_batches=total_batches,
                            total_pages=total_pages,
                            processed_pages=processed_pages,
                            progress_bar=progress_bar,
                            last_append_end_time=last_append_end_time,
                            formula_enable=formula_enable,
                            table_enable=table_enable,
                            on_doc_ready=on_doc_ready,
                            client_side_output_generation=client_side_output_generation,
                        )

                        processed_pages += processed_count

                    except Exception:
                        _cleanup_prefetch_future(next_future)
                        next_future = None
                        raise

            else:
                while processed_pages < total_pages:
                    current_future = submit_next_batch()

                    if current_future is None:
                        break

                    rendered_batch = current_future.result()

                    (
                        progress_bar,
                        last_append_end_time,
                        processed_count,
                    ) = _analyze_rendered_pipeline_batch(
                        rendered_batch,
                        total_batches=total_batches,
                        total_pages=total_pages,
                        processed_pages=processed_pages,
                        progress_bar=progress_bar,
                        last_append_end_time=last_append_end_time,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                        on_doc_ready=on_doc_ready,
                        client_side_output_generation=client_side_output_generation,
                    )

                    processed_pages += processed_count

        finally:
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=True, cancel_futures=True)

            if progress_bar is not None:
                progress_bar.close()

        total_cost = round(time.time() - total_start, 2)

        if total_cost > 0:
            logger.debug(
                f"processing-window multi-file infer finished, "
                f"cost: {total_cost}, "
                f"speed: {round(total_pages / total_cost, 3)} page/s"
            )

    finally:
        for context in doc_contexts:
            if not context["closed"]:
                close_pdfium_document(context["pdf_doc"])
                context["closed"] = True


def batch_image_analyze(
    images_with_extra_info: List[Tuple[Image.Image, bool, str]],
    formula_enable=True,
    table_enable=True,
):
    from .batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()
    device = get_device()

    if str(device).startswith("npu"):
        try:
            import torch_npu

            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        batch_ratio = 16
    elif gpu_memory >= 16:
        batch_ratio = 8
    elif gpu_memory >= 8:
        batch_ratio = 4
    elif gpu_memory >= 6:
        batch_ratio = 2
    else:
        batch_ratio = 1

    logger.info(
        f"GPU Memory: {gpu_memory} GB, Batch Ratio: {batch_ratio}."
    )

    import torch
    from packaging import version

    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    if device_type.lower() in ["corex"]:
        enable_ocr_det_batch = False
    else:
        enable_ocr_det_batch = True
        if version.parse(torch.__version__) >= version.parse("2.8.0"):
            os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

    batch_model = BatchAnalyze(
        model_manager,
        batch_ratio,
        formula_enable,
        table_enable,
        enable_ocr_det_batch,
    )
    results = batch_model(images_with_extra_info)
    clean_memory(get_device())
    return results
