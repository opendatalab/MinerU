# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import atexit
import multiprocessing
import os
import threading
import time
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from PIL import Image, ImageOps

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.os_env_config import get_load_images_timeout, get_load_images_threads
from mineru.utils.pdf_reader import image_to_b64str, image_to_bytes, page_to_image
from mineru.utils.enum_class import ImageType
from mineru.utils.hash_utils import str_sha256
from mineru.utils.pdf_page_id import get_end_page_id
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
    pdfium_guard,
)

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from concurrent.futures.process import BrokenProcessPool


DEFAULT_PDF_IMAGE_DPI = 200
# DEFAULT_PDF_IMAGE_DPI = 144
MAX_PDF_RENDER_PROCESSES = 4
MIN_PAGES_PER_RENDER_PROCESS = 30
PDF_RENDER_TERMINATE_GRACE_PERIOD_SECONDS = 0.1
PDF_RENDER_KILL_JOIN_TIMEOUT_SECONDS = 0.1

_pdf_render_executor: ProcessPoolExecutor | None = None
_pdf_render_executor_lock = threading.Lock()


def pdf_page_to_image(
    page: pdfium.PdfPage,
    dpi=DEFAULT_PDF_IMAGE_DPI,
    image_type=ImageType.PIL,
) -> dict:
    """Convert pdfium.PdfDocument to image, Then convert the image to base64.

    Args:
        page (_type_): pdfium.PdfPage
        dpi (int, optional): reset the dpi of dpi. Defaults to DEFAULT_PDF_IMAGE_DPI.
        image_type (ImageType, optional): The type of image to return. Defaults to ImageType.PIL.

    Returns:
        dict:  {'img_base64': str, 'img_pil': pil_img, 'scale': float }
    """
    pil_img, scale = page_to_image(page, dpi=dpi)
    image_dict = {
        "scale": scale,
    }
    if image_type == ImageType.BASE64:
        image_dict["img_base64"] = image_to_b64str(pil_img)
    else:
        image_dict["img_pil"] = pil_img

    return image_dict


def _load_images_from_pdf_worker(
    pdf_bytes, dpi, start_page_id, end_page_id, image_type
):
    """用于进程池的包装函数"""
    return load_images_from_pdf_core(
        pdf_bytes, dpi, start_page_id, end_page_id, image_type
    )


def _calculate_render_process_count(total_pages: int, threads: int, cpu_count=None) -> int:
    requested_threads = max(1, threads)
    available_cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    page_limited_threads = max(1, total_pages // MIN_PAGES_PER_RENDER_PROCESS)
    return min(
        available_cpus,
        requested_threads,
        MAX_PDF_RENDER_PROCESSES,
        page_limited_threads,
    )


def _build_render_page_ranges(
    start_page_id: int,
    end_page_id: int,
    process_count: int,
) -> list[tuple[int, int]]:
    total_pages = end_page_id - start_page_id + 1
    base_pages, remainder = divmod(total_pages, process_count)
    page_ranges = []
    current_page = start_page_id

    for process_idx in range(process_count):
        pages_in_range = base_pages + (1 if process_idx < remainder else 0)
        range_end = current_page + pages_in_range - 1
        page_ranges.append((current_page, range_end))
        current_page = range_end + 1

    return page_ranges


def _get_render_process_plan(
    start_page_id: int,
    end_page_id: int,
    threads: int,
    cpu_count=None,
) -> tuple[int, list[tuple[int, int]]]:
    total_pages = end_page_id - start_page_id + 1
    actual_threads = _calculate_render_process_count(total_pages, threads, cpu_count)
    return actual_threads, _build_render_page_ranges(
        start_page_id, end_page_id, actual_threads
    )


def _get_pdf_render_pool_capacity(cpu_count=None) -> int:
    available_cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    configured_threads = max(1, get_load_images_threads())
    return min(
        available_cpus,
        configured_threads,
        MAX_PDF_RENDER_PROCESSES,
    )


def _create_pdf_render_executor(max_workers: int) -> ProcessPoolExecutor:
    if is_windows_environment():
        return ProcessPoolExecutor(max_workers=max_workers)

    start_method = multiprocessing.get_start_method()
    if start_method == "fork":
        logger.debug(
            "PDF image rendering switches multiprocessing start method from fork to spawn"
        )
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    return ProcessPoolExecutor(max_workers=max_workers)


def _get_pdf_render_executor() -> ProcessPoolExecutor:
    global _pdf_render_executor

    with _pdf_render_executor_lock:
        if _pdf_render_executor is None:
            max_workers = _get_pdf_render_pool_capacity()
            _pdf_render_executor = _create_pdf_render_executor(max_workers=max_workers)
            logger.debug(
                f"Created persistent PDF render executor with max_workers={max_workers}"
            )
        return _pdf_render_executor


def _recycle_pdf_render_executor(
    executor: ProcessPoolExecutor | None,
    *,
    terminate_processes: bool,
) -> None:
    global _pdf_render_executor

    if executor is None:
        return

    with _pdf_render_executor_lock:
        if _pdf_render_executor is executor:
            _pdf_render_executor = None

    if terminate_processes:
        _terminate_executor_processes(executor)
    executor.shutdown(wait=False, cancel_futures=True)


def shutdown_pdf_render_executor() -> None:
    global _pdf_render_executor

    with _pdf_render_executor_lock:
        executor = _pdf_render_executor
        _pdf_render_executor = None

    if executor is not None:
        _recycle_pdf_render_executor(
            executor,
            terminate_processes=True,
        )


atexit.register(shutdown_pdf_render_executor)


def _load_images_from_pdf_bytes_range(
    pdf_bytes: bytes,
    dpi=DEFAULT_PDF_IMAGE_DPI,
    start_page_id=0,
    end_page_id=0,
    image_type=ImageType.PIL,
    timeout=None,
    threads=None,
):
    if end_page_id < start_page_id:
        return []

    if timeout is None:
        timeout = get_load_images_timeout()
    if threads is None:
        threads = get_load_images_threads()

    actual_threads, page_ranges = _get_render_process_plan(
        start_page_id,
        end_page_id,
        threads,
    )

    logger.debug(
        f"PDF image rendering uses {actual_threads} processes for pages "
        f"{start_page_id + 1}-{end_page_id + 1}: {page_ranges}"
    )

    executor = _get_pdf_render_executor()
    recycle_executor = False
    try:
        futures = []
        future_to_range = {}
        for range_start, range_end in page_ranges:
            future = executor.submit(
                _load_images_from_pdf_worker,
                pdf_bytes,
                dpi,
                range_start,
                range_end,
                image_type,
            )
            futures.append(future)
            future_to_range[future] = range_start

        _, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
        if not_done:
            recycle_executor = True
            raise TimeoutError(
                f"PDF image rendering timeout after {timeout}s "
                f"for pages {start_page_id + 1}-{end_page_id + 1}"
            )

        all_results = []
        for future in futures:
            range_start = future_to_range[future]
            images_list = future.result()
            all_results.append((range_start, images_list))

        all_results.sort(key=lambda x: x[0])
        images_list = []
        for _, imgs in all_results:
            images_list.extend(imgs)

        return images_list
    except BrokenProcessPool:
        recycle_executor = True
        raise
    finally:
        if recycle_executor:
            logger.warning("Recycling persistent PDF render executor after render failure")
            _recycle_pdf_render_executor(
                executor,
                terminate_processes=True,
            )


async def aio_load_images_from_pdf_bytes_range(
    pdf_bytes: bytes,
    dpi=DEFAULT_PDF_IMAGE_DPI,
    start_page_id=0,
    end_page_id=0,
    image_type=ImageType.PIL,
    timeout=None,
    threads=None,
):
    return await asyncio.to_thread(
        _load_images_from_pdf_bytes_range,
        pdf_bytes,
        dpi=dpi,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        image_type=image_type,
        timeout=timeout,
        threads=threads,
    )


def _terminate_executor_processes(executor):
    """强制终止 ProcessPoolExecutor 中的所有子进程"""
    processes = list(getattr(executor, "_processes", {}).values())
    if not processes:
        return

    alive_processes = []
    for process in processes:
        if not process.is_alive():
            continue
        try:
            process.terminate()
        except Exception:
            pass
        alive_processes.append(process)

    deadline = time.monotonic() + PDF_RENDER_TERMINATE_GRACE_PERIOD_SECONDS
    for process in alive_processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.join(timeout=remaining)
        except Exception:
            pass

    for process in alive_processes:
        if not process.is_alive():
            continue
        try:
            kill_process = getattr(process, "kill", None)
            if callable(kill_process):
                kill_process()
            else:
                process.terminate()
        except Exception:
            pass

    for process in alive_processes:
        if not process.is_alive():
            continue
        try:
            process.join(timeout=PDF_RENDER_KILL_JOIN_TIMEOUT_SECONDS)
        except Exception:
            pass


def load_images_from_pdf_core(
    pdf_bytes: bytes,
    dpi=DEFAULT_PDF_IMAGE_DPI,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,  # PIL or BASE64
):
    images_list = []
    pdf_doc = None
    try:
        with pdfium_guard():
            pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
            pdf_page_num = len(pdf_doc)
            end_page_id = get_end_page_id(end_page_id, pdf_page_num)

            for index in range(start_page_id, end_page_id + 1):
                # logger.debug(f"Converting page {index}/{pdf_page_num} to image")
                page = pdf_doc[index]
                image_dict = pdf_page_to_image(page, dpi=dpi, image_type=image_type)
                images_list.append(image_dict)
    finally:
        close_pdfium_document(pdf_doc)

    return images_list


def load_images_from_pdf_doc(
    pdf_doc: pdfium.PdfDocument,
    dpi=DEFAULT_PDF_IMAGE_DPI,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,
    pdf_bytes: bytes | None = None,
    timeout=None,
    threads=None,
):
    pdf_page_num = get_pdfium_document_page_count(pdf_doc)
    normalized_end_page_id = get_end_page_id(end_page_id, pdf_page_num)

    if pdf_bytes is not None:
        return _load_images_from_pdf_bytes_range(
            pdf_bytes,
            dpi=dpi,
            start_page_id=start_page_id,
            end_page_id=normalized_end_page_id,
            image_type=image_type,
            timeout=timeout,
            threads=threads,
        )

    images_list = []
    with pdfium_guard():
        for index in range(start_page_id, normalized_end_page_id + 1):
            page = pdf_doc[index]
            image_dict = pdf_page_to_image(page, dpi=dpi, image_type=image_type)
            images_list.append(image_dict)

    return images_list


def cut_image(
    bbox: tuple,
    page_num: int,
    page_pil_img,
    return_path,
    image_writer: FileBasedDataWriter,
    scale=2,
):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""

    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 老版本返回不带bucket的路径
    img_path = f"{return_path}_{filename}" if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f"{str_sha256(img_path)}.jpg"
    # img_hash256_path = f'{img_path}.jpg'

    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    image_writer.write(img_hash256_path, img_bytes)
    return img_hash256_path


def get_crop_img(bbox: tuple, pil_img, scale=2):
    scale_bbox = normalize_to_int_bbox([float(v) * scale for v in bbox])
    if scale_bbox is None:
        return pil_img.crop((0, 0, 0, 0))
    return pil_img.crop(tuple(scale_bbox))


def get_crop_np_img(bbox: tuple, input_img, scale=2):
    if isinstance(input_img, Image.Image):
        np_img = np.asarray(input_img)
    elif isinstance(input_img, np.ndarray):
        np_img = input_img
    else:
        raise ValueError("Input must be a pillow object or a numpy array.")

    height, width = np_img.shape[:2]
    scale_bbox = normalize_to_int_bbox(
        [float(v) * scale for v in bbox],
        image_size=(height, width),
    )
    if scale_bbox is None:
        return np_img[0:0, 0:0]

    return np_img[scale_bbox[1] : scale_bbox[3], scale_bbox[0] : scale_bbox[2]]


def images_bytes_to_pdf_bytes(image_bytes):
    # 内存缓冲区
    pdf_buffer = BytesIO()

    # 载入并转换所有图像为 RGB 模式
    image = Image.open(BytesIO(image_bytes))
    # 根据 EXIF 信息自动转正（处理手机拍摄的带 Orientation 标记的图片）
    image = ImageOps.exif_transpose(image) or image
    # 只在必要时转换
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 第一张图保存为 PDF，其余追加
    image.save(
        pdf_buffer,
        format="PDF",
        resolution=DEFAULT_PDF_IMAGE_DPI,
        quality=95,
        subsampling=0,
    )

    # 获取 PDF bytes 并重置指针（可选）
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
