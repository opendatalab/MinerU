# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import atexit
import multiprocessing
import os
import threading
import time
from concurrent.futures import ALL_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import math
from typing import Any, Callable, Literal

import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from PIL import Image

from ....model.flash.pdf.pdfium import close_pdfium_child, pdfium_guard
from ....model.flash._shared.image import image_to_b64str
from ....model.flash.pdf.raster import page_to_image
from ....types import BBox, IntBBox
from ....utils.geometry import normalize_to_int_bbox
from ....utils.platform import is_windows_environment


class ImageType:
    """限定 PDF 页面渲染支持的两种图像返回形态。"""

    PIL = "pil_img"
    BASE64 = "base64_img"


def _positive_int_env(name: str, default: int) -> int:
    """读取正整数环境变量，缺失或非法时返回默认值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def get_load_images_timeout() -> int:
    """返回 PDF 批量渲染的超时秒数。"""
    return _positive_int_env("MINERU_PDF_RENDER_TIMEOUT", 300)


def get_load_images_threads() -> int:
    """返回 PDF 批量渲染允许使用的进程数。"""
    return _positive_int_env("MINERU_PDF_RENDER_THREADS", 3)


DEFAULT_PDF_IMAGE_DPI = 200
# DEFAULT_PDF_IMAGE_DPI = 144
MAX_PDF_RENDER_PROCESSES = 3
MIN_PAGES_PER_RENDER_PROCESS = 30
PDF_RENDER_PROCESS_SPAWN_DELAY_SECONDS = 0.1
PDF_RENDER_TERMINATE_GRACE_PERIOD_SECONDS = 0.1
PDF_RENDER_KILL_JOIN_TIMEOUT_SECONDS = 0.1

_pdf_render_executor: ProcessPoolExecutor | None = None
_pdf_render_executor_lock = threading.Lock()
_pdf_render_atexit_registered = False
_pdf_render_spawn_submit_lock = threading.Lock()
_pdf_render_spawn_submit_executor_id: int | None = None
_pdf_render_spawn_submit_count = 0


def pdf_page_to_image(
    page: pdfium.PdfPage,
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    image_type: Literal["pil_img", "base64_img"] = ImageType.PIL,
) -> dict[str, Any]:
    """将单个 PDFium 页面渲染为 Pillow 图像或 base64 载荷。"""
    pil_img, scale = page_to_image(page, dpi=dpi)
    image_dict: dict[str, Any] = {"scale": scale}
    if image_type == ImageType.BASE64:
        try:
            image_dict["img_base64"] = image_to_b64str(pil_img)
        finally:
            pil_img.close()
    else:
        image_dict["img_pil"] = pil_img

    return image_dict


def _load_images_from_pdf_worker(
    pdf_bytes: bytes,
    dpi: int,
    start_page_id: int,
    end_page_id: int,
    image_type: Literal["pil_img", "base64_img"],
) -> list[dict[str, Any]]:
    """用于进程池的包装函数"""
    started_at = time.monotonic()
    worker_pid = os.getpid()
    page_range = f"{start_page_id + 1}-{end_page_id + 1}"
    logger.debug(
        f"PDF render worker started: pid={worker_pid}, pages={page_range}, dpi={dpi}, "
        f"image_type={image_type}, pdf_bytes={len(pdf_bytes)}"
    )
    try:
        images = load_images_from_pdf_core(pdf_bytes, dpi, start_page_id, end_page_id, image_type)
    except Exception:
        elapsed = time.monotonic() - started_at
        logger.exception(f"PDF render worker failed: pid={worker_pid}, pages={page_range}, elapsed={elapsed:.3f}s")
        raise

    elapsed = time.monotonic() - started_at
    logger.debug(
        f"PDF render worker completed: pid={worker_pid}, pages={page_range}, elapsed={elapsed:.3f}s, images={len(images)}"
    )
    return images


def _close_image_dicts(images_list: list[dict[str, Any]] | None) -> None:
    """关闭 image dict 中的 PIL 图片，供异常清理路径释放已生成的图像资源。"""
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is None:
            continue
        try:
            pil_img.close()
        except Exception:
            pass


def _calculate_render_process_count(total_pages: int, threads: int, cpu_count: int | None = None) -> int:
    """按页数、配置和 CPU 数量计算实际渲染进程数。"""
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
    """把闭区间页范围均匀拆分为指定数量的子范围。"""
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
    cpu_count: int | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """同时返回实际进程数及其页面分片计划。"""
    total_pages = end_page_id - start_page_id + 1
    actual_threads = _calculate_render_process_count(total_pages, threads, cpu_count)
    return actual_threads, _build_render_page_ranges(start_page_id, end_page_id, actual_threads)


def _get_pdf_render_pool_capacity(cpu_count: int | None = None) -> int:
    """返回持久 PDF 渲染进程池的最大容量。"""
    available_cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    configured_threads = max(1, get_load_images_threads())
    return min(
        available_cpus,
        configured_threads,
        MAX_PDF_RENDER_PROCESSES,
    )


def _exit_pdf_render_worker_when_parent_exits() -> None:
    """等待父进程退出后立即终止孤立渲染 worker。"""
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    parent.join()
    os._exit(1)


def _install_pdf_render_parent_exit_watcher() -> None:
    """在渲染 worker 中安装父进程退出监听线程。"""
    watcher = threading.Thread(
        target=_exit_pdf_render_worker_when_parent_exits,
        name="mineru-pdf-render-parent-exit-watcher",
        daemon=True,
    )
    watcher.start()


def _create_pdf_render_executor(max_workers: int) -> ProcessPoolExecutor:
    """使用安全 multiprocessing 上下文创建 PDF 渲染进程池。"""
    if is_windows_environment():
        return ProcessPoolExecutor(max_workers=max_workers, initializer=_install_pdf_render_parent_exit_watcher)

    start_method = multiprocessing.get_start_method()
    if start_method != "spawn":
        logger.debug(f"PDF image rendering switches multiprocessing start method from {start_method} to spawn")
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_install_pdf_render_parent_exit_watcher,
        )

    return ProcessPoolExecutor(max_workers=max_workers, initializer=_install_pdf_render_parent_exit_watcher)


def _is_pdf_render_pool_still_spawning_workers(executor: ProcessPoolExecutor) -> bool:
    """判断渲染进程池是否还可能因为 submit 而继续创建新的 worker。"""
    max_workers = getattr(executor, "_max_workers", None)
    if max_workers is None or max_workers <= 1:
        return False

    processes = getattr(executor, "_processes", None)
    process_count = 0 if processes is None else len(processes)
    return process_count < max_workers


def _submit_pdf_render_task(
    executor: ProcessPoolExecutor,
    fn: Callable[..., Any],
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """提交 PDF 渲染任务；冷启动补 worker 时串行 submit 并错开 100ms。"""
    global _pdf_render_spawn_submit_executor_id, _pdf_render_spawn_submit_count

    with _pdf_render_spawn_submit_lock:
        should_throttle = _is_pdf_render_pool_still_spawning_workers(executor)
        if should_throttle:
            executor_id = id(executor)
            if _pdf_render_spawn_submit_executor_id != executor_id:
                _pdf_render_spawn_submit_executor_id = executor_id
                _pdf_render_spawn_submit_count = 0
            elif _pdf_render_spawn_submit_count > 0:
                time.sleep(PDF_RENDER_PROCESS_SPAWN_DELAY_SECONDS)

        future = executor.submit(fn, *args, **kwargs)

        if should_throttle:
            _pdf_render_spawn_submit_count += 1

        return future


def _get_pdf_render_future_states(
    future_to_range: dict[Future[Any], tuple[int, int]],
) -> list[dict[str, Any]]:
    """汇总每个渲染 Future 的页面范围和运行状态。"""
    states = []
    for future, (range_start, range_end) in future_to_range.items():
        try:
            if future.cancelled():
                state = "cancelled"
            elif future.done():
                state = "done"
            elif future.running():
                state = "running"
            else:
                state = "pending"
        except Exception:
            state = "unknown"
        states.append(
            {
                "pages": f"{range_start + 1}-{range_end + 1}",
                "state": state,
            }
        )
    return states


def _get_pdf_render_worker_states(executor: ProcessPoolExecutor) -> list[dict[str, Any]]:
    """读取进程池 worker 的 pid、存活状态和退出码。"""
    try:
        process_map = getattr(executor, "_processes", None) or {}
        processes = list(process_map.values())
    except Exception:
        return []

    states = []
    for process in processes:
        try:
            pid = getattr(process, "pid", None)
        except Exception:
            pid = None
        try:
            exit_code = getattr(process, "exitcode", None)
        except Exception:
            exit_code = None
        try:
            alive = process.is_alive()
        except Exception:
            alive = None
        states.append(
            {
                "pid": pid,
                "alive": alive,
                "exit_code": exit_code,
            }
        )
    return sorted(states, key=lambda state: (state["pid"] is None, state["pid"] or 0))


def _get_pdf_render_executor() -> ProcessPoolExecutor:
    """惰性创建并复用持久 PDF 渲染进程池。"""
    global _pdf_render_atexit_registered, _pdf_render_executor

    with _pdf_render_executor_lock:
        if _pdf_render_executor is None:
            if not _pdf_render_atexit_registered:
                atexit.register(shutdown_pdf_render_executor)
                _pdf_render_atexit_registered = True
            max_workers = _get_pdf_render_pool_capacity()
            _pdf_render_executor = _create_pdf_render_executor(max_workers=max_workers)
            logger.debug(f"Created persistent PDF render executor with max_workers={max_workers}")
        return _pdf_render_executor


def _recycle_pdf_render_executor(
    executor: ProcessPoolExecutor | None,
    *,
    terminate_processes: bool,
) -> None:
    """从全局缓存移除指定进程池并按需终止 worker。"""
    global _pdf_render_executor

    if executor is None:
        return

    with _pdf_render_executor_lock:
        if _pdf_render_executor is executor:
            _pdf_render_executor = None

    if terminate_processes:
        try:
            _terminate_executor_processes(executor)
        except Exception as exc:
            logger.warning(f"Failed to terminate PDF render executor processes: {exc}")
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        logger.warning(f"Failed to shutdown PDF render executor: {exc}")


def shutdown_pdf_render_executor() -> None:
    """关闭当前持久 PDF 渲染进程池。"""
    global _pdf_render_executor

    with _pdf_render_executor_lock:
        executor = _pdf_render_executor
        _pdf_render_executor = None

    if executor is not None:
        _recycle_pdf_render_executor(
            executor,
            terminate_processes=True,
        )


def load_images_from_pdf_bytes_range(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    start_page_id: int = 0,
    end_page_id: int = 0,
    image_type: Literal["pil_img", "base64_img"] = ImageType.PIL,
    timeout: int | None = None,
    threads: int | None = None,
) -> list[dict[str, Any]]:
    """在持久进程池中批量渲染指定闭区间页面。"""
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
        f"PDF image rendering uses {actual_threads} processes for pages {start_page_id + 1}-{end_page_id + 1}: {page_ranges}"
    )

    render_started_at = time.monotonic()
    executor = _get_pdf_render_executor()
    executor_id = hex(id(executor))
    parent_pid = os.getpid()
    parent_thread = threading.current_thread().name
    recycle_executor = False
    collected_image_lists = []
    try:
        futures: list[Future[Any]] = []
        future_to_range: dict[Future[Any], tuple[int, int]] = {}
        for range_start, range_end in page_ranges:
            page_range = f"{range_start + 1}-{range_end + 1}"
            logger.debug(
                f"Submitting PDF render task: parent_pid={parent_pid}, thread={parent_thread}, "
                f"executor={executor_id}, pages={page_range}"
            )
            future = _submit_pdf_render_task(
                executor,
                _load_images_from_pdf_worker,
                pdf_bytes,
                dpi,
                range_start,
                range_end,
                image_type,
            )
            futures.append(future)
            future_to_range[future] = (range_start, range_end)
            logger.debug(
                f"Submitted PDF render task: parent_pid={parent_pid}, thread={parent_thread}, executor={executor_id}, "
                f"pages={page_range}, workers={_get_pdf_render_worker_states(executor)}"
            )

        _, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
        if not_done:
            recycle_executor = True
            elapsed = time.monotonic() - render_started_at
            logger.warning(
                f"PDF image rendering timed out: timeout={timeout}s, elapsed={elapsed:.3f}s, parent_pid={parent_pid}, "
                f"thread={parent_thread}, executor={executor_id}, pages={start_page_id + 1}-{end_page_id + 1}, "
                f"futures={_get_pdf_render_future_states(future_to_range)}, "
                f"workers={_get_pdf_render_worker_states(executor)}"
            )
            raise TimeoutError(f"PDF image rendering timeout after {timeout}s for pages {start_page_id + 1}-{end_page_id + 1}")

        all_results = []
        for future in futures:
            range_start, _ = future_to_range[future]
            images_list = future.result()
            collected_image_lists.append(images_list)
            all_results.append((range_start, images_list))

        all_results.sort(key=lambda x: x[0])
        images_list = []
        for _, imgs in all_results:
            images_list.extend(imgs)

        collected_image_lists.clear()
        elapsed = time.monotonic() - render_started_at
        logger.debug(
            f"PDF image rendering completed: elapsed={elapsed:.3f}s, parent_pid={parent_pid}, thread={parent_thread}, "
            f"executor={executor_id}, pages={start_page_id + 1}-{end_page_id + 1}, images={len(images_list)}"
        )
        return images_list
    except BrokenProcessPool:
        recycle_executor = True
        raise
    except Exception:
        for images_list in collected_image_lists:
            _close_image_dicts(images_list)
        raise
    finally:
        if recycle_executor:
            logger.warning("Recycling persistent PDF render executor after render failure")
            _recycle_pdf_render_executor(
                executor,
                terminate_processes=True,
            )


def _terminate_executor_processes(executor: ProcessPoolExecutor) -> None:
    """强制终止 ProcessPoolExecutor 中的所有子进程"""
    # executor.shutdown() 后 _processes 会被置空，重复回收时直接视为无进程。
    process_map = getattr(executor, "_processes", None) or {}
    processes = list(process_map.values())
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
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    image_type: Literal["pil_img", "base64_img"] = ImageType.PIL,
) -> list[dict[str, Any]]:
    """在当前进程中依次渲染 PDF 页面范围。"""
    images_list = []
    pdf_doc = None
    try:
        with pdfium_guard():
            pdf_doc = pdfium.PdfDocument(pdf_bytes)
            pdf_page_num = len(pdf_doc)
            end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
            if end_page_id > pdf_page_num - 1:
                end_page_id = pdf_page_num - 1

            for index in range(start_page_id, end_page_id + 1):
                # logger.debug(f"Converting page {index}/{pdf_page_num} to image")
                page = None
                try:
                    page = pdf_doc[index]
                    image_dict = pdf_page_to_image(page, dpi=dpi, image_type=image_type)
                    images_list.append(image_dict)
                finally:
                    close_pdfium_child(page)
    finally:
        if pdf_doc is not None:
            with pdfium_guard():
                pdf_doc.close()

    return images_list


def _model_input_bbox(item: dict[str, Any]) -> IntBBox:
    """把模型输入项的 bbox 向外取整为裁图整数坐标。"""
    bbox = item["bbox"]
    assert bbox is not None
    xmin, ymin, xmax, ymax = [float(value) for value in bbox]
    return math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax)


def crop_img(
    input_res: dict[str, Any],
    input_img: Image.Image | np.ndarray,
    crop_paste_x: int = 0,
    crop_paste_y: int = 0,
) -> tuple[Image.Image | np.ndarray, list[int]]:
    """按模型 bbox 裁图并在四周补白，返回坐标回投所需参数。"""
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = _model_input_bbox(input_res)
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
    if isinstance(input_img, np.ndarray):
        return_image: Image.Image | np.ndarray = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255
        cropped_img = input_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        return_image[
            crop_paste_y : crop_paste_y + (crop_ymax - crop_ymin),
            crop_paste_x : crop_paste_x + (crop_xmax - crop_xmin),
        ] = cropped_img
    else:
        return_image = Image.new("RGB", (crop_new_width, crop_new_height), "white")
        cropped_img = input_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))
    useful_list = [
        crop_paste_x,
        crop_paste_y,
        crop_xmin,
        crop_ymin,
        crop_xmax,
        crop_ymax,
        crop_new_width,
        crop_new_height,
    ]
    return return_image, useful_list


def get_crop_img(bbox: BBox, pil_img: Image.Image, scale: float = 2.0) -> Image.Image:
    """按缩放 bbox 裁剪 Pillow 图像。"""
    scale_bbox = normalize_to_int_bbox([float(v) * scale for v in bbox])
    if scale_bbox is None:
        return pil_img.crop((0, 0, 0, 0))
    return pil_img.crop(tuple(scale_bbox))


def get_crop_np_img(bbox: BBox, input_img: Image.Image | np.ndarray, scale: float = 2.0) -> np.ndarray:
    """按缩放 bbox 裁剪 Pillow 或 NumPy 图像并返回数组。"""
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


__all__ = [
    "ImageType",
    "crop_img",
    "get_crop_img",
    "get_crop_np_img",
    "load_images_from_pdf_bytes_range",
    "load_images_from_pdf_core",
    "pdf_page_to_image",
    "shutdown_pdf_render_executor",
]
