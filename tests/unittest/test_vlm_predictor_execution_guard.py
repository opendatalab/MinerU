# Copyright (c) Opendatalab. All rights reserved.
import atexit
import asyncio
import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _stub(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _noop(*args: object, **kwargs: object) -> None:
    del args, kwargs


class _DataWriter:
    pass


class _MinerUClient:
    pass


def _load_subject() -> ModuleType:
    """Load the production module without optional model/runtime dependencies."""
    stubs = {
        "pypdfium2": _stub("pypdfium2"),
        "loguru": _stub("loguru", logger=SimpleNamespace()),
        "tqdm": _stub("tqdm", tqdm=_noop),
        "mineru.backend.vlm.utils": _stub(
            "mineru.backend.vlm.utils",
            enable_custom_logits_processors=_noop,
            set_default_gpu_memory_utilization=_noop,
            set_default_batch_size=_noop,
            set_lmdeploy_backend=_noop,
            mod_kwargs_by_device_type=_noop,
        ),
        "mineru.backend.vlm.model_output_to_middle_json": _stub(
            "mineru.backend.vlm.model_output_to_middle_json",
            append_page_blocks_to_middle_json=_noop,
            finalize_middle_json=_noop,
            init_middle_json=_noop,
        ),
        "mineru.backend.utils.runtime_utils": _stub(
            "mineru.backend.utils.runtime_utils",
            exclude_progress_bar_idle_time=_noop,
        ),
        "mineru.data.data_reader_writer": _stub(
            "mineru.data.data_reader_writer",
            DataWriter=_DataWriter,
        ),
        "mineru.utils.pdf_image_tools": _stub(
            "mineru.utils.pdf_image_tools",
            aio_load_images_from_pdf_bytes_range=_noop,
            load_images_from_pdf_doc=_noop,
        ),
        "mineru.utils.check_sys_env": _stub(
            "mineru.utils.check_sys_env",
            is_mac_os_version_supported=_noop,
        ),
        "mineru.utils.config_reader": _stub(
            "mineru.utils.config_reader",
            get_device=_noop,
            get_processing_window_size=_noop,
        ),
        "mineru.utils.enum_class": _stub(
            "mineru.utils.enum_class",
            ImageType=SimpleNamespace(),
        ),
        "mineru.utils.pdfium_guard": _stub(
            "mineru.utils.pdfium_guard",
            close_pdfium_document=_noop,
            get_pdfium_document_page_count=_noop,
            open_pdfium_document=_noop,
        ),
        "mineru.utils.models_download_utils": _stub(
            "mineru.utils.models_download_utils",
            auto_download_and_get_model_root_path=_noop,
        ),
        "mineru_vl_utils": _stub(
            "mineru_vl_utils",
            MinerUClient=_MinerUClient,
        ),
    }
    source_path = Path(__file__).parents[2] / "mineru/backend/vlm/vlm_analyze.py"
    spec = importlib.util.spec_from_file_location(
        "mineru.backend.vlm._execution_guard_test_subject",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {source_path}")
    subject = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs), patch.object(atexit, "register"):
        spec.loader.exec_module(subject)
    return subject


SUBJECT = _load_subject()


def _predictor(client_module: str) -> SimpleNamespace:
    client_type = type("FakeClient", (), {"__module__": client_module})
    return SimpleNamespace(client=client_type())


def _sync_peak(predictor: SimpleNamespace) -> int:
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    active = 0
    peak = 0

    def work() -> None:
        nonlocal active, peak
        with SUBJECT.predictor_execution_guard(predictor):
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=0.1)
            except threading.BrokenBarrierError:
                pass
            finally:
                with counter_lock:
                    active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(work) for _ in range(2)]
        for future in futures:
            future.result(timeout=1)
    return peak


async def _async_peak(predictor: SimpleNamespace) -> int:
    both_entered = asyncio.Event()
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        async with SUBJECT.aio_predictor_execution_guard(predictor):
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_entered.set()
            try:
                await asyncio.wait_for(both_entered.wait(), timeout=0.1)
            except TimeoutError:
                pass
            finally:
                active -= 1

    await asyncio.gather(work(), work())
    return peak


class TestPredictorExecutionGuard(unittest.TestCase):
    def test_transformers_gets_one_per_predictor_lock(self) -> None:
        predictor = _predictor("custom.client")

        SUBJECT._maybe_enable_serial_execution(predictor, "transformers")
        first_lock = predictor._mineru_execution_lock
        SUBJECT._maybe_enable_serial_execution(predictor, "transformers")

        self.assertIs(first_lock, predictor._mineru_execution_lock)

    def test_async_transformers_execution_is_serialized(self) -> None:
        predictor = _predictor("mineru_vl_utils.vlm_client.transformers_client")
        SUBJECT._maybe_enable_serial_execution(predictor)

        self.assertEqual(asyncio.run(_async_peak(predictor)), 1)

    def test_sync_transformers_execution_is_serialized(self) -> None:
        predictor = _predictor("custom.client")
        SUBJECT._maybe_enable_serial_execution(predictor, "transformers")

        self.assertEqual(_sync_peak(predictor), 1)

    def test_mlx_execution_remains_serialized(self) -> None:
        predictor = _predictor("mineru_vl_utils.vlm_client.mlx_client")
        SUBJECT._maybe_enable_serial_execution(predictor)

        self.assertEqual(_sync_peak(predictor), 1)

    def test_concurrent_backends_remain_unlocked(self) -> None:
        cases = (
            ("vllm-engine", "mineru_vl_utils.vlm_client.vllm_engine_client"),
            ("vllm-async-engine", "mineru_vl_utils.vlm_client.vllm_async_engine_client"),
            ("lmdeploy-engine", "mineru_vl_utils.vlm_client.lmdeploy_engine_client"),
            ("http-client", "mineru_vl_utils.vlm_client.http_client"),
        )

        for backend, client_module in cases:
            with self.subTest(backend=backend):
                predictor = _predictor(client_module)
                SUBJECT._maybe_enable_serial_execution(predictor, backend)

                self.assertFalse(hasattr(predictor, "_mineru_execution_lock"))
                self.assertEqual(_sync_peak(predictor), 2)
                self.assertEqual(asyncio.run(_async_peak(predictor)), 2)


if __name__ == "__main__":
    unittest.main()
