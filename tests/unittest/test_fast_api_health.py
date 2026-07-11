# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import json
import sys
import types

import pytest


def install_fast_api_dependency_stubs() -> None:
    loguru = types.ModuleType("loguru")

    class Logger:
        def remove(self) -> None:
            pass

        def add(self, *args: object, **kwargs: object) -> None:
            pass

        def info(self, *args: object, **kwargs: object) -> None:
            pass

        def warning(self, *args: object, **kwargs: object) -> None:
            pass

        def error(self, *args: object, **kwargs: object) -> None:
            pass

        def exception(self, *args: object, **kwargs: object) -> None:
            pass

    loguru.logger = Logger()
    sys.modules["loguru"] = loguru

    uvicorn = types.ModuleType("uvicorn")

    class Config:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class Server:
        should_exit = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self) -> None:
            pass

    uvicorn.Config = Config
    uvicorn.Server = Server
    sys.modules["uvicorn"] = uvicorn

    common = types.ModuleType("mineru.cli.common")
    common.aio_do_parse = None
    common.do_parse = None
    common.image_suffixes = [".png", ".jpeg", ".jpg"]
    common.office_suffixes = [".docx", ".pptx", ".xlsx"]
    common.pdf_suffixes = [".pdf"]
    common.normalize_upload_filename = lambda name: name
    common.normalize_task_stem = lambda name: name
    common.read_fn = lambda path: b""
    common.uniquify_task_stems = lambda stems: (stems, False)
    sys.modules["mineru.cli.common"] = common

    api_request = types.ModuleType("mineru.cli.api_request")

    class ParseRequestOptions:
        pass

    api_request.ParseRequestOptions = ParseRequestOptions
    api_request.parse_request_form = lambda: None
    sys.modules["mineru.cli.api_request"] = api_request

    public_http_client_policy = types.ModuleType("mineru.cli.public_http_client_policy")
    public_http_client_policy.configure_public_http_client_policy = lambda *args, **kwargs: None
    public_http_client_policy.is_public_bind_host = lambda host: False
    public_http_client_policy.warn_if_public_http_client_policy = lambda *args, **kwargs: None
    sys.modules["mineru.cli.public_http_client_policy"] = public_http_client_policy

    output_paths = types.ModuleType("mineru.cli.output_paths")
    output_paths.resolve_parse_dir = lambda *args, **kwargs: "/tmp"
    sys.modules["mineru.cli.output_paths"] = output_paths

    api_protocol = types.ModuleType("mineru.cli.api_protocol")
    api_protocol.API_PROTOCOL_VERSION = "1.0"
    api_protocol.DEFAULT_MAX_CONCURRENT_REQUESTS = 1
    api_protocol.DEFAULT_PROCESSING_WINDOW_SIZE = 1
    sys.modules["mineru.cli.api_protocol"] = api_protocol

    backend_options = types.ModuleType("mineru.cli.backend_options")
    backend_options.DEFAULT_HYBRID_EFFORT = "auto"
    sys.modules["mineru.cli.backend_options"] = backend_options

    vlm_preload = types.ModuleType("mineru.cli.vlm_preload")
    vlm_preload.maybe_preload_vlm_model = lambda *args, **kwargs: None
    vlm_preload.split_service_and_model_config = lambda config: ({}, config)
    sys.modules["mineru.cli.vlm_preload"] = vlm_preload

    vlm_analyze = types.ModuleType("mineru.backend.vlm.vlm_analyze")
    vlm_analyze.shutdown_cached_models = lambda: None
    sys.modules["mineru.backend.vlm.vlm_analyze"] = vlm_analyze

    cli_parser = types.ModuleType("mineru.utils.cli_parser")
    cli_parser.arg_parse = lambda *args, **kwargs: None
    sys.modules["mineru.utils.cli_parser"] = cli_parser

    check_sys_env = types.ModuleType("mineru.utils.check_sys_env")
    check_sys_env.is_mac_environment = lambda: False
    sys.modules["mineru.utils.check_sys_env"] = check_sys_env

    config_reader = types.ModuleType("mineru.utils.config_reader")
    config_reader.get_max_concurrent_requests = lambda default: default
    config_reader.get_processing_window_size = lambda default: default
    sys.modules["mineru.utils.config_reader"] = config_reader

    guess_suffix_or_lang = types.ModuleType("mineru.utils.guess_suffix_or_lang")
    guess_suffix_or_lang.guess_suffix_by_path = lambda path: ".pdf"
    sys.modules["mineru.utils.guess_suffix_or_lang"] = guess_suffix_or_lang

    pdf_image_tools = types.ModuleType("mineru.utils.pdf_image_tools")
    pdf_image_tools.shutdown_pdf_render_executor = lambda: None
    sys.modules["mineru.utils.pdf_image_tools"] = pdf_image_tools

    version = types.ModuleType("mineru.version")
    version.__version__ = "test"
    sys.modules["mineru.version"] = version


try:
    from mineru.cli import fast_api
except ModuleNotFoundError:
    install_fast_api_dependency_stubs()
    sys.modules.pop("mineru.cli.fast_api", None)
    from mineru.cli import fast_api


def build_task(task_id: str) -> fast_api.AsyncParseTask:
    return fast_api.AsyncParseTask(
        task_id=task_id,
        status=fast_api.TASK_PENDING,
        backend="vlm",
        file_names=["sample.pdf"],
        created_at=fast_api.utc_now_iso(),
        output_dir="/tmp/mineru-test-output",
        effort=fast_api.DEFAULT_HYBRID_EFFORT,
        parse_method="auto",
        lang_list=["en"],
        formula_enable=True,
        table_enable=True,
        image_analysis=False,
        server_url=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        response_format_zip=False,
        return_original_file=False,
        client_side_output_generation=False,
        start_page_id=0,
        end_page_id=99999,
        upload_names=["sample.pdf"],
        uploads=["/tmp/sample.pdf"],
    )


async def build_started_manager(monkeypatch: pytest.MonkeyPatch) -> fast_api.AsyncTaskManager:
    monkeypatch.setattr(fast_api, "_request_semaphore", None)
    manager = fast_api.AsyncTaskManager(fast_api.app)
    await manager.start()
    return manager


@pytest.mark.asyncio
async def test_ordinary_task_failure_does_not_mark_worker_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = await build_started_manager(monkeypatch)
    task = build_task("ordinary-failure")
    manager.tasks[task.task_id] = task
    manager.task_events[task.task_id] = asyncio.Event()

    async def fail_parse_task(_task: fast_api.AsyncParseTask) -> None:
        raise Exception("single document parse failed")

    monkeypatch.setattr(manager, "_run_task", fail_parse_task)

    try:
        await manager._process_task(task.task_id)

        assert task.status == fast_api.TASK_FAILED
        assert task.error == "single document parse failed"
        assert manager.last_worker_error is None
        assert manager.is_healthy()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_engine_dead_task_failure_marks_worker_unhealthy_and_health_endpoint_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EngineDeadError(Exception):
        pass

    manager = await build_started_manager(monkeypatch)
    task = build_task("fatal-failure")
    manager.tasks[task.task_id] = task
    manager.task_events[task.task_id] = asyncio.Event()

    async def fail_parse_task(_task: fast_api.AsyncParseTask) -> None:
        raise EngineDeadError("EngineCore is dead")

    monkeypatch.setattr(manager, "_run_task", fail_parse_task)
    previous_task_manager = getattr(fast_api.app.state, "task_manager", None)
    fast_api.app.state.task_manager = manager

    try:
        await manager._process_task(task.task_id)
        health_response = await fast_api.health_check()
        health_payload = json.loads(health_response.body)

        assert task.status == fast_api.TASK_FAILED
        assert task.error == "EngineCore is dead"
        assert manager.last_worker_error == "EngineCore is dead"
        assert not manager.is_healthy()
        assert health_response.status_code == 503
        assert health_payload["status"] == "unhealthy"
        assert health_payload["error"] == "EngineCore is dead"
    finally:
        fast_api.app.state.task_manager = previous_task_manager
        await manager.shutdown()


@pytest.mark.asyncio
async def test_fatal_error_wakes_waiters_and_start_clears_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EngineDeadError(Exception):
        pass

    manager = await build_started_manager(monkeypatch)
    fatal_task = build_task("fatal-failure")
    waiting_task = build_task("waiting-task")
    for task in (fatal_task, waiting_task):
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()

    async def fail_parse_task(_task: fast_api.AsyncParseTask) -> None:
        raise EngineDeadError("AsyncLLMEngine has failed unrecoverably")

    monkeypatch.setattr(manager, "_run_task", fail_parse_task)
    waiter = asyncio.create_task(manager.wait_for_terminal_state(waiting_task.task_id))

    try:
        await asyncio.sleep(0)
        await manager._process_task(fatal_task.task_id)

        with pytest.raises(fast_api.TaskWaitAbortedError, match="AsyncLLMEngine has failed unrecoverably"):
            await waiter
        assert not manager.is_healthy()

        await manager.start()

        successful_task = build_task("successful-task")
        manager.tasks[successful_task.task_id] = successful_task
        manager.task_events[successful_task.task_id] = asyncio.Event()

        async def succeed_parse_task(_task: fast_api.AsyncParseTask) -> None:
            _task.status = fast_api.TASK_COMPLETED
            _task.completed_at = fast_api.utc_now_iso()

        monkeypatch.setattr(manager, "_run_task", succeed_parse_task)
        await manager._process_task(successful_task.task_id)

        assert successful_task.status == fast_api.TASK_COMPLETED
        assert manager.last_worker_error is None
        assert manager.is_healthy()
    finally:
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_task_propagates_without_marking_worker_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = await build_started_manager(monkeypatch)
    task = build_task("cancelled-task")
    manager.tasks[task.task_id] = task
    manager.task_events[task.task_id] = asyncio.Event()

    async def cancel_parse_task(_task: fast_api.AsyncParseTask) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(manager, "_run_task", cancel_parse_task)

    try:
        with pytest.raises(asyncio.CancelledError):
            await manager._process_task(task.task_id)

        assert task.status == fast_api.TASK_PENDING
        assert manager.last_worker_error is None
        assert manager.is_healthy()
    finally:
        await manager.shutdown()
