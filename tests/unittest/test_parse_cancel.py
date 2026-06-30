import asyncio
import json
import os
import tempfile
import unittest

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from mineru.cli import api_client, fast_api
from mineru.backend.hybrid import hybrid_analyze
from mineru.backend.vlm import vlm_analyze
from mineru.cli.vllm_cancellation import (
    VllmCancellationRegistry,
    patch_http_vlm_client_for_cancellation,
    vllm_request_context,
)


class FakeRequest:
    def url_for(self, name, **path_params):
        task_id = path_params["task_id"]
        if name.endswith("status"):
            return f"http://testserver/tasks/{task_id}"
        if name.endswith("result"):
            return f"http://testserver/tasks/{task_id}/result"
        raise AssertionError(f"unexpected route name: {name}")


class AbortHandle:
    def __init__(self):
        self.aborted: list[str] = []

    async def abort(self, request_id: str):
        self.aborted.append(request_id)


def make_task(task_id="task-1", status=None, backend="vlm-vllm-engine"):
    return fast_api.AsyncParseTask(
        task_id=task_id,
        status=status or fast_api.TASK_PENDING,
        backend=backend,
        file_names=["sample"],
        created_at="2026-06-25T00:00:00+00:00",
        output_dir="/tmp/mineru-test-output",
        effort="medium",
        parse_method="auto",
        lang_list=["ch"],
        formula_enable=True,
        table_enable=True,
        image_analysis=True,
        server_url=None,
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=False,
        return_original_file=False,
        client_side_output_generation=False,
        start_page_id=0,
        end_page_id=99999,
        upload_names=["sample.pdf"],
        uploads=["/tmp/sample.pdf"],
    )


class ParseCancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_pending_task_marks_cancelled_and_terminal(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task()
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()

        result = await manager.cancel_task(task.task_id)

        self.assertEqual(result.task.status, fast_api.TASK_CANCELLED)
        self.assertIsNotNone(result.task.completed_at)
        self.assertEqual(result.abort_count, 0)
        self.assertTrue(fast_api.is_task_terminal(result.task.status))
        self.assertTrue(manager.task_events[task.task_id].is_set())

    async def test_cancel_processing_task_aborts_active_vllm_requests(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PROCESSING)
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()
        abort_handle = AbortHandle()
        manager.cancellation_registry.register_request(
            task_id=task.task_id,
            file_name="sample",
            request_id="vllm-req-1",
            abort_handle=abort_handle,
        )

        result = await manager.cancel_task(task.task_id)

        self.assertEqual(result.task.status, fast_api.TASK_CANCELLED)
        self.assertEqual(result.abort_count, 1)
        self.assertEqual(result.aborted_request_ids, ["vllm-req-1"])
        self.assertEqual(abort_handle.aborted, ["vllm-req-1"])

    async def test_cancelled_task_is_not_processed(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_CANCELLED)
        task.cancel_requested = True
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()

        original_run_parse_job = fast_api.run_parse_job

        async def fail_if_called(**kwargs):
            raise AssertionError("cancelled task should not be parsed")

        fast_api.run_parse_job = fail_if_called
        try:
            await manager._process_task(task.task_id)
        finally:
            fast_api.run_parse_job = original_run_parse_job

        self.assertEqual(task.status, fast_api.TASK_CANCELLED)
        self.assertTrue(manager.task_events[task.task_id].is_set())

    async def test_worker_error_after_cancel_stays_cancelled(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PROCESSING)
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()

        async def fail_after_cancel(task):
            task.cancel_requested = True
            task.cancel_reason = "Task cancellation requested"
            raise RuntimeError("vllm returned 500")

        manager._run_task = fail_after_cancel

        await manager._process_task(task.task_id)

        self.assertEqual(task.status, fast_api.TASK_CANCELLED)
        self.assertEqual(task.error, "Task cancellation requested")
        self.assertTrue(manager.task_events[task.task_id].is_set())

    async def test_run_task_keeps_old_run_parse_job_wrapper_signature_compatible(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PENDING)
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()
        calls = []
        original_run_parse_job = fast_api.run_parse_job

        async def old_signature_wrapper(output_dir, uploads, request_options, config):
            calls.append((output_dir, uploads, request_options, config))
            self.assertIs(
                request_options.mineru_cancellation_registry,
                manager.cancellation_registry,
            )
            return ["sample"]

        fast_api.run_parse_job = old_signature_wrapper
        try:
            await manager._run_task(task)
        finally:
            fast_api.run_parse_job = original_run_parse_job

        self.assertEqual(len(calls), 1)
        self.assertEqual(task.status, fast_api.TASK_COMPLETED)

    async def test_http_vlm_client_patch_injects_and_tracks_request_id(self):
        class FakeHttpClient:
            chat_url = "http://vllm.test/v1/chat/completions"
            server_url = ""
            server_headers = None

            def __init__(self):
                self.started = asyncio.Event()
                self.finish = asyncio.Event()
                self.last_request_body = None

            def build_request_body(self, *args, **kwargs):
                return {"model": "mineru-test", "messages": []}

            async def aio_predict(self, *args, **kwargs):
                self.last_request_body = self.build_request_body()
                self.started.set()
                await self.finish.wait()
                return "ok"

        client = FakeHttpClient()
        registry = VllmCancellationRegistry()
        patch_http_vlm_client_for_cancellation(client)

        async def run_predict():
            with vllm_request_context(registry, "task-1", "sample"):
                return await client.aio_predict()

        task = asyncio.create_task(run_predict())
        await client.started.wait()

        request_ids = registry.get_active_request_ids("task-1", "sample")
        self.assertEqual(len(request_ids), 1)
        self.assertEqual(client.last_request_body["request_id"], request_ids[0])
        self.assertRegex(request_ids[0], r"^mineru-task1-[0-9a-f]{12}$")

        client.finish.set()
        self.assertEqual(await task, "ok")
        self.assertEqual(registry.get_active_request_ids("task-1", "sample"), [])

    async def test_http_vlm_client_abort_cancels_local_request_task(self):
        class FakeHttpClient:
            chat_url = "http://vllm.test/v1/chat/completions"
            server_url = ""
            server_headers = None

            def __init__(self):
                self.started = asyncio.Event()
                self.finish = asyncio.Event()

            def build_request_body(self, *args, **kwargs):
                return {"model": "mineru-test", "messages": []}

            async def aio_predict(self, *args, **kwargs):
                self.build_request_body()
                self.started.set()
                await self.finish.wait()
                return "ok"

        client = FakeHttpClient()
        registry = VllmCancellationRegistry()
        patch_http_vlm_client_for_cancellation(client)

        async def run_predict():
            with vllm_request_context(registry, "task-1", "sample"):
                return await client.aio_predict()

        task = asyncio.create_task(run_predict())
        await client.started.wait()
        request_ids = registry.get_active_request_ids("task-1", "sample")

        aborted_request_ids = await registry.abort_requests("task-1", "sample")

        self.assertEqual(aborted_request_ids, request_ids)
        with self.assertRaises(asyncio.CancelledError):
            await task


class ParseCancelPayloadTests(unittest.TestCase):
    def test_status_payload_includes_cancellation_and_vllm_request_ids(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PROCESSING)
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()
        manager.cancellation_registry.register_request(
            task_id=task.task_id,
            file_name="sample",
            request_id="vllm-req-1",
            abort_handle=AbortHandle(),
        )

        payload = manager.build_status_payload(task, FakeRequest())

        self.assertFalse(payload["cancel_requested"])
        self.assertEqual(
            payload["files"]["sample"]["active_vllm_request_ids"],
            ["vllm-req-1"],
        )

    def test_api_client_parses_cancel_response_payload(self):
        response = api_client.parse_cancel_response_payload(
            {
                "task_id": "task-1",
                "status": "cancelled",
                "abort_count": 2,
                "aborted_vllm_request_ids": ["req-1", "req-2"],
            }
        )

        self.assertEqual(response.task_id, "task-1")
        self.assertEqual(response.status, "cancelled")
        self.assertEqual(response.abort_count, 2)
        self.assertEqual(response.aborted_vllm_request_ids, ("req-1", "req-2"))


class ParsePauseRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_registry_blocks_until_resume_and_cleans_up(self):
        registry = fast_api.PauseRegistry()
        task = make_task(status=fast_api.TASK_PROCESSING)
        signalled = asyncio.Event()
        registry.track_task(task, lambda task_id: signalled.set())

        registry.request_pause(task.task_id)
        self.assertTrue(registry.is_pause_requested(task.task_id))

        checkpoint = {"completed_until_page": 4, "next_start_page": 5}
        registry.mark_paused(task.task_id, checkpoint)
        self.assertEqual(task.status, fast_api.TASK_PAUSED)
        self.assertEqual(task.checkpoint, checkpoint)
        self.assertTrue(signalled.is_set())

        waiter = asyncio.create_task(registry.wait_until_resumed(task.task_id))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        registry.resume(task.task_id)
        await asyncio.wait_for(waiter, timeout=1)
        self.assertFalse(registry.is_pause_requested(task.task_id))
        self.assertEqual(task.status, fast_api.TASK_PROCESSING)
        self.assertIsNotNone(task.resumed_at)

        registry.cleanup_task(task.task_id)
        self.assertFalse(registry.is_pause_requested(task.task_id))


class ParseCheckpointStoreTests(unittest.TestCase):
    def test_checkpoint_store_writes_window_partials_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as output_dir:
            store = fast_api.CheckpointStore()

            checkpoint = store.write_processing_checkpoint(
                task_output_dir=output_dir,
                task_id="task-1",
                file_name="sample",
                backend="vlm-http-client",
                parse_method="auto",
                status=fast_api.TASK_PAUSED,
                page_count=10,
                window_size=4,
                window_index=1,
                window_start=4,
                window_end=7,
                middle_json={"pdf_info": [{"page_idx": 0}]},
                model_output=[{"page": 5}],
                window_result=[{"page": 5}],
            )

            self.assertEqual(checkpoint["completed_until_page"], 8)
            self.assertEqual(checkpoint["next_start_page"], 9)
            self.assertEqual(checkpoint["artifacts"]["latest_window"], "pause_resume/windows/window-0001.json")

            loaded = store.load_checkpoint(output_dir)
            self.assertEqual(loaded["task_id"], "task-1")
            self.assertEqual(loaded["status"], fast_api.TASK_PAUSED)

            with open(f"{output_dir}/pause_resume/middle_json_partial.json", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"pdf_info": [{"page_idx": 0}]})
            with open(f"{output_dir}/pause_resume/model_output_partial.json", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), [{"page": 5}])

    def test_checkpoint_store_clears_pause_resume_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            store = fast_api.CheckpointStore()
            store.write_processing_checkpoint(
                task_output_dir=output_dir,
                task_id="task-1",
                file_name="sample",
                backend="vlm-http-client",
                parse_method="auto",
                status=fast_api.TASK_PAUSED,
                page_count=10,
                window_size=4,
                window_index=1,
                window_start=4,
                window_end=7,
                middle_json={"pdf_info": [{"page_idx": 0}]},
                model_output=[{"page": 5}],
                window_result=[{"page": 5}],
            )

            store.clear_checkpoint(output_dir)

            self.assertFalse(os.path.exists(f"{output_dir}/pause_resume"))


class ParsePauseManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_processing_task_marks_pause_requested(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PROCESSING)
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()
        manager.pause_registry.track_task(task, manager._signal_task_event)

        result = await manager.pause_task(task.task_id)

        self.assertIs(result, task)
        self.assertEqual(task.status, fast_api.TASK_PAUSE_REQUESTED)
        self.assertIsNotNone(task.pause_requested_at)
        self.assertTrue(manager.pause_registry.is_pause_requested(task.task_id))
        self.assertTrue(manager.task_events[task.task_id].is_set())

    async def test_resume_paused_task_marks_processing(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PAUSED)
        task.checkpoint = {"next_start_page": 5}
        manager.tasks[task.task_id] = task
        manager.task_events[task.task_id] = asyncio.Event()
        manager.pause_registry.track_task(task, manager._signal_task_event)
        manager.pause_registry.request_pause(task.task_id)
        manager.pause_registry.mark_paused(task.task_id, task.checkpoint)

        result = await manager.resume_task(task.task_id)

        self.assertIs(result, task)
        self.assertEqual(task.status, fast_api.TASK_PROCESSING)
        self.assertIsNotNone(task.resumed_at)
        self.assertFalse(manager.pause_registry.is_pause_requested(task.task_id))

    async def test_pause_terminal_task_is_rejected(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_COMPLETED)
        manager.tasks[task.task_id] = task

        with self.assertRaises(ValueError):
            await manager.pause_task(task.task_id)

    async def test_cancel_paused_task_clears_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = fast_api.AsyncTaskManager(FastAPI())
            task = make_task(status=fast_api.TASK_PAUSED)
            task.output_dir = output_dir
            task.checkpoint = {"next_start_page": 10}
            manager.tasks[task.task_id] = task
            manager.task_events[task.task_id] = asyncio.Event()
            manager.pause_registry.track_task(task, manager._signal_task_event)
            manager.checkpoint_store.write_processing_checkpoint(
                task_output_dir=output_dir,
                task_id=task.task_id,
                file_name="sample",
                backend="vlm-http-client",
                parse_method="auto",
                status=fast_api.TASK_PAUSED,
                page_count=10,
                window_size=1,
                window_index=8,
                window_start=8,
                window_end=8,
                middle_json={"pdf_info": [{"page_idx": 8}]},
                model_output=[{"page": 9}],
                window_result=[{"page": 9}],
            )

            result = await manager.cancel_task(task.task_id)

            self.assertEqual(result.task.status, fast_api.TASK_CANCELLED)
            self.assertIsNone(result.task.checkpoint)
            self.assertFalse(os.path.exists(f"{output_dir}/pause_resume"))

    async def test_result_endpoint_returns_accepted_for_paused_task(self):
        manager = fast_api.AsyncTaskManager(FastAPI())
        task = make_task(status=fast_api.TASK_PAUSED)
        manager.tasks[task.task_id] = task
        original_task_manager = getattr(fast_api.app.state, "task_manager", None)
        fast_api.app.state.task_manager = manager
        try:
            response = await fast_api.get_async_task_result(
                task.task_id,
                FakeRequest(),
                BackgroundTasks(),
            )
        finally:
            if original_task_manager is None:
                delattr(fast_api.app.state, "task_manager")
            else:
                fast_api.app.state.task_manager = original_task_manager

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 202)


class ParsePauseAnalyzeHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_vlm_window_hook_writes_checkpoint_and_waits_for_resume(self):
        with tempfile.TemporaryDirectory() as output_dir:
            registry = fast_api.PauseRegistry()
            task = make_task(status=fast_api.TASK_PAUSE_REQUESTED)
            registry.track_task(task)
            registry.request_pause(task.task_id)

            hook_task = asyncio.create_task(
                vlm_analyze._checkpoint_window_and_maybe_pause(
                    checkpoint_store=fast_api.CheckpointStore(),
                    pause_registry=registry,
                    task_output_dir=output_dir,
                    task_id=task.task_id,
                    file_name="sample",
                    backend="http-client",
                    parse_method="vlm",
                    page_count=6,
                    window_size=2,
                    window_index=0,
                    window_start=0,
                    window_end=1,
                    middle_json={"pdf_info": [{"page_idx": 0}]},
                    model_output=[{"page": 1}],
                    window_result=[{"page": 1}],
                )
            )
            await asyncio.sleep(0.05)

            self.assertFalse(hook_task.done())
            self.assertEqual(task.status, fast_api.TASK_PAUSED)
            self.assertEqual(task.checkpoint["completed_until_page"], 2)

            registry.resume(task.task_id)
            await asyncio.wait_for(hook_task, timeout=1)
            self.assertEqual(task.status, fast_api.TASK_PROCESSING)

    async def test_hybrid_window_hook_writes_checkpoint_and_waits_for_resume(self):
        with tempfile.TemporaryDirectory() as output_dir:
            registry = fast_api.PauseRegistry()
            task = make_task(status=fast_api.TASK_PAUSE_REQUESTED, backend="hybrid-vllm-engine")
            registry.track_task(task)
            registry.request_pause(task.task_id)

            hook_task = asyncio.create_task(
                hybrid_analyze._checkpoint_window_and_maybe_pause(
                    checkpoint_store=fast_api.CheckpointStore(),
                    pause_registry=registry,
                    task_output_dir=output_dir,
                    task_id=task.task_id,
                    file_name="sample",
                    backend="vllm-engine",
                    parse_method="auto",
                    page_count=6,
                    window_size=2,
                    window_index=0,
                    window_start=0,
                    window_end=1,
                    middle_json={"pdf_info": [{"page_idx": 0}]},
                    model_output=[{"page": 1}],
                    window_result=[{"page": 1}],
                )
            )
            await asyncio.sleep(0.05)

            self.assertFalse(hook_task.done())
            self.assertEqual(task.status, fast_api.TASK_PAUSED)
            self.assertEqual(task.checkpoint["next_start_page"], 3)

            registry.resume(task.task_id)
            await asyncio.wait_for(hook_task, timeout=1)
            self.assertEqual(task.status, fast_api.TASK_PROCESSING)


if __name__ == "__main__":
    unittest.main()
