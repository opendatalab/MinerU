# Copyright (c) Opendatalab. All rights reserved.
"""Unit tests for request priority scheduling in mineru-api.

Covers the task-level priority queue of ``AsyncTaskManager``:
- priority ordering under backlog,
- submission-order FIFO for equal priorities,
- ``get_queued_ahead`` semantics under priority ordering,
- the slot-first dispatcher (regression: under sparse arrivals a naive
  dequeue-then-acquire dispatcher hands freed slots out in submission
  order, silently nullifying priorities),
- execution-slot release on task failure.
"""
import asyncio
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from mineru.cli import fast_api


def _make_task(task_id: str, priority: int = 0) -> fast_api.AsyncParseTask:
    return fast_api.AsyncParseTask(
        task_id=task_id,
        status=fast_api.TASK_PENDING,
        backend="pipeline",
        file_names=[f"{task_id}.pdf"],
        created_at=fast_api.utc_now_iso(),
        output_dir=f"/tmp/{task_id}",
        effort="high",
        parse_method="auto",
        lang_list=["ch"],
        formula_enable=True,
        table_enable=True,
        image_analysis=True,
        server_url=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        response_format_zip=False,
        return_original_file=False,
        client_side_output_generation=False,
        priority=priority,
        start_page_id=0,
        end_page_id=99999,
        upload_names=[f"{task_id}.pdf"],
        uploads=[],
    )


class _FakeApp:
    class state:
        config = {}


class _ParseJobController:
    """Stands in for run_parse_job: records start order, blocks on gates."""

    def __init__(self):
        self.started: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.fail_ids: set[str] = set()

    async def run(self, **kwargs: Any) -> list[str]:
        task_id = kwargs["request_options"].task_id
        if task_id in self.fail_ids:
            raise RuntimeError(f"simulated parse failure: {task_id}")
        self.started.append(task_id)
        gate = asyncio.Event()
        self.gates[task_id] = gate
        await gate.wait()
        return [task_id]


async def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


async def _wait_all_started(
    controller: _ParseJobController, count: int, timeout: float = 5.0
) -> None:
    """Release gates as tasks progress so each freed slot admits the next one."""
    deadline = time.monotonic() + timeout
    while len(controller.started) < count:
        for gate in controller.gates.values():
            gate.set()
        if time.monotonic() > deadline:
            raise AssertionError("not all tasks started within timeout")
        await asyncio.sleep(0.01)


@pytest.fixture()
def task_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[fast_api.AsyncTaskManager]:
    monkeypatch.setenv("MINERU_API_TASK_RETENTION_SECONDS", "0")
    original_semaphore = fast_api._request_semaphore
    manager = fast_api.AsyncTaskManager(_FakeApp())
    yield manager
    fast_api._request_semaphore = original_semaphore


def test_backlog_dequeues_high_priority_first(
    task_manager: fast_api.AsyncTaskManager,
) -> None:
    async def scenario() -> list[str]:
        for task_id, priority in (("low", 0), ("high", 10), ("mid", 5)):
            await task_manager.submit(_make_task(task_id, priority))
        order = []
        for _ in range(3):
            _neg_priority, _submit_order, task_id = await task_manager.queue.get()
            order.append(task_id)
        return order

    assert asyncio.run(scenario()) == ["high", "mid", "low"]


def test_same_priority_keeps_submission_order(
    task_manager: fast_api.AsyncTaskManager,
) -> None:
    async def scenario() -> list[str]:
        for task_id in ("first", "second", "third"):
            await task_manager.submit(_make_task(task_id, 5))
        order = []
        for _ in range(3):
            _neg_priority, _submit_order, task_id = await task_manager.queue.get()
            order.append(task_id)
        return order

    assert asyncio.run(scenario()) == ["first", "second", "third"]


def test_queued_ahead_respects_priority(
    task_manager: fast_api.AsyncTaskManager,
) -> None:
    async def scenario() -> dict[str, int]:
        await task_manager.submit(_make_task("p10", 10))
        await task_manager.submit(_make_task("p5a", 5))
        await task_manager.submit(_make_task("p5b", 5))
        await task_manager.submit(_make_task("p0", 0))
        return {
            task_id: task_manager.get_queued_ahead(task_id)
            for task_id in ("p10", "p5a", "p5b", "p0")
        }

    assert asyncio.run(scenario()) == {"p10": 0, "p5a": 1, "p5b": 2, "p0": 3}


def test_slot_first_dispatcher_prefers_high_priority_under_sparse_arrivals(
    task_manager: fast_api.AsyncTaskManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the slot-first dispatcher.

    With one execution slot busy and an *empty* queue, submitting a
    default-priority task followed by a high-priority task must give the
    high-priority task the first freed slot. A dequeue-then-acquire
    dispatcher would spawn processors in submission order and the
    semaphore's FIFO wake-up would hand the slot to the low-priority task.
    """
    controller = _ParseJobController()
    monkeypatch.setattr(fast_api, "run_parse_job", controller.run)
    fast_api._request_semaphore = asyncio.Semaphore(1)

    async def scenario() -> list[str]:
        await task_manager.start()
        try:
            await task_manager.submit(_make_task("busy", 0))
            await _wait_until(lambda: controller.started == ["busy"])

            # Sparse arrivals while the only slot is busy.
            await task_manager.submit(_make_task("low", 0))
            await task_manager.submit(_make_task("high", 10))
            await asyncio.sleep(0.05)

            controller.gates["busy"].set()
            await _wait_all_started(controller, 3)
            return list(controller.started)
        finally:
            for gate in controller.gates.values():
                gate.set()
            await task_manager.shutdown()

    assert asyncio.run(scenario()) == ["busy", "high", "low"]


def test_held_slot_released_on_task_failure(
    task_manager: fast_api.AsyncTaskManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _ParseJobController()
    controller.fail_ids = {"boom"}
    monkeypatch.setattr(fast_api, "run_parse_job", controller.run)
    fast_api._request_semaphore = asyncio.Semaphore(1)

    async def scenario() -> None:
        await task_manager.start()
        try:
            await task_manager.submit(_make_task("boom", 0))
            await _wait_until(
                lambda: task_manager.get("boom").status == fast_api.TASK_FAILED
            )

            # The dispatcher may already hold the freed slot while waiting for
            # new work, so a functional check is used instead of the semaphore
            # value: a newly submitted task must still get executed, proving
            # the slot was not leaked by the failed task.
            await task_manager.submit(_make_task("next", 0))
            await _wait_until(lambda: "next" in controller.started)
            controller.gates["next"].set()
            await _wait_until(
                lambda: task_manager.get("next").status == fast_api.TASK_COMPLETED
            )
        finally:
            for gate in controller.gates.values():
                gate.set()
            await task_manager.shutdown()

    asyncio.run(scenario())


def test_no_semaphore_runs_all_tasks_without_slot_gating(
    task_manager: fast_api.AsyncTaskManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a concurrency cap every submitted task runs unconditionally."""
    controller = _ParseJobController()
    monkeypatch.setattr(fast_api, "run_parse_job", controller.run)
    fast_api._request_semaphore = None

    async def scenario() -> list[str]:
        await task_manager.start()
        try:
            for task_id, priority in (("low", 0), ("high", 10)):
                await task_manager.submit(_make_task(task_id, priority))
            await _wait_until(lambda: len(controller.started) == 2)
            return sorted(controller.started)
        finally:
            for gate in controller.gates.values():
                gate.set()
            await task_manager.shutdown()

    assert asyncio.run(scenario()) == ["high", "low"]
