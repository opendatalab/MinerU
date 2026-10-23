"""SDK 和同步包装在等待文件但没有上传动作时只轮询，避免重复传输或提前清理。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_file_parse import Upstream, application, submit
from test_tasks_sdk import bind_transport, parse, task_snapshot, zip_output

from mineru.parser import MinerUApiParser


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["success", "lost_accepted", "lost_retry"])
def test_sdk_waits_without_upload_and_preserves_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, outcome: str
) -> None:
    """上传目标延后出现、成功后延迟确认和响应丢失均不触发重复创建或无限上传。"""
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")
    puts = 0
    queries = 0
    creates = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """按阶段提供无上传动作和待重传快照，所有结果仍由正式 SDK 消费。"""
        nonlocal puts, queries, creates
        if request.method == "POST":
            creates += 1
            return httpx.Response(202, json=task_snapshot("waiting-file"))
        if request.method == "PUT":
            puts += 1
            queries = 0
            assert request.content == b"pdf" and request.headers["content-length"] == "3"
            if puts == 1 and outcome != "success":
                raise httpx.ReadError("upload response lost", request=request)
            return httpx.Response(200)
        if request.url.host == "storage.test":
            return httpx.Response(200, content=zip_output())
        queries += 1
        if puts == 0:
            snapshot = task_snapshot("waiting-file", upload=queries >= 2)
        elif outcome == "lost_retry" and puts == 1:
            snapshot = task_snapshot("waiting-file", upload=queries >= 3)
        else:
            snapshot = task_snapshot("waiting-file" if queries < 3 else "pending" if queries == 3 else "completed")
        return httpx.Response(200, json=snapshot)

    bind_transport(monkeypatch, handle)
    result = parse(MinerUApiParser(api_url="https://api.test"), source, asynchronous)
    assert len(result.pages) == 1 and creates == 1
    assert puts == (2 if outcome == "lost_retry" else 1)


def test_file_parse_keeps_source_until_input_is_ready(tmp_path: Path) -> None:
    """缓存命中后的服务端准备期间保留本地源文件，不把 upload=null 当成已就绪。"""
    upstream = Upstream()
    upstream.cache_hit = True
    upstream.ready_delay = 1000
    app = application(tmp_path, upstream)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(submit, client, wait_timeout=0)
        try:
            deadline = time.monotonic() + 3
            while True:
                record = app.state.service.records.get("request_1")
                if record is not None and record.remote is not None:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            state = client.get("/v1/file_parse/request_1").json()
            assert state["status"] == "waiting-file"
            assert record.task.files[0].upload is None
            retained = Path(record.source_path)
            assert retained.is_file() and upstream.uploads == 0
            assert not pending.done()
        finally:
            upstream.ready_delay = 0
        response = pending.result(timeout=3)
        assert response.status_code in {200, 202}
        assert client.get("/v1/file_parse/request_1").json()["status"] == "completed"
        assert record.source_path is None and not retained.exists()
        assert upstream.uploads == 0 and len(upstream.tasks) == 1
