from __future__ import annotations

from fastapi.testclient import TestClient

from test_v1_router import _FakeV1Upstream, _make_router, _upload_file


def test_v1_router_upload_affinity_keeps_same_content_on_same_worker() -> None:
    """同一文件（相同 sha256sum）重复上传应落在同一 worker：内容寻址 affinity 的稳定性。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)
    token = "affinity-stable"
    content = b"pdf-stable-content"

    with TestClient(router_app) as client:
        first_file = _upload_file(client, token=token, content=content)
        second_file = _upload_file(client, token=token, content=content)
        first_worker = router_app.state.registry.get("file", first_file).worker_id
        second_worker = router_app.state.registry.get("file", second_file).worker_id

    assert first_worker == second_worker


def test_v1_router_upload_affinity_spreads_same_caller_files() -> None:
    """同一调用方上传多个不同文件应分散到多个 worker，不再固定单一 worker（原 pinning 缺陷）。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)
    token = "affinity-spread"

    with TestClient(router_app) as client:
        file_ids = [_upload_file(client, token=token, content=f"pdf-{index}".encode()) for index in range(16)]
        worker_ids = {router_app.state.registry.get("file", file_id).worker_id for file_id in file_ids}

    assert len(worker_ids) >= 2


def test_v1_router_upload_without_sha256sum_still_lands_on_healthy_worker() -> None:
    """缺少 sha256sum 时应回退到请求级随机数，上传仍落在健康 worker 上。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)

    with TestClient(router_app) as client:
        healthy_ids = {worker.worker_id for worker in router_app.state.worker_pool.healthy_workers()}
        headers = {"authorization": "Bearer no-sha"}
        create = client.post(
            "/v1/uploads",
            headers=headers,
            json={
                "filename": "no-sha.pdf",
                "bytes": 4,
                "mime_type": "application/pdf",
                "purpose": "parse",
            },
        )
        assert create.status_code == 200, create.text
        upload_id = create.json()["id"]
        assert client.put(f"/v1/uploads/{upload_id}/content", headers=headers, content=b"body").status_code == 200
        complete = client.post(f"/v1/uploads/{upload_id}/complete", headers=headers, json={})
        assert complete.status_code == 200, complete.text
        route = router_app.state.registry.get("file", complete.json()["file"]["id"])

    assert route.worker_id in healthy_ids
