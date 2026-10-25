"""Tasks 回归测试共享的真实上传与产物读取工具，不模拟旧 HTTP 协议。"""

import hashlib
import io
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient
from mineru.task_types import TaskOutputFormat, task_status_url
from mineru.types import Tier
from mineru.parser.api_server import FileStore, _ExecutionRecord, _ExecutionRequest
from mineru.parser.base import ParseResult
from mineru.parser.task_outputs import build_task_output
from mineru.kit.router.resources import ExecutionRoute, ResourceRegistry, utc_now_iso
from mineru.kit.router.route_state import ExecutionRouteState
from mineru.kit.router.workers import JobReservation
from mineru.tasks.execution import TaskExecution, new_execution_id


def router_execution(
    registry: ResourceRegistry,
    *,
    worker_id: str,
    upstream_id: str,
    owner_scope: str = "anonymous",
    file_id: str = "file-test",
    file_name: str = "test.pdf",
    reservation: JobReservation | None = None,
) -> ExecutionRoute:
    """为路由生命周期测试登记完整执行快照，不先创建缺少类型状态的空记录。"""
    execution = TaskExecution(new_execution_id(), file_id, file_name, "pending", utc_now_iso())
    return registry.register_execution(
        owner_scope=owner_scope,
        worker_id=worker_id,
        upstream_id=upstream_id,
        state=ExecutionRouteState(execution=execution, reservation=reservation),
    )


def task_url(client: TestClient | httpx.Client | httpx.AsyncClient, task: dict[str, Any]) -> str:
    """从客户端部署地址和任务 ID 构造固定查询路径，不依赖响应链接。"""
    return task_status_url(str(client.base_url), task["task_id"])


def pending_files(task: dict[str, Any]) -> list[dict[str, Any]]:
    """从真实逐文件快照选取带上传动作的条目，省略字段时只轮询。"""
    return [item for item in task["files"] if item.get("upload") is not None]


def stored_source(store: FileStore, path: Path) -> dict[str, str]:
    """为执行层单元测试登记已校验源字节，不创建公开上传资源。"""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_id = store.register_source_file(path.name, path, sha)
    return {"type": "file_id", "file_id": file_id}


def execution_request(
    store: FileStore,
    source: dict[str, str],
    *,
    tier: Tier | None = None,
    page_range: str | None = None,
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    output_formats: list[TaskOutputFormat] | None = None,
    file_name: str | None = None,
) -> _ExecutionRequest:
    """为单文件执行测试分配结果身份，源 ID 与结果 ID 始终独立。"""
    if file_name is None:
        file_name = (
            store.get_file(source["file_id"]).filename
            if source["type"] == "file_id"
            else source["url"].rsplit("/", 1)[-1].split("?")[0] or "unknown"
        )
    return _ExecutionRequest(
        source=source,
        file_id=store.reserve_file_id(),
        file_name=file_name,
        tier=tier,
        page_range=page_range,
        ocr_mode=ocr_mode,
        output_formats=output_formats or ["default"],
    )


def execution_output(store: FileStore, record: _ExecutionRecord, name: str) -> bytes:
    """从唯一 Tasks 产物读取成员，避免断言已经移除的多产物引用。"""
    with zipfile.ZipFile(io.BytesIO(store.read_file_data(record.result_file_id))) as archive:
        return archive.read(name)


def build_default_zip(result: ParseResult) -> bytes:
    """使用真实 Tasks 产物构建器，保存 Middle JSON 和图片素材。"""
    with tempfile.TemporaryDirectory() as directory:
        return build_task_output(result, ["default"], Path(directory)).read_bytes()


def submit_task(client: TestClient, path: Path, **options: Any) -> dict[str, Any]:
    """通过清单和返回链接上传文件，任何失败都保留真实 HTTP 响应供断言。"""
    data = path.read_bytes()
    response = client.post(
        "/v1/tasks",
        json={
            "files": [
                {
                    "file_name": path.name,
                    "bytes": len(data),
                    "sha256sum": hashlib.sha256(data).hexdigest(),
                }
            ],
            **options,
        },
    )
    assert response.status_code == 202, response.text
    task = response.json()
    for link in pending_files(task):
        response = client.put(link.get("upload").get("url"), content=data, headers=link.get("upload")["headers"])
        assert response.status_code < 400, response.text
    return task


def wait_task(client: TestClient, task: dict[str, Any], timeout: float = 5) -> dict[str, Any]:
    """等待父任务终态，避免使用内部执行 ID 代替公开 task_id。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(task_url(client, task))
        assert response.status_code == 200, response.text
        task = response.json()
        if task["status"] in {"completed", "partial", "failed", "canceled"}:
            return task
        time.sleep(0.01)
    raise AssertionError(f"Task did not finish: {task}")
