"""Tasks 回归测试共享的真实上传与产物读取工具，不模拟旧 HTTP 协议。"""

import hashlib
import io
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from mineru.parser.api_server import FileStore, _JobRecord
from mineru.parser.base import ParseResult
from mineru.parser.task_outputs import build_task_output


def stored_source(store: FileStore, path: Path) -> dict[str, str]:
    """为执行层单元测试登记已校验源字节，不创建公开上传资源。"""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_id = store.register_source_file(path.name, path, sha)
    return {"type": "file_id", "file_id": file_id}


def execution_output(store: FileStore, record: _JobRecord, name: str) -> bytes:
    """从唯一 Tasks 产物读取成员，避免断言已经移除的多产物引用。"""
    with zipfile.ZipFile(io.BytesIO(store.read_file_data(record.task_result_file_id))) as archive:
        return archive.read(name)


def build_default_zip(result: ParseResult) -> bytes:
    """使用真实 Tasks 产物构建器，保留可用原始模型输出。"""
    with tempfile.TemporaryDirectory() as directory:
        return build_task_output(result, ["default"], Path(directory), include_model_output=True).read_bytes()


def submit_task(client: TestClient, path: Path, **options: Any) -> dict[str, Any]:
    """通过清单和返回链接上传文件，任何失败都保留真实 HTTP 响应供断言。"""
    data = path.read_bytes()
    response = client.post(
        "/v1/tasks",
        json={
            "manifest": [
                {
                    "client_file_id": "file",
                    "filename": path.name,
                    "bytes": len(data),
                    "sha256sum": hashlib.sha256(data).hexdigest(),
                }
            ],
            **options,
        },
    )
    assert response.status_code == 202, response.text
    task = response.json()
    for link in task["uploads"]:
        response = client.put(link["upload_url"], content=data, headers=link["upload_headers"])
        assert response.status_code < 400, response.text
    return task


def wait_task(client: TestClient, task: dict[str, Any], timeout: float = 5) -> dict[str, Any]:
    """等待父任务终态，避免使用内部执行 ID 代替公开 task_id。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(task["status_url"])
        assert response.status_code == 200, response.text
        task = response.json()
        if task["status"] in {"completed", "partial", "failed", "canceled"}:
            return task
        time.sleep(0.01)
    raise AssertionError(f"Task did not finish: {task}")
