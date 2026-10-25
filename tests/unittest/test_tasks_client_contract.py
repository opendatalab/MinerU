"""共享 Tasks 字段后仍保留 SDK 与同步网关各自的响应消费边界。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from mineru.kit.file_parse.client import TaskClient
from mineru.kit.file_parse.models import GatewayError, ParseOptions, RemoteTask, fingerprint
from mineru.kit.file_parse.settings import FileParseSettings
from mineru.parser.api_client import MinerUApiParser, _V1APIError


def _pending_task() -> dict[str, Any]:
    """构造客户端所需的最小单文件详情，不要求它携带服务端管理字段。"""
    return {
        "task_id": "task-one",
        "status": "pending",
        "output_formats": ["default"],
        "files": [{"file_id": "file-one", "file_name": "file.pdf", "bytes": 3, "status": "pending"}],
    }


@pytest.fixture
def clients(tmp_path: Path) -> Iterator[tuple[MinerUApiParser, TaskClient]]:
    """使用禁止联网的传输构造真实客户端，并在每项测试结束后关闭连接池。"""

    def reject_request(request: httpx.Request) -> httpx.Response:
        """响应契约测试只能消费给定快照，不允许发送任何请求。"""
        raise AssertionError(f"Unexpected network request: {request.method}")

    api_url = "https://api.test/gateway"
    gateway = TaskClient(
        FileParseSettings(api_url=api_url, state_dir=tmp_path),
        transport=httpx.MockTransport(reject_request),
    )
    try:
        yield MinerUApiParser(api_url=api_url), gateway
    finally:
        asyncio.run(gateway.close())


@pytest.mark.parametrize(
    "task_patch,file_patch,sdk_accepts,gateway_accepts",
    [
        pytest.param({}, {}, True, True, id="minimal-detail"),
        pytest.param({}, {"bytes": "3"}, False, True, id="numeric-string-size"),
        pytest.param({"output_formats": ["default", "default"]}, {}, False, True, id="duplicate-formats"),
        pytest.param({}, {"error": {"code": "remote-code"}}, True, True, id="partial-error-details"),
        pytest.param(
            {"status": "completed"},
            {"status": "completed", "result_url": ""},
            True,
            False,
            id="empty-completed-result-url",
        ),
        pytest.param({}, {"result_url": "https://storage.test/early"}, True, False, id="premature-result-url"),
        pytest.param(
            {"status": "waiting-file"},
            {"status": "waiting-file", "upload": {"url": "https://storage.test/input"}},
            True,
            False,
            id="partial-upload-descriptor",
        ),
        pytest.param(
            {"status": "waiting-file"},
            {
                "status": "waiting-file",
                "url": "https://source.test/file.pdf",
                "upload": {"url": "https://storage.test/input", "headers": {}, "expires_at": 9999999999},
            },
            True,
            True,
            id="unconsumed-source-url",
        ),
    ],
)
def test_snapshot_consumers_keep_existing_acceptance_boundaries(
    clients: tuple[MinerUApiParser, TaskClient],
    task_patch: dict[str, Any],
    file_patch: dict[str, Any],
    sdk_accepts: bool,
    gateway_accepts: bool,
) -> None:
    """实际快照入口保留各自宽容度，不把服务端完整响应模型强加给消费者。"""
    sdk, gateway = clients
    expected = _pending_task()
    payload = _pending_task()
    payload.update(task_patch)
    payload["files"][0].update(file_patch)
    response = httpx.Response(200, json=payload)
    if sdk_accepts:
        assert sdk._snapshot(response, expected) == payload
    else:
        with pytest.raises(_V1APIError) as sdk_error:
            sdk._snapshot(response, expected)
        assert sdk_error.value.code == "invalid_response"

    if gateway_accepts:
        actual = gateway._snapshot(response.content, RemoteTask.model_validate(expected), ["default"])
        assert actual.task_id == expected["task_id"]
        assert actual.files[0].bytes == 3
        assert actual.output_formats == payload["output_formats"]
        assert actual.files[0].error == payload["files"][0].get("error")
    else:
        with pytest.raises(GatewayError) as gateway_error:
            gateway._snapshot(response.content, RemoteTask.model_validate(expected), ["default"])
        assert gateway_error.value.code == "upstream_protocol_error"
        assert gateway_error.value.status_code == 502


def test_sdk_preserves_response_extensions_and_does_not_reuse_missing_fields(
    clients: tuple[MinerUApiParser, TaskClient],
) -> None:
    """身份投影只负责校验，SDK 仍返回完整新快照且不会补回上次的可选字段。"""
    sdk, _ = clients
    expected = _pending_task()
    expected["files"][0]["error"] = {"code": "stale-error"}
    payload = _pending_task()
    payload["provider_extension"] = {"queue": "priority"}
    payload["files"][0]["provider_extension"] = {"attempt": 2}
    payload["files"][0]["parse"] = {"duration_ms": 25, "provider_timing": "accepted"}

    actual = sdk._snapshot(httpx.Response(200, json=payload), expected)

    assert actual == payload
    assert "error" not in actual["files"][0]
    assert "upload" not in actual["files"][0]


@pytest.mark.parametrize(
    "file_patch,message",
    [
        ({"file_id": "", "status": "unknown"}, "Inconsistent Tasks response"),
        ({"file_name": 7, "status": "unknown"}, "Inconsistent Tasks response"),
        ({"file_id": 7}, "Inconsistent Tasks response"),
        ({"status": "partial"}, "Invalid task file state"),
        ({"status": "unknown", "upload": [], "file_id": "changed"}, "Invalid task file state"),
        ({"status": "running", "upload": {}, "file_id": "changed"}, "Invalid task upload descriptor"),
        ({"file_id": "changed", "bytes": 4}, "Task input identity changed"),
    ],
)
def test_sdk_preserves_identity_error_priority(
    clients: tuple[MinerUApiParser, TaskClient], file_patch: dict[str, Any], message: str
) -> None:
    """同时存在多个响应错误时，仍先报告原有身份、状态和上传边界中的首个错误。"""
    sdk, _ = clients
    payload = _pending_task()
    payload["files"][0].update(file_patch)
    with pytest.raises(_V1APIError) as error:
        sdk._snapshot(httpx.Response(200, json=payload), _pending_task())
    assert error.value.code == "invalid_response"
    assert error.value.message == message


@pytest.mark.parametrize("status", ["pending", "completed"])
def test_creation_receipt_keeps_local_recovery_context(clients: tuple[MinerUApiParser, TaskClient], status: str) -> None:
    """短创建回执从请求补齐大小和格式，已完成回执也可先保存再获取详情。"""
    sdk, gateway = clients
    receipt = {
        "task_id": "task-one",
        "status": status,
        "files": [{"file_id": "file-one", "file_name": "file.pdf", "status": status}],
    }
    request = {"files": [{"file_name": "file.pdf", "bytes": 3}], "output_formats": ["default"]}
    sdk_task = sdk._created_task(httpx.Response(202, json=receipt), request)
    gateway_task = gateway._created_snapshot(json.dumps(receipt).encode(), request)

    assert sdk_task["files"][0]["bytes"] == 3
    assert "result_url" not in sdk_task["files"][0]
    restored = RemoteTask.model_validate_json(gateway_task.model_dump_json())
    assert restored.task_id == "task-one"
    assert restored.files[0].bytes == 3
    assert restored.output_formats == ["default"]
    assert restored.files[0].result_url is None
    assert restored.details_loaded is False


@pytest.mark.parametrize("page_range,normalized_range", [(" ALL ", ""), (" 1-3 ", "1-3")])
def test_parse_options_normalization_preserves_request_identity(page_range: str, normalized_range: str) -> None:
    """共享选项模型后仍保留格式去重顺序和页范围归一化，等价请求指纹相同。"""
    options = ParseOptions(page_range=page_range, output_formats=["html", "default", "html", "docx", "default"])
    normalized = ParseOptions(page_range=normalized_range, output_formats=["html", "default", "docx"])
    assert options.page_range == normalized_range
    assert options.output_formats == ["html", "default", "docx"]
    assert fingerprint("file.pdf", 3, "a" * 64, options) == fingerprint("file.pdf", 3, "a" * 64, normalized)


@pytest.mark.parametrize("extra", [{"wait_timeout": 10}, {"include_model_output": False}])
def test_parse_options_rejects_fields_outside_parse_identity(extra: dict[str, Any]) -> None:
    """等待预算和已移除参数不能混入解析身份或被继承模型静默忽略。"""
    with pytest.raises(ValueError, match="extra_forbidden"):
        ParseOptions(**extra)
