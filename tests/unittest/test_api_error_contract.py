"""公共错误协议在本地、代理和级联部署中的结构与失败处理回归。"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_task_tiers import _deployment

from mineru.errors import MineruError
from mineru.kit.router.proxy import RouterProxyError, passthrough_response, stream_upstream
from mineru.kit.router.tasks import _checked_json
from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.tasks.contracts import ErrorDetail, ErrorResponse
from mineru.tasks.errors import ApiServerError


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": None},
        {"error": []},
        {"error": {"message": "Missing code"}},
        {"error": {"code": None, "message": "Null code"}},
        {"error": {"code": "", "message": "Empty code"}},
        {"error": {"code": 123, "message": "Numeric code"}},
        {"error": {"code": "invalid_request"}},
        {"error": {"code": "invalid_request", "message": None}},
        {"error": {"code": "invalid_request", "message": "Bad param", "param": []}},
        {"error": {"code": "invalid_request", "message": "Old schema", "type": "invalid_request_error"}},
        {"error": {"code": "invalid_request", "message": "Conflicting response"}, "task_id": "task-1"},
    ],
)
def test_malformed_upstream_errors_are_protocol_errors(payload: dict[str, Any]) -> None:
    """缺少字段、旧字段和矛盾响应必须被拒绝，Router 不补造业务错误或泄漏校验异常。"""
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(payload)
    response = httpx.Response(400, json=payload)
    with pytest.raises(ApiServerError) as task_error:
        _checked_json(response)
    assert task_error.value.status_code == 502
    assert task_error.value.error.code == "invalid_upstream_response"
    with pytest.raises(RouterProxyError) as proxy_error:
        passthrough_response(response)
    assert proxy_error.value.status_code == 502
    assert proxy_error.value.code == "invalid_upstream_response"


@pytest.mark.parametrize("fields", [{}, {"param": None}, {"param": "files.0.page_range"}])
def test_upstream_error_normalization_preserves_http_and_param(fields: dict[str, Any]) -> None:
    """普通代理和任务代理保留有效错误，重新编码正文时清除压缩头并统一省略空参数。"""
    detail = {"code": "page_range_invalid", "message": "Invalid page range", **fields}
    response = httpx.Response(
        400,
        content=gzip.compress(json.dumps({"error": detail}).encode()),
        headers={"content-encoding": "gzip", "content-type": "application/json", "x-request-id": "request-1"},
    )
    expected = {key: value for key, value in detail.items() if value is not None}
    with pytest.raises(ApiServerError) as task_error:
        _checked_json(response)
    assert task_error.value.status_code == 400
    assert task_error.value.error.model_dump(exclude_none=True) == expected
    forwarded = passthrough_response(response)
    assert forwarded.status_code == 400
    assert json.loads(forwarded.body) == {"error": expected}
    assert forwarded.headers["x-request-id"] == "request-1"
    assert forwarded.headers["content-type"] == "application/json"
    assert "content-encoding" not in forwarded.headers


@pytest.mark.parametrize("outcome", ["valid", "malformed", "timeout", "disconnected"])
def test_download_proxy_validates_errors_and_closes_upstream(outcome: str) -> None:
    """下载代理遇到错误先验证正文并关闭上游，畸形正文不继续以文件流返回。"""
    closed: list[bool] = []
    payload = b"not json" if outcome == "malformed" else b'{"error":{"code":"file_not_found","message":"Expired","param":null}}'

    class ErrorStream(httpx.AsyncByteStream):
        """记录下载错误流是否按时关闭，防止异常分支泄漏连接。"""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """返回受控的上游错误正文，或模拟读取失败。"""
            if outcome == "timeout":
                raise httpx.ReadTimeout("Upstream timed out")
            if outcome == "disconnected":
                raise httpx.ReadError("Upstream disconnected")
            yield payload

        async def aclose(self) -> None:
            """记录上游资源关闭。"""
            closed.append(True)

    async def scenario() -> None:
        """只替换网络传输，运行真实的下载代理与结构校验。"""
        transport = httpx.MockTransport(lambda _: httpx.Response(404, stream=ErrorStream()))
        async with httpx.AsyncClient(transport=transport) as client:
            pool = SimpleNamespace(client=client)
            worker = SimpleNamespace(base_url="http://worker", worker_id="worker")
            request = Request({"type": "http", "headers": []})
            if outcome != "valid":
                with pytest.raises(RouterProxyError) as error:
                    await stream_upstream(pool, worker, "GET", "/v1/files/file-1/content", request=request)
                assert error.value.status_code == (504 if outcome == "timeout" else 502)
                assert (
                    error.value.code
                    == {
                        "malformed": "invalid_upstream_response",
                        "timeout": "upstream_timeout",
                        "disconnected": "upstream_unavailable",
                    }[outcome]
                )
            else:
                response = await stream_upstream(pool, worker, "GET", "/v1/files/file-1/content", request=request)
                assert response.status_code == 404
                assert json.loads(response.body) == {"error": {"code": "file_not_found", "message": "Expired"}}
            assert closed == [True]

    asyncio.run(scenario())


def test_authentication_errors_use_public_contract(tmp_path: Path) -> None:
    """认证中间件也使用简化协议，认证成功后仍保留资源不存在的状态码。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), api_key="secret")
    with TestClient(app) as client:
        unauthorized = client.get("/v1/tasks/missing")
        assert unauthorized.status_code == 401
        assert unauthorized.json() == {"error": {"code": "invalid_api_key", "message": "Invalid or missing API key"}}
        missing = client.get("/v1/tasks/missing", headers={"Authorization": "Bearer secret"})
        assert missing.status_code == 404
        assert set(missing.json()["error"]) == {"code", "message"}


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
def test_public_error_schema_and_request_failures(tmp_path: Path, deployment: str) -> None:
    """真实请求覆盖校验、缺失资源、上传和重复取消，OpenAPI 与输出字段保持一致。"""

    async def scenario() -> None:
        """复用真实服务生命周期，在每一层验证相同的错误字段约束。"""
        async with _deployment(tmp_path, deployment, {"worker": ("flash",)}) as (client, _app, _apps, _submissions):
            schema = (await client.get("/openapi.json")).json()["components"]["schemas"]["ErrorDetail"]
            assert set(schema["properties"]) == {"code", "message", "param"}
            assert set(schema["required"]) == {"code", "message"}
            assert schema["properties"]["code"]["minLength"] == 1
            assert schema["additionalProperties"] is False
            invalid = await client.post("/v1/tasks", json={"files": []})
            assert invalid.status_code == 400
            assert invalid.json()["error"]["param"] == "files"
            assert "type" not in invalid.json()["error"]
            for path, method, code in (
                ("/v1/tasks/missing", "GET", "task_not_found"),
                ("/v1/tasks/missing/cancel", "POST", "task_not_found"),
                ("/v1/files/missing", "GET", "file_not_found"),
                ("/v1/files/missing/content", "GET", "file_not_found"),
            ):
                missing = await client.request(method, path)
                assert missing.status_code == 404, missing.text
                assert set(missing.json()["error"]) == {"code", "message"}
                assert missing.json()["error"]["code"] == code
            created = await client.post("/v1/tasks", json={"files": [{"file_name": "a.html", "bytes": 3}]})
            assert created.status_code == 202, created.text
            task = created.json()
            upload = task["files"][0]["upload"]
            bad_upload = await client.put(upload["url"], content=b"x", headers=upload["headers"])
            assert bad_upload.status_code == 400, bad_upload.text
            assert bad_upload.json()["error"]["code"] == "upload_size_mismatch"
            assert set(bad_upload.json()["error"]) == {"code", "message"}
            path = f"/v1/tasks/{task['task_id']}/cancel"
            assert (await client.post(path)).status_code == 200
            conflict = await client.post(path)
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "task_already_terminal"
            assert set(conflict.json()["error"]) == {"code", "message"}

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
@pytest.mark.parametrize("structured", [False, True])
def test_file_errors_survive_query_and_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment: str, structured: bool
) -> None:
    """解析失败经过逐跳查询和取消仍保留错误原因，未上传文件可独立取消。"""
    detail: dict[str, str] = {"code": "parse_failed", "message": "Cannot parse input"}
    if structured:
        detail = {"code": "page_range_invalid", "message": "Invalid page range", "param": "files.0.page_range"}

    async def fail_parse(path: str, **kwargs: Any) -> ParseResult:
        """在真实执行出口注入业务异常或通用异常，保留完整上传、调度和查询流程。"""
        if structured:
            raise MineruError(detail["code"], detail["message"], param=detail["param"])
        raise RuntimeError(detail["message"])

    monkeypatch.setattr(api_server, "parse_async", fail_parse)

    async def scenario() -> None:
        """第一文件失败后取消仍在等待上传的第二文件，同时检查父任务错误序列化。"""
        async with _deployment(tmp_path, deployment, {"worker": ("flash",)}) as (client, app, _apps, _submissions):
            data = b"<html><body>input</body></html>"
            created = await client.post(
                "/v1/tasks",
                json={"files": [{"file_name": name, "bytes": len(data)} for name in ("a.html", "b.html")]},
            )
            assert created.status_code == 202, created.text
            task = created.json()
            upload = task["files"][0]["upload"]
            assert (await client.put(upload["url"], content=data, headers=upload["headers"])).status_code == 200
            path = f"/v1/tasks/{task['task_id']}"
            async with asyncio.timeout(10):
                while True:
                    response = await client.get(path)
                    assert response.status_code == 200, response.text
                    snapshot = response.json()
                    if snapshot["files"][0]["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)
            assert snapshot["files"][0]["error"] == detail
            app.state.task_store._records[task["task_id"]].error = ErrorDetail(code="task_refresh_failed", message="Task error")
            assert (await client.get(path)).json()["error"] == {"code": "task_refresh_failed", "message": "Task error"}
            canceled = await client.post(path + "/cancel")
            assert canceled.status_code == 200, canceled.text
            assert canceled.json()["files"][0]["error"] == detail
            assert canceled.json()["files"][1]["status"] == "canceled"

    asyncio.run(scenario())
