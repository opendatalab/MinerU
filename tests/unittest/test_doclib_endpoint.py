from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from mineru.doclib import client as client_module
from mineru.doclib.client import DoclibClient
from mineru.doclib.endpoint import (
    DOCLIB_UDS_BASE_URL,
    ENDPOINT_VERSION,
    EndpointInfo,
    EndpointTransport,
    read_endpoint_file,
    write_endpoint_file,
)
from mineru.errors import MineruError, ServerNotRunningError


def _status_payload(server_id: str, pid: int) -> dict[str, object]:
    return {
        "running": True,
        "pid": pid,
        "server_id": server_id,
        "socket_path": "",
        "data_dir": "",
        "sqlite_path": "",
        "log_path": "",
    }


class _FakeClient:
    def __init__(self, server_id: str, *, pid: int = 123) -> None:
        self.server_id = server_id
        self.pid = pid
        self.paths: list[str] = []

    def get(self, path: str, *, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        self.paths.append(path)
        return httpx.Response(200, json=_status_payload(self.server_id, self.pid), request=httpx.Request("GET", path))

    def close(self) -> None:
        pass


def test_endpoint_file_round_trips_transports(tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"

    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="server-123",
        transports=[
            EndpointTransport(type="uds", path=str(tmp_path / "doclib.sock")),
            EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980"),
        ],
    )

    payload = json.loads(endpoint.read_text(encoding="utf-8"))
    assert payload["version"] == ENDPOINT_VERSION
    assert payload["pid"] == 123
    assert payload["server_id"] == "server-123"
    assert read_endpoint_file(endpoint) == EndpointInfo(
        version=ENDPOINT_VERSION,
        pid=123,
        server_id="server-123",
        transports=[
            EndpointTransport(type="uds", path=str(tmp_path / "doclib.sock")),
            EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980"),
        ],
    )


def test_endpoint_pid_is_diagnostic_only(tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "version": ENDPOINT_VERSION,
                "pid": "not-a-pid",
                "server_id": "server-123",
                "transports": [{"type": "tcp", "base_url": "http://127.0.0.1:15980"}],
            }
        ),
        encoding="utf-8",
    )

    assert read_endpoint_file(endpoint) == EndpointInfo(
        version=ENDPOINT_VERSION,
        pid=None,
        server_id="server-123",
        transports=[EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980")],
    )


def test_read_endpoint_file_accepts_legacy_v1_with_pid(tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "pid": 123,
                "transports": [{"type": "tcp", "base_url": "http://127.0.0.1:15980"}],
            }
        ),
        encoding="utf-8",
    )

    assert read_endpoint_file(endpoint) == EndpointInfo(
        version=1,
        pid=123,
        server_id=None,
        transports=[EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980")],
    )


def test_doclib_client_rejects_explicit_uds_and_tcp() -> None:
    with pytest.raises(ValueError, match="socket_path and base_url"):
        DoclibClient(socket_path="/tmp/doclib.sock", base_url="http://127.0.0.1:15980")


@pytest.mark.parametrize(
    ("override", "expected_transport"),
    [
        ({"socket_path": "/tmp/explicit.sock"}, EndpointTransport(type="uds", path="/tmp/explicit.sock")),
        ({"base_url": "http://127.0.0.1:17000"}, EndpointTransport(type="tcp", base_url="http://127.0.0.1:17000")),
    ],
)
def test_doclib_client_explicit_transport_bypasses_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: dict[str, str],
    expected_transport: EndpointTransport,
) -> None:
    transports: list[EndpointTransport] = []

    def _record_transport(transport: EndpointTransport, *, timeout: int) -> None:
        transports.append(transport)

    monkeypatch.setattr(client_module, "read_endpoint_file", lambda path: pytest.fail(f"unexpected discovery read: {path}"))
    monkeypatch.setattr(client_module, "_client_for_transport", _record_transport)

    DoclibClient(endpoint_path=tmp_path / "missing.json", **override)

    assert transports == [expected_transport]


def test_doclib_client_discovers_uds_then_tcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="server-123",
        transports=[
            EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980"),
            EndpointTransport(type="uds", path=str(tmp_path / "doclib.sock")),
        ],
    )
    calls: list[dict] = []

    class _Transport:
        def __init__(self, *, uds: str) -> None:
            self.uds = uds

    class _Client:
        def __init__(self, *, transport: object | None = None, base_url: str, timeout: int, trust_env: bool) -> None:
            calls.append({"transport": transport, "base_url": base_url, "timeout": timeout, "trust_env": trust_env})

        def close(self) -> None:
            pass

    monkeypatch.setattr(client_module, "uds_available", lambda: True)
    monkeypatch.setattr(client_module.httpx, "HTTPTransport", _Transport)
    monkeypatch.setattr(client_module.httpx, "Client", _Client)

    client = DoclibClient(endpoint_path=endpoint, timeout=5)
    client.close()

    assert calls[0]["base_url"] == DOCLIB_UDS_BASE_URL
    assert calls[0]["transport"].uds == str(tmp_path / "doclib.sock")
    assert calls[1] == {"transport": None, "base_url": "http://127.0.0.1:15980", "timeout": 5, "trust_env": False}


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "{}",
        "not json",
        json.dumps({"version": 1, "pid": "not-a-pid", "transports": []}),
    ],
)
def test_doclib_client_does_not_infer_transports_without_valid_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str | None
) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    if contents is not None:
        endpoint.write_text(contents, encoding="utf-8")
    inferred_transports: list[EndpointTransport] = []

    def _reject_inferred_transport(transport: EndpointTransport, *, timeout: int) -> None:
        inferred_transports.append(transport)
        return None

    monkeypatch.setattr(client_module, "_client_for_transport", _reject_inferred_transport)

    client = DoclibClient(endpoint_path=endpoint)

    assert inferred_transports == []
    with pytest.raises(ServerNotRunningError):
        client.get_server_status()


def test_doclib_client_lazily_validates_discovered_server_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="expected-server",
        transports=[EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980")],
    )
    fake_client = _FakeClient("expected-server", pid=456)
    monkeypatch.setattr(client_module, "_client_for_transport", lambda transport, timeout: fake_client)

    client = DoclibClient(endpoint_path=endpoint)

    assert fake_client.paths == []
    status = client.get_server_status()
    assert status.server_id == "expected-server"
    assert fake_client.paths == ["/api/v1/server/status", "/api/v1/server/status"]


def test_doclib_client_validates_legacy_endpoint_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "pid": 123,
                "transports": [{"type": "tcp", "base_url": "http://127.0.0.1:15980"}],
            }
        ),
        encoding="utf-8",
    )
    fake_client = _FakeClient("", pid=123)
    monkeypatch.setattr(client_module, "_client_for_transport", lambda transport, timeout: fake_client)

    status = DoclibClient(endpoint_path=endpoint).get_server_status()

    assert status.pid == 123
    assert fake_client.paths == ["/api/v1/server/status", "/api/v1/server/status"]


def test_doclib_client_rejects_legacy_endpoint_pid_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "pid": 123,
                "transports": [{"type": "tcp", "base_url": "http://127.0.0.1:15980"}],
            }
        ),
        encoding="utf-8",
    )
    fake_client = _FakeClient("", pid=456)
    monkeypatch.setattr(client_module, "_client_for_transport", lambda transport, timeout: fake_client)

    with pytest.raises(MineruError) as exc_info:
        DoclibClient(endpoint_path=endpoint).get_server_status()

    assert exc_info.value.code == "server_instance_mismatch"
    assert fake_client.paths == ["/api/v1/server/status"]


def test_doclib_client_rejects_discovered_server_id_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="expected-server",
        transports=[EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980")],
    )
    fake_client = _FakeClient("other-server")
    monkeypatch.setattr(client_module, "_client_for_transport", lambda transport, timeout: fake_client)

    client = DoclibClient(endpoint_path=endpoint)

    with pytest.raises(MineruError) as exc_info:
        client.get_server_status()
    assert exc_info.value.code == "server_instance_mismatch"
    assert fake_client.paths == ["/api/v1/server/status"]


def test_doclib_client_tries_next_transport_after_server_id_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    first_url = "http://127.0.0.1:15980"
    second_url = "http://127.0.0.1:15981"
    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="expected-server",
        transports=[
            EndpointTransport(type="tcp", base_url=first_url),
            EndpointTransport(type="tcp", base_url=second_url),
        ],
    )
    fake_clients = {
        first_url: _FakeClient("other-server"),
        second_url: _FakeClient("expected-server"),
    }

    def _fake_client_for_transport(transport: EndpointTransport, *, timeout: int) -> _FakeClient:
        assert transport.base_url is not None
        return fake_clients[transport.base_url]

    monkeypatch.setattr(client_module, "_client_for_transport", _fake_client_for_transport)

    status = DoclibClient(endpoint_path=endpoint).get_server_status()

    assert status.server_id == "expected-server"
    assert fake_clients[first_url].paths == ["/api/v1/server/status"]
    assert fake_clients[second_url].paths == ["/api/v1/server/status", "/api/v1/server/status"]
