from __future__ import annotations

import json

import pytest

from mineru.config import PatchedConfig
from mineru.doclib import client as client_module
from mineru.doclib.client import DoclibClient
from mineru.doclib.endpoint import (
    DOCLIB_UDS_BASE_URL,
    EndpointTransport,
    config_transports,
    read_endpoint_file,
    write_endpoint_file,
)


def test_endpoint_file_round_trips_transports(tmp_path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"

    write_endpoint_file(
        endpoint,
        pid=123,
        transports=[
            EndpointTransport(type="uds", path=str(tmp_path / "doclib.sock")),
            EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980"),
        ],
    )

    payload = json.loads(endpoint.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["pid"] == 123
    assert read_endpoint_file(endpoint) == [
        EndpointTransport(type="uds", path=str(tmp_path / "doclib.sock")),
        EndpointTransport(type="tcp", base_url="http://127.0.0.1:15980"),
    ]


def test_config_transports_uses_tcp_when_uds_disabled(tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "uds": {"enabled": False, "path": str(tmp_path / "doclib.sock")},
            "tcp": {"enabled": True, "host": "127.0.0.1", "port": 17000},
        }
    )

    assert config_transports(cfg) == [EndpointTransport(type="tcp", base_url="http://127.0.0.1:17000")]


def test_doclib_client_rejects_explicit_uds_and_tcp() -> None:
    with pytest.raises(ValueError, match="socket_path and base_url"):
        DoclibClient(socket_path="/tmp/doclib.sock", base_url="http://127.0.0.1:15980")


def test_doclib_client_discovers_uds_then_tcp(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    endpoint = tmp_path / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint,
        pid=123,
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
        def __init__(self, *, transport=None, base_url: str, timeout: int, trust_env: bool) -> None:
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
