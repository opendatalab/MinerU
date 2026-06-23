"""Local doclib endpoint discovery helpers."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import Config, config, resolve_tcp_enabled, resolve_uds_enabled

TransportType = Literal["uds", "tcp"]


@dataclass(frozen=True)
class EndpointTransport:
    type: TransportType
    path: str | None = None
    base_url: str | None = None


DOCLIB_UDS_BASE_URL = "http://mineru"
ENDPOINT_VERSION = 1


def uds_available() -> bool:
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
        return True
    except Exception:
        return False


def default_endpoint_path(cfg: Config | None = None) -> str:
    cfg = config if cfg is None else cfg
    return os.path.expanduser(cfg.doclib.endpoint_path)


def write_endpoint_file(path: str | os.PathLike[str], *, pid: int, transports: list[EndpointTransport]) -> None:
    endpoint_path = Path(path).expanduser()
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ENDPOINT_VERSION,
        "pid": pid,
        "transports": [_transport_to_payload(transport) for transport in transports],
    }
    tmp_path = endpoint_path.with_name(f"{endpoint_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, endpoint_path)


def remove_endpoint_file(path: str | os.PathLike[str]) -> None:
    try:
        Path(path).expanduser().unlink()
    except OSError:
        pass


def read_endpoint_file(path: str | os.PathLike[str]) -> list[EndpointTransport]:
    endpoint_path = Path(path).expanduser()
    try:
        payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    transports = payload.get("transports")
    if not isinstance(transports, list):
        return []
    return [_parse_transport(item) for item in transports if isinstance(item, dict)]


def config_transports(cfg: Config | None = None) -> list[EndpointTransport]:
    cfg = config if cfg is None else cfg
    transports: list[EndpointTransport] = []
    if resolve_uds_enabled(cfg) and uds_available():
        transports.append(EndpointTransport(type="uds", path=os.path.expanduser(cfg.doclib.uds.path)))
    if resolve_tcp_enabled(cfg):
        transports.append(EndpointTransport(type="tcp", base_url=f"http://{cfg.doclib.tcp.host}:{cfg.doclib.tcp.port}"))
    return transports


def _transport_to_payload(transport: EndpointTransport) -> dict[str, str]:
    if transport.type == "uds":
        return {"type": "uds", "path": transport.path or ""}
    return {"type": "tcp", "base_url": transport.base_url or ""}


def _parse_transport(item: dict[str, Any]) -> EndpointTransport:
    transport_type = item.get("type")
    if transport_type == "uds":
        path = item.get("path")
        return EndpointTransport(type="uds", path=path if isinstance(path, str) else None)
    if transport_type == "tcp":
        base_url = item.get("base_url")
        return EndpointTransport(type="tcp", base_url=base_url if isinstance(base_url, str) else None)
    return EndpointTransport(type="tcp", base_url=None)


__all__ = [
    "DOCLIB_UDS_BASE_URL",
    "ENDPOINT_VERSION",
    "EndpointTransport",
    "config_transports",
    "default_endpoint_path",
    "read_endpoint_file",
    "remove_endpoint_file",
    "uds_available",
    "write_endpoint_file",
]
