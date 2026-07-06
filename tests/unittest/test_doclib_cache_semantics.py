from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from mineru.config import LogConfig, ManagedParseServerConfig
from mineru.doclib.background.compaction import Compaction
from mineru.doclib.background.device_monitor import DeviceMonitor
from mineru.doclib.background.ingest import IngestWorkerPool
from mineru.doclib.background.parse_server_health import (
    ParseServerHealth,
    ParseServerHealthCheck,
    api_server_args_for_tier,
    select_available_managed_port,
    start_managed_parse_server,
    stop_managed_parse_server,
)
from mineru.doclib.background.watch import WatchLoop
from mineru.doclib.config_defaults import CONFIG_DEFAULTS
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.file_io import FileStat, get_file_stat
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.locators import ContentCursor
from mineru.doclib.server import DoclibServer, _ReadPlan
from mineru.doclib.services import parse_svc as parse_svc_module
from mineru.doclib.services.cleanup_svc import CleanupService
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_svc import (
    FileRefreshResult,
    ParseFailure,
    ParseService,
    _local_parse_server_url,
    _resolve_default_tier,
    expand_page_range,
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    parse_batch_json_path,
    parse_image_sidecar_dir,
)
from mineru.doclib.services.scan_svc import ScanService
from mineru.doclib.services.search_svc import SearchService
from mineru.doclib.types import DocContentExportRequest, FileInfo, InvalidateRequest, ParseResponse, WatchRequest
from mineru.errors import InvalidRequestError, MineruError, NotFoundError
from mineru.parser import backend_for_tier, resolve_tier_and_backend
from mineru.parser.api_client import _V1APIError
from mineru.parser.base import ParseResult
from mineru.schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span, Tier
from mineru.utils.image_payload import ImagePayloadCache


class _Cursor:
    def __init__(self, rowcount: int, lastrowid: int | None = None) -> None:
        self.rowcount = rowcount
        self.lastrowid = lastrowid


class _FakeDB:
    def __init__(
        self,
        *,
        parses: list[dict[str, Any]],
        file_row: dict[str, Any] | None,
        doc_row: dict[str, Any] | None = None,
    ) -> None:
        self.parses = parses
        self.file_row = file_row
        self.doc_row = doc_row
        self.updated_priorities: list[int] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _Cursor:
        if sql.startswith("UPDATE parses SET status=?, updated_at=?"):
            status, _, sha256, done_status, *rest = params
            tier = rest[0] if rest else None
            rowcount = 0
            for row in self.parses:
                if row["sha256"] != sha256 or row["status"] != done_status:
                    continue
                if tier and row["tier"] != tier:
                    continue
                row["status"] = status
                rowcount += 1
            return _Cursor(rowcount)
        if sql.startswith("UPDATE parses SET priority=1"):
            parse_id = params[1]
            self.updated_priorities.append(parse_id)
            for row in self.parses:
                if row["id"] == parse_id:
                    row["priority"] = 1
                    return _Cursor(1)
        if sql.startswith("UPDATE parses SET status=?, error_code=?, error_msg=?"):
            status, error_code, error_msg, updated_at, parse_id = params
            for row in self.parses:
                if row["id"] == parse_id:
                    row["status"] = status
                    row["error_code"] = error_code
                    row["error_msg"] = error_msg
                    row["locked_at"] = None
                    row["updated_at"] = updated_at
                    return _Cursor(1)
        if sql.startswith("UPDATE parses SET status=?, done_at=?"):
            status, done_at, via, updated_at, parse_id = params
            for row in self.parses:
                if row["id"] == parse_id:
                    row["status"] = status
                    row["done_at"] = done_at
                    row["via"] = via
                    row["locked_at"] = None
                    row["updated_at"] = updated_at
                    return _Cursor(1)
        return _Cursor(0)

    async def execute_insert(self, sql: str, params: tuple[Any, ...]) -> int:
        if sql.startswith("INSERT INTO parses"):
            parse_id = max((row.get("id", 0) for row in self.parses), default=0) + 1
            sha256, tier, page_range, status, privacy, created_at, updated_at = params
            self.parses.append(
                {
                    "id": parse_id,
                    "sha256": sha256,
                    "tier": tier,
                    "page_range": page_range,
                    "status": status,
                    "privacy": privacy,
                    "priority": 1,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "done_at": None,
                }
            )
            return parse_id
        return 0

    async def fetchone(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        if sql.startswith("SELECT * FROM files WHERE path="):
            path = params[0]
            if self.file_row and self.file_row["path"] == path and self.file_row["status"] == "active":
                return self.file_row
        if sql.startswith("SELECT page_count FROM docs WHERE sha256=") or sql.startswith("SELECT * FROM docs WHERE sha256="):
            sha256 = params[0]
            if self.doc_row and self.doc_row["sha256"] == sha256:
                return self.doc_row
        if sql.startswith("SELECT * FROM files WHERE sha256="):
            sha256 = params[0]
            if self.file_row and self.file_row["sha256"] == sha256 and self.file_row["status"] == "active":
                return self.file_row
        if sql.startswith("SELECT status FROM parses WHERE id=") or sql.startswith("SELECT * FROM parses WHERE id="):
            parse_id = params[0]
            for row in self.parses:
                if row["id"] == parse_id:
                    return row
        return None

    async def fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if sql.startswith("SELECT * FROM parses WHERE id IN"):
            ids = set(params)
            return [row for row in self.parses if row["id"] in ids]
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND tier=? AND status=?"):
            sha256, tier, status = params
            rows = [row for row in self.parses if row["sha256"] == sha256 and row["tier"] == tier and row["status"] == status]
            return sorted(rows, key=lambda row: row["done_at"] or 0, reverse=True)
        if sql.startswith("SELECT page_range, done_at FROM parses WHERE sha256=? AND tier=? AND status=?"):
            sha256, tier, status = params
            rows = [
                {"page_range": row["page_range"], "done_at": row["done_at"]}
                for row in self.parses
                if row["sha256"] == sha256 and row["tier"] == tier and row["status"] == status
            ]
            return sorted(rows, key=lambda row: row["done_at"] or 0, reverse=True)
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN (?, ?)"):
            sha256, tier, *statuses = params
            return [row for row in self.parses if row["sha256"] == sha256 and row["tier"] == tier and row["status"] in statuses]
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND status=?"):
            sha256, status = params
            rows = [row for row in self.parses if row["sha256"] == sha256 and row["status"] == status]
            return sorted(rows, key=lambda row: row["done_at"], reverse=True)
        return []


class _FakeFTS:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.replaced: list[dict[str, Any]] = []

    async def delete(self, sha256: str) -> None:
        self.deleted.append(sha256)

    async def replace(self, **kwargs: Any) -> None:
        self.replaced.append(kwargs)


class _FilenameOnlyFTS:
    async def upsert_filename(self, file_id: int, stem: str, ext: str) -> None:
        pass


class _OrderRecordingDB:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.events: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.db, name)

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        if sql.startswith("UPDATE files SET sha256=?"):
            self.events.append("update_file_sha")
        elif sql.startswith("INSERT INTO parses"):
            self.events.append("insert_parse")
        return await self.db.execute(sql, params)


def _write_batch(data_dir: Path, sha256: str, tier: Tier, page_range: str, done_at: int, json_pages: list[dict]) -> None:
    path = Path(parse_batch_json_path(str(data_dir), sha256, tier, page_range, done_at))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pages": json_pages}), encoding="utf-8")


def _image_page(image_path: str) -> PageInfo:
    image_span = Span(
        type=ContentType.IMAGE,
        bbox=(1, 1, 20, 20),
        image_path=image_path,
    )
    body = Block(
        index=0,
        type=BlockType.IMAGE_BODY,
        bbox=(1, 1, 20, 20),
        lines=[Line(bbox=(1, 1, 20, 20), spans=[image_span])],
    )
    image_block = Block(index=0, type=BlockType.IMAGE, bbox=(1, 1, 20, 20), blocks=[body])
    return PageInfo(page_idx=0, page_size=(100, 100), para_blocks=[image_block], _backend="vlm")


def _image_result(image_path: str, image_bytes: bytes = b"image-bytes") -> ParseResult:
    """构造带顶层图片缓存的 ParseResult，避免在 span 中保留 base64。"""
    image_cache = ImagePayloadCache()
    image_cache.register_bytes(image_bytes, "jpeg", image_path=image_path)
    return ParseResult(pages=[_image_page(image_path)], _image_cache=image_cache)


def _text_page(text: str) -> PageInfo:
    span = Span(type=ContentType.TEXT, bbox=(1, 1, 20, 10), content=text)
    line = Line(bbox=(1, 1, 20, 10), spans=[span])
    block = Block(index=0, type=BlockType.TEXT, bbox=(1, 1, 20, 10), lines=[line])
    return PageInfo(page_idx=0, page_size=(100, 100), para_blocks=[block], _backend="vlm")


def test_load_pages_from_done_batches_keeps_newest_page_idx(tmp_path: Path) -> None:
    sha256 = "a" * 64
    tier = "standard"
    older_page = {"page_idx": 1, "page_size": [100, 100], "para_blocks": []}
    older_duplicate = {"page_idx": 2, "page_size": [100, 100], "para_blocks": []}
    newer_duplicate = {"page_idx": 2, "page_size": [200, 200], "para_blocks": []}

    _write_batch(tmp_path, sha256, tier, "1~2", 1000, [older_page, older_duplicate])
    _write_batch(tmp_path, sha256, tier, "2", 2000, [newer_duplicate])

    done_rows = [
        {"page_range": "2", "done_at": 2000},
        {"page_range": "1~2", "done_at": 1000},
    ]

    pages = load_pages_from_done_batches(str(tmp_path), sha256, tier, done_rows)

    assert [page.page_idx for page in pages] == [1, 2]
    assert pages[0].page_size == (100, 100)
    assert pages[1].page_size == (200, 200)


def test_parser_tier_backend_mapping_is_parser_layer_only() -> None:
    assert backend_for_tier("flash") == "flash"
    assert backend_for_tier("standard") == "pipeline"
    assert backend_for_tier("pro") == "hybrid-engine"
    assert resolve_tier_and_backend(tier=None) == ("pro", "hybrid-engine")
    assert resolve_tier_and_backend(tier="pro", backend="vlm-auto-engine") == ("pro", "vlm-engine")


def test_managed_api_server_args_use_tier_and_selected_port_for_process_start() -> None:
    assert api_server_args_for_tier("standard", host="127.0.0.1", port=16580) == [
        "--tier",
        "standard",
        "--host",
        "127.0.0.1",
        "--port",
        "16580",
    ]
    assert api_server_args_for_tier("pro", host="127.0.0.2", port=16581) == [
        "--tier",
        "pro",
        "--host",
        "127.0.0.2",
        "--port",
        "16581",
    ]


def test_managed_parse_server_port_selection_tries_configured_range(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_calls: list[tuple[str, int]] = []

    class _Socket:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.bound_port: int | None = None

        def setsockopt(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            bind_calls.append(address)
            _host, port = address
            if port in (16580, 16581):
                raise OSError(98, "in use")
            self.bound_port = port

        def getsockname(self) -> tuple[str, int]:
            assert self.bound_port is not None
            return ("127.0.0.1", self.bound_port)

        def close(self) -> None:
            return None

    monkeypatch.setattr("mineru.doclib.background.parse_server_health.socket.socket", _Socket)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.errno.EADDRINUSE", 98)

    port = select_available_managed_port("127.0.0.1", 16580, strict_port=False, port_probe_count=3)

    assert port == 16582
    assert bind_calls == [("127.0.0.1", 16580), ("127.0.0.1", 16581), ("127.0.0.1", 16582)]


def test_default_tier_error_mentions_remote_when_remote_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    health = ParseServerHealth(
        local_healthy=False,
        local_supported_tiers=["flash"],
        remote_healthy=True,
        remote_supported_tiers=["standard", "pro"],
    )
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.get_health", lambda: health)

    with pytest.raises(ParseFailure) as exc_info:
        _resolve_default_tier(remote=False)

    assert exc_info.value.code == "quality_tier_unavailable"
    assert "--remote" in exc_info.value.message
    assert "local parse-server" in exc_info.value.message
    assert "--tier flash" in exc_info.value.message


def test_default_tier_error_omits_remote_when_remote_is_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    health = ParseServerHealth(
        local_healthy=False,
        local_supported_tiers=["flash"],
        remote_healthy=False,
        remote_supported_tiers=[],
    )
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.get_health", lambda: health)

    with pytest.raises(ParseFailure) as exc_info:
        _resolve_default_tier(remote=False)

    assert exc_info.value.code == "quality_tier_unavailable"
    assert "--remote" not in exc_info.value.message
    assert "local parse-server" in exc_info.value.message
    assert "--tier flash" in exc_info.value.message


def test_managed_local_parse_server_url_uses_health_managed_url() -> None:
    health = ParseServerHealth(managed_url="http://127.0.0.1:16582")

    assert _local_parse_server_url("managed", health) == "http://127.0.0.1:16582"


def test_parse_server_health_probe_disables_env_proxy_for_local_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class _Response:
        status_code = 200

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"data": [{"id": "standard"}]}

    class _AsyncClient:
        def __init__(self, *, timeout: int, trust_env: bool) -> None:
            calls.append(trust_env)

        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> _Response:
            return _Response()

    monkeypatch.setattr("mineru.doclib.background.parse_server_health.httpx.AsyncClient", _AsyncClient)
    checker = ParseServerHealthCheck(None, interval_sec=1, probe_timeout_sec=2, startup_grace_sec=3, stop_timeout_sec=4)

    assert asyncio.run(checker._probe("http://127.0.0.1:16580")) == (True, ["standard"])
    assert asyncio.run(checker._probe("https://staging.mineru.org.cn/api")) == (True, ["standard"])
    assert calls == [False, True]


def test_start_managed_parse_server_selects_port_and_writes_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parse_stdout_log_path = tmp_path / "doclib.parse-server.stdout.log"
    parse_stderr_log_path = tmp_path / "doclib.parse-server.stderr.log"
    popen_calls: list[dict[str, Any]] = []

    class _Proc:
        pid = 12345

    def _popen(*args: Any, **kwargs: Any) -> _Proc:
        popen_calls.append({"args": args, "kwargs": kwargs})
        cmd = args[0]
        assert "--tier" in cmd
        assert cmd[cmd.index("--tier") + 1] == "standard"
        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "127.0.0.2"
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "16582"
        assert kwargs["stdout"] is not subprocess.DEVNULL
        assert kwargs["stderr"] is not subprocess.DEVNULL
        assert kwargs["stdout"] is not kwargs["stderr"]
        assert kwargs["stdin"] is subprocess.PIPE
        assert kwargs["env"]["MINERU_MANAGED_PARSE_SERVER"] == "1"
        kwargs["stdout"].write("parse helper stdout\n")
        kwargs["stdout"].flush()
        kwargs["stderr"].write("parse helper stderr\n")
        kwargs["stderr"].flush()
        return _Proc()

    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582
    )
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    proc, url = start_managed_parse_server(
        tier="standard",
        managed_cfg=ManagedParseServerConfig(host="127.0.0.2", port=16580, port_probe_count=3),
        log_cfg=LogConfig(
            parse_server_stdout_path=str(parse_stdout_log_path),
            parse_server_stderr_path=str(parse_stderr_log_path),
        ),
        marker="test",
    )

    assert proc.pid == 12345
    assert url == "http://127.0.0.2:16582"
    assert popen_calls
    assert parse_stdout_log_path.read_text(encoding="utf-8").endswith("parse helper stdout\n")
    assert parse_stderr_log_path.read_text(encoding="utf-8").endswith("parse helper stderr\n")


def test_stop_managed_parse_server_closes_stdin_then_terminates_then_kills() -> None:
    events: list[str] = []

    class _Stdin:
        closed = False

        def close(self) -> None:
            events.append("stdin.close")
            self.closed = True

    class _Proc:
        pid = 12345
        stdin = _Stdin()
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            self.waits += 1
            events.append(f"wait:{timeout}")
            if self.waits < 3:
                raise subprocess.TimeoutExpired(cmd=["parse-server"], timeout=timeout)

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    stop_managed_parse_server(_Proc(), timeout_sec=4, reason="test")

    assert events == ["stdin.close", "wait:4", "terminate", "wait:4", "kill", "wait:4"]


def test_managed_parse_server_restart_writes_stdout_and_stderr_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parse_stdout_log_path = tmp_path / "doclib.parse-server.stdout.log"
    parse_stderr_log_path = tmp_path / "doclib.parse-server.stderr.log"

    class _ConfigSvc:
        async def get(self, key: str) -> str:
            assert key == "parse_server.local.managed_tier"
            return "standard"

    class _Proc:
        pid = 12345

    def _popen(*args: Any, **kwargs: Any) -> _Proc:
        cmd = args[0]
        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "127.0.0.2"
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "16582"
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] is not None
        assert kwargs["stdout"] is not subprocess.DEVNULL
        assert kwargs["stderr"] is not subprocess.DEVNULL
        assert kwargs["stdout"] is not kwargs["stderr"]
        kwargs["stdout"].write("parse restart stdout\n")
        kwargs["stdout"].flush()
        kwargs["stderr"].write("parse restart stderr\n")
        kwargs["stderr"].flush()
        return _Proc()

    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.get_parse_server_stdout_log_path", lambda: str(parse_stdout_log_path)
    )
    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.get_parse_server_stderr_log_path", lambda: str(parse_stderr_log_path)
    )
    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582
    )
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    checker = ParseServerHealthCheck(
        _ConfigSvc(),
        interval_sec=1,
        probe_timeout_sec=2,
        startup_grace_sec=3,
        stop_timeout_sec=4,
        managed_parse_server=ManagedParseServerConfig(host="127.0.0.2", port=16580, port_probe_count=3),
    )
    health = ParseServerHealth()

    asyncio.run(checker._try_restart_managed(health))

    assert health.managed_url == "http://127.0.0.2:16582"
    assert parse_stdout_log_path.read_text(encoding="utf-8").endswith("parse restart stdout\n")
    assert parse_stderr_log_path.read_text(encoding="utf-8").endswith("parse restart stderr\n")


def test_managed_parse_server_restart_stops_recorded_proc_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    old_proc = object()

    class _ConfigSvc:
        async def get(self, key: str) -> str:
            assert key == "parse_server.local.managed_tier"
            return "standard"

    class _Proc:
        pid = 67890

    def _stop(proc: object, *, timeout_sec: int, reason: str) -> None:
        assert proc is old_proc
        assert timeout_sec == 4
        assert reason == "restart"
        events.append("stop")

    def _popen(*args: Any, **kwargs: Any) -> _Proc:
        assert events == ["stop"]
        events.append("start")
        return _Proc()

    monkeypatch.setattr("mineru.doclib.background.parse_server_health.stop_managed_parse_server", _stop)
    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582
    )
    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.open_managed_parse_server_logs",
        lambda *args, **kwargs: nullcontext((None, None)),
    )
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    checker = ParseServerHealthCheck(
        _ConfigSvc(),
        interval_sec=1,
        probe_timeout_sec=2,
        startup_grace_sec=3,
        stop_timeout_sec=4,
        managed_parse_server=ManagedParseServerConfig(host="127.0.0.2", port=16580, port_probe_count=3),
    )
    health = ParseServerHealth(managed_proc=old_proc)

    asyncio.run(checker._try_restart_managed(health))

    assert events == ["stop", "start"]
    assert health.managed_proc.pid == 67890


def test_managed_parse_server_tier_change_detection_triggers_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    class _ConfigSvc:
        async def get(self, key: str) -> str:
            raise AssertionError(f"unexpected config read: {key}")

    checker = ParseServerHealthCheck(
        _ConfigSvc(),
        interval_sec=1,
        probe_timeout_sec=2,
        startup_grace_sec=3,
        stop_timeout_sec=4,
        managed_parse_server=ManagedParseServerConfig(host="127.0.0.2", port=16580, port_probe_count=3),
    )

    async def _restart(
        health_arg: ParseServerHealth,
        *,
        reason: str,
        marker: str | None,
        count_restart: bool,
    ) -> None:
        assert health_arg is health
        calls.append((reason, marker or "", count_restart))

    monkeypatch.setattr(checker, "_try_restart_managed", _restart)
    health = ParseServerHealth(running_managed_tier="standard")

    restarted = asyncio.run(checker._try_restart_managed_for_tier_change(health, "pro"))
    unchanged = asyncio.run(checker._try_restart_managed_for_tier_change(health, "standard"))

    assert restarted is True
    assert unchanged is False
    assert calls == [("tier-change", "tier change standard->pro", False)]


def test_managed_parse_server_tier_change_restart_uses_desired_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    old_proc = object()

    class _ConfigSvc:
        async def get(self, key: str) -> str:
            assert key == "parse_server.local.managed_tier"
            return "pro"

    class _Proc:
        pid = 24680

    def _stop(proc: object, *, timeout_sec: int, reason: str) -> None:
        assert proc is old_proc
        assert timeout_sec == 4
        assert reason == "tier-change"
        events.append("stop")

    def _start(*, tier: Tier, managed_cfg: ManagedParseServerConfig, log_cfg: LogConfig | None, marker: str) -> tuple[_Proc, str]:
        assert tier == "pro"
        assert managed_cfg.host == "127.0.0.2"
        assert log_cfg is None
        assert marker == "tier change standard->pro"
        events.append("start")
        return _Proc(), "http://127.0.0.2:16582"

    monkeypatch.setattr("mineru.doclib.background.parse_server_health.stop_managed_parse_server", _stop)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.start_managed_parse_server", _start)

    checker = ParseServerHealthCheck(
        _ConfigSvc(),
        interval_sec=1,
        probe_timeout_sec=2,
        startup_grace_sec=3,
        stop_timeout_sec=4,
        managed_parse_server=ManagedParseServerConfig(host="127.0.0.2", port=16580, port_probe_count=3),
    )
    health = ParseServerHealth(managed_proc=old_proc, running_managed_tier="standard", restart_count=2)

    asyncio.run(
        checker._try_restart_managed(
            health,
            reason="tier-change",
            marker="tier change standard->pro",
            count_restart=False,
        )
    )

    assert events == ["stop", "start"]
    assert health.managed_proc.pid == 24680
    assert health.running_managed_tier == "pro"
    assert health.restart_count == 2


def test_get_file_stat_returns_typed_file_stat(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("content", encoding="utf-8")

    stat = asyncio.run(get_file_stat(str(source)))

    assert isinstance(stat, FileStat)
    assert stat.size_bytes == source.stat().st_size
    assert stat.mtime_ms == int(source.stat().st_mtime * 1000)


def test_config_defaults_are_code_backed_and_unset_removes_override(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        seeded_rows = await db.fetchall("SELECT key, value FROM config ORDER BY key")
        assert seeded_rows == []

        assert await service.get("parse_server.local.mode") == CONFIG_DEFAULTS["parse_server.local.mode"]
        config, sources = await service.get_all_with_sources()
        assert config["parse_server.local.mode"] == CONFIG_DEFAULTS["parse_server.local.mode"]
        assert sources["parse_server.local.mode"] == "default"

        await service.set("parse_server.local.mode", "managed")
        assert await service.get("parse_server.local.mode") == "managed"
        config, sources = await service.get_all_with_sources()
        assert config["parse_server.local.mode"] == "managed"
        assert sources["parse_server.local.mode"] == "override"

        await service.unset("parse_server.local.mode")
        assert await service.get("parse_server.local.mode") == CONFIG_DEFAULTS["parse_server.local.mode"]
        config, sources = await service.get_all_with_sources()
        assert config["parse_server.local.mode"] == CONFIG_DEFAULTS["parse_server.local.mode"]
        assert sources["parse_server.local.mode"] == "default"
        assert await db.fetchall("SELECT key, value FROM config ORDER BY key") == []

    asyncio.run(_run())


def test_config_service_rejects_invalid_known_config_values(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        invalid_values = [
            ("parse_server.local.mode", "totally-invalid-mode"),
            ("parse_server.local.managed_tier", "ultra"),
            ("parse_server.remote.url", "not-a-url"),
            ("parse_server.remote.url", "ftp://example.com/api"),
            ("parse_server.local.self_hosted_url", "not-a-url"),
        ]
        for key, value in invalid_values:
            with pytest.raises(InvalidRequestError) as exc_info:
                await service.set(key, value)
            assert exc_info.value.code == "invalid_config_value"
            assert exc_info.value.param == "value"

        assert await service.get("parse_server.local.mode") == CONFIG_DEFAULTS["parse_server.local.mode"]
        assert await service.get("parse_server.local.managed_tier") == CONFIG_DEFAULTS["parse_server.local.managed_tier"]
        assert await service.get("parse_server.remote.url") == CONFIG_DEFAULTS["parse_server.remote.url"]
        assert await service.get("parse_server.local.self_hosted_url") == CONFIG_DEFAULTS["parse_server.local.self_hosted_url"]
        assert await db.fetchall("SELECT key, value FROM config ORDER BY key") == []

    asyncio.run(_run())


def test_config_service_accepts_valid_url_config_values(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        await service.set("parse_server.remote.url", "https://example.com/api")
        await service.set("parse_server.local.self_hosted_url", "http://127.0.0.1:16580")
        await service.set("parse_server.local.self_hosted_url", "")

        assert await service.get("parse_server.remote.url") == "https://example.com/api"
        assert await service.get("parse_server.local.self_hosted_url") == ""

    asyncio.run(_run())


def test_remote_api_target_prefers_config_api_key_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            values = {
                "parse_server.remote.url": "https://mineru.net/api",
                "parse_server.remote.api_key": "config-key",
            }
            return values[key]

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_ConfigService(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )

        base_url, api_key, via = await service._resolve_api_target("remote", "pro")

        assert base_url == "https://mineru.net/api"
        assert api_key == "config-key"
        assert via == "remote"

    monkeypatch.setenv("MINERU_API_KEY", "env-key")
    monkeypatch.setattr(
        "mineru.doclib.background.parse_server_health.get_health",
        lambda: SimpleNamespace(remote_healthy=True),
    )

    asyncio.run(_run())


def test_data_dir_is_not_runtime_kv_config(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        config, sources = await service.get_all_with_sources()

        assert "data_dir" not in CONFIG_DEFAULTS
        assert "data_dir" not in config
        assert "data_dir" not in sources

        with pytest.raises(InvalidRequestError, match="Unknown config key: data_dir"):
            await service.get("data_dir")
        with pytest.raises(InvalidRequestError, match="Unknown config key: data_dir"):
            await service.set("data_dir", "/tmp/mineru")
        with pytest.raises(InvalidRequestError, match="Unknown config key: data_dir"):
            await service.unset("data_dir")

    asyncio.run(_run())


def test_remote_parse_server_default_url_is_declared_once() -> None:
    default_url = CONFIG_DEFAULTS["parse_server.remote.url"]
    doclib_dir = Path(__file__).parents[2] / "mineru" / "doclib"
    occurrences = [path for path in doclib_dir.rglob("*.py") if default_url in path.read_text(encoding="utf-8")]

    assert occurrences == [doclib_dir / "config_defaults.py"]


def test_compaction_uses_configured_data_dir(tmp_path: Path) -> None:
    sha256 = "b" * 64
    tier = "standard"
    older_page = {"page_idx": 0, "content": "old"}
    older_duplicate = {"page_idx": 1, "content": "old"}
    newer_duplicate = {"page_idx": 1, "content": "new"}

    _write_batch(tmp_path, sha256, tier, "1~2", 1000, [older_page, older_duplicate])
    _write_batch(tmp_path, sha256, tier, "2", 2000, [newer_duplicate])

    compaction = Compaction(db=None, interval_sec=600, data_dir=str(tmp_path))
    done_rows = [
        {"page_range": "2", "done_at": 2000},
        {"page_range": "1~2", "done_at": 1000},
    ]

    asyncio.run(compaction._compact_json(sha256, tier, ["1~2"], done_rows, 2000))

    compacted_path = Path(parse_batch_json_path(str(tmp_path), sha256, tier, "1~2", 2000))
    compacted = json.loads(compacted_path.read_text(encoding="utf-8"))

    assert compacted["schema_version"] == MIDDLE_JSON_SCHEMA_VERSION
    assert compacted["pages"] == [older_page, newer_duplicate]
    assert sorted(path.name for path in compacted_path.parent.glob("*.json")) == ["1~2_2000.json"]


def test_invalidate_deletes_fts_when_no_done_batches_remain(tmp_path: Path) -> None:
    sha256 = "c" * 64
    parses = [{"sha256": sha256, "tier": "standard", "page_range": "1", "status": "done", "done_at": 1000}]
    db = _FakeDB(parses=parses, file_row={"sha256": sha256, "status": "active", "filename": "doc.pdf"})
    fts = _FakeFTS()
    service = ParseService(db=db, fts=fts, config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    count = asyncio.run(service.invalidate(sha256, "standard"))

    assert count == 1
    assert parses[0]["status"] == "superseded"
    assert fts.deleted == [sha256]
    assert fts.replaced == []


def test_invalidate_rebuilds_fts_from_highest_remaining_done_tier(tmp_path: Path) -> None:
    sha256 = "d" * 64
    _write_batch(tmp_path, sha256, "flash", "1", 1000, [{"page_idx": 1, "page_size": [100, 100], "para_blocks": []}])
    _write_batch(tmp_path, sha256, "standard", "1", 2000, [{"page_idx": 1, "page_size": [200, 200], "para_blocks": []}])
    parses = [
        {"sha256": sha256, "tier": "flash", "page_range": "1", "status": "done", "done_at": 1000},
        {"sha256": sha256, "tier": "standard", "page_range": "1", "status": "done", "done_at": 2000},
    ]
    db = _FakeDB(parses=parses, file_row={"sha256": sha256, "status": "active", "filename": "doc.pdf"})
    fts = _FakeFTS()
    service = ParseService(db=db, fts=fts, config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    count = asyncio.run(service.invalidate(sha256, "standard"))

    assert count == 1
    assert fts.deleted == []
    assert len(fts.replaced) == 1
    assert fts.replaced[0]["sha256"] == sha256
    assert fts.replaced[0]["tier"] == "flash"


def test_ingest_binds_file_sha_before_creating_parse_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, Any]]:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        recording_db = _OrderRecordingDB(db)
        service = ParseService(
            db=recording_db,
            fts=_FilenameOnlyFTS(),
            config_svc=_NoRulesConfig(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4\n")

        async def _metadata(path: str) -> dict[str, Any]:
            return {
                "page_count": 1,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _metadata)

        row = await service.ingest_file(str(source))
        dangling_parses = await db.fetchall(
            "SELECT p.id FROM parses p LEFT JOIN files f ON f.sha256=p.sha256 AND f.status='active' WHERE f.id IS NULL",
        )

        assert row is not None
        assert recording_db.events.index("update_file_sha") < recording_db.events.index("insert_parse")
        assert dangling_parses == []

    asyncio.run(_run())


def test_request_parse_explicit_image_ingests_and_queues_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, Any]]:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "scan.png"
        source.write_bytes(b"png-bytes")

        async def _metadata(path: str) -> dict[str, Any]:
            return {
                "page_count": 1,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 1,
            }

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _metadata)

        result = await service.request_parse(str(source), tier="flash")
        file_row = await db.fetchone("SELECT path, ext, sha256, status FROM files WHERE path=?", (str(source),))
        doc_row = await db.fetchone(
            "SELECT short_id, file_type, page_count, is_image_based FROM docs WHERE sha256=?",
            (result.sha256,),
        )
        parse_rows = await db.fetchall("SELECT tier, page_range, status FROM parses WHERE sha256=?", (result.sha256,))

        assert result.status == "pending"
        assert result.tier == "flash"
        assert doc_row is not None
        assert result.short_id == doc_row["short_id"]
        assert file_row is not None
        assert file_row["ext"] == "png"
        assert file_row["sha256"] == result.sha256
        assert file_row["status"] == "active"
        assert doc_row == {"short_id": result.short_id, "file_type": "image", "page_count": 1, "is_image_based": 1}
        assert parse_rows == [{"tier": "flash", "page_range": "1", "status": "pending"}]

    asyncio.run(_run())


@pytest.mark.parametrize("ext", ["txt", "html", "docx", "pptx", "xlsx"])
@pytest.mark.parametrize("tier", ["standard", "pro"])
def test_request_parse_rejects_quality_tiers_for_non_pdf_image_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext: str,
    tier: Tier,
) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, Any]]:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / f"sample.{ext}"
        source.write_text("content", encoding="utf-8")

        async def _metadata(path: str) -> dict[str, Any]:
            return {
                "page_count": 1,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _metadata)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier=tier)

        assert exc_info.value.code == "tier_unsupported_for_file_type"
        assert exc_info.value.param == "tier"
        assert tier in exc_info.value.message
        assert ext in exc_info.value.message

    asyncio.run(_run())


@pytest.mark.parametrize("ext", ["txt", "html", "docx", "pptx", "xlsx"])
def test_request_parse_rejects_remote_for_non_pdf_image_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext: str,
) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, Any]]:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / f"sample.{ext}"
        source.write_text("content", encoding="utf-8")

        async def _metadata(path: str) -> dict[str, Any]:
            return {
                "page_count": 1,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _metadata)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), remote=True)

        assert exc_info.value.code == "remote_unsupported_for_file_type"
        assert exc_info.value.param == "remote"
        assert ext in exc_info.value.message

    asyncio.run(_run())


@pytest.mark.parametrize("ext", ["pdf", "png"])
def test_request_parse_rejects_flash_tier_for_remote_quality_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext: str,
) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, Any]]:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / f"sample.{ext}"
        source.write_bytes(b"%PDF-1.4\n" if ext == "pdf" else b"png-bytes")

        async def _metadata(path: str) -> dict[str, Any]:
            return {
                "page_count": 1,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": int(ext != "pdf"),
            }

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _metadata)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash", remote=True)

        assert exc_info.value.code == "tier_unsupported_for_remote"
        assert exc_info.value.param == "tier"
        assert "flash" in exc_info.value.message
        assert "remote" in exc_info.value.message

    asyncio.run(_run())


def test_request_parse_rejects_unsupported_file_type_before_parse_response(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=None,
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.unsupported"
        source.write_text("content", encoding="utf-8")

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash")

        assert exc_info.value.code == "file_type_unsupported"
        assert exc_info.value.param == "path"

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("ext", "target_ext"),
    [
        ("doc", "docx"),
        ("ppt", "pptx"),
        ("xls", "xlsx"),
    ],
)
def test_request_parse_rejects_legacy_office_with_conversion_hint(tmp_path: Path, ext: str, target_ext: str) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=None,
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / f"sample.{ext}"
        source.write_bytes(b"office")

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash")

        assert exc_info.value.code == "file_type_unsupported"
        assert exc_info.value.param == "path"
        assert f".{ext} files are not supported" in exc_info.value.message
        assert f".{target_ext}" in exc_info.value.message

    asyncio.run(_run())


def test_request_parse_raises_ingest_failed_when_file_row_has_no_sha(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=None,
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        stat = source.stat()
        now = 1000
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
            (str(source), source.name, "pdf", stat.st_size, int(stat.st_mtime * 1000), "active", now, now),
        )

        async def _refresh_file(*args: Any, **kwargs: Any) -> FileRefreshResult:
            return FileRefreshResult(
                file=FileInfo(
                    path=str(source),
                    filename=source.name,
                    ext="pdf",
                    size_bytes=stat.st_size,
                    mtime_ms=int(stat.st_mtime * 1000),
                    sha256=None,
                    status="active",
                    first_seen_at=now,
                    updated_at=now,
                ),
                status="known",
            )

        service.refresh_file = _refresh_file  # type: ignore[method-assign]

        with pytest.raises(MineruError) as exc_info:
            await service.request_parse(str(source), tier="flash")

        assert exc_info.value.code == "ingest_failed"
        assert exc_info.value.param == "path"

    asyncio.run(_run())


def test_force_request_reuses_active_and_creates_only_uncovered_parse(tmp_path: Path) -> None:
    sha256 = "e" * 64
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    stat = source.stat()
    path = str(source)
    parses = [
        {
            "id": 10,
            "sha256": sha256,
            "tier": "standard",
            "page_range": "1~5",
            "status": "done",
            "priority": 0,
            "done_at": 1000,
            "created_at": 900,
        },
        {
            "id": 11,
            "sha256": sha256,
            "tier": "standard",
            "page_range": "6~8",
            "status": "pending",
            "priority": 0,
            "done_at": None,
            "created_at": 1100,
        },
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": path,
            "filename": "doc.pdf",
            "sha256": sha256,
            "status": "active",
            "ext": "pdf",
            "mtime_ms": int(stat.st_mtime * 1000),
            "size_bytes": stat.st_size,
            "first_seen_at": 100,
            "updated_at": 100,
        },
        doc_row={"sha256": sha256, "short_id": "eeeeeee", "page_count": 10},
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    result = asyncio.run(service.request_parse(path, tier="standard", page_range="1~10", force=True))

    assert isinstance(result, ParseResponse)
    assert result.wait_parse_ids == [11, 12]
    assert result.reused_parse_ids == [11]
    assert result.created_parse_ids == [12]
    assert result.page_range == "1~10"
    assert result.short_id == "eeeeeee"
    assert result.status == "pending"
    assert result.cache_hit is False
    assert db.updated_priorities == [11]
    assert parses[-1]["page_range"] == "1~5,9~10"


def test_list_parse_records_by_ids_returns_precise_status(tmp_path: Path) -> None:
    parses = [
        {"id": 1, "sha256": "f" * 64, "tier": "standard", "page_range": "1~5", "status": "done", "done_at": 1000},
        {
            "id": 2,
            "sha256": "f" * 64,
            "tier": "standard",
            "page_range": "6~10",
            "status": "failed",
            "error_code": "parse_failed",
            "error_msg": "boom",
        },
    ]
    db = _FakeDB(parses=parses, file_row=None)
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    result = asyncio.run(service.list_parse_records(ids=[2, 1]))

    assert result["parses"] == [
        {
            "id": 2,
            "sha256": "f" * 64,
            "tier": "standard",
            "page_range": "6~10",
            "status": "failed",
            "done_at": None,
            "created_at": None,
            "updated_at": None,
            "error": {"code": "parse_failed", "message": "boom"},
        },
        {
            "id": 1,
            "sha256": "f" * 64,
            "tier": "standard",
            "page_range": "1~5",
            "status": "done",
            "done_at": 1000,
            "created_at": None,
            "updated_at": None,
            "error": None,
        },
    ]


def test_filter_pages_by_user_range_uses_one_based_page_numbers() -> None:
    pages = [PageInfo(page_idx=0), PageInfo(page_idx=1), PageInfo(page_idx=2)]

    selected = filter_pages_by_user_range(pages, "1")

    assert [page.page_idx for page in selected] == [0]


def test_expand_page_range_uses_available_subset_and_merges_ranges() -> None:
    assert expand_page_range("1~10,3,4~5", 5) == "1~5"


def test_expand_page_range_rejects_empty_available_subset() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        expand_page_range("6~10", 5)

    assert exc_info.value.code == "page_range_invalid"


def test_remap_api_result_pages_to_non_contiguous_page_range() -> None:
    from mineru.doclib.services.parse_svc import _remap_api_result_pages_to_page_range

    result = ParseResult(pages=[PageInfo(page_idx=0), PageInfo(page_idx=1), PageInfo(page_idx=2)])

    _remap_api_result_pages_to_page_range(result, "11,13~14")

    assert [page.page_idx for page in result.pages] == [10, 12, 13]


def test_remap_api_result_pages_refreshes_attached_export_cache() -> None:
    from mineru.doclib.services.parse_svc import _remap_api_result_pages_to_page_range

    result = ParseResult.from_dict({"pages": [{"page_idx": 0}]})
    result.attach_export_images({"images/figure.png": b"figure-bytes"})

    _remap_api_result_pages_to_page_range(result, "5")

    assert result.to_dict()["pages"][0]["page_idx"] == 4
    assert result.images() == {"images/figure.png": b"figure-bytes"}


def test_remap_api_result_pages_rejects_count_mismatch() -> None:
    from mineru.doclib.services.parse_svc import ParseFailure, _remap_api_result_pages_to_page_range

    result = ParseResult(pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)])

    with pytest.raises(ParseFailure) as exc:
        _remap_api_result_pages_to_page_range(result, "11~13")

    assert exc.value.code == "parse_page_remap_failed"


def test_ensure_doc_record_extends_short_id_on_prefix_conflict(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        sha_a = "a" * 64
        sha_b = ("a" * 7) + ("b" * 57)

        for sha256 in (sha_a, sha_b):
            await parse_svc_module.ensure_doc_record(
                db,
                sha256=sha256,
                size_bytes=10,
                file_type="pdf",
                page_count=1,
                title=None,
                author=None,
                subject=None,
                keywords=None,
                error_code=None,
                error_msg=None,
                first_seen_at=1000,
                updated_at=1000,
            )

        first = await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha_a,))
        second = await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha_b,))

        assert first == {"short_id": "a" * 7}
        assert second == {"short_id": ("a" * 7) + "b"}

    asyncio.run(_run())


def test_ensure_ingested_rebinds_changed_text_file_to_new_sha(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"

        source.write_text("old", encoding="utf-8")
        first = await service.ensure_ingested(str(source))

        assert first is not None
        assert first["sha256"] == hashlib.sha256(b"old").hexdigest()

        source.write_text("new content", encoding="utf-8")
        discovered = await service.refresh_file(str(source))
        changed_row = await db.fetchone("SELECT * FROM files WHERE path=?", (str(source),))

        assert discovered.status == "changed"
        assert discovered.needs_ingest is True
        assert discovered.file is not None
        assert discovered.file.sha256 is None
        assert changed_row is not None
        assert changed_row["sha256"] is None

        second = await service.ensure_ingested(str(source))

        assert second is not None
        assert second["sha256"] == hashlib.sha256(b"new content").hexdigest()
        fts_row = await db.fetchone("SELECT sha256, tier, filename FROM fts_contents WHERE sha256=?", (second["sha256"],))
        assert fts_row == {"sha256": second["sha256"], "tier": "flash", "filename": "note.txt"}

    asyncio.run(_run())


def test_refresh_file_marks_missing_known_path_deleted(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"

        source.write_text("content", encoding="utf-8")
        ingested = await service.ensure_ingested(str(source))
        assert ingested is not None
        original_sha = ingested["sha256"]

        source.unlink()
        refreshed = await service.refresh_file(str(source))
        row = await db.fetchone("SELECT sha256, status, deleted_at FROM files WHERE path=?", (str(source),))

        assert refreshed.status == "deleted"
        assert refreshed.file is not None
        assert refreshed.file.sha256 == original_sha
        assert refreshed.file.status == "deleted"
        assert row is not None
        assert row["sha256"] == original_sha
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None

    asyncio.run(_run())


def test_ensure_ingested_maps_read_permission_error_to_file_permission_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "locked.pdf"
        source.write_bytes(b"%PDF-1.7\n")

        async def _permission_denied(path: str) -> str:
            raise PermissionError("permission denied")

        monkeypatch.setattr(parse_svc_module, "compute_sha256", _permission_denied)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.ensure_ingested(str(source))

        row = await db.fetchone("SELECT sha256, error_code, error_msg FROM files WHERE path=?", (str(source),))

        assert exc_info.value.code == "file_permission_denied"
        assert exc_info.value.param == "path"
        assert row is not None
        assert row["sha256"] is None
        assert row["error_code"] == "file_permission_denied"
        assert row["error_msg"] == "permission denied"

    asyncio.run(_run())


def test_refresh_file_records_stat_error_without_marking_deleted(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"

        source.write_text("content", encoding="utf-8")
        ingested = await service.ensure_ingested(str(source))
        assert ingested is not None

        async def _permission_denied(path: str) -> dict[str, Any]:
            raise PermissionError("permission denied")

        monkeypatch.setattr(parse_svc_module, "get_file_stat", _permission_denied)

        refreshed = await service.refresh_file(str(source))
        row = await db.fetchone("SELECT status, error_code, error_msg, deleted_at FROM files WHERE path=?", (str(source),))

        assert refreshed.status == "error"
        assert refreshed.file is not None
        assert refreshed.file.status == "active"
        assert refreshed.file.error_code == "file_permission_denied"
        assert row is not None
        assert row["status"] == "active"
        assert row["error_code"] == "file_permission_denied"
        assert row["error_msg"] == "permission denied"
        assert row["deleted_at"] is None

    asyncio.run(_run())


def test_request_parse_maps_stat_permission_error_without_existing_file_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "no_exec_dir" / "inside.pdf"

        async def _permission_denied(path: str) -> dict[str, Any]:
            raise PermissionError(f"permission denied: {path}")

        monkeypatch.setattr(parse_svc_module, "get_file_stat", _permission_denied)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash")

        assert exc_info.value.code == "file_permission_denied"
        assert exc_info.value.param == "path"
        assert str(source) in exc_info.value.message

    asyncio.run(_run())


def test_ingest_worker_skips_files_with_blocking_stat_errors(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"

        source.write_text("content", encoding="utf-8")
        assert await service.ensure_ingested(str(source)) is not None

        async def _permission_denied(path: str) -> dict[str, Any]:
            raise PermissionError("permission denied")

        monkeypatch.setattr(parse_svc_module, "get_file_stat", _permission_denied)

        await service.refresh_file(str(source))

        worker = IngestWorkerPool(service, num_workers=1, lock_timeout_sec=60)
        assert await worker._acquire_task() is None

        row = await db.fetchone("SELECT error_code, error_msg, locked_at FROM files WHERE path=?", (str(source),))
        assert row is not None
        assert row["error_code"] == "file_permission_denied"
        assert row["error_msg"] == "permission denied"
        assert row["locked_at"] is None

    asyncio.run(_run())


def test_ingest_worker_preserves_mineru_permission_errors(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"
        now = 1000
        file_id = await db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(source), source.name, "txt", 10, now, now, now),
        )

        worker = IngestWorkerPool(service, num_workers=1, lock_timeout_sec=60)
        await worker._handle_ingest_error(
            {"id": file_id, "path": str(source), "watch_id": None},
            InvalidRequestError("file_permission_denied", "permission denied", "path"),
        )

        row = await db.fetchone("SELECT error_code, error_msg, locked_at FROM files WHERE id=?", (file_id,))
        assert row is not None
        assert row["error_code"] == "file_permission_denied"
        assert row["error_msg"] == "permission denied"
        assert row["locked_at"] is None
        assert await worker._acquire_task() is None

    asyncio.run(_run())


def test_ingest_worker_preserves_precise_file_access_errors(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"
        now = 1000
        file_id = await db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(source), source.name, "txt", 10, now, now, now),
        )

        worker = IngestWorkerPool(service, num_workers=1, lock_timeout_sec=60)
        await worker._handle_ingest_error(
            {"id": file_id, "path": str(source), "watch_id": None},
            PermissionError("permission denied"),
        )

        row = await db.fetchone("SELECT error_code, error_msg, locked_at FROM files WHERE id=?", (file_id,))
        assert row is not None
        assert row["error_code"] == "file_permission_denied"
        assert row["error_msg"] == "permission denied"
        assert row["locked_at"] is None
        assert await worker._acquire_task() is None

    asyncio.run(_run())


def test_ingest_worker_skips_any_file_with_existing_error(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "note.txt"
        now = 1000
        await db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, error_code, error_msg, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(source), source.name, "txt", 10, now, "ingest_failed", "boom", now, now),
        )

        worker = IngestWorkerPool(service, num_workers=1, lock_timeout_sec=60)

        assert await worker._acquire_task() is None
        row = await db.fetchone("SELECT error_code, locked_at FROM files WHERE path=?", (str(source),))
        assert row == {"error_code": "ingest_failed", "locked_at": None}

    asyncio.run(_run())


def test_search_filters_by_tier_min_tier_and_file_type(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        fts = FTSManager(db)
        service = SearchService(db, fts)
        now = 1000
        docs = [
            ("1" * 64, "flash", "pdf", "flash.pdf", 2),
            ("2" * 64, "standard", "pdf", "standard.pdf", 12),
            ("3" * 64, "pro", "docx", "pro.docx", 23),
        ]

        for sha256, tier, file_type, filename, page_count in docs:
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 10, file_type, page_count, now, now),
            )
            await db.execute(
                "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(tmp_path / filename), filename, filename.rsplit(".", 1)[1], 10, now, sha256, "active", now, now),
            )
            await fts.replace(sha256=sha256, tier=tier, text="needle content", title="", author="", filename=filename)

        exact_results, exact_total = await service.search("needle", tier="standard")
        min_results, min_total = await service.search("needle", min_tier="standard")
        type_results, type_total = await service.search("needle", file_type="docx")

        assert exact_total == 1
        assert [row["tier"] for row in exact_results] == ["standard"]
        assert [row["short_id"] for row in exact_results] == ["2" * 7]
        assert [row["page_count"] for row in exact_results] == [12]
        assert min_total == 2
        assert {row["tier"] for row in min_results} == {"standard", "pro"}
        assert type_total == 1
        assert [row["filename"] for row in type_results] == ["pro.docx"]
        assert [row["short_id"] for row in type_results] == ["3" * 7]
        assert [row["page_count"] for row in type_results] == [23]

    asyncio.run(_run())


def test_search_prefers_active_paths_and_falls_back_to_non_active_paths(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        fts = FTSManager(db)
        service = SearchService(db, fts)
        now = 1000
        sha_active = "4" * 64
        sha_deleted = "5" * 64

        for sha256 in (sha_active, sha_deleted):
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 10, "pdf", 7 if sha256 == sha_active else 9, now, now),
            )
            await fts.replace(
                sha256=sha256, tier="standard", text="fallback needle", title="", author="", filename=f"{sha256[0]}.pdf"
            )

        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "active.pdf"), "active.pdf", "pdf", 10, now, sha_active, "active", now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "deleted-copy.pdf"), "deleted-copy.pdf", "pdf", 10, now, sha_active, "deleted", now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "deleted-only.pdf"), "deleted-only.pdf", "pdf", 10, now, sha_deleted, "deleted", now, now),
        )

        results, total = await service.search("fallback")
        paths_by_sha = {row["sha256"]: row["paths"] for row in results}

        assert total == 2
        assert paths_by_sha[sha_active] == [str(tmp_path / "active.pdf")]
        assert paths_by_sha[sha_deleted] == [str(tmp_path / "deleted-only.pdf")]
        assert {row["sha256"]: row["page_count"] for row in results} == {sha_active: 7, sha_deleted: 9}

    asyncio.run(_run())


def test_find_filters_by_ext(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        fts = FTSManager(db)
        service = SearchService(db, fts)
        now = 1000
        sha_pdf = "a" * 64
        sha_docx = "b" * 64

        for sha256, filename, ext in ((sha_pdf, "report.pdf", "pdf"), (sha_docx, "report.docx", "docx")):
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 10, ext, 11 if ext == "pdf" else 5, now, now),
            )
            file_id = await db.execute_insert(
                "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(tmp_path / filename), filename, ext, 10, now, sha256, "active", now, now),
            )
            await fts.upsert_filename(file_id, filename, ext)

        results, total = await service.search_filenames("report", ext="pdf")

        assert total == 1
        assert [row["filename"] for row in results] == ["report.pdf"]
        assert [row["page_count"] for row in results] == [11]

    asyncio.run(_run())


def test_find_probes_and_filters_deleted_paths_without_rescan(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        fts = FTSManager(db)
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db,
            fts=fts,
            config_svc=config_svc,
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        server = DoclibServer(
            SimpleNamespace(
                db=db,
                fts=fts,
                parse_svc=parse_svc,
                search_svc=SearchService(db, fts),
                telemetry_svc=None,
            )
        )
        source = tmp_path / "bench1.txt"
        source.write_text("hello", encoding="utf-8")
        assert await parse_svc.ensure_ingested(str(source)) is not None

        source.unlink()
        result = await server.find("bench1")
        row = await db.fetchone("SELECT status, deleted_at FROM files WHERE path=?", (str(source),))

        assert result.total == 0
        assert result.results == []
        assert row is not None
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None

    asyncio.run(_run())


def test_refresh_file_marks_only_current_file_unreachable_when_watch_root_is_missing(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        root = tmp_path / "removable"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=True)
        source = root / "note.txt"
        other = root / "other.txt"

        source.write_text("content", encoding="utf-8")
        other.write_text("other", encoding="utf-8")
        assert await service.ensure_ingested(str(source), watch_id=watch["id"]) is not None
        assert await service.ensure_ingested(str(other), watch_id=watch["id"]) is not None

        source.unlink()
        other.unlink()
        root.rmdir()
        refreshed = await service.refresh_file(str(source), watch_id=watch["id"])

        source_row = await db.fetchone(
            "SELECT status, error_code, error_msg, deleted_at FROM files WHERE path=?", (str(source),)
        )
        other_row = await db.fetchone("SELECT status FROM files WHERE path=?", (str(other),))
        watch_row = await db.fetchone("SELECT status, unreachable_at FROM watches WHERE id=?", (watch["id"],))

        assert refreshed.status == "unreachable"
        assert source_row is not None
        assert source_row["status"] == "unreachable"
        assert source_row["error_code"] is None
        assert source_row["error_msg"] is None
        assert source_row["deleted_at"] is None
        assert other_row == {"status": "active"}
        assert watch_row is not None
        assert watch_row["status"] == "unreachable"
        assert watch_row["unreachable_at"] is not None

    asyncio.run(_run())


def test_remove_watch_converts_unreachable_files_to_deleted_and_unbinds_watch(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        root = tmp_path / "removable"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=True)
        source = root / "note.txt"
        source.write_text("content", encoding="utf-8")

        assert await service.ensure_ingested(str(source), watch_id=watch["id"]) is not None
        now = 123456
        await db.execute(
            "INSERT INTO scans (path, kind, source, watch_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(root), "watch", "cli", watch["id"], "done", now, now),
        )
        await db.execute("UPDATE files SET status=? WHERE path=?", ("unreachable", str(source)))

        await config_svc.remove_watch(str(root))
        row = await db.fetchone("SELECT watch_id, status, deleted_at FROM files WHERE path=?", (str(source),))
        watch_row = await db.fetchone("SELECT * FROM watches WHERE id=?", (watch["id"],))
        scan_row = await db.fetchone("SELECT path, kind, watch_id FROM scans WHERE path=?", (str(root),))

        assert row is not None
        assert row["watch_id"] is None
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None
        assert watch_row is None
        assert scan_row == {"path": str(root), "kind": "watch", "watch_id": None}

    asyncio.run(_run())


def test_watch_scan_refreshes_known_active_paths_before_walk(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        fts = FTSManager(db)
        service = ParseService(
            db=db, fts=fts, config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        watch_loop = WatchLoop(db=db, config_svc=config_svc, parse_svc=service, scan_interval_sec=300)
        root = tmp_path / "watched"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=False)
        source = root / "note.txt"

        source.write_text("content", encoding="utf-8")
        ingested = await service.ensure_ingested(str(source), watch_id=watch["id"])
        assert ingested is not None

        source.unlink()
        await watch_loop._initial_scan(str(root), watch["id"])
        row = await db.fetchone("SELECT status, deleted_at FROM files WHERE path=?", (str(source),))

        assert row is not None
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None

    asyncio.run(_run())


def test_watch_scan_marks_root_unreachable_without_refreshing_all_files(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        fts = FTSManager(db)
        service = ParseService(
            db=db, fts=fts, config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        watch_loop = WatchLoop(db=db, config_svc=config_svc, parse_svc=service, scan_interval_sec=300)
        root = tmp_path / "removable"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=True)
        source = root / "note.txt"

        source.write_text("content", encoding="utf-8")
        ingested = await service.ensure_ingested(str(source), watch_id=watch["id"])
        assert ingested is not None

        source.unlink()
        root.rmdir()
        await watch_loop._initial_scan(str(root), watch["id"])
        file_row = await db.fetchone("SELECT status FROM files WHERE path=?", (str(source),))
        watch_row = await db.fetchone("SELECT status, unreachable_at FROM watches WHERE id=?", (watch["id"],))

        assert file_row == {"status": "active"}
        assert watch_row is not None
        assert watch_row["status"] == "unreachable"
        assert watch_row["unreachable_at"] is not None

    asyncio.run(_run())


def test_device_monitor_queues_watch_scan_when_removable_watch_recovers(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        fts = FTSManager(db)
        parse_svc = ParseService(
            db=db, fts=fts, config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        monitor = DeviceMonitor(db=db, config_svc=config_svc, interval_sec=5, scan_svc=scan_svc)
        root = tmp_path / "removable"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=True)
        source = root / "note.txt"

        source.write_text("content", encoding="utf-8")
        ingested = await parse_svc.ensure_ingested(str(source), watch_id=watch["id"])
        assert ingested is not None

        source.unlink()
        root.rmdir()
        await monitor._poll_once()
        unreachable_watch = await db.fetchone("SELECT status FROM watches WHERE id=?", (watch["id"],))
        unreachable_file = await db.fetchone("SELECT status FROM files WHERE path=?", (str(source),))

        assert unreachable_watch == {"status": "unreachable"}
        assert unreachable_file == {"status": "unreachable"}

        root.mkdir()
        await monitor._poll_once()
        recovered_watch = await db.fetchone("SELECT status FROM watches WHERE id=?", (watch["id"],))
        recovered_file = await db.fetchone("SELECT status FROM files WHERE path=?", (str(source),))
        scan = await db.fetchone(
            "SELECT path, kind, source, watch_id, status FROM scans WHERE watch_id=?",
            (watch["id"],),
        )

        assert recovered_watch == {"status": "active"}
        assert recovered_file == {"status": "active"}
        assert scan == {
            "path": str(root),
            "kind": "watch",
            "source": "system",
            "watch_id": watch["id"],
            "status": "pending",
        }

    asyncio.run(_run())


def test_scan_service_creates_and_processes_manual_file_scan(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        source = tmp_path / "note.txt"
        source.write_text("content", encoding="utf-8")

        scan = await scan_svc.create_scan(str(source), kind="manual", source="cli")
        task = await scan_svc.acquire_task()

        assert task is not None
        assert task["id"] == scan.id
        assert await scan_svc.process_scan(task) is True

        done = await scan_svc.get_scan(scan.id)
        file_row = await db.fetchone("SELECT path, sha256, status FROM files WHERE path=?", (str(source),))

        assert done.status == "done"
        assert done.files_seen == 1
        assert done.files_refreshed == 1
        assert done.files_new == 1
        assert file_row is not None
        assert file_row["sha256"] is None
        assert file_row["status"] == "active"

    asyncio.run(_run())


def test_scan_service_directory_scan_applies_excludes_and_counts_unsupported(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        root = tmp_path / "docs"
        root.mkdir()
        included = root / "included.txt"
        excluded = root / "excluded.txt"
        unsupported = root / "image.png"
        included.write_text("included", encoding="utf-8")
        excluded.write_text("excluded", encoding="utf-8")
        unsupported.write_text("unsupported", encoding="utf-8")
        await config_svc.add_rule("exclude-test", "exclude", f"*{excluded.name}")

        scan = await scan_svc.create_scan(str(root), kind="manual", source="cli")
        task = await scan_svc.acquire_task()
        assert task is not None
        assert await scan_svc.process_scan(task) is True

        done = await scan_svc.get_scan(scan.id)
        rows = await db.fetchall("SELECT path FROM files ORDER BY path")

        assert done.files_seen == 3
        assert done.files_refreshed == 1
        assert done.files_new == 1
        assert done.files_excluded == 1
        assert done.files_unsupported == 1
        assert rows == [{"path": str(included)}]

    asyncio.run(_run())


def test_refresh_file_ignores_office_temp_lock_files(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=ConfigService(db),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "~$package-size.xlsx"
        source.write_bytes(b"\x15Microsoft Office lock")

        refreshed = await service.refresh_file(str(source), ensure_ingested=True)
        row = await db.fetchone("SELECT path FROM files WHERE path=?", (str(source),))

        assert refreshed.status == "unsupported"
        assert refreshed.file is None
        assert row is None

    asyncio.run(_run())


def test_ensure_ingested_ignores_existing_office_temp_lock_file_row(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=ConfigService(db),
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "~$package-size.xlsx"
        source.write_bytes(b"\x15Microsoft Office lock")
        now = 1000
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, watch_id, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(source), source.name, "xlsx", source.stat().st_size, now, None, "active", now, now),
        )

        ingested = await service.ensure_ingested(str(source))

        assert ingested is None

    asyncio.run(_run())


def test_scan_service_directory_scan_ignores_office_temp_lock_files(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        root = tmp_path / "docs"
        root.mkdir()
        regular = root / "regular.txt"
        lock_file = root / "~$package-size.xlsx"
        regular.write_text("regular", encoding="utf-8")
        lock_file.write_bytes(b"\x15Microsoft Office lock")

        scan = await scan_svc.create_scan(str(root), kind="manual", source="cli")
        task = await scan_svc.acquire_task()
        assert task is not None
        assert await scan_svc.process_scan(task) is True

        done = await scan_svc.get_scan(scan.id)
        rows = await db.fetchall("SELECT path FROM files ORDER BY path")

        assert done.files_seen == 2
        assert done.files_refreshed == 1
        assert done.files_new == 1
        assert done.files_unsupported == 1
        assert rows == [{"path": str(regular)}]

    asyncio.run(_run())


def test_watch_event_ignores_office_temp_lock_files(tmp_path: Path) -> None:
    async def _run() -> None:
        class _ConfigService:
            async def is_path_excluded(self, path: str) -> bool:
                return False

        class _ParseService:
            def __init__(self) -> None:
                self.refreshed: list[str] = []

            async def refresh_file(self, filepath: str, watch_id: int) -> None:
                self.refreshed.append(filepath)

        parse_svc = _ParseService()
        watch_loop = WatchLoop(
            db=SimpleNamespace(),
            config_svc=_ConfigService(),
            parse_svc=parse_svc,
            scan_interval_sec=300,
        )
        lock_file = tmp_path / "~$package-size.xlsx"
        lock_file.write_bytes(b"\x15Microsoft Office lock")

        await watch_loop._handle_event(str(lock_file), watch_id=1)

        assert parse_svc.refreshed == []

    asyncio.run(_run())


def test_watch_event_ignores_image_files(tmp_path: Path) -> None:
    async def _run() -> None:
        class _ConfigService:
            async def is_path_excluded(self, path: str) -> bool:
                return False

        class _ParseService:
            def __init__(self) -> None:
                self.refreshed: list[str] = []

            async def refresh_file(self, filepath: str, watch_id: int) -> None:
                self.refreshed.append(filepath)

        parse_svc = _ParseService()
        watch_loop = WatchLoop(
            db=SimpleNamespace(),
            config_svc=_ConfigService(),
            parse_svc=parse_svc,
            scan_interval_sec=300,
        )
        image_file = tmp_path / "scan.png"
        image_file.write_bytes(b"png-bytes")

        await watch_loop._handle_event(str(image_file), watch_id=1)

        assert parse_svc.refreshed == []

    asyncio.run(_run())


def test_scan_service_reuses_pending_scan_for_same_kind_and_path(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        root = tmp_path / "docs"
        root.mkdir()

        first = await scan_svc.create_scan(str(root), kind="manual", source="cli")
        second = await scan_svc.create_scan(str(root), kind="manual", source="sdk")
        scans = await scan_svc.list_scans()

        assert second.id == first.id
        assert len(scans) == 1

    asyncio.run(_run())


def test_scan_service_does_not_reuse_stale_running_scan_for_same_kind_and_path(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1)
        root = tmp_path / "docs"
        root.mkdir()

        await db.execute(
            "INSERT INTO scans (path, kind, source, status, locked_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(root), "manual", "cli", "running", 1, 10, 10),
        )

        fresh = await scan_svc.create_scan(str(root), kind="manual", source="sdk")
        scans = await scan_svc.list_scans()

        assert len(scans) == 2
        assert fresh.id != scans[-1].id

    asyncio.run(_run())


def test_watch_ids_are_non_negative_for_cli_arguments(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        root = tmp_path / "watch-root"
        root.mkdir()

        watch = await config_svc.add_watch(str(root))

        assert watch["id"] >= 0

    asyncio.run(_run())


def test_watch_ids_are_stable_standard_library_hashes(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        root = tmp_path / "watch-root"
        root.mkdir()

        watch = await config_svc.add_watch(str(root))
        expected = int.from_bytes(hashlib.blake2b(str(root).encode("utf-8"), digest_size=8).digest(), "big") & ((1 << 63) - 1)

        assert watch["id"] == expected

    asyncio.run(_run())


def test_doclib_server_list_responses_include_pagination_metadata(tmp_path: Path) -> None:
    class _NoopParseService:
        async def ensure_ingested(self, path: str) -> None:
            return None

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        server = DoclibServer(SimpleNamespace(db=db, scan_svc=scan_svc, parse_svc=_NoopParseService()))

        for index in range(3):
            sha256 = str(index + 1) * 64
            now = 1000 + index
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, file_type, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 10 + index, "pdf", now, now),
            )
            await db.execute(
                "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(tmp_path / f"doc-{index}.pdf"), f"doc-{index}.pdf", "pdf", 10 + index, now, sha256, "active", now, now),
            )
            await db.execute(
                "INSERT INTO parses (sha256, tier, page_range, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sha256, "standard", "1", "pending", now, now),
            )
            await db.execute(
                "INSERT INTO scans (path, kind, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(tmp_path / f"doc-{index}.pdf"), "manual", "cli", "done", now, now),
            )

        parses = await server.list_parses(limit=1, offset=1)
        files = await server.list_files(limit=1, offset=1)
        scans = await server.list_scans(limit=1, offset=1)
        docs = await server.list_docs(file_type="pdf", limit=1, offset=1)

        assert parses.total == 3
        assert parses.limit == 1
        assert parses.offset == 1
        assert len(parses.parses) == 1
        assert parses.parses[0].short_id == "2" * 7

        assert files.total == 3
        assert files.limit == 1
        assert files.offset == 1
        assert len(files.files) == 1
        assert files.files[0].short_id == "2" * 7

        file_detail = await server.get_file_by_path(str(tmp_path / "doc-1.pdf"))
        assert file_detail.file.short_id == "2" * 7
        assert len(file_detail.active_parses) == 1
        assert file_detail.active_parses[0].short_id == "2" * 7

        assert scans.total == 3
        assert scans.limit == 1
        assert scans.offset == 1
        assert len(scans.scans) == 1

        assert docs.total == 3
        assert docs.limit == 1
        assert docs.offset == 1
        assert len(docs.docs) == 1

    asyncio.run(_run())


def test_doclib_server_accepts_short_id_for_sha256_doc_inputs(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        fts = FTSManager(db)
        parse_svc = ParseService(
            db=db,
            fts=fts,
            config_svc=config_svc,
            data_dir=str(tmp_path),
            parse_lock_timeout_sec=1800,
        )
        server = DoclibServer(SimpleNamespace(db=db, data_dir=str(tmp_path), parse_svc=parse_svc))
        now = 1000
        sha256 = "a" * 64
        short_id = "aaaaaaa"
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, short_id, 12, "pdf", 1, now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "doc.pdf"), "doc.pdf", "pdf", 12, now, sha256, "active", now, now),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, done_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sha256, "standard", "1", "done", "local", now, now, now),
        )
        _write_batch(
            tmp_path,
            sha256,
            "standard",
            "1",
            now,
            ParseResult([_text_page("hello short id")]).to_dict(skip_defaults=True)["pages"],
        )

        doc_by_short_id = await server.get_doc(short_id, expand_files=True)
        doc_by_sha256 = await server.get_doc(sha256, expand_files=True)
        parses_by_short_id = await server.list_parses(doc_ref=short_id)
        parses_by_sha256 = await server.list_parses(doc_ref=sha256)
        missing_parses = await server.list_parses(doc_ref="ccccccc")
        content = await server.get_doc_content(short_id, tier="standard", page_range="1")
        export_path = tmp_path / "out.md"
        exported = await server.export_doc_content(
            short_id,
            DocContentExportRequest(tier="standard", page_range="1", output=str(export_path)),
        )
        invalidated = await server.invalidate(InvalidateRequest(doc_ref=short_id, tier="standard"))

        assert doc_by_short_id.sha256 == sha256
        assert doc_by_short_id.short_id == short_id
        assert doc_by_short_id.files is not None
        assert doc_by_short_id.files[0].short_id == short_id
        assert doc_by_sha256.short_id == short_id
        assert parses_by_short_id.total == 1
        assert parses_by_short_id.parses[0].sha256 == sha256
        assert parses_by_sha256.total == 1
        assert parses_by_sha256.parses[0].short_id == short_id
        assert missing_parses.total == 0
        assert content.sha256 == sha256
        assert content.short_id == short_id
        assert "hello short id" in content.content
        assert exported.sha256 == sha256
        assert exported.short_id == short_id
        assert export_path.read_text(encoding="utf-8")
        assert invalidated.sha256 == sha256
        assert invalidated.short_id == short_id
        assert invalidated.invalidated_count == 1

        with pytest.raises(NotFoundError) as missing_doc_exc:
            await server.get_doc("ccccccc")
        assert missing_doc_exc.value.code == "doc_not_found"
        assert missing_doc_exc.value.param == "doc_ref"

    asyncio.run(_run())


def test_doc_content_invalid_after_cursor_returns_invalid_locator(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.sqlite"))
        await db.initialize()
        server = DoclibServer(SimpleNamespace(db=db, data_dir=str(tmp_path)))
        now = 1000
        sha256 = "a" * 64
        short_id = "aaaaaaa"
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, short_id, 12, "pdf", 1, now, now),
        )
        try:
            with pytest.raises(InvalidRequestError) as exc_info:
                await server.get_doc_content(short_id, tier="standard", after="invalid-cursor-for-param-test")

            assert exc_info.value.code == "invalid_locator"
            assert exc_info.value.param == "after"
            assert "Invalid doclib content cursor" in exc_info.value.message
        finally:
            await db.close()

    asyncio.run(_run())


def test_read_locator_out_of_range_page_returns_page_range_invalid(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.sqlite"))
        await db.initialize()
        server = DoclibServer(SimpleNamespace(db=db, data_dir=str(tmp_path)))
        now = 1000
        sha256 = "a" * 64
        short_id = "aaaaaaa"
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, short_id, 12, "pdf", 4, now, now),
        )
        try:
            with pytest.raises(InvalidRequestError) as exc_info:
                await server.read_content(f"doc:{short_id}/tier:flash/page:500")

            assert exc_info.value.code == "page_range_invalid"
            assert exc_info.value.param == "page_range"
            assert exc_info.value.message == "Page range does not select any pages: 500"
        finally:
            await db.close()

    asyncio.run(_run())


def test_get_file_by_path_missing_doclib_record_message_is_not_disk_file_not_found(tmp_path: Path) -> None:
    class _ParseSvc:
        async def ensure_ingested(self, path: str) -> None:
            return None

    async def _run() -> None:
        db = _FakeDB(parses=[], file_row=None)
        server = DoclibServer(SimpleNamespace(db=db, parse_svc=_ParseSvc()))

        with pytest.raises(NotFoundError) as exc_info:
            await server.get_file_by_path(str(tmp_path / "sample.png"))

        assert exc_info.value.code == "file_not_found"
        assert exc_info.value.param == "path"
        assert "File record" in exc_info.value.message
        assert "doclib" in exc_info.value.message

    asyncio.run(_run())


def test_scan_service_cleanup_keeps_latest_terminal_scans_and_active_tasks(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)

        for index in range(1002):
            status = "done" if index % 2 == 0 else "failed"
            await db.execute(
                "INSERT INTO scans (path, kind, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"/tmp/doc-{index}.pdf", "manual", "cli", status, index, index),
            )
        await db.execute(
            "INSERT INTO scans (path, kind, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("/tmp/pending.pdf", "manual", "cli", "pending", 2000, 2000),
        )
        await db.execute(
            "INSERT INTO scans (path, kind, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("/tmp/running.pdf", "manual", "cli", "running", 2001, 2001),
        )

        await scan_svc.cleanup_terminal_scan_logs()

        terminal_count = await db.fetchone("SELECT COUNT(*) AS cnt FROM scans WHERE status IN (?, ?)", ("done", "failed"))
        active_count = await db.fetchone("SELECT COUNT(*) AS cnt FROM scans WHERE status IN (?, ?)", ("pending", "running"))
        oldest = await db.fetchone("SELECT MIN(created_at) AS created_at FROM scans WHERE status IN (?, ?)", ("done", "failed"))

        assert terminal_count == {"cnt": 1000}
        assert active_count == {"cnt": 2}
        assert oldest == {"created_at": 2}

    asyncio.run(_run())


def test_watch_loop_queues_watch_scan_when_scan_service_is_available(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        parse_svc = ParseService(
            db=db, fts=FTSManager(db), config_svc=config_svc, data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        scan_svc = ScanService(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_lock_timeout_sec=1800)
        watch_loop = WatchLoop(db=db, config_svc=config_svc, parse_svc=parse_svc, scan_interval_sec=300, scan_svc=scan_svc)
        root = tmp_path / "watched"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=False)

        await watch_loop._initial_scan(str(root), watch["id"])
        scans = await scan_svc.list_scans(kind="watch", watch_id=watch["id"])

        assert len(scans) == 1
        assert scans[0].path == str(root)
        assert scans[0].status == "pending"

    asyncio.run(_run())


def test_cleanup_deleted_files_does_not_cleanup_orphan_docs(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        cleanup_svc = CleanupService(db=db, data_dir=str(tmp_path / "data"))
        now = 1000
        sha256 = "1" * 64

        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (sha256, sha256[:7], 7, now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, deleted_at, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "deleted.txt"), "deleted.txt", "txt", 7, now, sha256, "deleted", now, now, now),
        )
        file_row = await db.fetchone("SELECT id FROM files WHERE sha256=?", (sha256,))
        assert file_row is not None
        await db.execute(
            "INSERT INTO fts_filenames (file_id, filename, ext) VALUES (?, ?, ?)",
            (file_row["id"], "deleted", "txt"),
        )

        count = await cleanup_svc.cleanup_deleted(dry_run=False)
        doc = await db.fetchone("SELECT sha256 FROM docs WHERE sha256=?", (sha256,))
        fts_name = await db.fetchone("SELECT file_id FROM fts_filenames WHERE file_id=?", (file_row["id"],))
        file_after = await db.fetchone("SELECT id FROM files WHERE id=?", (file_row["id"],))

        assert count == 1
        assert file_after is None
        assert fts_name is None
        assert doc == {"sha256": sha256}

    asyncio.run(_run())


def test_cleanup_orphans_keeps_docs_referenced_by_deleted_or_unreachable_files(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        cleanup_svc = CleanupService(db=db, data_dir=str(tmp_path / "data"))
        now = 1000
        deleted_sha = "2" * 64
        unreachable_sha = "3" * 64
        orphan_sha = "4" * 64

        for sha256 in (deleted_sha, unreachable_sha, orphan_sha):
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 7, now, now),
            )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, deleted_at, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "deleted.txt"), "deleted.txt", "txt", 7, now, deleted_sha, "deleted", now, now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "unreachable.txt"), "unreachable.txt", "txt", 7, now, unreachable_sha, "unreachable", now, now),
        )

        count = await cleanup_svc.cleanup_orphans(dry_run=False)
        deleted_doc = await db.fetchone("SELECT sha256 FROM docs WHERE sha256=?", (deleted_sha,))
        unreachable_doc = await db.fetchone("SELECT sha256 FROM docs WHERE sha256=?", (unreachable_sha,))
        orphan_doc = await db.fetchone("SELECT sha256 FROM docs WHERE sha256=?", (orphan_sha,))

        assert count == 1
        assert deleted_doc == {"sha256": deleted_sha}
        assert unreachable_doc == {"sha256": unreachable_sha}
        assert orphan_doc is None

    asyncio.run(_run())


def test_watch_loop_starts_watcher_before_initial_scan() -> None:
    async def _run() -> None:
        class _ConfigService:
            async def list_watches(self) -> list[dict[str, Any]]:
                return [{"id": 1, "path": "/watched", "status": "active"}]

        order: list[str] = []
        watch_loop = WatchLoop(
            db=None,
            config_svc=_ConfigService(),
            parse_svc=None,
            scan_interval_sec=0,
        )

        async def _watch_one(path: str, watch_id: int) -> None:
            order.append("watch")
            await asyncio.Event().wait()

        async def _initial_scan(path: str, watch_id: int) -> None:
            order.append("initial")
            watch_loop.running = False

        watch_loop._watch_one = _watch_one
        watch_loop._initial_scan = _initial_scan

        await watch_loop.run()
        await watch_loop.stop()

        assert order[:2] == ["watch", "initial"]

    asyncio.run(_run())


def test_watch_loop_wakeup_triggers_next_poll_without_interval_delay() -> None:
    async def _run() -> None:
        poll_count = 0
        second_poll = asyncio.Event()

        class _ConfigService:
            async def list_watches(self) -> list[dict[str, Any]]:
                nonlocal poll_count
                poll_count += 1
                if poll_count == 2:
                    second_poll.set()
                return []

        watch_loop = WatchLoop(
            db=None,
            config_svc=_ConfigService(),
            parse_svc=None,
            scan_interval_sec=300,
        )
        task = asyncio.create_task(watch_loop.run())

        while poll_count == 0:
            await asyncio.sleep(0)

        watch_loop.wakeup()
        await asyncio.wait_for(second_poll.wait(), timeout=1)
        await watch_loop.stop()
        await asyncio.wait_for(task, timeout=1)

        assert poll_count == 2

    asyncio.run(_run())


def test_add_watch_wakes_background_watch_loop() -> None:
    async def _run() -> None:
        class _ConfigService:
            async def add_watch(self, path: str, removable: bool, label: str | None) -> dict[str, Any]:
                return {
                    "id": 1,
                    "path": path,
                    "label": label,
                    "removable": int(removable),
                    "enabled": 1,
                    "recursive": 1,
                    "status": "active",
                }

        class _WatchLoop:
            def __init__(self) -> None:
                self.wakeup_count = 0

            def wakeup(self) -> None:
                self.wakeup_count += 1

        watch = _WatchLoop()
        state = SimpleNamespace(config_svc=_ConfigService(), telemetry_svc=None, watch=watch)
        server = DoclibServer(state)

        info = await server.add_watch(WatchRequest(path="/watched", removable=True, label="Docs"))

        assert info.id == 1
        assert watch.wakeup_count == 1

    asyncio.run(_run())


def test_add_watch_queues_initial_watch_scan() -> None:
    async def _run() -> None:
        class _ConfigService:
            async def add_watch(self, path: str, removable: bool, label: str | None) -> dict[str, Any]:
                return {
                    "id": 1,
                    "path": path,
                    "label": label,
                    "removable": int(removable),
                    "enabled": 1,
                    "recursive": 1,
                    "status": "active",
                }

        class _ScanService:
            def __init__(self) -> None:
                self.created_scans: list[dict[str, Any]] = []

            async def create_scan(self, path: str, *, kind: str, source: str, watch_id: int) -> object:
                self.created_scans.append({"path": path, "kind": kind, "source": source, "watch_id": watch_id})
                return object()

        scan_svc = _ScanService()
        state = SimpleNamespace(config_svc=_ConfigService(), telemetry_svc=None, watch=None, scan_svc=scan_svc)
        server = DoclibServer(state)

        await server.add_watch(WatchRequest(path="/watched", removable=True, label="Docs"))

        assert scan_svc.created_scans == [
            {"path": "/watched", "kind": "watch", "source": "watch", "watch_id": 1}
        ]

    asyncio.run(_run())


def test_remove_watch_wakes_background_watch_loop() -> None:
    async def _run() -> None:
        class _DB:
            async def fetchone(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
                return {"id": params[0]}

        class _ConfigService:
            def __init__(self) -> None:
                self.removed_watch_ids: list[int] = []

            async def remove_watch_by_id(self, watch_id: int) -> None:
                self.removed_watch_ids.append(watch_id)

        class _WatchLoop:
            def __init__(self) -> None:
                self.wakeup_count = 0

            def wakeup(self) -> None:
                self.wakeup_count += 1

        config_svc = _ConfigService()
        watch = _WatchLoop()
        state = SimpleNamespace(db=_DB(), config_svc=config_svc, telemetry_svc=None, watch=watch)
        server = DoclibServer(state)

        response = await server.remove_watch(42)

        assert response.removed is True
        assert config_svc.removed_watch_ids == [42]
        assert watch.wakeup_count == 1

    asyncio.run(_run())


def test_server_status_includes_watch_stats_and_error_summary(tmp_path: Path) -> None:
    class _ParseQueue:
        async def get_queue_length(self) -> int:
            return 2

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        (tmp_path / "watched").mkdir()
        watch = await config_svc.add_watch(str(tmp_path / "watched"), removable=True)
        now = 1000
        sha_done = "5" * 64
        sha_pending = "6" * 64

        for sha256 in (sha_done, sha_pending):
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 7, now, now),
            )
        await db.execute(
            "UPDATE docs SET error_code=? WHERE sha256=?",
            ("open_failed", sha_pending),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, watch_id, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "watched" / "done.pdf"), "done.pdf", "pdf", 7, now, sha_done, watch["id"], "active", now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, watch_id, status, error_code, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(tmp_path / "watched" / "pending.pdf"),
                "pending.pdf",
                "pdf",
                7,
                now,
                sha_pending,
                watch["id"],
                "active",
                "stat_failed",
                now,
                now,
            ),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, watch_id, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "watched" / "need-ingest.pdf"), "need-ingest.pdf", "pdf", 7, now, watch["id"], "active", now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, watch_id, status, error_code, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(tmp_path / "watched" / "blocked-ingest.pdf"),
                "blocked-ingest.pdf",
                "pdf",
                7,
                now,
                watch["id"],
                "active",
                "ingest_failed",
                now,
                now,
            ),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha_done, "standard", "1", "done", "local", now, now),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, error_code, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sha_pending, "standard", "1", "failed", "local", "parse_failed", now, now),
        )
        for index in range(6):
            await db.execute(
                "INSERT INTO scans (path, kind, source, status, files_seen, files_new, error_code, error_msg, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(tmp_path / f"scan-{index}.pdf"),
                    "manual",
                    "cli",
                    "done",
                    index,
                    index + 1,
                    "scan_failed" if index == 5 else None,
                    "hidden message",
                    now + index,
                    now + index,
                ),
            )
        await db.execute(
            "INSERT INTO scans (path, kind, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "still-running.pdf"), "manual", "cli", "running", now + 99, now + 99),
        )

        state = SimpleNamespace(
            db=db,
            config_svc=config_svc,
            parse_svc=_ParseQueue(),
            start_time=0,
            pid=123,
            socket_path="/tmp/doclib.sock",
            data_dir=str(tmp_path / "data"),
            watch=SimpleNamespace(running=False),
            scan_workers=SimpleNamespace(running=False, num_workers=1),
            ingest_workers=SimpleNamespace(running=False, num_workers=2),
            parse_workers=SimpleNamespace(running=False, num_workers=2),
            device_monitor=SimpleNamespace(running=False),
            compaction=SimpleNamespace(running=False),
            health_check=SimpleNamespace(running=False),
        )
        status = await DoclibServer(state).get_server_status()

        assert status.watch_count == 1
        assert status.ingest_queue_length == 1
        assert len(status.watch_stats) == 1
        stats = status.watch_stats[0]
        assert stats.watch_id == watch["id"]
        assert stats.total_files == 4
        assert stats.active_files == 4
        assert stats.pending_ingest_files == 1
        assert stats.file_error_count == 2
        assert stats.doc_count == 2
        assert stats.parse_done_count == 1
        assert stats.parse_failed_count == 1
        assert [scan.path for scan in status.recent_scans] == [
            str(tmp_path / "still-running.pdf"),
            *[str(tmp_path / f"scan-{index}.pdf") for index in range(5, 1, -1)],
        ]
        assert status.recent_scans[0].status == "running"
        assert status.recent_scans[1].error_code == "scan_failed"
        assert status.recent_scans[1].error_msg == "hidden message"
        assert status.last_scan_at == now + 5
        assert status.error_summary is not None
        assert [(bucket.code, bucket.count) for bucket in status.error_summary.file_errors] == [
            ("ingest_failed", 1),
            ("stat_failed", 1),
        ]
        assert [(bucket.code, bucket.count) for bucket in status.error_summary.doc_errors] == [("open_failed", 1)]
        assert [(bucket.code, bucket.count) for bucket in status.error_summary.parse_errors] == [("parse_failed", 1)]

    asyncio.run(_run())


def test_forget_path_deletes_file_row_and_filename_fts_without_deleting_doc(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        cleanup_svc = CleanupService(db=db, data_dir=str(tmp_path / "data"))
        now = 1000
        sha256 = "7" * 64
        path = str(tmp_path / "doc.pdf")

        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (sha256, sha256[:7], 7, now, now),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, "standard", "1", "done", "local", now, now),
        )
        await db.execute(
            "INSERT INTO fts_contents (sha256, tier, text, filename) VALUES (?, ?, ?, ?)",
            (sha256, "standard", "content", "doc.pdf"),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path, "doc.pdf", "pdf", 7, now, sha256, "active", now, now),
        )
        file_row = await db.fetchone("SELECT id FROM files WHERE path=?", (path,))
        assert file_row is not None
        await db.execute(
            "INSERT INTO fts_filenames (file_id, filename, ext) VALUES (?, ?, ?)",
            (file_row["id"], "doc", "pdf"),
        )

        dry_run = await cleanup_svc.forget_path(path, dry_run=True)
        assert dry_run["matched_as"] == "file"
        assert dry_run["forgotten_files"] == 1
        assert await db.fetchone("SELECT id FROM files WHERE path=?", (path,)) is not None

        result = await cleanup_svc.forget_path(path, dry_run=False)

        assert result["matched_as"] == "file"
        assert result["forgotten_files"] == 1
        assert await db.fetchone("SELECT id FROM files WHERE path=?", (path,)) is None
        assert await db.fetchone("SELECT file_id FROM fts_filenames WHERE file_id=?", (file_row["id"],)) is None
        assert await db.fetchone("SELECT sha256 FROM docs WHERE sha256=?", (sha256,)) == {"sha256": sha256}
        assert await db.fetchone("SELECT sha256 FROM parses WHERE sha256=?", (sha256,)) == {"sha256": sha256}
        assert await db.fetchone("SELECT sha256 FROM fts_contents WHERE sha256=?", (sha256,)) == {"sha256": sha256}

    asyncio.run(_run())


def test_forget_directory_matches_historical_prefix_rows(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        cleanup_svc = CleanupService(db=db, data_dir=str(tmp_path / "data"))
        now = 1000
        root = tmp_path / "project"
        sha_a = "8" * 64
        sha_b = "9" * 64

        for sha256 in (sha_a, sha_b):
            await db.execute(
                "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sha256, sha256[:7], 7, now, now),
            )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(root / "a.pdf"), "a.pdf", "pdf", 7, now, sha_a, "active", now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(root / "nested" / "b.pdf"), "b.pdf", "pdf", 7, now, sha_b, "deleted", now, now),
        )

        result = await cleanup_svc.forget_path(str(root), dry_run=False)

        assert result["matched_as"] == "directory"
        assert result["forgotten_files"] == 2
        rows = await db.fetchall("SELECT path FROM files")
        assert rows == []

    asyncio.run(_run())


def test_forget_rejects_watch_root_and_warns_under_active_watch(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        config_svc = ConfigService(db)
        cleanup_svc = CleanupService(db=db, data_dir=str(tmp_path / "data"))
        root = tmp_path / "watched"
        root.mkdir()
        watch = await config_svc.add_watch(str(root), removable=False)
        path = str(root / "doc.pdf")
        now = 1000

        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("a" * 64, "a" * 7, 7, now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, watch_id, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path, "doc.pdf", "pdf", 7, now, "a" * 64, watch["id"], "active", now, now),
        )

        try:
            await cleanup_svc.forget_path(str(root), dry_run=True)
        except InvalidRequestError as exc:
            assert exc.param == "path"
        else:
            raise AssertionError("watch root forget should be rejected")

        result = await cleanup_svc.forget_path(path, dry_run=True)

        assert result["matched_as"] == "file"
        assert result["forgotten_files"] == 1
        assert result["warnings"] == ["Path is under an active watch and may be rediscovered on the next scan."]

    asyncio.run(_run())


def test_ingest_records_doc_error_when_metadata_extraction_fails(tmp_path: Path, monkeypatch) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=_NoRulesConfig(), data_dir=str(tmp_path / "data"), parse_lock_timeout_sec=1800
        )
        source = tmp_path / "broken.pdf"
        source.write_bytes(b"%PDF-1.4\nnot actually a valid pdf")

        async def _fail_metadata(path: str) -> dict[str, Any]:
            raise parse_svc_module.MetadataExtractionError("open_failed", "metadata boom")

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _fail_metadata)

        row = await service.ingest_file(str(source))

        doc = await db.fetchone("SELECT error_code, error_msg FROM docs WHERE sha256=?", (row["sha256"],))
        assert doc is not None
        assert doc["error_code"] == "open_failed"
        assert doc["error_msg"] == "metadata boom"

    asyncio.run(_run())


def test_process_doc_rejects_existing_office_temp_lock_file_task(tmp_path: Path) -> None:
    sha256 = "c" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "flash",
        "page_range": "1",
        "status": "parsing",
        "privacy": "local",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/~$package-size.xlsx",
            "sha256": sha256,
            "status": "active",
            "filename": "~$package-size.xlsx",
            "title": "",
            "author": "",
        },
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    async def _unexpected_parse(file_row: dict, tier: Tier, page_range: str) -> ParseResult:
        raise AssertionError("temporary lock files should not reach parser execution")

    service._parse_via_local = _unexpected_parse  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is False
    assert parses[0]["status"] == "failed"
    assert parses[0]["error_code"] == "file_type_unsupported"
    assert "temporary lock" in parses[0]["error_msg"]


def test_process_doc_marks_empty_page_result_failed(tmp_path: Path) -> None:
    sha256 = "a" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "flash",
        "page_range": "1",
        "status": "parsing",
        "privacy": "local",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/doc.pdf",
            "sha256": sha256,
            "status": "active",
            "filename": "doc.pdf",
            "title": "",
            "author": "",
        },
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    async def _empty_parse(file_row: dict, tier: Tier, page_range: str) -> ParseResult:
        return ParseResult(pages=[])

    service._parse_via_local = _empty_parse  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is False
    assert parses[0]["status"] == "failed"
    assert parses[0]["error_code"] == "parse_empty"
    assert list(tmp_path.rglob("*.json")) == []


def test_process_doc_preserves_remote_api_error_code(tmp_path: Path) -> None:
    sha256 = "b" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "pro",
        "page_range": "1",
        "status": "parsing",
        "privacy": "remote",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/doc.pdf",
            "sha256": sha256,
            "status": "active",
            "filename": "doc.pdf",
            "title": "",
            "author": "",
        },
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    async def _parse_via_api(
        file_row: dict[str, object],
        tier: Tier,
        page_range: str,
        privacy: str,
    ) -> tuple[ParseResult, str]:
        raise _V1APIError(
            "invalid_api_key",
            "Remote authentication failed: user authenticate failed",
            param="parse_server.remote.api_key",
        )

    service._parse_via_api = _parse_via_api  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is False
    assert parses[0]["status"] == "failed"
    assert parses[0]["error_code"] == "invalid_api_key"
    assert parses[0]["error_msg"] == "Remote authentication failed: user authenticate failed"


def test_process_doc_fails_when_batch_json_cannot_be_written(tmp_path: Path) -> None:
    sha256 = "b" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "flash",
        "page_range": "1",
        "status": "parsing",
        "privacy": "local",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/doc.pdf",
            "sha256": sha256,
            "status": "active",
            "filename": "doc.pdf",
            "title": "",
            "author": "",
        },
    )
    fts = _FakeFTS()
    service = ParseService(db=db, fts=fts, config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)
    blocked_output_dir = tmp_path / "parsed" / sha256[:2] / sha256 / "flash"
    blocked_output_dir.parent.mkdir(parents=True)
    blocked_output_dir.write_text("not a directory", encoding="utf-8")

    async def _parse(file_row: dict, tier: Tier, page_range: str) -> ParseResult:
        return ParseResult(pages=[PageInfo(page_idx=0)])

    service._parse_via_local = _parse  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is False
    assert parses[0]["status"] == "failed"
    assert parses[0]["error_code"] == "parse_json_write_failed"
    assert parses[0]["done_at"] is None
    assert fts.replaced == []


def test_process_doc_writes_cached_image_sidecars(tmp_path: Path) -> None:
    sha256 = "c" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "flash",
        "page_range": "1",
        "status": "parsing",
        "privacy": "local",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/doc.pdf",
            "sha256": sha256,
            "status": "active",
            "filename": "doc.pdf",
            "title": "",
            "author": "",
        },
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path), parse_lock_timeout_sec=1800)

    async def _parse(file_row: dict, tier: Tier, page_range: str) -> ParseResult:
        return _image_result("figures/cache-hit.jpg", b"fresh-image")

    async def _skip_fts(*args: object, **kwargs: object) -> None:
        return None

    async def _skip_docs_meta(*args: object, **kwargs: object) -> None:
        return None

    service._parse_via_local = _parse  # type: ignore[method-assign]
    service._maybe_update_fts = _skip_fts  # type: ignore[method-assign]
    service._maybe_update_docs_meta = _skip_docs_meta  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is True
    sidecar = tmp_path / "parsed" / sha256[:2] / sha256 / "flash" / "images" / "figures" / "cache-hit.jpg"
    assert sidecar.read_bytes() == b"fresh-image"
    batch_path = Path(parse_batch_json_path(str(tmp_path), sha256, "flash", "1", parses[0]["done_at"]))
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    image_span = batch["pages"][0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]
    assert image_span["image_path"] == "figures/cache-hit.jpg"
    assert "image_base64" not in image_span


def test_load_pages_from_done_batches_keeps_existing_image_sidecar(tmp_path: Path) -> None:
    sha256 = "e" * 64
    tier = "standard"
    page = _image_page("figures/existing.jpg")
    _write_batch(tmp_path, sha256, tier, "1", 1000, ParseResult([page]).to_dict(skip_defaults=True)["pages"])
    sidecar = tmp_path / "parsed" / sha256[:2] / sha256 / tier / "images" / "figures" / "existing.jpg"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(b"existing-sidecar")

    load_pages_from_done_batches(str(tmp_path), sha256, tier, [{"page_range": "1", "done_at": 1000}])

    assert sidecar.read_bytes() == b"existing-sidecar"


def test_progressive_markdown_uses_public_image_sidecar_prefix(tmp_path: Path) -> None:
    sha256 = "e" * 64
    tier = "standard"
    page = _image_page("figures/rendered.jpg")
    _write_batch(tmp_path, sha256, tier, "1", 1000, ParseResult([page]).to_dict(skip_defaults=True)["pages"])
    image_dir = tmp_path / "parsed" / sha256[:2] / sha256 / tier / "images"
    db = _FakeDB(
        parses=[
            {
                "sha256": sha256,
                "tier": tier,
                "status": "done",
                "page_range": "1",
                "done_at": 1000,
            }
        ],
        file_row=None,
    )
    server = DoclibServer(SimpleNamespace(data_dir=str(tmp_path), db=db))
    plan = _ReadPlan(
        sha256=sha256,
        short_id="eeeeeee",
        tier=tier,
        page_range=None,
        after=None,
        locator=None,
        context=0,
        limit=30000,
        format="markdown",
        no_marker=False,
    )

    response = asyncio.run(server._execute_read_plan(plan))

    assert str(image_dir) not in response.content
    assert "![](images/figures/rendered.jpg)" in response.content


def test_doclib_office_image_asset_reads_cached_sidecar(tmp_path: Path) -> None:
    sha256 = "f" * 64
    tier = "standard"
    image_dir = Path(parse_image_sidecar_dir(str(tmp_path), sha256, tier))
    image_path = "figures/office.png"
    sidecar = image_dir / image_path
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
    )

    image_span = Span(type=ContentType.IMAGE, bbox=(1, 1, 20, 20), image_path=image_path)
    body = Block(
        index=0,
        type=BlockType.IMAGE_BODY,
        bbox=(1, 1, 20, 20),
        lines=[Line(bbox=(1, 1, 20, 20), spans=[image_span])],
    )
    image_block = Block(index=0, type=BlockType.IMAGE, bbox=(1, 1, 20, 20), blocks=[body])
    page = PageInfo(page_idx=0, page_size=(100, 100), para_blocks=[image_block], _backend="office")
    server = DoclibServer(SimpleNamespace(data_dir=str(tmp_path), db=None))
    plan = _ReadPlan(
        sha256=sha256,
        short_id="fffffff",
        tier=tier,
        page_range=None,
        after=None,
        locator="doc:fffffff/tier:standard/page:1/block:1",
        context=0,
        limit=30000,
        format="image",
        no_marker=False,
        image_format="png",
        target=ContentCursor(short_id="fffffff", tier=tier, page_no=1, block_no=1),
    )

    asset = asyncio.run(server._render_office_image_asset(plan, page))

    assert asset.mime_type == "image/png"
    assert Path(asset.path).suffix == ".png"
    with Image.open(asset.path) as image:
        assert image.format == "PNG"


def test_doclib_office_image_asset_transcodes_to_requested_format(tmp_path: Path) -> None:
    sha256 = "f" * 64
    tier = "standard"
    image_dir = Path(parse_image_sidecar_dir(str(tmp_path), sha256, tier))
    image_path = "figures/office.png"
    sidecar = image_dir / image_path
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
    )

    image_span = Span(type=ContentType.IMAGE, bbox=(1, 1, 20, 20), image_path=image_path)
    body = Block(
        index=0,
        type=BlockType.IMAGE_BODY,
        bbox=(1, 1, 20, 20),
        lines=[Line(bbox=(1, 1, 20, 20), spans=[image_span])],
    )
    image_block = Block(index=0, type=BlockType.IMAGE, bbox=(1, 1, 20, 20), blocks=[body])
    page = PageInfo(page_idx=0, page_size=(100, 100), para_blocks=[image_block], _backend="office")
    server = DoclibServer(SimpleNamespace(data_dir=str(tmp_path), db=None))
    plan = _ReadPlan(
        sha256=sha256,
        short_id="fffffff",
        tier=tier,
        page_range=None,
        after=None,
        locator="doc:fffffff/tier:standard/page:1/block:1",
        context=0,
        limit=30000,
        format="image",
        no_marker=False,
        image_format="jpeg",
        target=ContentCursor(short_id="fffffff", tier=tier, page_no=1, block_no=1),
    )

    asset = asyncio.run(server._render_office_image_asset(plan, page))

    assert asset.mime_type == "image/jpeg"
    assert Path(asset.path).suffix == ".jpg"
    with Image.open(asset.path) as image:
        assert image.format == "JPEG"
