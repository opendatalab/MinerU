from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.server import DoclibServer
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_svc import ParseService
from mineru.doclib.services.scan_svc import ScanService, _ScanCounters
from mineru.doclib.telemetry import (
    TelemetryContext,
    TelemetryService,
    TelemetryStore,
    duration_bucket,
    file_size_bucket,
    pages_bucket,
    reset_telemetry_context,
    results_bucket,
    set_telemetry_context,
)
from mineru.doclib.telemetry.constants import CONSENT_DISABLED, CONSENT_UNSET
from mineru.doclib.telemetry.payload import build_period_payload, compact_json_bytes, dimensions_hash
from mineru.doclib.telemetry.transport import signed_headers
from mineru.doclib.telemetry import service as telemetry_service_module
from mineru.doclib.types import ParseRequest
from mineru.doclib.types import TelemetryObservation, TelemetryObservationsRequest


def test_bucket_helpers_use_documented_boundaries() -> None:
    assert duration_bucket(999) == "lt_1s"
    assert duration_bucket(1_000) == "1_5s"
    assert duration_bucket(5_000) == "5_30s"
    assert duration_bucket(120_000) == "2_10m"
    assert duration_bucket(600_000) == "gt_10m"

    assert pages_bucket(0) == "1"
    assert pages_bucket(5) == "2_5"
    assert pages_bucket(501) == "gt_500"

    assert file_size_bucket(1024 * 1024 - 1) == "lt_1mb"
    assert file_size_bucket(10 * 1024 * 1024) == "10_50mb"
    assert results_bucket(0) == "0"
    assert results_bucket(101) == "gt_100"


def test_store_initializes_installation_id_and_consent_state(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        await store.initialize()

        installation_id = await store.installation_id()
        assert installation_id.startswith("inst_")
        assert await store.installation_id() == installation_id
        assert await store.consent_state() == CONSENT_UNSET

    asyncio.run(_run())


def test_status_counts_all_pending_periods_not_just_flush_limit(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()

        for index in range(12):
            await service.record_count("search.request.count", timestamp_ms=1_700_000_000_000 + index * 3_600_000)

        status = await service.status()

        assert status["pending_periods"] == 12
        assert status["pending_metrics"] == 12

    asyncio.run(_run())


def test_service_records_aggregates_and_uses_context_dimensions(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()

        token = set_telemetry_context(TelemetryContext(source="cli", caller="agent"))
        try:
            await service.record_count("parse.request.count", timestamp_ms=1_700_000_000_000)
            await service.record_duration_bucket(
                "parse.duration_bucket.count",
                duration_ms=42_000,
                dimensions={"status": "succeeded", "tier": "flash"},
                timestamp_ms=1_700_000_000_000,
            )
        finally:
            reset_telemetry_context(token)

        periods = await store.list_periods_for_flush()
        assert len(periods) == 1
        aggregates = await store.list_aggregates(*periods[0])
        by_name = {item.metric_name: item for item in aggregates}

        assert by_name["parse.request.count"].metric_value == 1
        assert by_name["parse.request.count"].dimensions == {"caller": "agent", "source": "cli"}
        assert by_name["parse.duration_bucket.count"].dimensions == {
            "bucket": "30_120s",
            "caller": "agent",
            "source": "cli",
            "status": "succeeded",
            "tier": "flash",
        }

    asyncio.run(_run())


def test_disabled_consent_clears_and_blocks_aggregates(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()

        await service.record_count("search.request.count")
        assert await store.list_periods_for_flush()

        await store.set_consent_state(CONSENT_DISABLED)
        assert await store.list_periods_for_flush() == []

        await service.record_count("search.request.count")
        assert await store.list_periods_for_flush() == []

    asyncio.run(_run())


def test_payload_is_compact_and_dimension_hash_is_order_stable() -> None:
    first = dimensions_hash({"source": "cli", "caller": "agent"})
    second = dimensions_hash({"caller": "agent", "source": "cli"})
    assert first == second

    payload = build_period_payload(
        batch_id="tb_test",
        installation_id="inst_test",
        period_start=1_700_000_000_000,
        period_end=1_700_003_600_000,
        context={"app": "mineru"},
        metrics=[{"name": "parse.request.count", "value": 1, "dimensions": {"source": "cli"}}],
    )
    raw = compact_json_bytes(payload)
    assert b" " not in raw
    assert b'"api_version":"v2"' in raw
    assert b'"period_start":"2023-11-14T22:13:20.000Z"' in raw


def test_telemetry_service_preview_returns_next_period_body(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()

        await service.record_count("parse.request.count", timestamp_ms=1_700_000_000_000)
        body = await service.preview_body()

        assert body["batch_id"] == "tb_preview"
        assert body["installation_id"].startswith("inst_")
        assert body["period_start"] == "2023-11-14T22:00:00.000Z"
        assert body["metrics"] == [
            {"name": "parse.request.count", "value": 1, "dimensions": {"caller": "unknown", "source": "unknown"}}
        ]

    asyncio.run(_run())


def test_signed_headers_match_staging_api_contract() -> None:
    body = b'{"api_version":"v2"}'
    headers = signed_headers(body, timestamp=1_700_000_000_000)

    assert headers["Content-Type"] == "application/json"
    assert headers["X-Track-App-Key"] == "213a83db-44c8-4218-90c0-ded685cca86e"
    assert headers["X-Track-Ts"] == "1700000000000"
    assert headers["X-Track-Sign"] == "a64fb6b8ad0f9aa89866685f9a849e7fa316742e7ee147cc22e1789d225370ea"


def test_flush_once_sends_enabled_period_and_deletes_on_success(monkeypatch, tmp_path) -> None:
    async def _run() -> None:
        captured: list[dict] = []

        async def _send(payload: dict) -> str:
            captured.append(payload)
            return "success"

        monkeypatch.setattr(telemetry_service_module, "send_payload", _send)
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()
        await store.set_consent_state("enabled")
        await service.record_count("search.request.count", timestamp_ms=1_700_000_000_000)

        result = await service.flush_once()

        assert result.status == "success"
        assert result.attempted == 1
        assert len(captured) == 1
        assert captured[0]["metrics"] == [
            {"name": "search.request.count", "value": 1, "dimensions": {"caller": "unknown", "source": "unknown"}}
        ]
        assert await store.list_periods_for_flush() == []

    asyncio.run(_run())


def test_flush_once_keeps_period_on_retry(monkeypatch, tmp_path) -> None:
    async def _run() -> None:
        async def _send(_payload: dict) -> str:
            return "retry"

        monkeypatch.setattr(telemetry_service_module, "send_payload", _send)
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        service = TelemetryService(store)
        await service.initialize()
        await store.set_consent_state("enabled")
        await service.record_count("find.request.count", timestamp_ms=1_700_000_000_000)

        result = await service.flush_once()

        assert result.status == "failed"
        assert result.attempted == 1
        assert await store.list_periods_for_flush()

    asyncio.run(_run())


def test_parse_route_records_request_finished_and_duration_metrics(tmp_path) -> None:
    class _ParseService:
        async def request_parse(self, *args, **kwargs) -> dict:
            return {
                "sha256": "abc",
                "tier": "standard",
                "page_range": "1",
                "status": "pending",
                "cache_hit": False,
                "wait_parse_ids": [1],
                "created_parse_ids": [1],
                "reused_parse_ids": [],
            }

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        telemetry = TelemetryService(store)
        await telemetry.initialize()
        server = DoclibServer(SimpleNamespace(parse_svc=_ParseService(), telemetry_svc=telemetry))

        await server.ensure_parse(ParseRequest(path="/tmp/a.pdf", tier="standard"))
        body = await telemetry.preview_body()

        metric_names = {metric["name"] for metric in body["metrics"]}
        assert "parse.request.count" in metric_names
        assert "parse.finished.count" in metric_names
        assert "parse.duration_bucket.count" in metric_names
        finished = next(metric for metric in body["metrics"] if metric["name"] == "parse.finished.count")
        assert finished["dimensions"] == {
            "caller": "unknown",
            "source": "unknown",
            "status": "queued",
            "tier": "standard",
        }

    asyncio.run(_run())


def test_search_route_records_result_bucket_metrics(tmp_path) -> None:
    class _SearchService:
        async def search(self, **kwargs) -> tuple[list[dict], int]:
            return [], 0

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        telemetry = TelemetryService(store)
        await telemetry.initialize()
        server = DoclibServer(SimpleNamespace(search_svc=_SearchService(), telemetry_svc=telemetry))

        await server.search("needle")
        body = await telemetry.preview_body()

        by_name = {metric["name"]: metric for metric in body["metrics"]}
        assert by_name["search.request.count"]["value"] == 1
        assert by_name["search.finished.count"]["dimensions"]["status"] == "succeeded"
        assert by_name["search.results_bucket.count"]["dimensions"] == {
            "bucket": "0",
            "caller": "unknown",
            "source": "unknown",
        }

    asyncio.run(_run())


def test_parse_service_process_doc_records_task_metrics(tmp_path) -> None:
    class _Fts:
        async def get_tier(self, sha256: str) -> str | None:
            return None

        async def replace(self, **kwargs) -> None:
            return None

        async def upsert_filename(self, *args, **kwargs) -> None:
            return None

    class _Result:
        def to_dict(self, *, skip_defaults: bool = True) -> dict:
            return {"pages": [{"page_idx": 0, "blocks": []}]}

        def markdown(self, *, add_markers: bool = False) -> str:
            return "content"

    class _ParseService(ParseService):
        async def _parse_via_local(self, file_row, tier, page_range):
            return _Result()

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        telemetry = TelemetryService(store)
        await telemetry.initialize()
        config_svc = ConfigService(db)
        parse_svc = _ParseService(
            db,
            _Fts(),
            config_svc,
            str(tmp_path),
            parse_lock_timeout_sec=30,
            telemetry_svc=telemetry,
        )
        now = 1_700_000_000_000
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("sha", "sha", 1, "pdf", 1, now, now),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "a.pdf"), "a.pdf", "pdf", 1, now, "sha", now, now),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("sha", "flash", "1", "parsing", now, now),
        )
        task = await db.fetchone("SELECT * FROM parses WHERE sha256=?", ("sha",))

        assert await parse_svc.process_doc(task)
        body = await telemetry.preview_body()
        names = {metric["name"] for metric in body["metrics"]}

        assert "parse_task.execute.count" in names
        assert "parse_task.write.count" in names
        assert "parse_task.finished.count" in names
        assert "parse_task.files.count" in names
        assert "parse_task.pages.count" in names
        finished = next(metric for metric in body["metrics"] if metric["name"] == "parse_task.finished.count")
        assert finished["dimensions"] == {"status": "succeeded", "tier": "flash"}

    asyncio.run(_run())


def test_scan_service_process_scan_records_task_metrics(tmp_path) -> None:
    class _ScanService(ScanService):
        async def _run_scan(self, task):
            counters = _ScanCounters()
            counters.files_seen = 2
            counters.files_refreshed = 1
            counters.files_new = 1
            return counters

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        telemetry = TelemetryService(store)
        await telemetry.initialize()
        scan_svc = _ScanService(
            db,
            ConfigService(db),
            SimpleNamespace(),
            scan_lock_timeout_sec=30,
            telemetry_svc=telemetry,
        )
        now = 1_700_000_000_000
        await db.execute(
            "INSERT INTO scans (path, kind, source, status, started_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path), "manual", "cli", "running", now, now, now),
        )
        task = await db.fetchone("SELECT * FROM scans LIMIT 1")

        assert await scan_svc.process_scan(task)
        body = await telemetry.preview_body()
        by_name = {metric["name"]: metric for metric in body["metrics"]}

        assert by_name["scan.finished.count"]["dimensions"] == {"status": "succeeded"}
        assert by_name["scan.files.count"]["value"] >= 1

    asyncio.run(_run())


def test_parse_wait_observation_uses_parse_rows_for_tier_and_pages_bucket(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        telemetry = TelemetryService(store)
        await telemetry.initialize()
        server = DoclibServer(SimpleNamespace(db=db, telemetry_svc=telemetry))
        now = 1_700_000_000_000
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("sha", "sha", 1, "pdf", 10, now, now),
        )
        await db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("sha", "standard", "1~3", "done", now, now),
        )
        row = await db.fetchone("SELECT id FROM parses WHERE sha256=?", ("sha",))

        await server.record_observations(
            TelemetryObservationsRequest(
                observations=[
                    TelemetryObservation(
                        metric_name="parse.wait",
                        parse_ids=[row["id"]],
                        duration_ms=1234,
                        dimensions={"status": "succeeded"},
                    )
                ]
            )
        )
        body = await telemetry.preview_body()
        by_name = {metric["name"]: metric for metric in body["metrics"]}

        assert by_name["parse.wait.count"]["dimensions"] == {
            "caller": "unknown",
            "pages_bucket": "2_5",
            "source": "unknown",
            "status": "succeeded",
            "tier": "standard",
        }
        assert by_name["parse.wait_duration_bucket.count"]["dimensions"]["bucket"] == "1_5s"

    asyncio.run(_run())
