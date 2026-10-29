"""验证应用配置、公式渲染、探测状态和托管进程的实例归属。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mineru import config as configuration
from mineru.config import Config, LogConfig, ManagedParseServerConfig
from mineru.doclib import app as doclib_app
from mineru.doclib.background import parse_server_health as probes
from mineru.doclib.core.fts import strip_sep
from mineru.doclib.core.parse_artifacts import parse_batch_json_path, write_parse_payload
from mineru.doclib.parse_server_state import ParseServerHealth, ProbeResult, ProbeState
from mineru.doclib.services.parse_errors import ParseFailure
from mineru.doclib.services.parse_policy import resolve_default_tier
from mineru.types import BlockType, EquationBlock, MiddleJson, PageInfo


async def _seed_formula(state: Any, root: Path, sha: str) -> None:
    """登记两个可读档位的合法缓存，供读取、导出和失效索引回退共同消费。"""
    source = root / "formula.pdf"
    source.write_bytes(b"cached source")
    await state.db.execute(
        "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
        "VALUES (?, ?, 13, 'pdf', 1, 1, 1)",
        (sha, sha[:7]),
    )
    await state.db.execute(
        "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
        "VALUES (?, 'formula.pdf', 'pdf', 13, 1, ?, 'active', 1, 1)",
        (str(source), sha),
    )
    middle = MiddleJson(
        is_full_document=True,
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[EquationBlock(type=BlockType.EQUATION, index=0, bbox=(0.1, 0.1, 0.9, 0.2), content="E=mc^2")],
            )
        ],
        metadata={"file_suffix": "pdf", "producer": {"name": "test", "version": "test"}},
    )
    for tier, done_at in (("flash", 1000), ("standard", 2000)):
        path = Path(parse_batch_json_path(state.data_dir, sha, tier, "1", done_at))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parse_payload(path, middle.model_dump(mode="json", by_alias=True))
        await state.db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, done_at, created_at, updated_at) "
            "VALUES (?, ?, '1', 'done', ?, 1, 1)",
            (sha, tier, done_at),
        )


def test_two_apps_keep_rendering_and_health_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """相同文档按各自配置读取、导出和重建 FTS，关闭一个应用不改变另一个健康状态。"""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path / "home"))

    def skip_background(*args: object, **kwargs: object) -> None:
        """保留完整实例组装，仅固定本用例的后台调度时序。"""

    def forbidden_default() -> Config:
        """显式宿主配置路径不得再读取进程默认配置。"""
        raise AssertionError("Hosted code read process configuration")

    monkeypatch.setattr(doclib_app, "_create_background_task", skip_background)
    configs = []
    for label in ("A", "B"):
        root = tmp_path / label
        cfg = Config()
        cfg.doclib.data_dir = str(root / "data")
        cfg.doclib.sqlite.path = str(root / "library.sqlite")
        cfg.render.latex_delimiters.display.left = f"LEFT-{label}"
        cfg.render.latex_delimiters.display.right = f"RIGHT-{label}"
        configs.append(cfg)
    monkeypatch.setattr(configuration, "get_config", forbidden_default)
    monkeypatch.setattr(doclib_app, "get_config", forbidden_default)
    monkeypatch.setattr(probes, "get_config", forbidden_default)
    first_app, second_app = [doclib_app.create_app(cfg) for cfg in configs]
    configs[0].render.latex_delimiters.display.left = "changed outside application"
    sha = "a" * 64
    with TestClient(second_app) as second:
        with TestClient(first_app) as first:
            clients = (("A", first), ("B", second))
            for label, client in clients:
                state = client.app.state.doclib_state
                assert state.health is state.health_check.health is state.ingest_svc.health
                assert state.health is state.parse_queue_svc.health is state.parse_execution_svc.health
                client.portal.call(_seed_formula, state, tmp_path / label, sha)
                state.health.local_mode = "self_hosted"
                state.health.local = ProbeState(
                    probe=ProbeResult(healthy=True, tiers=["basic" if label == "A" else "standard"])
                )
                client.portal.call(state.config_svc.set, "parse_server.local.mode", "self_hosted")
                content = client.get(f"/api/v1/docs/{sha}/content", params={"tier": "flash"})
                assert content.status_code == 200, content.text
                assert f"LEFT-{label}\nE=mc^2\nRIGHT-{label}" in content.json()["content"]
                output = tmp_path / label / "formula.md"
                exported = client.post(f"/api/v1/docs/{sha}/exports", json={"tier": "flash", "output": str(output)})
                assert exported.status_code == 200, exported.text
                assert f"LEFT-{label}" in output.read_text()
                invalidated = client.post("/api/v1/invalidate", json={"doc_ref": sha, "tier": "standard"})
                assert invalidated.status_code == 200, invalidated.text
                row = client.portal.call(state.db.fetchone, "SELECT tier,text FROM fts_contents WHERE sha256=?", (sha,))
                assert row["tier"] == "flash" and f"LEFT-{label}" in strip_sep(row["text"])
                status = client.get("/api/v1/server/status")
                assert status.status_code == 200, status.text
                assert status.json()["parse_server"]["local"]["healthy"]
            left, right = first_app.state.doclib_state, second_app.state.doclib_state
            assert left.health is not right.health
            assert resolve_default_tier(health=left.health) == "basic"
            assert resolve_default_tier(health=right.health) == "standard"
            with pytest.raises(ParseFailure) as error:
                right.parse_execution_svc._resolve_tier("basic", "local")
            assert error.value.code == "tier_mismatch"
        assert right.health.local.probe.healthy
        assert resolve_default_tier(health=right.health) == "standard"
        assert second.get(f"/api/v1/docs/{sha}/content", params={"tier": "flash"}).status_code == 200


def test_health_checker_stop_owns_only_its_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """停止一个探测器只清理对应托管进程，不接触另一个实例持有的进程。"""
    stopped: list[object] = []

    def stop_process(process: object, **kwargs: object) -> None:
        """记录实际交给进程清理入口的对象。"""
        stopped.append(process)

    monkeypatch.setattr(probes, "stop_managed_parse_server", stop_process)
    processes = [SimpleNamespace(pid=index) for index in (1, 2)]
    states = [ParseServerHealth(managed_proc=process) for process in processes]
    checkers = [
        probes.ParseServerHealthCheck(
            None,
            health=state,
            interval_sec=30,
            probe_timeout_sec=1,
            startup_grace_sec=30,
            startup_timeout_sec=60,
            stop_timeout_sec=1,
            managed_parse_server=ManagedParseServerConfig(),
            log_cfg=LogConfig(dir=str(tmp_path)),
        )
        for state in states
    ]
    asyncio.run(checkers[0].stop())
    assert stopped == [processes[0]] and states[0].managed_proc is None
    assert states[1].managed_proc is processes[1]
    asyncio.run(checkers[1].stop())
    assert stopped == processes and states[1].managed_proc is None
