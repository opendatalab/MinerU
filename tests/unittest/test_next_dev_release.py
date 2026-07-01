from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _load_release_module():
    module_path = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "next_dev_release.py"
    spec = importlib.util.spec_from_file_location("next_dev_release", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_version_uses_utc_date() -> None:
    module = _load_release_module()

    version = module.build_version("4.0.0", datetime(2026, 6, 25, 1, 2, 3, tzinfo=UTC))

    assert version == "4.0.0.dev2026062500"


def test_build_version_uses_daily_sequence() -> None:
    module = _load_release_module()

    version = module.build_version("4.0.0", datetime(2026, 6, 25, 1, 2, 3, tzinfo=UTC), sequence=3)

    assert version == "4.0.0.dev2026062503"


def test_build_version_rejects_sequence_that_breaks_date_ordering() -> None:
    module = _load_release_module()

    with pytest.raises(ValueError, match="between 0 and 99"):
        module.build_version("4.0.0", datetime(2026, 6, 25, 1, 2, 3, tzinfo=UTC), sequence=100)


def test_write_and_read_release_state(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"

    module.write_state(
        state_file,
        package="mineru-next-dev",
        version="4.0.0.dev2026062500",
        source_branch="next",
        source_commit="abc123",
        trigger="schedule",
        release_date="20260625",
        sequence=0,
        published_at="2026-06-25T02:20:00Z",
    )

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "abc123"
    assert payload["last_auto_date"] == "2026-06-25"
    assert payload["daily_sequences"] == {"20260625": 0}
    assert payload["releases"][0]["trigger"] == "schedule"

    state = module.read_state(state_file)
    assert state is not None
    assert state.package == "mineru-next-dev"
    assert state.version == "4.0.0.dev2026062500"
    assert state.source_branch == "next"
    assert state.source_commit == "abc123"
    assert state.published_at == "2026-06-25T02:20:00Z"
    assert state.last_auto_date == "2026-06-25"
    assert state.daily_sequences == {"20260625": 0}
    assert state.releases[0].source_commit == "abc123"


def test_read_release_state_accepts_legacy_shape(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "package": "mineru-next-dev",
                "version": "4.0.0.dev20260625",
                "source_branch": "next",
                "source_commit": "abc123",
                "published_at": "2026-06-25T02:20:00Z",
            }
        ),
        encoding="utf-8",
    )

    state = module.read_state(state_file)

    assert state is not None
    assert state.version == "4.0.0.dev20260625"
    assert state.last_auto_date is None
    assert state.daily_sequences == {}
    assert state.releases == []


def test_plan_release_initial_publish_uses_first_daily_sequence() -> None:
    module = _load_release_module()

    plan = module.plan_release(
        None,
        base_version="4.0.0",
        source_commit="abc123",
        trigger="schedule",
        now=datetime(2026, 6, 25, 1, 2, 3, tzinfo=UTC),
    )

    assert plan.should_publish is True
    assert plan.version == "4.0.0.dev2026062500"
    assert plan.release_date == "20260625"
    assert plan.sequence == 0


def test_plan_release_skips_second_automatic_release_on_same_utc_date(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"
    module.write_state(
        state_file,
        package="mineru-next-dev",
        version="4.0.0.dev2026062500",
        source_branch="next",
        source_commit="abc123",
        trigger="schedule",
        release_date="20260625",
        sequence=0,
        published_at="2026-06-25T02:20:00Z",
    )

    plan = module.plan_release(
        module.read_state(state_file),
        base_version="4.0.0",
        source_commit="def456",
        trigger="schedule",
        now=datetime(2026, 6, 25, 8, 0, 0, tzinfo=UTC),
    )

    assert plan.should_publish is False
    assert plan.reason == "automatic release already ran on 2026-06-25"


def test_plan_release_skips_manual_publish_for_already_published_commit(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"
    module.write_state(
        state_file,
        package="mineru-next-dev",
        version="4.0.0.dev2026062500",
        source_branch="next",
        source_commit="abc123",
        trigger="workflow_dispatch",
        release_date="20260625",
        sequence=0,
        published_at="2026-06-25T02:20:00Z",
    )

    plan = module.plan_release(
        module.read_state(state_file),
        base_version="4.0.0",
        source_commit="abc123",
        trigger="workflow_dispatch",
        now=datetime(2026, 6, 25, 8, 0, 0, tzinfo=UTC),
    )

    assert plan.should_publish is False
    assert plan.reason == "source commit was already published"


def test_plan_release_allows_forced_manual_publish_for_same_commit(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"
    module.write_state(
        state_file,
        package="mineru-next-dev",
        version="4.0.0.dev2026062500",
        source_branch="next",
        source_commit="abc123",
        trigger="workflow_dispatch",
        release_date="20260625",
        sequence=0,
        published_at="2026-06-25T02:20:00Z",
    )

    plan = module.plan_release(
        module.read_state(state_file),
        base_version="4.0.0",
        source_commit="abc123",
        trigger="workflow_dispatch",
        force_publish=True,
        now=datetime(2026, 6, 25, 8, 0, 0, tzinfo=UTC),
    )

    assert plan.should_publish is True
    assert plan.version == "4.0.0.dev2026062501"
    assert plan.sequence == 1


def test_patch_workspace_updates_package_name_and_version(tmp_path: Path) -> None:
    module = _load_release_module()
    pyproject_path = tmp_path / "pyproject.toml"
    version_file_path = tmp_path / "version.py"

    pyproject_path.write_text(
        '[project]\nname = "mineru"\ndynamic = ["version"]\n\n[tool.example]\nvalue = 1\n',
        encoding="utf-8",
    )
    version_file_path.write_text('__version__ = "3.2.1"\n', encoding="utf-8")

    module.patch_workspace(
        pyproject_path=pyproject_path,
        version_file_path=version_file_path,
        package_name="mineru-next-dev",
        version="4.0.0.dev20260625",
    )

    assert 'name = "mineru-next-dev"' in pyproject_path.read_text(encoding="utf-8")
    assert version_file_path.read_text(encoding="utf-8") == '__version__ = "4.0.0.dev20260625"\n'
