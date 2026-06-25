from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


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

    assert version == "4.0.0.dev20260625"


def test_write_and_read_release_state(tmp_path: Path) -> None:
    module = _load_release_module()
    state_file = tmp_path / "state.json"

    module.write_state(
        state_file,
        package="mineru-next-dev",
        version="4.0.0.dev20260625",
        source_branch="next",
        source_commit="abc123",
        published_at="2026-06-25T02:20:00Z",
    )

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "abc123"

    state = module.read_state(state_file)
    assert state is not None
    assert state.package == "mineru-next-dev"
    assert state.version == "4.0.0.dev20260625"
    assert state.source_branch == "next"
    assert state.source_commit == "abc123"
    assert state.published_at == "2026-06-25T02:20:00Z"


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
