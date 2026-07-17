from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest

from mineru.utils import models_download_utils
from mineru.utils.model_registry import MODEL_COMPLETE_MARKER, MINERU_2_5_PRO_2605_1_2B, ModelRepo, model_path_exists


def test_resolve_model_source_does_not_persist_env_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_download_utils.config.model, "source", "auto")
    monkeypatch.setattr(models_download_utils, "get_config_source", lambda path: "env")
    monkeypatch.setattr(models_download_utils, "_resolve_auto_model_source", lambda: "modelscope")

    def fail_persist(_model_source: str) -> None:
        raise AssertionError("env-provided auto must not be persisted")

    monkeypatch.setattr(models_download_utils, "_persist_resolved_model_source", fail_persist)

    assert models_download_utils.resolve_model_source(allow_auto=True) == "modelscope"


def test_resolve_model_source_can_treat_config_local_as_auto_for_explicit_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models_download_utils.config.model, "source", "local")
    monkeypatch.setattr(models_download_utils, "_resolve_auto_model_source", lambda: "huggingface")

    assert models_download_utils.resolve_model_source(allow_auto=True) == "local"
    assert models_download_utils.resolve_model_source(allow_auto=True, local_as_auto=True) == "huggingface"


def test_resolve_model_source_rejects_invalid_explicit_source() -> None:
    with pytest.raises(ValueError, match="Unsupported model source"):
        models_download_utils.resolve_model_source("invalid-source", allow_auto=True)


def test_download_model_repo_uses_required_path_patterns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="partial-repo",
        local_name="partial-repo",
        repos={"huggingface": "owner/partial-repo", "modelscope": "Owner/partial-repo"},
        paths={"weights": "models/weights"},
        download_mode="required_paths",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils, "resolve_model_source", lambda *args, **kwargs: "modelscope")

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> str:
        captured["source"] = model_source
        captured["repo"] = target_repo.name
        captured["patterns"] = patterns
        target = target_repo.local_dir() / "models/weights"
        target.mkdir(parents=True)
        (target / "model.bin").write_text("x", encoding="utf-8")
        return str(target_repo.local_dir())

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    root = models_download_utils.download_model_repo(repo, source="modelscope")

    assert root == repo.local_dir()
    assert captured == {
        "source": "modelscope",
        "repo": "partial-repo",
        "patterns": ["models/weights", "models/weights/*"],
    }
    assert (repo.local_dir() / "models/weights" / MODEL_COMPLETE_MARKER).is_file()


def test_download_model_repo_keeps_full_download_as_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="full-repo",
        local_name="full-repo",
        repos={"huggingface": "owner/full-repo", "modelscope": "Owner/full-repo"},
        paths={"config": "config.json"},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils, "resolve_model_source", lambda *args, **kwargs: "huggingface")

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> str:
        captured["source"] = model_source
        captured["repo"] = target_repo.name
        captured["patterns"] = patterns
        target = target_repo.local_dir() / "config.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        return str(target_repo.local_dir())

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    root = models_download_utils.download_model_repo(repo, source="huggingface")

    assert root == repo.local_dir()
    assert captured == {
        "source": "huggingface",
        "repo": "full-repo",
        "patterns": None,
    }
    assert (repo.local_dir() / MODEL_COMPLETE_MARKER).read_bytes() == b""


def test_vlm_repo_verify_requires_root_completion_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))

    repo_dir = MINERU_2_5_PRO_2605_1_2B.local_dir()
    repo_dir.mkdir(parents=True)
    assert MINERU_2_5_PRO_2605_1_2B.required_paths() == ()

    unmarked = models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B)

    assert unmarked.ready is False
    assert unmarked.missing_paths == [MODEL_COMPLETE_MARKER]

    (repo_dir / MODEL_COMPLETE_MARKER).touch()

    ready = models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B)

    assert ready.ready is True
    assert ready.missing_paths == []


def test_model_path_exists_requires_completion_marker_for_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights", "config": "config.json"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))

    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.bin").write_bytes(b"weights")
    assert model_path_exists(repo.weights) is False

    (weights_dir / MODEL_COMPLETE_MARKER).touch()
    assert model_path_exists(repo.weights) is True

    nested_weights = repo.weights.path("det")
    nested_weights.local_path().mkdir()
    (nested_weights.local_path() / "det.bin").write_bytes(b"weights")
    assert model_path_exists(nested_weights) is True

    config_path = repo.config.local_path()
    config_path.write_text("{}", encoding="utf-8")
    assert model_path_exists(repo.config) is True


def test_model_path_ensure_reuses_completed_directory_without_resolving_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.bin").write_bytes(b"weights")
    (weights_dir / MODEL_COMPLETE_MARKER).touch()

    def fail_resolve(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("ready model paths must not resolve a remote source")

    monkeypatch.setattr(models_download_utils, "resolve_model_source", fail_resolve)

    assert repo.weights.ensure() == weights_dir


def test_model_repo_ensure_reuses_completed_full_repo_without_resolving_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"config": "config.json", "weights": "model.bin"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    repo.local_dir().mkdir(parents=True)
    repo.config.local_path().write_text("{}", encoding="utf-8")
    repo.weights.local_path().write_bytes(b"weights")
    (repo.local_dir() / MODEL_COMPLETE_MARKER).touch()

    def fail_resolve(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("ready model repos must not resolve a remote source")

    monkeypatch.setattr(models_download_utils, "resolve_model_source", fail_resolve)

    assert repo.ensure() == repo.local_dir()


def test_model_path_ensure_repairs_unmarked_directory_and_writes_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "existing.bin").write_bytes(b"existing")
    calls = 0

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> str:
        nonlocal calls
        calls += 1
        assert model_source == "modelscope"
        assert patterns == ["models/weights", "models/weights/*"]
        (weights_dir / "missing.bin").write_bytes(b"repaired")
        return str(target_repo.local_dir())

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    assert repo.weights.ensure(source="modelscope") == weights_dir
    assert calls == 1
    assert (weights_dir / MODEL_COMPLETE_MARKER).is_file()


def test_local_source_rejects_unmarked_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils.config.model, "source", "local")
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "partial.bin").write_bytes(b"partial")

    with pytest.raises(FileNotFoundError, match="models/weights"):
        repo.weights.ensure()

    assert (weights_dir / MODEL_COMPLETE_MARKER).exists() is False


def test_download_failure_removes_stale_directory_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
        download_mode="required_paths",
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    marker = weights_dir / MODEL_COMPLETE_MARKER
    marker.touch()

    def fail_snapshot_download(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("download interrupted")

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fail_snapshot_download)

    with pytest.raises(RuntimeError, match="download interrupted"):
        models_download_utils.download_model_repo(repo, source="modelscope")

    assert marker.exists() is False


def test_full_download_failure_removes_root_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"config": "config.json"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    repo.local_dir().mkdir(parents=True)
    marker = repo.local_dir() / MODEL_COMPLETE_MARKER
    marker.touch()

    def fail_snapshot_download(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("download interrupted")

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fail_snapshot_download)

    with pytest.raises(RuntimeError, match="download interrupted"):
        models_download_utils.download_model_repo(repo, source="modelscope")

    assert marker.exists() is False


def test_nested_download_failure_invalidates_parent_directory_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    nested_weights = repo.weights.path("det")
    nested_weights.local_path().mkdir(parents=True)
    parent_marker = repo.weights.local_path() / MODEL_COMPLETE_MARKER
    parent_marker.touch()

    def fail_snapshot_download(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("download interrupted")

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fail_snapshot_download)

    with pytest.raises(RuntimeError, match="download interrupted"):
        models_download_utils.download_model_files(repo, [nested_weights], source="modelscope")

    assert parent_marker.exists() is False


def test_nested_download_success_restores_parent_directory_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    nested_weights = repo.weights.path("det")
    nested_weights.local_path().mkdir(parents=True)
    parent_marker = repo.weights.local_path() / MODEL_COMPLETE_MARKER
    parent_marker.touch()

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> str:
        assert model_source == "modelscope"
        assert target_repo is repo
        assert patterns == ["models/weights/det", "models/weights/det/*"]
        (nested_weights.local_path() / "det.bin").write_bytes(b"weights")
        return str(repo.local_dir())

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    models_download_utils.download_model_files(repo, [nested_weights], source="modelscope")

    assert parent_marker.is_file()
    assert (nested_weights.local_path() / MODEL_COMPLETE_MARKER).is_file()


def test_download_does_not_mark_metadata_only_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
        download_mode="required_paths",
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> str:
        assert model_source == "modelscope"
        assert target_repo is repo
        assert patterns == ["models/weights", "models/weights/*"]
        (weights_dir / ".cache/huggingface").mkdir(parents=True)
        (weights_dir / ".cache/huggingface/metadata").write_text("metadata", encoding="utf-8")
        (weights_dir / ".msc").write_text("metadata", encoding="utf-8")
        return str(repo.local_dir())

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="models/weights"):
        models_download_utils.download_model_repo(repo, source="modelscope")

    assert (weights_dir / MODEL_COMPLETE_MARKER).exists() is False


def test_huggingface_snapshot_rejects_missing_expected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = ModelRepo(
        name="repo",
        local_name="repo",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"weights": "models/weights"},
    )
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))

    class FakeApi:
        def list_repo_files(self, repo_id: str) -> list[str]:
            assert repo_id == "owner/repo"
            return ["models/weights/config.json", "models/weights/model.bin"]

    monkeypatch.setattr(models_download_utils, "HfApi", FakeApi)

    def fake_hf_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        assert repo_id == "owner/repo"
        target = Path(kwargs["local_dir"]) / "models/weights"
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return kwargs["local_dir"]

    monkeypatch.setattr(models_download_utils, "hf_snapshot_download", fake_hf_snapshot_download)

    with pytest.raises(FileNotFoundError, match="models/weights/model.bin"):
        models_download_utils._snapshot_download(
            "huggingface",
            repo,
            ["models/weights", "models/weights/*"],
        )
