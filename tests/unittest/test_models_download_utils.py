from __future__ import annotations

from pathlib import Path

import pytest

from mineru.utils import models_download_utils
from mineru.utils.model_registry import MINERU_2_5_PRO_2605_1_2B, ModelRepo


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


def test_vlm_repo_verify_requires_core_model_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models_download_utils.config.model, "base_dir", str(tmp_path / "models"))

    repo_dir = MINERU_2_5_PRO_2605_1_2B.local_dir()
    repo_dir.mkdir(parents=True)
    for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (repo_dir / filename).write_text("{}", encoding="utf-8")

    missing_weight = models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B)

    assert missing_weight.ready is False
    assert missing_weight.missing_paths == ["model.safetensors"]

    (repo_dir / "model.safetensors").write_bytes(b"weights")

    ready = models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B)

    assert ready.ready is True
    assert ready.missing_paths == []
