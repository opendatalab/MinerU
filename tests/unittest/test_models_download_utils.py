from __future__ import annotations

import pytest

from mineru.utils import models_download_utils


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
