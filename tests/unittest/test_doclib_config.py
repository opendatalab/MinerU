from __future__ import annotations

import pytest

from mineru.config import (
    Config,
    PatchedConfig,
    apply_env_overrides,
    get_env,
    interpolate_env,
    load_config,
)


def test_interpolate_env_supports_required_and_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TEST_ROOT", "/tmp/mineru-test")

    data = interpolate_env(
        {
            "doclib": {
                "data_dir": "${MINERU_TEST_ROOT}/data",
                "log": {"path": "${MISSING_LOG:-'/tmp/default.log'}"},
            },
            "paths": ["${MINERU_TEST_ROOT}/a", "${MISSING_PATH:-/tmp/b}"],
        }
    )

    assert data == {
        "doclib": {
            "data_dir": "/tmp/mineru-test/data",
            "log": {"path": "/tmp/default.log"},
        },
        "paths": ["/tmp/mineru-test/a", "/tmp/b"],
    }


def test_interpolate_env_rejects_missing_required_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_TEST_MISSING", raising=False)

    with pytest.raises(ValueError, match="MINERU_TEST_MISSING"):
        interpolate_env("${MINERU_TEST_MISSING}")


def test_load_config_reads_yaml_and_interpolates_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TEST_DATA", str(tmp_path / "data"))
    config_file = tmp_path / "mineru.yaml"
    config_file.write_text(
        """
doclib:
  data_dir: ${MINERU_TEST_DATA}
  http:
    enabled: true
    port: 18080
  sqlite:
    path: ${MINERU_TEST_DATA}/mineru.db
""",
        encoding="utf-8",
    )

    data = load_config(str(config_file))

    assert data["doclib"]["data_dir"] == str(tmp_path / "data")
    assert data["doclib"]["http"]["enabled"] is True
    assert data["doclib"]["http"]["port"] == 18080
    assert data["doclib"]["sqlite"]["path"] == str(tmp_path / "data" / "mineru.db")


def test_apply_env_overrides_uses_greedy_field_path_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("TEST_MINERU_DOCLIB_HTTP_ENABLED", "true")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_HTTP_PORT", "15990")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_COMPACTION_INTERVAL_SEC", "5")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_MMAP_SIZE", "0")
    monkeypatch.setenv("TEST_MINERU_UNKNOWN_FIELD", "ignored")
    monkeypatch.setenv("TEST_MINERU_CONFIG", "/tmp/ignored.yaml")

    cfg = apply_env_overrides(Config(), prefix=prefix)

    assert cfg.doclib.http.enabled is True
    assert cfg.doclib.http.port == 15990
    assert cfg.doclib.compaction_interval_sec == 5
    assert cfg.doclib.sqlite.mmap_size == 0


def test_get_env_returns_default_and_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_TEST_ENV", raising=False)

    assert get_env("MINERU_TEST_ENV", "fallback") == "fallback"
    with pytest.raises(ValueError, match="MINERU_TEST_ENV"):
        get_env("MINERU_TEST_ENV")


def test_patched_config_returns_validated_deep_patch() -> None:
    cfg = PatchedConfig(doclib={"http": {"port": "16000"}, "sqlite": {"cache_size": "-1"}})

    assert cfg.doclib.http.port == 16000
    assert cfg.doclib.sqlite.cache_size == -1
