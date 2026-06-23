from __future__ import annotations

import pytest

from mineru.config import (
    Config,
    PatchedConfig,
    _apply_env_overrides,
    _default_config_path,
    _default_data_path,
    _default_uds_path,
    _interpolate_env,
    _load_config,
    _read_config,
    resolve_tcp_enabled,
    resolve_uds_enabled,
)
from mineru.doclib.config_defaults import CONFIG_DEFAULTS


def test_interpolate_env_supports_required_and_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TEST_ROOT", "/tmp/mineru-test")

    data = _interpolate_env(
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
        _interpolate_env("${MINERU_TEST_MISSING}")


def test_load_config_reads_yaml_and_interpolates_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TEST_DATA", str(tmp_path / "data"))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
doclib:
  data_dir: ${MINERU_TEST_DATA}
  tcp:
    enabled: true
    port: 18080
  sqlite:
    path: ${MINERU_TEST_DATA}/doclib.db
""",
        encoding="utf-8",
    )

    data = _load_config(str(config_file))

    assert data["doclib"]["data_dir"] == str(tmp_path / "data")
    assert data["doclib"]["tcp"]["enabled"] is True
    assert data["doclib"]["tcp"]["port"] == 18080
    assert data["doclib"]["sqlite"]["path"] == str(tmp_path / "data" / "doclib.db")


def test_apply_env_overrides_uses_greedy_field_path_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("TEST_MINERU_DOCLIB_TCP_ENABLED", "true")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_TCP_PORT", "15990")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_COMPACTION_INTERVAL_SEC", "5")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SCAN_INTERVAL_SEC", "7")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_DEVICE_CHECK_INTERVAL_SEC", "11")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_INGEST_LOCK_TIMEOUT_SEC", "13")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_LOCK_TIMEOUT_SEC", "17")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SCAN_LOCK_TIMEOUT_SEC", "19")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_HEALTH_CHECK_INTERVAL_SEC", "23")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_PROBE_TIMEOUT_SEC", "29")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_STARTUP_GRACE_SEC", "31")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_STOP_TIMEOUT_SEC", "37")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_MMAP_SIZE", "0")
    monkeypatch.setenv("TEST_MINERU_UNKNOWN_FIELD", "ignored")
    monkeypatch.setenv("TEST_MINERU_CONFIG", "/tmp/ignored.yaml")

    cfg = _apply_env_overrides(Config(), prefix=prefix)

    assert cfg.doclib.tcp.enabled is True
    assert cfg.doclib.tcp.port == 15990
    assert cfg.doclib.compaction_interval_sec == 5
    assert cfg.doclib.scan_interval_sec == 7
    assert cfg.doclib.device_check_interval_sec == 11
    assert cfg.doclib.ingest_lock_timeout_sec == 13
    assert cfg.doclib.parse_lock_timeout_sec == 17
    assert cfg.doclib.scan_lock_timeout_sec == 19
    assert cfg.doclib.parse_server_health_check_interval_sec == 23
    assert cfg.doclib.parse_server_probe_timeout_sec == 29
    assert cfg.doclib.parse_server_startup_grace_sec == 31
    assert cfg.doclib.parse_server_stop_timeout_sec == 37
    assert cfg.doclib.sqlite.mmap_size == 0


def test_default_paths_derive_from_mineru_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_HOME", "/tmp/mineru-home")

    assert _default_data_path() == "/tmp/mineru-home/data"
    assert _default_config_path() == "/tmp/mineru-home/config.yaml"


def test_read_config_uses_default_config_under_mineru_home(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    mineru_home = tmp_path / "mineru-home"
    mineru_home.mkdir()
    monkeypatch.setenv("MINERU_HOME", str(mineru_home))
    monkeypatch.delenv("MINERU_CONFIG", raising=False)
    (mineru_home / "config.yaml").write_text(
        """
doclib:
  data_dir: /tmp/ignored-data-dir
  tcp:
    port: 18080
""",
        encoding="utf-8",
    )

    data = _read_config()

    assert data["doclib"]["tcp"]["port"] == 18080
    assert data["doclib"]["data_dir"] == "/tmp/ignored-data-dir"


def test_apply_env_overrides_can_override_doclib_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("MINERU_HOME", "/tmp/mineru-home")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_DATA_DIR", "/tmp/ignored-data-dir")

    cfg = _apply_env_overrides(Config(), prefix=prefix)

    assert cfg.doclib.data_dir == "/tmp/ignored-data-dir"


def test_default_uds_path_uses_mineru_home_on_unix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config.platform.system", lambda: "Linux")
    monkeypatch.setenv("MINERU_HOME", "/tmp/mineru-home")

    assert _default_uds_path() == "/tmp/mineru-home/doclib.sock"


def test_default_uds_path_uses_mineru_home_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config.platform.system", lambda: "Windows")
    monkeypatch.setenv("MINERU_HOME", r"C:\MinerU")

    assert _default_uds_path() == r"C:\MinerU/doclib.sock"


def test_default_transport_prefers_uds_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)

    cfg = Config()

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert resolve_uds_enabled(cfg) is True
    assert resolve_tcp_enabled(cfg) is False


def test_default_transport_falls_back_to_tcp_when_uds_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: False)

    cfg = Config()

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert resolve_uds_enabled(cfg) is False
    assert resolve_tcp_enabled(cfg) is True


def test_transport_enabled_accepts_auto_and_explicit_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)

    auto_cfg = Config(doclib={"uds": {"enabled": "auto"}, "tcp": {"enabled": "auto"}})
    explicit_cfg = Config(doclib={"uds": {"enabled": False}, "tcp": {"enabled": True}})

    assert auto_cfg.doclib.uds.enabled == "auto"
    assert auto_cfg.doclib.tcp.enabled == "auto"
    assert resolve_uds_enabled(auto_cfg) is True
    assert resolve_tcp_enabled(auto_cfg) is False
    assert explicit_cfg.doclib.uds.enabled is False
    assert explicit_cfg.doclib.tcp.enabled is True
    assert resolve_uds_enabled(explicit_cfg) is False
    assert resolve_tcp_enabled(explicit_cfg) is True


def test_transport_enabled_accepts_auto_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)
    monkeypatch.setenv("TEST_MINERU_DOCLIB_UDS_ENABLED", "auto")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_TCP_ENABLED", "auto")

    cfg = _apply_env_overrides(Config(doclib={"uds": {"enabled": False}, "tcp": {"enabled": True}}), prefix=prefix)

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert resolve_uds_enabled(cfg) is True
    assert resolve_tcp_enabled(cfg) is False


def test_patched_config_returns_validated_deep_patch() -> None:
    cfg = PatchedConfig(doclib={"tcp": {"port": "16000"}, "sqlite": {"cache_size": "-1"}})

    assert cfg.doclib.tcp.port == 16000
    assert cfg.doclib.sqlite.cache_size == -1


def test_interval_and_timeout_config_is_startup_config_not_runtime_kv() -> None:
    startup_only_keys = {
        "default_tier",
        "scan_interval_sec",
        "device_check_interval_sec",
        "ingest_lock_timeout_sec",
        "parse_lock_timeout_sec",
        "scan_lock_timeout_sec",
        "compaction_interval_sec",
        "parse_server_health_check_interval_sec",
        "parse_server_probe_timeout_sec",
        "parse_server_startup_grace_sec",
        "parse_server_stop_timeout_sec",
    }

    assert startup_only_keys.isdisjoint(CONFIG_DEFAULTS)
