from __future__ import annotations

from pathlib import Path

import pytest

from mineru.config import (
    Config,
    LoadedConfig,
    LogConfig,
    PatchedConfig,
    _collect_env_overrides,
    _interpolate_env,
    _load_config,
    _load_effective_config,
    update_config_file,
)
from mineru.doclib.config_defaults import CONFIG_DEFAULTS


def test_interpolate_env_supports_required_and_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_TEST_ROOT", "/tmp/mineru-test")

    data = _interpolate_env(
        {
            "doclib": {
                "data_dir": "${MINERU_TEST_ROOT}/data",
                "log": {"app_path": "${MISSING_LOG:-'/tmp/default.log'}"},
            },
            "paths": ["${MINERU_TEST_ROOT}/a", "${MISSING_PATH:-/tmp/b}"],
        }
    )

    assert data == {
        "doclib": {
            "data_dir": "/tmp/mineru-test/data",
            "log": {"app_path": "/tmp/default.log"},
        },
        "paths": ["/tmp/mineru-test/a", "/tmp/b"],
    }


def test_interpolate_env_rejects_missing_required_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_TEST_MISSING", raising=False)

    with pytest.raises(ValueError, match="MINERU_TEST_MISSING"):
        _interpolate_env("${MINERU_TEST_MISSING}")


def test_load_config_reads_yaml_and_interpolates_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_STARTUP_TIMEOUT_SEC", "601")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_PARSE_SERVER_STOP_TIMEOUT_SEC", "37")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_BUSY_TIMEOUT_MS", "1000")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_LOCK_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_LOCK_RETRY_BASE_DELAY_MS", "25")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_SQLITE_MMAP_SIZE", "0")
    monkeypatch.setenv("TEST_MINERU_UNKNOWN_FIELD", "ignored")
    monkeypatch.setenv("TEST_MINERU_CONFIG", "/tmp/ignored.yaml")

    overrides, paths = _collect_env_overrides(prefix=prefix)
    cfg = Config(**overrides)

    assert ("config",) not in paths
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
    assert cfg.doclib.parse_server_startup_timeout_sec == 601
    assert cfg.doclib.parse_server_stop_timeout_sec == 37
    assert cfg.doclib.sqlite.busy_timeout_ms == 1000
    assert cfg.doclib.sqlite.lock_retry_attempts == 4
    assert cfg.doclib.sqlite.lock_retry_base_delay_ms == 25
    assert cfg.doclib.sqlite.mmap_size == 0


def test_read_config_uses_default_config_under_mineru_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    loaded = _load_effective_config()

    assert loaded.config_file == str(mineru_home / "config.yaml")
    assert loaded.config_file_exists is True
    assert loaded.config.doclib.tcp.port == 18080
    assert loaded.config.doclib.data_dir == "/tmp/ignored-data-dir"


def test_default_config_path_exists_even_when_file_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mineru_home = tmp_path / "mineru-home"
    mineru_home.mkdir()
    monkeypatch.setenv("MINERU_HOME", str(mineru_home))
    monkeypatch.delenv("MINERU_CONFIG", raising=False)

    loaded = _load_effective_config()

    assert loaded.config_file == str(mineru_home / "config.yaml")
    assert loaded.config_file_exists is False
    assert loaded.config == Config()


def test_apply_env_overrides_can_override_doclib_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("MINERU_HOME", "/tmp/mineru-home")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_DATA_DIR", "/tmp/ignored-data-dir")

    overrides, _paths = _collect_env_overrides(prefix=prefix)
    cfg = Config(**overrides)

    assert cfg.doclib.data_dir == "/tmp/ignored-data-dir"


def test_default_doclib_data_dir_uses_doclib_directory() -> None:
    cfg = Config()

    assert cfg.doclib.data_dir.endswith(".mineru/doclib")


def test_default_model_config_uses_mineru_model_directory() -> None:
    cfg = Config()

    assert cfg.model.base_dir.endswith(".mineru/models")
    assert cfg.model.source == "auto"


def test_load_effective_config_tracks_file_and_env_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
model:
  base_dir: /tmp/file-models
  source: huggingface
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MINERU_CONFIG", str(config_file))
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "modelscope")

    loaded = _load_effective_config()

    assert loaded.config.model.base_dir == "/tmp/file-models"
    assert loaded.config.model.source == "modelscope"
    assert loaded.sources[("model", "base_dir")] == "file"
    assert loaded.sources[("model", "source")] == "env"


def test_update_config_file_deep_merges_with_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
doclib:
  tcp:
    port: 18080
model:
  base_dir: /tmp/models
  source: auto
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mineru.config._loaded_config",
        LoadedConfig(config=Config(), sources={}, config_file=str(config_file), config_file_exists=True),
    )

    update_config_file({"model": {"source": "huggingface"}})

    data = _load_config(str(config_file))
    assert data["doclib"]["tcp"]["port"] == 18080
    assert data["model"] == {"base_dir": "/tmp/models", "source": "huggingface"}
    assert not list(tmp_path.glob(".config.*.tmp"))


def test_default_transport_prefers_uds_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)

    cfg = Config()

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert cfg.doclib.resolved_uds_enabled is True
    assert cfg.doclib.resolved_tcp_enabled is False


def test_default_transport_falls_back_to_tcp_when_uds_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: False)

    cfg = Config()

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert cfg.doclib.resolved_uds_enabled is False
    assert cfg.doclib.resolved_tcp_enabled is True


def test_transport_enabled_accepts_auto_and_explicit_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)

    auto_cfg = Config(doclib={"uds": {"enabled": "auto"}, "tcp": {"enabled": "auto"}})
    explicit_cfg = Config(doclib={"uds": {"enabled": False}, "tcp": {"enabled": True}})

    assert auto_cfg.doclib.uds.enabled == "auto"
    assert auto_cfg.doclib.tcp.enabled == "auto"
    assert auto_cfg.doclib.resolved_uds_enabled is True
    assert auto_cfg.doclib.resolved_tcp_enabled is False
    assert explicit_cfg.doclib.uds.enabled is False
    assert explicit_cfg.doclib.tcp.enabled is True
    assert explicit_cfg.doclib.resolved_uds_enabled is False
    assert explicit_cfg.doclib.resolved_tcp_enabled is True


def test_transport_enabled_accepts_auto_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setattr("mineru.config._uds_available", lambda: True)
    monkeypatch.setenv("TEST_MINERU_DOCLIB_UDS_ENABLED", "auto")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_TCP_ENABLED", "auto")

    overrides, _paths = _collect_env_overrides(prefix=prefix)
    cfg = Config(**overrides)

    assert cfg.doclib.uds.enabled == "auto"
    assert cfg.doclib.tcp.enabled == "auto"
    assert cfg.doclib.resolved_uds_enabled is True
    assert cfg.doclib.resolved_tcp_enabled is False


def test_patched_config_returns_validated_deep_patch() -> None:
    cfg = PatchedConfig(doclib={"tcp": {"port": "16000"}, "sqlite": {"cache_size": "-1"}})

    assert cfg.doclib.tcp.port == 16000
    assert cfg.doclib.sqlite.cache_size == -1


def test_tcp_config_exposes_port_probe_count() -> None:
    cfg = Config(doclib={"tcp": {"port_probe_count": "5"}})

    assert Config().doclib.tcp.port_probe_count == 100
    assert cfg.doclib.tcp.port_probe_count == 5


def test_managed_parse_server_config_is_startup_config() -> None:
    cfg = Config(
        doclib={
            "managed_parse_server": {
                "host": "127.0.0.2",
                "port": "16581",
                "strict_port": "true",
                "port_probe_count": "5",
            }
        }
    )

    assert Config().doclib.managed_parse_server.host == "127.0.0.1"
    assert Config().doclib.managed_parse_server.port == 16580
    assert Config().doclib.managed_parse_server.strict_port is False
    assert Config().doclib.managed_parse_server.port_probe_count == 100
    assert cfg.doclib.managed_parse_server.host == "127.0.0.2"
    assert cfg.doclib.managed_parse_server.port == 16581
    assert cfg.doclib.managed_parse_server.strict_port is True
    assert cfg.doclib.managed_parse_server.port_probe_count == 5


def test_log_config_exposes_separate_log_paths() -> None:
    defaults = LogConfig()

    assert defaults.dir.endswith("logs")
    assert defaults.app_path is None
    assert defaults.access_path is None
    assert defaults.stdout_path is None
    assert defaults.stderr_path is None
    assert defaults.parse_server_stdout_path is None
    assert defaults.parse_server_stderr_path is None
    assert defaults.resolved_app_path.endswith("logs/doclib.log")
    assert defaults.resolved_access_path.endswith("logs/doclib.access.log")
    assert defaults.resolved_stdout_path.endswith("logs/doclib.stdout.log")
    assert defaults.resolved_stderr_path.endswith("logs/doclib.stderr.log")
    assert defaults.resolved_parse_server_stdout_path.endswith("logs/doclib.parse-server.stdout.log")
    assert defaults.resolved_parse_server_stderr_path.endswith("logs/doclib.parse-server.stderr.log")

    cfg = Config(
        doclib={
            "log": {
                "dir": "/tmp/mineru-logs",
                "app_path": "/tmp/app.log",
                "access_path": "/tmp/access.log",
                "stdout_path": "/tmp/stdout.log",
                "stderr_path": "/tmp/stderr.log",
                "parse_server_stdout_path": "/tmp/parse-server.stdout.log",
                "parse_server_stderr_path": "/tmp/parse-server.stderr.log",
            }
        }
    )

    assert cfg.doclib.log.dir == "/tmp/mineru-logs"
    assert cfg.doclib.log.app_path == "/tmp/app.log"
    assert cfg.doclib.log.access_path == "/tmp/access.log"
    assert cfg.doclib.log.stdout_path == "/tmp/stdout.log"
    assert cfg.doclib.log.stderr_path == "/tmp/stderr.log"
    assert cfg.doclib.log.parse_server_stdout_path == "/tmp/parse-server.stdout.log"
    assert cfg.doclib.log.parse_server_stderr_path == "/tmp/parse-server.stderr.log"
    assert cfg.doclib.log.resolved_app_path == "/tmp/app.log"
    assert cfg.doclib.log.resolved_access_path == "/tmp/access.log"
    assert cfg.doclib.log.resolved_stdout_path == "/tmp/stdout.log"
    assert cfg.doclib.log.resolved_stderr_path == "/tmp/stderr.log"
    assert cfg.doclib.log.resolved_parse_server_stdout_path == "/tmp/parse-server.stdout.log"
    assert cfg.doclib.log.resolved_parse_server_stderr_path == "/tmp/parse-server.stderr.log"


def test_log_config_dir_derives_unspecified_log_paths() -> None:
    cfg = LogConfig(dir="/tmp/mineru-logs", stderr_path="/tmp/custom-stderr.log")

    assert cfg.app_path is None
    assert cfg.access_path is None
    assert cfg.stdout_path is None
    assert cfg.stderr_path == "/tmp/custom-stderr.log"
    assert cfg.resolved_app_path == "/tmp/mineru-logs/doclib.log"
    assert cfg.resolved_access_path == "/tmp/mineru-logs/doclib.access.log"
    assert cfg.resolved_stdout_path == "/tmp/mineru-logs/doclib.stdout.log"
    assert cfg.resolved_stderr_path == "/tmp/custom-stderr.log"
    assert cfg.resolved_parse_server_stdout_path == "/tmp/mineru-logs/doclib.parse-server.stdout.log"
    assert cfg.resolved_parse_server_stderr_path == "/tmp/mineru-logs/doclib.parse-server.stderr.log"


def test_log_config_dir_override_derives_paths_in_deep_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("TEST_MINERU_DOCLIB_LOG_DIR", "/tmp/env-logs")

    overrides, _paths = _collect_env_overrides(prefix=prefix)
    env_cfg = Config(**overrides)
    patched_cfg = PatchedConfig(doclib={"log": {"dir": "/tmp/patched-logs"}})

    assert env_cfg.doclib.log.app_path is None
    assert env_cfg.doclib.log.access_path is None
    assert env_cfg.doclib.log.stdout_path is None
    assert env_cfg.doclib.log.stderr_path is None
    assert env_cfg.doclib.log.parse_server_stdout_path is None
    assert env_cfg.doclib.log.parse_server_stderr_path is None
    assert env_cfg.doclib.log.resolved_app_path == "/tmp/env-logs/doclib.log"
    assert env_cfg.doclib.log.resolved_access_path == "/tmp/env-logs/doclib.access.log"
    assert env_cfg.doclib.log.resolved_stdout_path == "/tmp/env-logs/doclib.stdout.log"
    assert env_cfg.doclib.log.resolved_stderr_path == "/tmp/env-logs/doclib.stderr.log"
    assert env_cfg.doclib.log.resolved_parse_server_stdout_path == "/tmp/env-logs/doclib.parse-server.stdout.log"
    assert env_cfg.doclib.log.resolved_parse_server_stderr_path == "/tmp/env-logs/doclib.parse-server.stderr.log"
    assert patched_cfg.doclib.log.app_path is None
    assert patched_cfg.doclib.log.access_path is None
    assert patched_cfg.doclib.log.stdout_path is None
    assert patched_cfg.doclib.log.stderr_path is None
    assert patched_cfg.doclib.log.parse_server_stdout_path is None
    assert patched_cfg.doclib.log.parse_server_stderr_path is None
    assert patched_cfg.doclib.log.resolved_app_path == "/tmp/patched-logs/doclib.log"
    assert patched_cfg.doclib.log.resolved_access_path == "/tmp/patched-logs/doclib.access.log"
    assert patched_cfg.doclib.log.resolved_stdout_path == "/tmp/patched-logs/doclib.stdout.log"
    assert patched_cfg.doclib.log.resolved_stderr_path == "/tmp/patched-logs/doclib.stderr.log"
    assert patched_cfg.doclib.log.resolved_parse_server_stdout_path == "/tmp/patched-logs/doclib.parse-server.stdout.log"
    assert patched_cfg.doclib.log.resolved_parse_server_stderr_path == "/tmp/patched-logs/doclib.parse-server.stderr.log"


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
        "parse_server_startup_timeout_sec",
        "parse_server_stop_timeout_sec",
    }

    assert startup_only_keys.isdisjoint(CONFIG_DEFAULTS)
