from __future__ import annotations

from pathlib import Path

import pytest

from mineru.config import (
    Config,
    LLMAidedConfig,
    LogConfig,
    PatchedConfig,
    _apply_env_overrides,
    _interpolate_env,
    _load_config,
    _read_config,
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

    data = _read_config()

    assert data["doclib"]["tcp"]["port"] == 18080
    assert data["doclib"]["data_dir"] == "/tmp/ignored-data-dir"


def test_apply_env_overrides_can_override_doclib_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = "TEST_MINERU_"
    monkeypatch.setenv("MINERU_HOME", "/tmp/mineru-home")
    monkeypatch.setenv("TEST_MINERU_DOCLIB_DATA_DIR", "/tmp/ignored-data-dir")

    cfg = _apply_env_overrides(Config(), prefix=prefix)

    assert cfg.doclib.data_dir == "/tmp/ignored-data-dir"


def test_llm_aided_config_defaults_to_disabled() -> None:
    """验证 LLM 辅助功能缺省关闭且不会要求 API 凭据。"""
    llm_config = Config().llm_aided

    assert llm_config.api_key == ""
    assert llm_config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert llm_config.model == "qwen3.5-plus"
    assert llm_config.enable_thinking is None
    assert llm_config.max_concurrency == 16
    assert llm_config.features.title_leveling is False
    assert llm_config.features.cross_page_table_cell_merge is False


def test_llm_aided_config_reads_yaml_shape(tmp_path: Path) -> None:
    """验证 config.yaml 的共享连接参数和独立功能开关进入强类型配置。"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm_aided:
  api_key: secret
  base_url: https://example.test/v1
  model: test-model
  enable_thinking: false
  max_concurrency: 24
  features:
    title_leveling: true
    cross_page_table_cell_merge: true
""".strip(),
        encoding="utf-8",
    )

    cfg = Config(**_load_config(str(config_file)))

    assert cfg.llm_aided.api_key == "secret"
    assert cfg.llm_aided.base_url == "https://example.test/v1"
    assert cfg.llm_aided.model == "test-model"
    assert cfg.llm_aided.enable_thinking is False
    assert cfg.llm_aided.max_concurrency == 24
    assert cfg.llm_aided.features.title_leveling is True
    assert cfg.llm_aided.features.cross_page_table_cell_merge is True


def test_llm_aided_config_supports_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 LLM 连接参数和深层功能开关均可由环境变量覆盖。"""
    prefix = "TEST_MINERU_"
    monkeypatch.setenv(f"{prefix}LLM_AIDED_API_KEY", "env-secret")
    monkeypatch.setenv(f"{prefix}LLM_AIDED_MODEL", "env-model")
    monkeypatch.setenv(f"{prefix}LLM_AIDED_ENABLE_THINKING", "false")
    monkeypatch.setenv(f"{prefix}LLM_AIDED_MAX_CONCURRENCY", "32")
    monkeypatch.setenv(f"{prefix}LLM_AIDED_FEATURES_TITLE_LEVELING", "true")
    monkeypatch.setenv(f"{prefix}LLM_AIDED_FEATURES_CROSS_PAGE_TABLE_CELL_MERGE", "true")

    cfg = _apply_env_overrides(Config(), prefix=prefix)

    assert cfg.llm_aided.api_key == "env-secret"
    assert cfg.llm_aided.model == "env-model"
    assert cfg.llm_aided.enable_thinking is False
    assert cfg.llm_aided.max_concurrency == 32
    assert cfg.llm_aided.features.title_leveling is True
    assert cfg.llm_aided.features.cross_page_table_cell_merge is True


@pytest.mark.parametrize("feature", ["title_leveling", "cross_page_table_cell_merge"])
def test_llm_aided_config_requires_credentials_for_enabled_feature(feature: str) -> None:
    """验证任一 LLM 功能启用时空 API key 会在配置边界报错。"""
    with pytest.raises(ValueError, match="llm_aided.api_key"):
        LLMAidedConfig(features={feature: True})


@pytest.mark.parametrize("value", [0, -1, 1.5, "invalid", True])
def test_llm_aided_config_rejects_invalid_max_concurrency(value: object) -> None:
    """验证并发数无论功能是否启用都必须是正整数。"""
    with pytest.raises(ValueError):
        LLMAidedConfig(max_concurrency=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["api_key", "base_url", "model"])
def test_llm_aided_config_rejects_each_empty_connection_field(field_name: str) -> None:
    """验证启用功能时共享连接参数中的任一空值都会被拒绝。"""
    values = {
        "api_key": "secret",
        "base_url": "https://example.test/v1",
        "model": "test-model",
    }
    values[field_name] = " "

    with pytest.raises(ValueError, match=f"llm_aided.{field_name}"):
        LLMAidedConfig(
            **values,
            features={"title_leveling": True},
        )


def test_legacy_llm_aided_root_key_is_ignored() -> None:
    """验证旧 llm-aided-config 不会再启用新的 YAML LLM 功能。"""
    cfg = Config.model_validate(
        {
            "llm-aided-config": {
                "title_aided": {
                    "api_key": "legacy",
                    "enable": True,
                }
            }
        }
    )

    assert cfg.llm_aided.features.title_leveling is False


def test_default_doclib_data_dir_uses_doclib_directory() -> None:
    cfg = Config()

    assert cfg.doclib.data_dir.endswith(".mineru/doclib")


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

    cfg = _apply_env_overrides(Config(doclib={"uds": {"enabled": False}, "tcp": {"enabled": True}}), prefix=prefix)

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

    env_cfg = _apply_env_overrides(Config(), prefix=prefix)
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
        "parse_server_stop_timeout_sec",
    }

    assert startup_only_keys.isdisjoint(CONFIG_DEFAULTS)
