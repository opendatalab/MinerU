"""验证配置装载时机、默认路径及进程缓存边界。"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mineru.config import Config, LoadedConfig

config_module = importlib.import_module("mineru.config")


@pytest.fixture(autouse=True)
def isolate_default_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """每项装载测试使用独立环境和空缓存，结束后恢复调用方原有默认配置。"""
    for name in tuple(os.environ):
        if name.startswith("MINERU_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    monkeypatch.setattr(config_module, "_loaded_config", None)


def test_import_does_not_open_configuration_file(tmp_path: Path) -> None:
    """通过独立进程审计文件读取，保证导入配置模块不会装载配置文件。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  source: local\n", encoding="utf-8")
    script = """
import importlib
import sys

def reject_configuration_read(event: str, args: tuple[object, ...]) -> None:
    \"\"\"配置文件只允许显式装载时读取。\"\"\"
    if event == "open" and args[0] == sys.argv[1]:
        raise AssertionError("configuration file read during import")

sys.addaudithook(reject_configuration_read)
module = importlib.import_module("mineru.config")
assert module._loaded_config is None
assert not hasattr(module, "config")
"""
    env = {**os.environ, "MINERU_CONFIG": str(config_path), "MINERU_MODEL_STACK": "removed"}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("accessor", ["get_config", "get_config_file_path", "get_config_file_exists", "get_config_source"])
def test_first_configuration_access_validates_missing_file(
    accessor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """任何默认配置或来源查询首次执行时都报告显式配置路径错误。"""
    monkeypatch.setenv("MINERU_CONFIG", str(tmp_path / "missing.yaml"))
    arguments = ("model.source",) if accessor == "get_config_source" else ()
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        getattr(config_module, accessor)(*arguments)
    assert config_module._loaded_config is None


def test_failed_default_load_can_retry_after_configuration_is_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """初始化失败不缓存半成品，修复配置后同一进程可以重新装载。"""
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("MINERU_CONFIG", str(config_path))
    with pytest.raises(FileNotFoundError):
        config_module.get_config()
    config_path.write_text("log:\n  level: debug\n", encoding="utf-8")
    assert config_module.get_config().log.level == "debug"


def test_default_paths_use_environment_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """所有依赖 MINERU_HOME 的默认路径在构造配置时解析，且不修改既有实例。"""
    first_home, second_home = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("MINERU_HOME", str(first_home))
    first = Config()
    monkeypatch.setenv("MINERU_HOME", str(second_home))
    second = Config()

    for configuration, root in ((first, first_home), (second, second_home)):
        assert Path(configuration.doclib.uds.path) == root / "doclib.sock"
        assert Path(configuration.doclib.log.dir) == root / "logs"
        assert Path(configuration.doclib.sqlite.path) == root / "doclib.db"
        assert Path(configuration.doclib.endpoint_path) == root / "doclib.endpoint.json"
        assert Path(configuration.doclib.data_dir) == root / "doclib"
        assert Path(configuration.model.base_dir) == root / "models"
    assert config_module._loaded_config is None


def test_independent_loads_do_not_replace_default_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """独立装载读取当前文件和环境，同时保持默认实例及其配置来源稳定。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  source: local\n", encoding="utf-8")
    default = config_module.get_config()
    assert default.model.source == "local"
    assert config_module.get_config_source("model.source") == "file"

    config_path.write_text("model:\n  source: huggingface\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "modelscope")
    loaded = config_module.load_config()
    assert loaded.config.model.source == "modelscope"
    assert loaded.sources[("model", "source")] == "env"
    assert loaded.config_file == str(config_path)
    assert loaded.config_file_exists is True
    assert config_module.load_config().config is not loaded.config
    assert config_module.get_config() is default
    assert default.model.source == "local"
    assert config_module.get_config_source("model.source") == "file"


def test_configuration_queries_share_first_loaded_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """路径查询触发的首次装载与后续配置及来源查询共享同一快照。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  source: local\n", encoding="utf-8")
    assert config_module.get_config_file_path() == str(config_path)
    monkeypatch.setenv("MINERU_CONFIG", str(tmp_path / "missing.yaml"))
    assert config_module.get_config().model.source == "local"
    assert config_module.get_config_file_path() == str(config_path)
    assert config_module.get_config_file_exists() is True
    assert config_module.get_config_source(("model", "source")) == "file"


def test_concurrent_first_access_loads_default_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """并发首次访问只能执行一次装载，并向所有线程返回同一配置实例。"""
    entered, release = threading.Event(), threading.Event()
    loaded_results: list[LoadedConfig] = []
    independent_loader = config_module.load_config

    def load_after_release() -> LoadedConfig:
        """阻塞首次装载，让其他线程在同一初始化锁上等待。"""
        entered.set()
        assert release.wait(5)
        loaded = independent_loader()
        loaded_results.append(loaded)
        return loaded

    monkeypatch.setattr(config_module, "load_config", load_after_release)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(config_module.get_config) for _ in range(8)]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert len(loaded_results) == 1
    assert all(configuration is loaded_results[0].config for configuration in results)
