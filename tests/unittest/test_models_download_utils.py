from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from mineru.model import download as models_download_utils
from mineru.model.registry import MODEL_COMPLETE_MARKER, MINERU_2_5_PRO_2605_1_2B, ModelRepo, model_path_exists


def _write_marker(directory: Path, files: dict[str, int]) -> None:
    """按持久化协议构造已确认清单，文件是否完整由各测试独立安排。"""
    (directory / MODEL_COMPLETE_MARKER).write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")


def test_resolve_model_source_does_not_persist_env_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_download_utils.get_config().model, "source", "auto")
    monkeypatch.setattr(models_download_utils, "get_config_source", lambda path: "env")
    monkeypatch.setattr(models_download_utils, "_resolve_auto_model_source", lambda: "modelscope")

    def fail_persist(_model_source: str) -> None:
        raise AssertionError("env-provided auto must not be persisted")

    monkeypatch.setattr(models_download_utils, "_persist_resolved_model_source", fail_persist)

    assert models_download_utils.resolve_model_source(allow_auto=True) == "modelscope"


def test_resolve_model_source_can_treat_config_local_as_auto_for_explicit_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models_download_utils.get_config().model, "source", "local")
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

    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils, "resolve_model_source", lambda *args, **kwargs: "modelscope")

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟 provider 确认局部仓库的预期权重文件。"""
        captured["source"] = model_source
        captured["repo"] = target_repo.name
        captured["patterns"] = patterns
        target = target_repo.local_dir() / "models/weights"
        target.mkdir(parents=True)
        (target / "model.bin").write_text("x", encoding="utf-8")
        return {"models/weights/model.bin": 1}

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

    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils, "resolve_model_source", lambda *args, **kwargs: "huggingface")

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟 provider 返回整仓文件清单。"""
        captured["source"] = model_source
        captured["repo"] = target_repo.name
        captured["patterns"] = patterns
        target = target_repo.local_dir() / "config.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        return {"config.json": 2}

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    root = models_download_utils.download_model_repo(repo, source="huggingface")

    assert root == repo.local_dir()
    assert captured == {
        "source": "huggingface",
        "repo": "full-repo",
        "patterns": None,
    }
    assert json.loads((repo.local_dir() / MODEL_COMPLETE_MARKER).read_text()) == {
        "version": 1,
        "files": {"config.json": 2},
    }


def test_vlm_repo_verify_requires_root_completion_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))

    repo_dir = MINERU_2_5_PRO_2605_1_2B.local_dir()
    repo_dir.mkdir(parents=True)
    assert MINERU_2_5_PRO_2605_1_2B.required_paths() == ()

    unmarked = models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B)

    assert unmarked.ready is False
    assert unmarked.missing_paths == [MODEL_COMPLETE_MARKER]

    (repo_dir / MODEL_COMPLETE_MARKER).touch()

    assert models_download_utils.verify_model_repo(MINERU_2_5_PRO_2605_1_2B).ready is False
    (repo_dir / "model.safetensors").write_bytes(b"weights")
    _write_marker(repo_dir, {"model.safetensors": 7})

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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))

    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.bin").write_bytes(b"weights")
    assert model_path_exists(repo.weights) is False

    _write_marker(weights_dir, {"model.bin": 7})
    assert model_path_exists(repo.weights) is True

    nested_weights = repo.weights.path("det")
    nested_weights.local_path().mkdir()
    (nested_weights.local_path() / "det.bin").write_bytes(b"weights")
    assert model_path_exists(nested_weights) is False
    _write_marker(weights_dir, {"model.bin": 7, "det/det.bin": 7})
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.bin").write_bytes(b"weights")
    _write_marker(weights_dir, {"model.bin": 7})

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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    repo.local_dir().mkdir(parents=True)
    repo.config.local_path().write_text("{}", encoding="utf-8")
    repo.weights.local_path().write_bytes(b"weights")
    _write_marker(repo.local_dir(), {"config.json": 2, "model.bin": 7})

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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()
    weights_dir.mkdir(parents=True)
    (weights_dir / "existing.bin").write_bytes(b"existing")
    calls = 0

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟增量补齐并返回完整请求目录的文件清单。"""
        nonlocal calls
        calls += 1
        assert model_source == "modelscope"
        assert patterns == ["models/weights", "models/weights/*"]
        (weights_dir / "missing.bin").write_bytes(b"repaired")
        return {"models/weights/existing.bin": 8, "models/weights/missing.bin": 8}

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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_download_utils.get_config().model, "source", "local")
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    nested_weights = repo.weights.path("det")
    nested_weights.local_path().mkdir(parents=True)
    parent_marker = repo.weights.local_path() / MODEL_COMPLETE_MARKER
    (nested_weights.local_path() / "det.bin").write_bytes(b"old")
    (repo.weights.local_path() / "rec.bin").write_bytes(b"recognizer")
    _write_marker(repo.weights.local_path(), {"det/det.bin": 3, "rec.bin": 10})

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟只更新检测器，父清单中的识别器仍必须保留。"""
        assert model_source == "modelscope"
        assert target_repo is repo
        assert patterns == ["models/weights/det", "models/weights/det/*"]
        (nested_weights.local_path() / "det.bin").write_bytes(b"weights")
        return {"models/weights/det/det.bin": 7}

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    models_download_utils.download_model_files(repo, [nested_weights], source="modelscope")

    assert parent_marker.is_file()
    assert json.loads(parent_marker.read_text())["files"] == {"det/det.bin": 7, "rec.bin": 10}
    assert model_path_exists(repo.weights)
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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    weights_dir = repo.weights.local_path()

    def fake_snapshot_download(model_source: str, target_repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟下载器只有元数据、没有任何已确认模型文件。"""
        assert model_source == "modelscope"
        assert target_repo is repo
        assert patterns == ["models/weights", "models/weights/*"]
        (weights_dir / ".cache/huggingface").mkdir(parents=True)
        (weights_dir / ".cache/huggingface/metadata").write_text("metadata", encoding="utf-8")
        (weights_dir / ".msc").write_text("metadata", encoding="utf-8")
        return {}

    monkeypatch.setattr(models_download_utils, "_snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="models/weights"):
        models_download_utils.download_model_repo(repo, source="modelscope")

    assert (weights_dir / MODEL_COMPLETE_MARKER).exists() is False


@pytest.mark.parametrize("legacy_types", [False, True], ids=["sdk-without-type", "legacy-sdk"])
@pytest.mark.parametrize("weight_state", ["complete", "missing", "truncated"])
def test_modelscope_snapshot_excludes_directories_and_checks_real_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legacy_types: bool,
    weight_state: str,
) -> None:
    """兼容新旧 SDK 的目录清单，同时保留空文件并拒绝缺失或截断的权重。"""
    repo = ModelRepo(name="repo", repos={"modelscope": "Owner/repo"}, paths={"layout": "Layout/PP-DocLayoutV2"})
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))
    layout = "Layout/PP-DocLayoutV2"
    payloads = {f"{layout}/config.json": b"{}", f"{layout}/model.safetensors": b"weights", f"{layout}/empty.txt": b""}
    directories = ["Layout", layout, "Unused"]
    # 目录项故意排在文件之后，确保过滤逻辑不依赖 SDK 返回顺序。
    entries = [{"Path": name, "Size": len(content)} for name, content in payloads.items()]
    entries.extend({"Path": name, "Size": 0} for name in directories)
    entries.append({"Path": "Unused/model.bin", "Size": 7})
    if legacy_types:
        for entry in entries:
            entry["Type"] = "tree" if entry["Path"] in directories else "blob"

    class FakeApi:
        def get_model_files(self, repo_id: str, *, revision: str, recursive: bool) -> list[dict[str, Any]]:
            """返回真实接口形状的递归清单，覆盖不再包含 Type 的新版 SDK。"""
            assert repo_id == "Owner/repo"
            assert revision == "master"
            assert recursive
            return entries

    monkeypatch.setitem(sys.modules, "modelscope.hub.api", SimpleNamespace(HubApi=FakeApi))
    monkeypatch.setitem(sys.modules, "modelscope.utils.constant", SimpleNamespace(DEFAULT_MODEL_REVISION="master"))
    requested: list[str] = []

    def fake_ms_snapshot_download(repo_id: str, *, local_dir: str, allow_patterns: list[str]) -> str:
        """模拟下载真实文件，目录项由 SDK 忽略，缺失和截断按测试参数注入。"""
        assert repo_id == "Owner/repo"
        requested.extend(allow_patterns)
        for name, content in payloads.items():
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("model.safetensors"):
                if weight_state == "missing":
                    continue
                if weight_state == "truncated":
                    content = content[:-1]
            target.write_bytes(content)
        return local_dir

    monkeypatch.setattr(models_download_utils, "ms_snapshot_download", fake_ms_snapshot_download)

    if weight_state != "complete":
        with pytest.raises(FileNotFoundError, match=r"missing or invalid: Layout/PP-DocLayoutV2/model\.safetensors$"):
            repo.layout.ensure(source="modelscope")
        assert not (repo.layout.local_path() / MODEL_COMPLETE_MARKER).exists()
    else:
        assert repo.layout.ensure(source="modelscope") == repo.layout.local_path()
        assert json.loads((repo.layout.local_path() / MODEL_COMPLETE_MARKER).read_text())["files"] == {
            name.removeprefix(f"{layout}/"): len(content) for name, content in payloads.items()
        }
    assert requested == list(payloads)


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
    monkeypatch.setattr(models_download_utils.get_config().model, "base_dir", str(tmp_path / "models"))

    class FakeApi:
        def model_info(self, repo_id: str, *, files_metadata: bool) -> SimpleNamespace:
            """提供与下载提交一致的远程文件名和大小。"""
            assert repo_id == "owner/repo"
            assert files_metadata
            return SimpleNamespace(
                sha="commit",
                siblings=[
                    SimpleNamespace(rfilename="models/weights/config.json", size=2),
                    SimpleNamespace(rfilename="models/weights/model.bin", size=7),
                ],
            )

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


def test_resolve_small_model_backend_explicit() -> None:
    """显式后端用于下载时不要求对应运行时已经安装。"""
    from mineru.model.runtime.device import resolve_small_model_backend

    assert resolve_small_model_backend("onnx") == "onnx"
    assert resolve_small_model_backend("torch") == "torch"


def test_resolve_small_model_backend_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """自动模式同时检查小模型依赖和可用设备。"""
    from mineru.model.runtime import device
    from mineru.config import get_config

    monkeypatch.setattr(get_config().model, "small_backend", "auto")
    monkeypatch.setattr(device, "module_available", lambda name: True)
    monkeypatch.setattr(device, "get_device", lambda: "cpu")
    assert device.resolve_small_model_backend() == "onnx"
    monkeypatch.setattr(device, "get_device", lambda: "mps")
    assert device.resolve_small_model_backend() == "torch"
    monkeypatch.setattr(device, "module_available", lambda name: name != "transformers")
    assert device.resolve_small_model_backend() == "onnx"


def test_resolve_small_model_backend_rejects_invalid_value() -> None:
    """旧栈名称不再作为小模型后端的别名。"""
    from mineru.model.runtime.device import resolve_small_model_backend

    with pytest.raises(ValueError, match="Unsupported small backend 'full'"):
        resolve_small_model_backend("full")
