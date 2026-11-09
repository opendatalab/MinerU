"""模型文件被清理、损坏及局部更新后的真实就绪行为回归。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NoReturn
from unittest.mock import Mock

import pytest

from mineru.model import download


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> download.ModelRepo:
    """将下载和离线检查隔离在临时模型目录中。"""
    monkeypatch.setattr(download.get_config().model, "base_dir", str(tmp_path))
    return download.ModelRepo(
        name="test-model",
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        paths={"config": "config.json", "weights": "model.safetensors"},
    )


def _write_files(root: Path, files: dict[str, bytes]) -> dict[str, int]:
    """准备实际模型文件，并返回与字节内容独立对应的预期大小。"""
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_bytes(content)
    return {name: len(content) for name, content in files.items()}


def _write_marker(root: Path, files: dict[str, int]) -> None:
    """直接按协议构造完成标记，使回归测试不依赖生产清单写入函数。"""
    (root / download.MODEL_COMPLETE_MARKER).write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")


@pytest.mark.parametrize(
    "remaining",
    [
        {},
        {"README.md": b"readme"},
        {"config.json": b"{}"},
        {"config.json": b"{}", "model.safetensors": b""},
        {".cache/huggingface/model.safetensors": b"cached", ".msc": b"metadata"},
        {"._____temp/model.safetensors": b"partial"},
    ],
)
def test_legacy_marker_cannot_confirm_remaining_files(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch, remaining: dict[str, bytes]
) -> None:
    """旧空标记及少量残留文件均不能在离线模式中被认定为完整模型。"""
    root = repo.local_dir()
    root.mkdir()
    _write_files(root, remaining)
    marker = root / download.MODEL_COMPLETE_MARKER
    marker.touch()
    monkeypatch.setattr(download.get_config().model, "source", "local")
    remote = Mock(side_effect=AssertionError("offline checks must not download"))
    monkeypatch.setattr(download, "_snapshot_download", remote)

    result = download.verify_model_repo(repo)

    assert result.ready is False
    assert result.missing_paths
    assert not download.model_path_exists(repo.path(""))
    for target in (repo, repo.path("")):
        with pytest.raises(FileNotFoundError, match="repair files"):
            target.ensure()
    assert marker.read_bytes() == b""
    remote.assert_not_called()


@pytest.mark.parametrize("source", ["auto", "huggingface", "modelscope", "local"])
def test_complete_legacy_model_stays_offline_without_marker_upgrade(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """完整旧模型在任意来源配置下均直接复用，仓库及目录加载不联网也不改写标记。"""
    _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights", "ocr/inference.pth": b"detector"})
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    marker.touch()
    monkeypatch.setattr(download.get_config().model, "source", source)
    resolve = Mock(side_effect=AssertionError("complete legacy models must stay offline"))
    monkeypatch.setattr(download, "resolve_model_source", resolve)

    assert download.verify_model_repo(repo).ready
    assert repo.ensure() == repo.local_dir()
    assert repo.path("").ensure() == repo.local_dir()
    assert repo.path("ocr").ensure() == repo.local_dir() / "ocr"
    assert repo.weights.ensure() == repo.weights.local_path()
    assert marker.read_bytes() == b""
    assert not (repo.local_dir() / "ocr" / download.MODEL_COMPLETE_MARKER).exists()
    resolve.assert_not_called()


def test_legacy_marker_still_requires_declared_config(repo: download.ModelRepo) -> None:
    """权重存在也不能掩盖 registry 已声明的必要配置文件缺失。"""
    _write_files(repo.local_dir(), {"model.safetensors": b"weights"})
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()

    assert download.verify_model_repo(repo).missing_paths == ["config.json"]


@pytest.mark.parametrize("damage", ["intact", "deleted", "empty", "broken_link"])
def test_legacy_weight_index_checks_all_shards(repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    """原始 VLM 没有静态逐文件清单时，仍通过自带权重索引检查每一个分片。"""
    raw_repo = download.ModelRepo(name=repo.name, repos=repo.repos)
    index = {"weight_map": {"layer.0": "model-00001.safetensors", "layer.1": "model-00002.safetensors"}}
    _write_files(
        raw_repo.local_dir(),
        {
            "config.json": b"{}",
            "model.safetensors.index.json": json.dumps(index).encode(),
            "model-00001.safetensors": b"first shard",
            "model-00002.safetensors": b"second shard",
        },
    )
    (raw_repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()
    second = raw_repo.local_dir() / "model-00002.safetensors"
    if damage != "intact":
        second.unlink()
        if damage == "empty":
            second.touch()
        elif damage == "broken_link":
            second.symlink_to("missing-target")
    resolve = Mock(side_effect=AssertionError("legacy verification must stay offline"))
    monkeypatch.setattr(download, "resolve_model_source", resolve)

    result = download.verify_model_repo(raw_repo)

    if damage == "intact":
        assert result.ready
        assert raw_repo.ensure() == raw_repo.local_dir()
    else:
        assert not result.ready
        assert result.missing_paths == ["model-00002.safetensors"]
    resolve.assert_not_called()


@pytest.mark.parametrize("index", ["{", "[]", '{"weight_map": {}}', '{"weight_map": {"layer": "../outside"}}'])
def test_legacy_broken_weight_index_is_rejected(repo: download.ModelRepo, index: str) -> None:
    """旧标记不能绕过损坏、空或包含不安全分片路径的权重索引。"""
    _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights"})
    (repo.local_dir() / "model.safetensors.index.json").write_text(index, encoding="utf-8")
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()

    assert download.verify_model_repo(repo).missing_paths == ["model.safetensors.index.json"]


def test_legacy_gguf_requires_both_registered_weights(repo: download.ModelRepo) -> None:
    """GGUF 旧目录必须同时具备主模型与投影文件，保留其中一个不能判为就绪。"""
    from mineru.model.registry import MINERU_2_5_PRO_2605_1_2B_GGUF

    gguf = MINERU_2_5_PRO_2605_1_2B_GGUF
    _write_files(gguf.local_dir(), {gguf.paths["main"]: b"main model"})
    (gguf.local_dir() / download.MODEL_COMPLETE_MARKER).touch()

    assert download.verify_model_repo(gguf).missing_paths == [gguf.paths["mmproj"]]

    _write_files(gguf.local_dir(), {gguf.paths["mmproj"]: b"projector"})
    assert download.verify_model_repo(gguf).ready


def test_legacy_child_marker_cannot_override_new_parent_manifest(repo: download.ModelRepo) -> None:
    """祖先已有新版清单时，旧子标记不能把缺少一个权重的目录降级放行。"""
    files = _write_files(repo.local_dir(), {"det/first.onnx": b"first", "det/second.onnx": b"second"})
    _write_marker(repo.local_dir(), files)
    (repo.local_dir() / "det" / download.MODEL_COMPLETE_MARKER).touch()
    (repo.local_dir() / "det/second.onnx").unlink()

    result = download._verify_paths(repo, [repo.path("det")])

    assert not result.ready
    assert result.missing_paths == ["det/second.onnx"]
    assert not download.model_path_exists(repo.path("det"))


def test_legacy_parent_still_checks_new_child_manifest(repo: download.ModelRepo) -> None:
    """旧父目录复用时也必须尊重已确认的新版子清单，不能忽略后来丢失的成员。"""
    _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights"})
    files = _write_files(repo.local_dir() / "det", {"first.onnx": b"first", "second.onnx": b"second"})
    _write_marker(repo.local_dir() / "det", files)
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()
    (repo.local_dir() / "det/second.onnx").unlink()

    assert download.verify_model_repo(repo).missing_paths == ["det/second.onnx"]
    assert not download.model_path_exists(repo.path("det"))


@pytest.mark.parametrize("damage", ["deleted", "truncated", "empty", "broken_link"])
@pytest.mark.parametrize("entry", ["repo", "path"])
def test_missing_payload_is_reported_offline_and_repaired_online(
    repo: download.ModelRepo,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    entry: str,
) -> None:
    """已完成模型缺失或截断时，两种入口均报告具体文件并通过一次下载恢复。"""
    contents = {"config.json": b"{}", "model.safetensors": b"complete model weights"}
    files = _write_files(repo.local_dir(), contents)
    _write_marker(repo.local_dir(), files)
    weight = repo.weights.local_path()
    weight.unlink()
    if damage == "truncated":
        weight.write_bytes(b"short")
    elif damage == "empty":
        weight.touch()
    elif damage == "broken_link":
        weight.symlink_to("missing-target")
    target = repo if entry == "repo" else repo.path("")
    monkeypatch.setattr(download.get_config().model, "source", "local")
    remote = Mock(side_effect=AssertionError("offline checks must not download"))
    monkeypatch.setattr(download, "_snapshot_download", remote)

    assert download.verify_model_repo(repo).missing_paths == ["model.safetensors"]
    assert not download.model_path_exists(repo.weights)
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        target.ensure()
    remote.assert_not_called()

    def repair(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟 provider 增量修复，并确认根目录请求没有被展开成空过滤器。"""
        assert source == "modelscope"
        assert target_repo is repo
        assert patterns is None
        assert not (repo.local_dir() / download.MODEL_COMPLETE_MARKER).exists()
        return _write_files(repo.local_dir(), contents)

    remote.side_effect = repair
    assert target.ensure(source="modelscope") == repo.local_dir()
    assert target.ensure(source="modelscope") == repo.local_dir()
    assert remote.call_count == 1
    assert download.verify_model_repo(repo).ready
    assert download.model_path_exists(repo.path(""))


@pytest.mark.parametrize(
    "marker_text",
    [
        " ",
        "{",
        "[]",
        '{"version": 2, "files": {"model.safetensors": 7}}',
        '{"version": true, "files": {"model.safetensors": 7}}',
        '{"version": 1, "files": {}}',
        '{"version": 1, "files": {"../outside": 7}}',
        '{"version": 1, "files": {"/outside": 7}}',
        '{"version": 1, "files": {"C:/outside": 7}}',
        '{"version": 1, "files": {"bad\\u0000path": 7}}',
        '{"version": 1, "files": {".cache/metadata": 7}}',
        '{"version": 1, "files": {"model.safetensors": -1}}',
        '{"version": 1, "files": {"model.safetensors": true}}',
    ],
)
def test_invalid_completion_manifest_is_not_ready(repo: download.ModelRepo, marker_text: str) -> None:
    """损坏或不受支持的清单不能通过本地目录存在性检查。"""
    _write_files(repo.local_dir(), {"model.safetensors": b"weights"})
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).write_text(marker_text, encoding="utf-8")

    assert not download.verify_model_repo(repo).ready
    assert not download.model_path_exists(repo.path(""))


def test_legacy_marker_is_upgraded_after_provider_confirmation(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧目录只有经 provider 确认完整后才升级，不能扫描残留文件自行认证。"""
    _write_files(repo.local_dir(), {"config.json": b"{}"})
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    marker.touch()

    def repair(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """下载器补齐原目录缺少的权重，并返回完整预期集合。"""
        assert not marker.exists()
        return _write_files(target_repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights"})

    remote = Mock(side_effect=repair)
    monkeypatch.setattr(download, "_snapshot_download", remote)

    repo.ensure(source="modelscope")

    assert remote.call_count == 1
    assert json.loads(marker.read_text())["files"] == {"config.json": 2, "model.safetensors": 7}
    assert download.verify_model_repo(repo).ready


def test_ready_model_reuses_local_manifest_without_source_resolution(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正常模型在整仓、根目录和子目录入口均通过纯本地校验复用。"""
    files = _write_files(repo.local_dir(), {"model/config.json": b"{}", "model/model.safetensors": b"weights"})
    _write_marker(repo.local_dir(), files)
    resolve = Mock(side_effect=AssertionError("ready models must not resolve a remote source"))
    monkeypatch.setattr(download, "resolve_model_source", resolve)

    assert repo.ensure() == repo.local_dir()
    assert repo.path("").ensure() == repo.local_dir()
    assert repo.path("model").ensure() == repo.local_dir() / "model"
    resolve.assert_not_called()


@pytest.mark.parametrize("damage", ["legacy", "deleted", "truncated"])
def test_required_directory_verifies_and_repairs_all_members(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """小模型目录保留配置时，必需权重缺失也必须通过一次增量下载补齐。"""
    directory_repo = download.ModelRepo(
        name=repo.name, repos=repo.repos, paths={"weights": "weights"}, download_mode="required_paths"
    )
    files = _write_files(directory_repo.local_dir() / "weights", {"config.json": b"{}", "model.onnx": b"weights"})
    marker = directory_repo.weights.local_path() / download.MODEL_COMPLETE_MARKER
    if damage == "legacy":
        marker.touch()
        (directory_repo.weights.local_path() / "model.onnx").unlink()
    else:
        _write_marker(directory_repo.weights.local_path(), files)
        (directory_repo.weights.local_path() / "model.onnx").unlink()
        if damage == "truncated":
            (directory_repo.weights.local_path() / "model.onnx").write_bytes(b"short")
    monkeypatch.setattr(download.get_config().model, "source", "local")

    assert not download.verify_model_repo(directory_repo).ready
    assert not download.model_path_exists(directory_repo.weights)
    with pytest.raises(FileNotFoundError, match="weights"):
        directory_repo.weights.ensure()

    def repair(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """只确认和修复所请求的小模型目录。"""
        assert patterns == ["weights", "weights/*"]
        return _write_files(target_repo.local_dir(), {"weights/config.json": b"{}", "weights/model.onnx": b"weights"})

    remote = Mock(side_effect=repair)
    monkeypatch.setattr(download, "_snapshot_download", remote)
    directory_repo.weights.ensure(source="modelscope")
    directory_repo.ensure(source="modelscope")

    assert remote.call_count == 1
    assert download.verify_model_repo(directory_repo).ready


def test_partial_repair_does_not_hide_missing_sibling(repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """检测器下载成功不能恢复仍缺少识别器的父目录完成状态。"""
    files = _write_files(repo.local_dir(), {"det/model.onnx": b"detector", "rec/model.onnx": b"recognizer"})
    _write_marker(repo.local_dir(), files)
    (repo.local_dir() / "rec/model.onnx").unlink()

    def repair(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟仅确认请求的检测器目录。"""
        assert patterns == ["det", "det/*"]
        return _write_files(target_repo.local_dir(), {"det/model.onnx": b"new detector"})

    monkeypatch.setattr(download, "_snapshot_download", repair)
    download.download_model_files(repo, [repo.path("det")], source="modelscope")

    assert download.model_path_exists(repo.path("det"))
    assert not (repo.local_dir() / download.MODEL_COMPLETE_MARKER).exists()
    assert not download.verify_model_repo(repo).ready


@pytest.mark.parametrize("fail", [False, True])
def test_single_file_update_invalidates_and_revalidates_parent_manifest(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """单文件更新也撤销父标记，成功后合并新大小，失败后保持未完成。"""
    files = _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights"})
    _write_marker(repo.local_dir(), files)
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER

    def update(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """验证标记在 provider 改写文件之前已经撤销。"""
        assert not marker.exists()
        if fail:
            raise RuntimeError("download interrupted")
        return _write_files(target_repo.local_dir(), {"model.safetensors": b"updated model weights"})

    monkeypatch.setattr(download, "_snapshot_download", update)
    if fail:
        with pytest.raises(RuntimeError, match="download interrupted"):
            download.download_model_files(repo, [repo.weights], source="modelscope")
        assert not marker.exists()
    else:
        download.download_model_files(repo, [repo.weights], source="modelscope")
        assert download.verify_model_repo(repo).ready
        assert json.loads(marker.read_text())["files"] == {"config.json": 2, "model.safetensors": 21}


def test_multiple_directory_updates_preserve_each_parent_manifest(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一次下载多个分支时，各父目录只合并自身请求，且完整根清单同步更新。"""
    files = _write_files(repo.local_dir(), {"a/nested/model.bin": b"a", "b/nested/model.bin": b"b"})
    _write_marker(repo.local_dir(), files)
    _write_marker(repo.local_dir() / "a", {"nested/model.bin": 1})
    _write_marker(repo.local_dir() / "b", {"nested/model.bin": 1})

    def update(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟分别补齐两个请求目录。"""
        return _write_files(target_repo.local_dir(), {"a/nested/model.bin": b"new a", "b/nested/model.bin": b"new b"})

    monkeypatch.setattr(download, "_snapshot_download", update)
    download.download_model_files(repo, [repo.path("a/nested"), repo.path("b/nested")], source="modelscope")

    assert download.verify_model_repo(repo).ready
    assert download.model_path_exists(repo.path("a"))
    assert download.model_path_exists(repo.path("b"))


def test_local_subdirectory_download_preserves_complete_legacy_parent(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """局部更新后可恢复通过本地必要文件检查的旧父标记，且不伪造新版清单。"""
    _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"weights", "det/model.onnx": b"detector"})
    (repo.local_dir() / download.MODEL_COMPLETE_MARKER).touch()

    def update(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """只确认检测器，父目录保留原有兼容语义。"""
        return {"det/model.onnx": 8}

    monkeypatch.setattr(download, "_snapshot_download", update)
    download.download_model_files(repo, [repo.path("det")], source="modelscope")

    assert download.model_path_exists(repo.path("det"))
    assert download.verify_model_repo(repo).ready
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""


def test_full_download_supersedes_stale_child_marker(repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """整仓确认后，旧子目录标记不能阻止新清单在离线加载中生效。"""
    _write_files(repo.local_dir(), {"nested/model.safetensors": b"old"})
    child_marker = repo.local_dir() / "nested" / download.MODEL_COMPLETE_MARKER
    child_marker.touch()

    def update(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """整仓下载同时确认子目录的新权重。"""
        return _write_files(target_repo.local_dir(), {"nested/model.safetensors": b"new weights"})

    monkeypatch.setattr(download, "_snapshot_download", update)
    repo.ensure(source="modelscope")
    resolve = Mock(side_effect=AssertionError("confirmed child paths must stay offline"))
    monkeypatch.setattr(download, "resolve_model_source", resolve)

    assert not child_marker.exists()
    assert repo.path("nested").ensure() == repo.local_dir() / "nested"
    assert download.verify_model_repo(repo).ready
    resolve.assert_not_called()


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_provider_repair_replaces_wrong_size_cached_payload(
    repo: download.ModelRepo,
    monkeypatch: pytest.MonkeyPatch,
    source: Literal["huggingface", "modelscope"],
) -> None:
    """SDK 根据缓存跳过已有文件时，失效权重必须先移除，完整文件继续复用。"""
    _write_files(repo.local_dir(), {"config.json": b"{}", "model.safetensors": b"short"})
    expected = {"config.json": 2, "model.safetensors": 7}
    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha="commit-id",
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in expected.items()],
    )
    monkeypatch.setattr(download, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(download, "ms_model_files", Mock(return_value=expected))
    monkeypatch.setattr(download, "ms_model_revision", Mock(return_value="a" * 40))

    def snapshot(repo_id: str, **kwargs: object) -> str:
        """模拟增量下载只补齐缺失文件，保留已验证的配置。"""
        assert repo.config.local_path().read_bytes() == b"{}"
        assert not repo.weights.local_path().exists()
        repo.weights.local_path().write_bytes(b"weights")
        return str(repo.local_dir())

    monkeypatch.setattr(download, "hf_snapshot_download", snapshot)
    monkeypatch.setattr(download, "ms_snapshot_download", snapshot)

    repo.ensure(source=source)

    assert download.verify_model_repo(repo).ready
    assert repo.weights.local_path().read_bytes() == b"weights"


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
@pytest.mark.parametrize("damage", ["deleted", "truncated", "intact"])
def test_provider_manifest_checks_expected_payload_before_marking_complete(
    repo: download.ModelRepo,
    monkeypatch: pytest.MonkeyPatch,
    source: Literal["huggingface", "modelscope"],
    damage: str,
) -> None:
    """两种来源均按远程大小检查预期文件，配置或 README 不能掩盖权重未下载。"""
    expected = {"config.json": 2, "README.md": 6, "model.safetensors": 7}
    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha="commit-id",
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in expected.items()],
    )
    monkeypatch.setattr(download, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(download, "ms_model_files", Mock(return_value=expected))
    monkeypatch.setattr(download, "ms_model_revision", Mock(return_value="a" * 40))

    def snapshot(repo_id: str, **kwargs: object) -> str:
        """模拟 SDK 返回成功但权重可能缺失或未完整落盘。"""
        contents = {"config.json": b"{}", "README.md": b"readme"}
        if damage != "deleted":
            contents["model.safetensors"] = b"short" if damage == "truncated" else b"weights"
        _write_files(repo.local_dir(), contents)
        if source == "huggingface":
            assert kwargs["revision"] == "commit-id"
        return str(repo.local_dir())

    monkeypatch.setattr(download, "hf_snapshot_download", snapshot)
    monkeypatch.setattr(download, "ms_snapshot_download", snapshot)

    if damage == "intact":
        download.download_model_repo(repo, source=source)
        assert download.verify_model_repo(repo).ready
        assert json.loads((repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_text())["files"] == expected
    else:
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            download.download_model_repo(repo, source=source)
        assert not (repo.local_dir() / download.MODEL_COMPLETE_MARKER).exists()


def test_manifest_publication_failure_does_not_leave_completion_marker(
    repo: download.ModelRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """原子替换失败时不留下完成标记或临时清单，避免后续误判。"""

    def snapshot(source: str, target_repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """先完成下载，使失败只发生在清单发布阶段。"""
        return _write_files(target_repo.local_dir(), {"model.safetensors": b"weights"})

    def fail_replace(path: Path, target: Path) -> NoReturn:
        """模拟进程无法原子发布文件。"""
        raise OSError("cannot publish completion manifest")

    monkeypatch.setattr(download, "_snapshot_download", snapshot)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot publish"):
        repo.ensure(source="modelscope")

    assert not download.verify_model_repo(repo).ready
    assert sorted(path.name for path in repo.local_dir().iterdir()) == ["model.safetensors"]
