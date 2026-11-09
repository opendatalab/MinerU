"""单文件下载记录、损坏修复与目录完成标记隔离的回归。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from huggingface_hub.utils import filter_repo_objects

from mineru.model import download


@pytest.fixture
def file_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> download.ModelRepo:
    """隔离一个与 ONNX 仓库相同、全部必需资源均为文件的模型仓库。"""
    monkeypatch.setattr(download.get_config().model, "base_dir", str(tmp_path))
    monkeypatch.setattr(download.get_config().model, "source", "modelscope")
    return download.ModelRepo(
        name="file-model",
        repos={"modelscope": "Owner/file-model"},
        paths={"config": "Layout/config.json", "weights": "Layout/model.onnx"},
        download_mode="required_paths",
    )


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """模拟远端按请求范围下载完整文件，同时返回独立的预期大小。"""
    payloads = {"Layout/config.json": b"{}", "Layout/model.onnx": b"12345678", "empty.txt": b""}

    def snapshot(source: str, repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """按真实下载模式选择文件，便于验证局部下载和整仓下载的区别。"""
        assert source == "modelscope"
        sizes = {}
        for name in filter_repo_objects(payloads, allow_patterns=patterns):
            target = repo.local_dir() / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            target.write_bytes(payloads[name])
            sizes[name] = len(payloads[name])
        return sizes

    result = Mock(side_effect=snapshot)
    monkeypatch.setattr(download, "_snapshot_download", result)
    return result


@pytest.mark.parametrize("entry", ["files", "repo"])
@pytest.mark.parametrize("damage", ["truncated", "deleted", "broken_link"])
def test_downloaded_file_damage_is_rejected_and_repaired(
    file_repo: download.ModelRepo, remote: Mock, monkeypatch: pytest.MonkeyPatch, entry: str, damage: str
) -> None:
    """首次成功下载后，所有单文件入口都必须拒绝损坏缓存并允许联网修复。"""
    if entry == "files":
        file_repo.config.ensure()
        file_repo.weights.ensure()
    else:
        file_repo.ensure()
    assert download.verify_model_repo(file_repo).ready
    assert not list(file_repo.local_dir().rglob(download.MODEL_COMPLETE_MARKER))
    calls = remote.call_count
    weight = file_repo.weights.local_path()
    if damage == "truncated":
        weight.write_bytes(b"1")
    else:
        weight.unlink()
        if damage == "broken_link":
            weight.symlink_to("missing.onnx")

    assert not download.model_path_exists(file_repo.weights)
    assert download.verify_model_repo(file_repo).missing_paths == ["Layout/model.onnx"]
    monkeypatch.setattr(download.get_config().model, "source", "local")
    with pytest.raises(FileNotFoundError, match="Layout/model.onnx"):
        file_repo.weights.ensure()
    assert remote.call_count == calls

    file_repo.weights.ensure(source="modelscope")
    assert remote.call_count == calls + 1
    assert weight.read_bytes() == b"12345678"
    assert download.verify_model_repo(file_repo).ready


def test_file_records_do_not_complete_parent_directories(file_repo: download.ModelRepo, remote: Mock) -> None:
    """局部文件记录不能把父目录或原始 VLM 整仓提升为已完成状态。"""
    file_repo.weights.ensure()
    assert download.model_path_exists(file_repo.weights)
    assert not download.model_path_exists(file_repo.path("Layout"))
    assert not download.verify_model_repo(replace(file_repo, download_mode="full")).ready
    assert not download.verify_model_repo(file_repo).ready
    assert remote.call_count == 1


def test_unmanaged_files_and_legacy_markers_still_work_offline(
    file_repo: download.ModelRepo, remote: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有新下载记录的手工模型与旧空标记保持原有离线使用边界。"""
    root = file_repo.local_dir()
    weight = file_repo.weights.local_path()
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"manual model")
    file_repo.config.local_path().write_bytes(b"{}")
    monkeypatch.setattr(download.get_config().model, "source", "local")
    assert file_repo.weights.ensure() == weight
    (root / download.MODEL_COMPLETE_MARKER).touch()
    assert file_repo.ensure() == root
    assert (root / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    remote.assert_not_called()


def test_corrupted_file_manifest_cannot_fall_back_to_nonempty_files(file_repo: download.ModelRepo, remote: Mock) -> None:
    """受管记录损坏时不降级放行，重新下载确认后才能恢复就绪。"""
    file_repo.weights.ensure()
    marker = file_repo.local_dir() / download.MODEL_FILES_MANIFEST
    marker.write_text("{", encoding="utf-8")
    assert not download.model_path_exists(file_repo.weights)
    with pytest.raises(ValueError, match="rebuild file records"):
        file_repo.weights.ensure()
    assert marker.read_text() == "{"
    file_repo.ensure()
    assert download.model_path_exists(file_repo.weights)
    assert json.loads(marker.read_text())["files"] == {"Layout/config.json": 2, "Layout/model.onnx": 8}
    assert remote.call_count == 3


def test_directory_refresh_replaces_obsolete_file_records(file_repo: download.ModelRepo, remote: Mock) -> None:
    """目录经远端完整确认后移除过时局部记录，避免已删除文件阻塞新快照。"""
    file_repo.weights.ensure()
    file_repo.weights.local_path().unlink()

    def snapshot(source: str, repo: download.ModelRepo, patterns: list[str] | None) -> dict[str, int]:
        """模拟整仓刷新后旧权重被新文件取代。"""
        assert patterns is None
        target = repo.local_dir() / "new-model.onnx"
        target.write_bytes(b"new weights")
        return {"new-model.onnx": 11}

    remote.side_effect = snapshot
    full_repo = replace(file_repo, paths={}, download_mode="full")
    full_repo.ensure()
    assert download.verify_model_repo(full_repo).ready
    assert not (full_repo.local_dir() / download.MODEL_FILES_MANIFEST).exists()


def test_confirmed_empty_file_is_valid_and_manifest_is_not_payload(file_repo: download.ModelRepo, remote: Mock) -> None:
    """远端确认的合法空文件可复用，文件记录本身不能作为模型 payload。"""
    empty = file_repo.path("empty.txt")
    empty.ensure()
    assert download.model_path_exists(empty)
    assert remote.call_count == 1
    empty.ensure()
    assert remote.call_count == 1
    assert not download.is_payload_path(download.MODEL_FILES_MANIFEST)


def test_failed_repair_keeps_other_managed_file_records(file_repo: download.ModelRepo, remote: Mock) -> None:
    """下载失败时仍保留原文件记录，其他受管文件不会退回非空即就绪的判断。"""
    file_repo.ensure()
    marker = file_repo.local_dir() / download.MODEL_FILES_MANIFEST
    before = marker.read_bytes()
    file_repo.weights.local_path().write_bytes(b"short")
    remote.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        file_repo.weights.ensure()
    assert marker.read_bytes() == before
    assert download.model_path_exists(file_repo.config)
    assert not download.model_path_exists(file_repo.weights)


def test_directory_refresh_keeps_file_records_outside_its_scope(file_repo: download.ModelRepo, remote: Mock) -> None:
    """目录清单只接管自身范围，其他位置的文件记录继续保留。"""
    file_repo.ensure()
    file_repo.path("empty.txt").ensure()
    file_repo.path("Layout").ensure()
    marker = file_repo.local_dir() / download.MODEL_FILES_MANIFEST
    assert json.loads(marker.read_text())["files"] == {"empty.txt": 0}
    assert download.model_path_exists(file_repo.path("Layout"))
    file_repo.weights.local_path().write_bytes(b"short")
    assert not download.model_path_exists(file_repo.weights)
