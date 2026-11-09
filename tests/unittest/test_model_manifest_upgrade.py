"""旧模型完成标记的后台联网迁移与离线兼容回归。"""

from __future__ import annotations

import json
import pickle
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests
from filelock import FileLock

from mineru.model import _manifest_upgrade as upgrade
from mineru.model import download

_REVISION = "a" * 40
_FILES = {"config.json": 2, "model.safetensors": 7}


def _write_model(root: Path, *, scope: str = "", source: str = "huggingface") -> None:
    """准备旧标记、真实文件与下载器缓存记录，不依赖实际网络或模型权重。"""
    directory = root / scope
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_bytes(b"{}")
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / download.MODEL_COMPLETE_MARKER).touch()
    names = [f"{scope}/{name}" if scope else name for name in _FILES]
    if source == "huggingface":
        for name in names:
            metadata = root / ".cache/huggingface/download" / f"{name}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(f"{_REVISION}\n{'b' * 64}\n1000\n", encoding="utf-8")
    else:
        (root / ".msc").write_bytes(pickle.dumps([{"Path": name, "Revision": _REVISION} for name in names]))


def _request(root: Path, *, scope: str = "") -> upgrade.ManifestUpgrade:
    """构造目录范围明确的后台迁移请求。"""
    lock_path = root.parent / ".locks" / f"{root.name}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    return upgrade.ManifestUpgrade(
        root=root,
        lock_path=lock_path,
        repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"},
        markers=(root / scope / download.MODEL_COMPLETE_MARKER,),
    )


def _join(workers: list[threading.Thread]) -> None:
    """等待已安排的测试线程结束，避免后台访问已清理的临时目录。"""
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[download.ModelRepo, list[threading.Thread]]]:
    """隔离仓库、调度状态和线程列表，所有测试在退出前回收自身任务。"""
    monkeypatch.setattr(download.get_config().model, "base_dir", str(tmp_path))
    monkeypatch.setattr(download.get_config().model, "source", "local")
    monkeypatch.setattr(upgrade, "_SCHEDULER", upgrade._UpgradeScheduler())
    repo = download.ModelRepo(name="repo", repos={"huggingface": "owner/repo", "modelscope": "Owner/repo"})
    workers: list[threading.Thread] = []

    def schedule(request: upgrade.ManifestUpgrade) -> threading.Thread | None:
        """记录真实后台线程，同时保留生产调度器的去重和冷却行为。"""
        worker = upgrade.schedule_manifest_upgrade(request)
        if worker is not None:
            workers.append(worker)
        return worker

    monkeypatch.setattr(download, "schedule_manifest_upgrade", schedule)
    yield repo, workers
    _join(workers)


@pytest.mark.parametrize("source", ["auto", "huggingface", "modelscope", "local"])
@pytest.mark.parametrize("entry", ["repo", "path"])
def test_online_legacy_model_upgrades_without_downloading_payload(
    environment: tuple[download.ModelRepo, list[threading.Thread]],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    entry: str,
) -> None:
    """任意来源配置均可后台升级旧标记，local 也只放行元数据请求。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    monkeypatch.setattr(download.get_config().model, "source", source)
    query = Mock(return_value=_FILES)
    snapshot = Mock(side_effect=AssertionError("manifest migration must not download model files"))
    resolve = Mock(side_effect=AssertionError("manifest migration must not change model source"))
    monkeypatch.setattr(upgrade, "_remote_manifest", query)
    monkeypatch.setattr(download, "_snapshot_download", snapshot)
    monkeypatch.setattr(download, "resolve_model_source", resolve)
    before = {name: (repo.local_dir() / name).read_bytes() for name in _FILES}

    target = repo if entry == "repo" else repo.path("")
    assert target.ensure() == repo.local_dir()
    _join(workers)

    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    assert json.loads(marker.read_text()) == {"version": 1, "files": _FILES}
    assert before == {name: (repo.local_dir() / name).read_bytes() for name in _FILES}
    assert download.get_config().model.source == source
    query.assert_called_once_with("huggingface", "owner/repo", _REVISION)
    snapshot.assert_not_called()
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("offline"), requests.Timeout("timeout"), ValueError("bad response")]
)
def test_offline_or_failed_query_preserves_usable_legacy_model(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """无网、超时或元数据损坏均不影响旧模型使用，且短期内不会反复请求。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    query = Mock(side_effect=error)
    monkeypatch.setattr(upgrade, "_remote_manifest", query)

    assert repo.ensure() == repo.local_dir()
    _join(workers)
    assert repo.ensure() == repo.local_dir()
    _join(workers)

    assert download.verify_model_repo(repo).ready
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    query.assert_called_once()


def test_offline_device_can_upgrade_after_retry_cooldown(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """设备从无网恢复联网后，冷却结束的下一次使用会重新尝试迁移。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    now = [1000.0]
    monkeypatch.setattr(upgrade, "monotonic", lambda: now[0])
    query = Mock(side_effect=[requests.ConnectionError("offline"), _FILES])
    monkeypatch.setattr(upgrade, "_remote_manifest", query)

    repo.ensure()
    _join(workers)
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    now[0] += upgrade._RETRY_COOLDOWN + 1
    repo.ensure()
    _join(workers)

    assert query.call_count == 2
    assert json.loads((repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_text())["files"] == _FILES


def test_pending_metadata_query_does_not_block_model_use(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """远端查询被阻塞时，模型加载仍可返回，重复调用也只保留一个后台任务。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def query(source: str, repo_id: str, revision: str) -> dict[str, int]:
        """将查询保持在进行中，以检测调用方是否错误等待网络。"""
        entered.set()
        assert release.wait(timeout=3)
        return _FILES

    remote = Mock(side_effect=query)
    monkeypatch.setattr(upgrade, "_remote_manifest", remote)

    def load() -> None:
        """模拟正常使用线程，通过事件报告 ensure 已经返回。"""
        repo.ensure()
        returned.set()

    caller = threading.Thread(target=load)
    caller.start()
    try:
        assert entered.wait(timeout=1)
        assert returned.wait(timeout=1)
        assert repo.ensure() == repo.local_dir()
        assert remote.call_count == 1
        assert all(worker.daemon for worker in workers)
    finally:
        release.set()
        caller.join(timeout=3)
        _join(workers)


def test_background_upgrade_uses_captured_model_directory(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """启动任务后配置上下文切换，也只更新最初使用的目录，不串到其他模型目录。"""
    repo, workers = environment
    root = repo.local_dir()
    _write_model(root)
    entered, release = threading.Event(), threading.Event()

    def query(source: str, repo_id: str, revision: str) -> dict[str, int]:
        """在主线程切换配置之前暂停元数据返回。"""
        entered.set()
        assert release.wait(timeout=3)
        return _FILES

    monkeypatch.setattr(upgrade, "_remote_manifest", query)
    repo.ensure()
    try:
        assert entered.wait(timeout=1)
        monkeypatch.setattr(download.get_config().model, "base_dir", str(tmp_path / "other-context"))
    finally:
        release.set()
        _join(workers)

    assert json.loads((root / download.MODEL_COMPLETE_MARKER).read_text())["files"] == _FILES
    assert not (tmp_path / "other-context").exists()


def test_concurrent_repositories_are_bounded_and_can_retry_when_slot_is_free(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """多个仓库同时使用时迁移有并发上限，被让位的仓库可在后续使用时再次安排。"""
    repo, workers = environment
    repos = [download.ModelRepo(name=f"repo-{index}", repos=repo.repos) for index in range(3)]
    for target in repos:
        _write_model(target.local_dir())
    release = threading.Event()

    def query(source: str, repo_id: str, revision: str) -> dict[str, int]:
        """暂不完成现有请求，验证新仓库不会无限创建后台线程。"""
        assert release.wait(timeout=3)
        return _FILES

    monkeypatch.setattr(upgrade, "_remote_manifest", query)
    try:
        for target in repos:
            target.ensure()
        assert len(workers) == 2
        assert (repos[2].local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    finally:
        release.set()
        _join(workers)
    repos[2].ensure()
    _join(workers)

    assert len(workers) == 3
    assert all(download.verify_model_repo(target).ready for target in repos)
    assert json.loads((repos[2].local_dir() / download.MODEL_COMPLETE_MARKER).read_text())["files"] == _FILES


def test_background_thread_start_failure_does_not_break_model_use(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """解释器退出或系统无法创建线程时，旧模型仍正常返回并释放任务配额。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("cannot start thread")))

    assert repo.ensure() == repo.local_dir()
    assert not workers
    assert not upgrade._SCHEDULER.active
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""


@pytest.mark.parametrize("files", [{"config.json": 2, "model.safetensors": 99}, {**_FILES, "missing.json": 10}])
def test_mismatched_remote_manifest_keeps_old_marker(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, files: dict[str, int]
) -> None:
    """远端多出文件或大小不一致时保留旧标记，不删除、补下载或改写已有模型。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    monkeypatch.setattr(upgrade, "_remote_manifest", Mock(return_value=files))

    repo.ensure()
    _join(workers)

    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    assert (repo.local_dir() / "model.safetensors").read_bytes() == b"weights"
    assert not (repo.local_dir() / "missing.json").exists()


def test_partial_directory_migration_does_not_publish_root_completion(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """局部目录只升级自己的完整范围，不将单个模型目录当作整个仓库。"""
    repo, workers = environment
    _write_model(repo.local_dir(), scope="weights")
    remote = {f"weights/{name}": size for name, size in _FILES.items()}
    monkeypatch.setattr(upgrade, "_remote_manifest", Mock(return_value={**remote, "other/model.bin": 100}))

    repo.path("weights").ensure()
    _join(workers)

    assert json.loads((repo.local_dir() / "weights" / download.MODEL_COMPLETE_MARKER).read_text())["files"] == _FILES
    assert not (repo.local_dir() / download.MODEL_COMPLETE_MARKER).exists()
    assert not download.verify_model_repo(repo).ready


@pytest.mark.parametrize("change", ["marker", "payload"])
def test_concurrent_update_is_not_overwritten_by_migration(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """请求网络期间其他操作更新标记或模型文件时，后台迁移不能覆盖新状态。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    other = {"version": 1, "files": {"model.safetensors": 7}}

    def query(source: str, repo_id: str, revision: str) -> dict[str, int]:
        """在网络返回之前模拟另一个下载或更新操作。"""
        if change == "marker":
            marker.write_text(json.dumps(other), encoding="utf-8")
        else:
            (repo.local_dir() / "model.safetensors").write_bytes(b"changed payload")
        return _FILES

    monkeypatch.setattr(upgrade, "_remote_manifest", query)
    repo.ensure()
    _join(workers)

    if change == "marker":
        assert json.loads(marker.read_text()) == other
    else:
        assert marker.read_bytes() == b""
        assert (repo.local_dir() / "model.safetensors").read_bytes() == b"changed payload"


def test_payload_is_rechecked_inside_repository_lock(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """网络返回后的第一次检查通过，也不能省略锁内对文件变更的第二次检查。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    monkeypatch.setattr(upgrade, "_remote_manifest", Mock(return_value=_FILES))
    verify = upgrade.missing_manifest_files
    calls = []

    def check(root: Path, files: dict[str, int]) -> list[str]:
        """在进入锁内复核前模拟大小发生变化。"""
        calls.append(root)
        if len(calls) == 2:
            (root / "model.safetensors").write_bytes(b"changed payload")
        return verify(root, files)

    monkeypatch.setattr(upgrade, "missing_manifest_files", check)
    repo.ensure()
    _join(workers)

    assert len(calls) == 2
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""


@pytest.mark.parametrize("failure", ["lock", "write"])
def test_busy_or_readonly_repository_keeps_legacy_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """锁被正常下载占用或目录不允许发布清单时，迁移安静让位并保留旧文件。"""
    root = tmp_path / "repo"
    _write_model(root)
    request = _request(root)
    monkeypatch.setattr(upgrade, "_SCHEDULER", upgrade._UpgradeScheduler())
    monkeypatch.setattr(upgrade, "_remote_manifest", Mock(return_value=_FILES))
    if failure == "write":
        monkeypatch.setattr(upgrade, "write_manifest", Mock(side_effect=PermissionError("read only")))
        worker = upgrade.schedule_manifest_upgrade(request)
        assert worker is not None
        _join([worker])
    else:
        with FileLock(str(request.lock_path)):
            worker = upgrade.schedule_manifest_upgrade(request)
            assert worker is not None
            _join([worker])

    assert (root / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    assert (root / "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize("cache", ["missing", "mixed", "unsafe_pickle"])
def test_unidentified_cached_revision_is_not_replaced_with_latest(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, cache: str
) -> None:
    """缺少可信旧版本、混合版本或包含可执行对象的缓存均跳过迁移。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    metadata = repo.local_dir() / ".cache/huggingface/download"
    if cache == "mixed":
        (metadata / "config.json.metadata").write_text(f"{'c' * 40}\netag\n1000\n", encoding="utf-8")
    else:
        for path in metadata.rglob("*.metadata"):
            path.unlink()
        if cache == "unsafe_pickle":
            (repo.local_dir() / ".msc").write_bytes(b"cos\nsystem\n.")
    query = Mock(side_effect=AssertionError("unknown cached revisions must not query latest"))
    monkeypatch.setattr(upgrade, "_remote_manifest", query)

    repo.ensure()
    _join(workers)

    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    query.assert_not_called()


def test_modelscope_cache_can_upgrade_using_metadata_only(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ModelScope 的旧缓存也能在 local 模式下通过带超时的元数据 GET 完成升级。"""
    repo, workers = environment
    _write_model(repo.local_dir(), source="modelscope")
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {
        "Code": 200,
        "Data": {
            "Files": [{"Path": name, "Size": size, "Type": "blob", "Revision": _REVISION} for name, size in _FILES.items()]
        },
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(upgrade.requests, "get", get)
    snapshot = Mock(side_effect=AssertionError("metadata upgrade cannot invoke snapshot_download"))
    monkeypatch.setattr(download, "ms_snapshot_download", snapshot)

    repo.ensure()
    _join(workers)

    assert json.loads((repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_text())["files"] == _FILES
    get.assert_called_once_with(
        "https://modelscope.cn/api/v1/models/Owner/repo/repo/files",
        params={"Revision": _REVISION, "Recursive": "True"},
        timeout=(upgrade._QUERY_CONNECT_TIMEOUT, upgrade._QUERY_TIMEOUT),
    )
    snapshot.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_modelscope_mixed_file_commits_must_match_cached_versions(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    """MS 正常的逐文件多 commit 缓存可升级；远端任一文件版本变化时保留旧标记。"""
    repo, workers = environment
    _write_model(repo.local_dir(), source="modelscope")
    revisions = {"config.json": "c" * 40, "model.safetensors": _REVISION}
    (repo.local_dir() / ".msc").write_bytes(
        pickle.dumps([{"Path": name, "Revision": revision} for name, revision in revisions.items()])
    )
    (repo.local_dir() / ".mv").write_text("Revision:v1.0,CreatedAt:1000", encoding="utf-8")
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {
        "Code": 200,
        "Data": {
            "Files": [
                {
                    "Path": name,
                    "Size": size,
                    "Type": "blob",
                    "Revision": "d" * 40 if changed and name == "model.safetensors" else revisions[name],
                }
                for name, size in _FILES.items()
            ]
        },
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(upgrade.requests, "get", get)

    repo.ensure()
    _join(workers)

    assert get.call_args.kwargs["params"]["Revision"] == "v1.0"
    marker = repo.local_dir() / download.MODEL_COMPLETE_MARKER
    if changed:
        assert marker.read_bytes() == b""
    else:
        assert json.loads(marker.read_text())["files"] == _FILES


def test_huggingface_query_pins_cached_revision_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF 请求明确固定旧 commit 并开启文件大小字段，避免隐式访问最新版或无限等待。"""
    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha=_REVISION, siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in _FILES.items()]
    )
    monkeypatch.setattr(upgrade, "HfApi", Mock(return_value=api))

    assert upgrade._remote_manifest("huggingface", "owner/repo", _REVISION) == _FILES
    api.model_info.assert_called_once_with(
        "owner/repo", revision=_REVISION, files_metadata=True, timeout=upgrade._QUERY_TIMEOUT
    )
    api.model_info.return_value.sha = "c" * 40
    assert upgrade._remote_manifest("huggingface", "owner/repo", _REVISION) == {}


def test_readonly_verification_does_not_schedule_network_upgrade(
    environment: tuple[download.ModelRepo, list[threading.Thread]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """show/verify 使用的只读检查不启动后台任务、不修改旧标记。"""
    repo, workers = environment
    _write_model(repo.local_dir())
    schedule = Mock(side_effect=AssertionError("readonly verification cannot schedule migration"))
    monkeypatch.setattr(download, "schedule_manifest_upgrade", schedule)

    assert download.verify_model_repo(repo).ready
    assert download.model_path_exists(repo.path(""))
    assert not workers
    assert (repo.local_dir() / download.MODEL_COMPLETE_MARKER).read_bytes() == b""
    schedule.assert_not_called()
