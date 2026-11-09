"""ModelScope 固定 commit 的元数据解析与下载一致性回归。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from mineru.model import download

_REVISION = "a" * 40


@pytest.mark.parametrize(
    ("commit", "entries", "expected"),
    [
        ({"Id": _REVISION}, [], _REVISION),
        ({"Id": "", "ShortId": _REVISION[:8]}, [{"Revision": _REVISION}], _REVISION),
        ({"ShortId": _REVISION[:8]}, [], _REVISION[:8]),
        ({"ShortId": _REVISION[:8]}, [{"Revision": _REVISION}, {"Revision": _REVISION[:8] + "b" * 32}], None),
    ],
)
def test_modelscope_commit_resolution_preserves_endpoint_and_auth(
    monkeypatch: pytest.MonkeyPatch, commit: dict[str, str], entries: list[dict[str, str]], expected: str | None
) -> None:
    """从同次仓库响应解析固定 commit，并沿用 SDK 的端点、请求头与认证。"""
    api = SimpleNamespace(
        endpoint="https://modelscope.example/", headers={"user-agent": "sdk"}, get_cookies=Mock(return_value={})
    )
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", SimpleNamespace(HubApi=Mock(return_value=api)))
    monkeypatch.setitem(sys.modules, "modelscope.utils.constant", SimpleNamespace(DEFAULT_MODEL_REVISION="master"))
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {"Code": 200, "Data": {"LatestCommitter": commit, "Files": entries}}
    request = Mock(return_value=response)
    monkeypatch.setattr(download.requests, "get", request)

    if expected is None:
        with pytest.raises(ValueError, match="ambiguous commit"):
            download.ms_model_revision("Owner/repo")
    else:
        assert download.ms_model_revision("Owner/repo") == expected
    request.assert_called_once_with(
        "https://modelscope.example/api/v1/models/Owner/repo/repo/files",
        params={"Revision": "master", "Recursive": "False"},
        headers=api.headers,
        cookies={},
        timeout=(5, 30),
    )
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("commit", [{}, {"Id": "master"}, {"ShortId": "latest"}, {"Id": 123}])
def test_modelscope_missing_commit_is_not_replaced_with_mutable_branch(
    monkeypatch: pytest.MonkeyPatch, commit: dict[str, object]
) -> None:
    """无法确认固定 commit 时明确失败，不能静默回退到会变化的 master。"""
    api = SimpleNamespace(endpoint="https://modelscope.example", headers={}, get_cookies=lambda: {})
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", SimpleNamespace(HubApi=lambda: api))
    monkeypatch.setitem(sys.modules, "modelscope.utils.constant", SimpleNamespace(DEFAULT_MODEL_REVISION="master"))
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {"Code": 200, "Data": {"LatestCommitter": commit, "Files": []}}
    monkeypatch.setattr(download.requests, "get", Mock(return_value=response))
    with pytest.raises(ValueError, match="fixed commit"):
        download.ms_model_revision("Owner/repo")


def test_modelscope_snapshot_wrapper_forwards_fixed_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """下载封装必须把与清单相同的 commit 传给 SDK。"""
    snapshot = Mock(return_value="local-model")
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=snapshot))
    assert download.ms_snapshot_download("Owner/repo", local_dir="local-model", revision=_REVISION) == "local-model"
    snapshot.assert_called_once_with("Owner/repo", local_dir="local-model", revision=_REVISION, allow_patterns=None)


def test_modelscope_branch_update_does_not_change_selected_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """分支在列清单后更新时仍下载原 commit，避免完整新文件被旧清单误拒绝。"""
    monkeypatch.setattr(download.get_config().model, "base_dir", str(tmp_path))
    repo = download.ModelRepo(name="repo", repos={"modelscope": "Owner/repo"}, paths={"layout": "Layout/weights"})
    branch = {"revision": _REVISION}
    monkeypatch.setattr(download, "ms_model_revision", lambda repo_id: branch["revision"])

    def files(repo_id: str, *, revision: str) -> dict[str, int]:
        """模拟读取旧 commit 后远端分支立即推进。"""
        assert revision == _REVISION
        branch["revision"] = "b" * 40
        return {"Layout/weights/model.bin": 7}

    def snapshot(repo_id: str, *, local_dir: str, revision: str, allow_patterns: list[str]) -> str:
        """按实际传入的 commit 选择内容，漏传固定版本时将得到不同大小的新权重。"""
        assert branch["revision"] != _REVISION
        target = Path(local_dir) / "Layout/weights/model.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"weights" if revision == _REVISION else b"new weights")
        return local_dir

    monkeypatch.setattr(download, "ms_model_files", files)
    monkeypatch.setattr(download, "ms_snapshot_download", snapshot)
    repo.layout.ensure(source="modelscope")
    assert (repo.layout.local_path() / "model.bin").read_bytes() == b"weights"
    assert download.model_path_exists(repo.layout)
