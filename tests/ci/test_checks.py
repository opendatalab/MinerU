"""用真实 Git 差异和模拟 wheel 验证 CI 分流及发布身份，不依赖 MinerU 运行时。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.version import Version

from scripts.ci.changes import changed_paths, classify_paths
from scripts.ci.check_release import source_version, verify_wheel

ROOT = Path(__file__).resolve().parents[2]


def git(root: Path, *args: str) -> str:
    """只在测试临时仓库执行 Git，避免触碰开发者的分支和暂存区。"""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """建立独立仓库，允许测试真实提交差异、标签校验和工作区漂移。"""
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.name", "CI tests")
    git(tmp_path, "config", "user.email", "ci@example.invalid")
    (tmp_path / "mineru").mkdir()
    (tmp_path / "mineru/version.py").write_text('__version__ = "4.0.12"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="mineru"\nrequires-python=">=3.11,<3.15"\nclassifiers=["Programming Language :: Python :: 3.11"]\n',
        encoding="utf-8",
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("paths", "code", "docs"),
    [
        (["docs/next/api/tasks.md"], False, True),
        (["docs/requirements.txt"], False, True),
        (["docs/images/example.png"], False, True),
        (["README_zh-CN.md"], False, True),
        (["mkdocs.yml"], False, True),
        (["mineru/parser/base.py"], True, False),
        (["scripts/http_task_example.py"], True, False),
        (["scripts/ci/changes.py"], True, True),
        ([".github/workflows/cli.yml"], True, False),
        (["docs/index.md", "mineru/config.py"], True, True),
        ([], False, False),
    ],
)
def test_change_classification(paths: list[str], code: bool, docs: bool) -> None:
    """真实产品、文档资源和公共分流脚本分别触发所需检查。"""
    assert classify_paths(paths) == {"code": code, "docs": docs}


def test_rename_out_of_code_is_not_misclassified_as_docs_only(repository: Path) -> None:
    """移动源码到文档目录仍包含原路径删除，不能绕过代码测试。"""
    base = git(repository, "rev-parse", "HEAD")
    (repository / "docs").mkdir()
    git(repository, "mv", "mineru/version.py", "docs/version.md")
    git(repository, "commit", "-m", "move")
    head = git(repository, "rev-parse", "HEAD")
    paths = changed_paths("pull_request", {"pull_request": {"base": {"sha": base}}}, head)
    assert paths == ["docs/version.md", "mineru/version.py"]
    assert classify_paths(paths) == {"code": True, "docs": True}


@pytest.mark.parametrize("event", [{"before": "0" * 40}, {"ref": "refs/tags/mineru-4.0.12-released"}])
def test_first_push_and_release_run_every_check(event: dict[str, str]) -> None:
    """没有可信差异基点的新分支和 release 必须完整验证。"""
    assert changed_paths("push", event, "1" * 40) is None
    assert changed_paths("workflow_dispatch", {}, "1" * 40) is None


def test_changes_script_writes_outputs_for_docs_only_push(repository: Path, tmp_path: Path) -> None:
    """执行真实 CLI 并检查 Actions 输出，验证 push 的多提交差异入口。"""
    base = git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("docs", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "docs")
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps({"before": base, "ref": "refs/heads/main"}), encoding="utf-8")
    output = tmp_path / "outputs"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/changes.py")],
        check=True,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_EVENT_PATH": str(payload),
            "GITHUB_SHA": git(repository, "rev-parse", "HEAD"),
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert output.read_text() == "code=false\ndocs=true\n"


@pytest.mark.parametrize("tag", ["mineru-4.0.13-released", "mineru-4.0.11-released", "v4.0.12", "mineru-bad-released"])
def test_release_rejects_mismatched_tag_without_rewriting_source(repository: Path, tag: str) -> None:
    """不同版本及非法标签必须失败，不能通过改源码或回退版本掩盖错误。"""
    version_file = repository / "mineru/version.py"
    before = version_file.read_bytes()
    with pytest.raises(ValueError):
        source_version(repository, tag)
    assert version_file.read_bytes() == before


def wheel(directory: Path, *, version: str = "4.0.12", requires_python: str = ">=3.11,<3.15") -> Path:
    """构造最小发行包以测试实际 METADATA 和摘要，无需安装模型依赖。"""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mineru-4.0.12-py3-none-any.whl"
    with ZipFile(target, "w") as archive:
        archive.writestr(
            "mineru-4.0.12.dist-info/METADATA",
            f"Name: mineru\nVersion: {version}\nRequires-Python: {requires_python}\n"
            "Classifier: Programming Language :: Python :: 3.11\n",
        )
    return target


@pytest.mark.parametrize("failure", ["version", "python", "multiple", "missing"])
def test_release_rejects_wrong_or_ambiguous_wheel(repository: Path, failure: str) -> None:
    """构建内容漂移、产物缺失或多个候选不能进入发布。"""
    directory = repository / "dist"
    artifact = wheel(
        directory,
        version="4.0.13" if failure == "version" else "4.0.12",
        requires_python=">=3.10" if failure == "python" else ">=3.11,<3.15",
    )
    if failure == "multiple":
        (directory / "another.whl").write_bytes(artifact.read_bytes())
    elif failure == "missing":
        artifact.unlink()
    with pytest.raises(ValueError):
        verify_wheel(repository, directory, Version("4.0.12"))


def test_release_cli_records_exact_commit_and_rejects_dirty_source(repository: Path) -> None:
    """端到端校验标签、源码 SHA 和 wheel 摘要，未提交的版本漂移必须失败。"""
    artifact = wheel(repository / "dist")
    head = git(repository, "rev-parse", "HEAD")
    command = [
        sys.executable,
        str(ROOT / "scripts/ci/check_release.py"),
        "--commit",
        head,
        "--tag",
        "mineru-4.0.12-released",
        "--wheel-dir",
        str(artifact.parent),
        "--output",
        "release.json",
    ]
    subprocess.run(command, check=True)
    record = json.loads((repository / "release.json").read_text())
    assert record["source_commit"] == head and record["version"] == "4.0.12"
    assert record["wheel"] == artifact.name and len(record["wheel_sha256"]) == 64
    (repository / "mineru/version.py").write_text('__version__ = "4.0.13"\n', encoding="utf-8")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert json.loads((repository / "release.json").read_text()) == record
