"""用模拟发布产物与临时 Git 仓库验证 issue 模板同步，无需发布或安装 MinerU。"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
SCRIPT = ROOT / "scripts" / "sync_issue_template_version.py"
WORKFLOW = ROOT / ".github" / "workflows" / "python-package.yml"
TEMPLATE_RELATIVE = TEMPLATE.relative_to(ROOT)


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    """复制真实模板，确保同步测试不会写入仓库中的表单。"""
    target = tmp_path / "bug_report.yml"
    target.write_bytes(TEMPLATE.read_bytes())
    return target


def _wheel(directory: Path, metadata: str = "Name: mineru\nVersion: 4.0.13\n") -> Path:
    """构造只含发行元数据的模拟 wheel，隔离模型依赖和网络访问。"""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mineru-test-py3-none-any.whl"
    with ZipFile(target, "w") as wheel:
        wheel.writestr("mineru-test.dist-info/METADATA", metadata)
    return target


def _run_sync(wheel: Path, template: Path, tag: str = "mineru-4.0.13-released") -> subprocess.CompletedProcess[str]:
    """通过真实脚本入口执行同步，覆盖参数解析、错误退出和文件写入。"""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--wheel", str(wheel), "--template", str(template), "--tag", tag],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("version", ["4.0.13", "4.0.100"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_new_release_preserves_other_text_and_line_endings(template_path: Path, version: str, newline: str) -> None:
    """验证 PEP 440 升级、注释保留、换行保留和重复运行不写文件。"""
    source = TEMPLATE.read_text(encoding="utf-8") + "# Unrelated version example: 4.0.12\n"
    original = source.replace("\n", newline).encode("utf-8")
    template_path.write_bytes(original)
    wheel = _wheel(template_path.parent, f"Name: mineru\nVersion: {version}\n")
    result = _run_sync(wheel, template_path, f"mineru-{version}-released")
    assert result.returncode == 0, result.stderr
    expected = original.replace(
        "Latest stable release: 4.0.12 | 最新正式版本：4.0.12".encode(),
        f"Latest stable release: {version} | 最新正式版本：{version}".encode(),
    )
    assert template_path.read_bytes() == expected
    modified = template_path.stat().st_mtime_ns
    repeated = _run_sync(wheel, template_path, f"mineru-{version}-released")
    assert repeated.returncode == 0, repeated.stderr
    assert template_path.stat().st_mtime_ns == modified


@pytest.mark.parametrize("version", ["4.0.12", "4.0.11", "4.0.9", "5.0.0a1", "5.0.0rc1", "5.0.0.dev1", "4.0.13+local"])
def test_older_and_non_stable_versions_do_not_change_hint(template_path: Path, version: str) -> None:
    """旧版回放、预发布和本地开发构建不能覆盖正式版本提示。"""
    original = template_path.read_bytes()
    wheel = _wheel(template_path.parent, f"Name: mineru\nVersion: {version}\n")
    result = _run_sync(wheel, template_path, f"mineru-{version}-released")
    assert result.returncode == 0, result.stderr
    assert template_path.read_bytes() == original


@pytest.mark.parametrize(
    "metadata",
    [
        "Name: other-project\nVersion: 4.0.13\n",
        "Version: 4.0.13\n",
        "Name: mineru\n",
        "Name: mineru\nVersion: invalid\n",
        "Name: mineru\nVersion: 4.0.13\nVersion: 4.0.14\n",
        "Name: mineru\nName: other-project\nVersion: 4.0.13\n",
    ],
)
def test_invalid_metadata_fails_without_writing(template_path: Path, metadata: str) -> None:
    """缺失、重复或不属于 MinerU 的发行元数据必须失败并保留原文件。"""
    original = template_path.read_bytes()
    result = _run_sync(_wheel(template_path.parent, metadata), template_path)
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert template_path.read_bytes() == original


@pytest.mark.parametrize("tag", ["mineru-4.0.14-released", "v4.0.13", "mineru-invalid-released"])
def test_invalid_or_mismatched_tag_fails_without_writing(template_path: Path, tag: str) -> None:
    """不接受标签与发布产物不一致或格式非法的同步请求。"""
    original = template_path.read_bytes()
    result = _run_sync(_wheel(template_path.parent), template_path, tag)
    assert result.returncode == 1
    assert template_path.read_bytes() == original


@pytest.mark.parametrize("failure", ["corrupt", "missing-metadata", "multiple-metadata", "no-wheel", "multiple-wheels"])
def test_ambiguous_or_broken_artifact_fails_without_writing(template_path: Path, failure: str) -> None:
    """损坏或含有多种候选产物的目录不能被当成唯一发布 wheel。"""
    original = template_path.read_bytes()
    directory = template_path.parent / "dist"
    wheel_path = _wheel(directory)
    if failure == "corrupt":
        wheel_path.write_bytes(b"invalid zip")
    elif failure == "missing-metadata":
        with ZipFile(wheel_path, "w") as wheel:
            wheel.writestr("unrelated.txt", "missing metadata")
    elif failure == "multiple-metadata":
        with ZipFile(wheel_path, "a") as wheel:
            wheel.writestr("other.dist-info/METADATA", "Name: other\nVersion: 4.0.13\n")
    elif failure == "no-wheel":
        wheel_path.unlink()
    else:
        shutil.copyfile(wheel_path, directory / "another.whl")
    result = _run_sync(directory, template_path)
    assert result.returncode == 1
    assert template_path.read_bytes() == original


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("# END MINERU_LATEST_STABLE_VERSION", "# missing marker"),
        (
            "# BEGIN MINERU_LATEST_STABLE_VERSION",
            "# BEGIN MINERU_LATEST_STABLE_VERSION\n      # BEGIN MINERU_LATEST_STABLE_VERSION",
        ),
        ("      # END MINERU_LATEST_STABLE_VERSION", "    # END MINERU_LATEST_STABLE_VERSION"),
        ("最新正式版本：4.0.12", "最新正式版本：4.0.11"),
        ("4.0.12", "5.0.0rc1"),
        ("id: software_version", "id: wrong_field"),
        ("  - type: textarea\n    id: software_version", "  - type: [textarea]\n    id: software_version"),
        ("id: device_mode", "id: software_version"),
        ("name: 🐛 Bug Report", "name: [invalid yaml"),
    ],
)
def test_invalid_template_fails_without_writing(template_path: Path, old: str, new: str) -> None:
    """损坏的标记、版本、字段归属和 YAML 在写入前被拒绝。"""
    source = template_path.read_text(encoding="utf-8").replace(old, new)
    template_path.write_text(source, encoding="utf-8")
    original = template_path.read_bytes()
    result = _run_sync(_wheel(template_path.parent), template_path)
    assert result.returncode == 1
    assert template_path.read_bytes() == original


def test_issue_form_accepts_installation_and_current_product_reports() -> None:
    """真实表单覆盖当前入口，安装失败也可以填写所有必填环境字段。"""
    form = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    ids = [field["id"] for field in form["body"] if "id" in field]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", field_id) for field_id in ids)
    fields = {field["id"]: field for field in form["body"] if "id" in field}
    for field in form["body"]:
        if field["type"] == "dropdown":
            options = field["attributes"]["options"]
            assert all(isinstance(option, str) and option.strip() for option in options)
            assert len(options) == len(set(options))
            assert "default" not in field["attributes"]
    assert fields["software_version"]["type"] == "textarea"
    assert fields["software_version"]["validations"]["required"] is True
    assert "value" not in fields["software_version"]["attributes"]
    assert "installation failed" in fields["software_version"]["attributes"]["description"]
    assert "3.14" in fields["python_version"]["attributes"]["options"]
    assert fields["tier"]["attributes"]["options"][:4] == ["flash", "basic", "standard", "advanced"]
    assert fields["parse_mode"]["attributes"]["options"][:3] == ["auto", "txt", "ocr"]
    assert not fields["parse_mode"].get("validations", {}).get("required", False)
    assert fields["device_mode"]["type"] == "textarea"
    assert fields["device_mode"]["validations"]["required"] is True
    diagnostics = fields["device_mode"]["attributes"]
    assert diagnostics["render"] == "text"
    assert "value" not in diagnostics
    assert "mineru-kit hardware info" in diagnostics["placeholder"]
    assert "mineru-kit hardware info" in diagnostics["description"]
    for guidance in (
        "installation failed",
        "does not support the command",
        "cannot access the server",
        "error output",
        "known environment",
    ):
        assert guidance in diagnostics["description"]
    assert "Installation / build | 安装或构建" in fields["usage_entry"]["attributes"]["options"]
    ai_field = next(field for field in form["body"] if "🤖" in field["attributes"].get("label", ""))
    assert all(option.get("required", False) is False for option in ai_field["attributes"]["options"])


def _workflow_script() -> str:
    """直接读取发版 job 的运行脚本，让集成测试执行 CI 中的原始逻辑。"""
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["sync-issue-template"]
    return next(step["run"] for step in job["steps"] if "run" in step)


def test_workflow_sync_requires_successful_tag_publication() -> None:
    """同步必须依赖发布成功，使用默认分支、发布产物和既有写入凭据。"""
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["sync-issue-template"]
    assert job["needs"] == ["publish"]
    assert "startsWith(github.ref, 'refs/tags/')" in job["if"]
    assert "needs.publish.result == 'success'" in job["if"]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["token"] == "${{ secrets.RELEASE_TOKEN }}"
    download = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/download-artifact@"))
    assert download["with"]["name"] == "wheel-file"
    assert job["env"]["RELEASE_TAG"] == "${{ github.ref_name }}"
    assert "${{" not in _workflow_script()


def _git(directory: Path, *args: str) -> str:
    """只在临时仓库调用 Git，返回输出便于断言提交与远端状态。"""
    return subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def release_checkout(tmp_path: Path) -> Path:
    """建立本地 bare 远端和专用 checkout，模拟机器人同步而不访问 GitHub。"""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "--initial-branch=master")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", str(remote), str(checkout))
    _git(checkout, "config", "user.name", "Release sync test")
    _git(checkout, "config", "user.email", "release-sync-test@example.invalid")
    for source in (TEMPLATE, SCRIPT):
        target = checkout / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (checkout / "untouched.txt").write_text("preserve unrelated content\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "seed release template")
    _git(checkout, "push", "origin", "HEAD:master")
    return checkout


def _workflow_environment(tmp_path: Path) -> dict[str, str]:
    """替换 uv 启动器以复用测试 Python，避免 Git 集成测试下载依赖。"""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    launcher = binaries / "uv"
    launcher.write_text("#!/bin/sh\nshift 4\n" + f'exec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    wheel = _wheel(tmp_path / "published-dist")
    return {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "RELEASE_BRANCH": "master",
        "RELEASE_TAG": "mineru-4.0.13-released",
        "RELEASE_WHEEL_DIR": str(wheel.parent),
    }


def _run_workflow(checkout: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """执行 CI 同步步骤本身，保留与 Actions 相同的失败终止行为。"""
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _workflow_script()],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_workflow_only_commits_template_and_repeated_run_is_noop(release_checkout: Path, tmp_path: Path) -> None:
    """真实提交只能包含表单，重复同步不能新增机器人提交。"""
    env = _workflow_environment(tmp_path)
    result = _run_workflow(release_checkout, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(release_checkout, "show", "--format=", "--name-only", "HEAD") == str(TEMPLATE_RELATIVE)
    assert "4.0.13" in (release_checkout / TEMPLATE_RELATIVE).read_text(encoding="utf-8")
    committed = _git(release_checkout, "rev-parse", "HEAD")
    repeated = _run_workflow(release_checkout, env)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert _git(release_checkout, "rev-parse", "HEAD") == committed
    assert _git(release_checkout, "status", "--porcelain") == ""


@pytest.mark.parametrize("newer_remote_version", [False, True])
def test_workflow_retry_preserves_concurrent_commits(
    release_checkout: Path, tmp_path: Path, newer_remote_version: bool
) -> None:
    """首次推送前模拟他人提交，验证重新生成能保留修改且不会回退较新版本。"""
    real_git = shutil.which("git")
    assert real_git is not None
    competing = tmp_path / "competing"
    _git(tmp_path, "clone", str(tmp_path / "remote.git"), str(competing))
    _git(competing, "config", "user.name", "Concurrent test")
    _git(competing, "config", "user.email", "concurrent-test@example.invalid")
    (competing / "concurrent.txt").write_text("keep this concurrent commit\n", encoding="utf-8")
    competing_template = competing / TEMPLATE_RELATIVE
    source = competing_template.read_text(encoding="utf-8") + "# Concurrent form edit\n"
    if newer_remote_version:
        source = source.replace("4.0.12", "4.0.14")
    competing_template.write_text(source, encoding="utf-8")
    _git(competing, "add", ".")
    _git(competing, "commit", "-m", "concurrent default branch update")
    env = _workflow_environment(tmp_path)
    wrapper = tmp_path / "bin" / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = push ] && [ ! -e "$SYNC_RACE_FLAG" ]; then\n'
        '  touch "$SYNC_RACE_FLAG"\n'
        '  "$REAL_GIT" -C "$COMPETING_CHECKOUT" push origin HEAD:master || exit 1\n'
        "fi\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env.update(REAL_GIT=real_git, COMPETING_CHECKOUT=str(competing), SYNC_RACE_FLAG=str(tmp_path / "race-injected"))
    result = _run_workflow(release_checkout, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Push attempt 1 failed" in result.stdout
    updated = (release_checkout / TEMPLATE_RELATIVE).read_text(encoding="utf-8")
    expected_version = "4.0.14" if newer_remote_version else "4.0.13"
    assert f"Latest stable release: {expected_version}" in updated
    assert "# Concurrent form edit" in updated
    assert (release_checkout / "concurrent.txt").read_text(encoding="utf-8") == "keep this concurrent commit\n"
    assert _git(release_checkout, "rev-parse", "HEAD") == _git(release_checkout, "rev-parse", "origin/master")


def test_workflow_reports_failure_after_three_rejected_pushes(release_checkout: Path, tmp_path: Path) -> None:
    """模拟持续拒绝推送，确认重试有界且以失败退出而非误报同步成功。"""
    real_git = shutil.which("git")
    assert real_git is not None
    env = _workflow_environment(tmp_path)
    wrapper = tmp_path / "bin" / "git"
    wrapper.write_text(
        '#!/bin/sh\nif [ "$1" = push ]; then\n  echo push >> "$PUSH_ATTEMPTS"\n  exit 1\nfi\nexec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    attempts = tmp_path / "push-attempts"
    env.update(REAL_GIT=real_git, PUSH_ATTEMPTS=str(attempts))
    result = _run_workflow(release_checkout, env)
    assert result.returncode == 1
    assert "::error::Could not synchronize the issue template after three push attempts." in result.stdout
    assert attempts.read_text(encoding="utf-8").splitlines() == ["push", "push", "push"]
    assert _git(release_checkout, "show", f"origin/master:{TEMPLATE_RELATIVE}") == TEMPLATE.read_text(encoding="utf-8").strip()
