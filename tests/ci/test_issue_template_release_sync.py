"""用模拟发行记录、wheel 和本地 Git 仓库验证表单同步，不导入 MinerU 或访问网络。"""

from __future__ import annotations

import json
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
from packaging.version import Version

from scripts.sync_issue_template_version import read_release_versions

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
TEMPLATES = (TEMPLATE, TEMPLATE.with_name("bug_report_zh.yml"))
SCRIPT = ROOT / "scripts" / "sync_issue_template_version.py"
WORKFLOW = ROOT / ".github" / "workflows" / "python-package.yml"
TEMPLATE_RELATIVE = TEMPLATE.relative_to(ROOT)
BASELINE = ("4.0.12", "4.0.11", "4.0.10", "4.0.9", "4.0.8")
REGION = re.compile(r"(?<=# BEGIN MINERU_RECENT_RELEASES\n).*?(?=      # END MINERU_RECENT_RELEASES)", re.DOTALL)


def _baseline_template(template: Path = TEMPLATE) -> str:
    """保留实际语言和表单结构，用固定起始版本隔离后续正式发版。"""
    source = template.read_text(encoding="utf-8")
    form = yaml.safe_load(source)
    field = next(item for item in form["body"] if item.get("id") == "software_version")
    fallback = field["attributes"]["options"][-1]
    options = "      options:\n" + "".join(
        "        - " + json.dumps(value, ensure_ascii=False) + "\n" for value in (*BASELINE, fallback)
    )
    options += f"      default: {len(BASELINE)}\n"
    updated, count = REGION.subn(options, source)
    assert count == 1
    return updated


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    """复制固定版本的英文表单，所有变更限制在临时目录。"""
    target = tmp_path / TEMPLATE.name
    target.write_text(_baseline_template(), encoding="utf-8")
    return target


def _wheel(directory: Path, metadata: str = "Name: mineru\nVersion: 4.0.13\n") -> Path:
    """构造最小发行产物，只保留脚本读取的包名和版本元数据。"""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mineru-test-py3-none-any.whl"
    with ZipFile(target, "w") as archive:
        archive.writestr("mineru-test.dist-info/METADATA", metadata)
    return target


def _release(version: str, **changes: object) -> dict[str, object]:
    """构造与 GitHub Release API 相同的发布身份和状态字段。"""
    return {
        "tag_name": f"mineru-{version}-released",
        "draft": False,
        "prerelease": False,
        "published_at": "2026-10-29T00:00:00Z",
        **changes,
    }


def _history(path: Path, versions: tuple[str, ...] = ("4.0.13", *BASELINE)) -> Path:
    """保存离线版本历史，测试和工作流替身都不请求真实 GitHub。"""
    path.write_text(json.dumps([_release(version) for version in versions]), encoding="utf-8")
    return path


def _options(path: Path) -> list[str]:
    """从完整表单读取实际下拉值，避免只验证版本字符串出现。"""
    form = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(field["attributes"]["options"] for field in form["body"] if field.get("id") == "software_version")


def _run_sync(
    wheel: Path, template: Path | list[Path], tag: str = "mineru-4.0.13-released", history: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """执行真实同步 CLI，覆盖参数、版本读取、多模板预校验与错误退出。"""
    history = history or _history(wheel.parent / "releases.json")
    command = [sys.executable, str(SCRIPT), "--wheel", str(wheel), "--tag", tag, "--releases-json", str(history)]
    for path in [template] if isinstance(template, Path) else template:
        command.extend(("--template", str(path)))
    return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)


def test_release_history_uses_semantic_order_and_only_public_stable_versions(tmp_path: Path) -> None:
    """乱序、重复和发布时间均不能改变语义排序，草稿及非正式版本被排除。"""
    history = tmp_path / "history.json"
    records = [
        _release("4.0.9"),
        _release("4.0.100"),
        _release("4.0.12"),
        _release("4.0.12"),
        _release("9.0.0", draft=True, published_at=None),
        _release("8.0.0", prerelease=True),
        _release("7.0.0rc1"),
        _release("7.0.0.dev1"),
        _release("7.0.0+local"),
        _release("unused", tag_name="unrelated-tag"),
    ]
    history.write_text(json.dumps(records), encoding="utf-8")
    assert read_release_versions(history) == tuple(map(Version, ("4.0.100", "4.0.12", "4.0.9")))


def test_paginated_release_history_includes_versions_from_every_page(tmp_path: Path) -> None:
    """使用 gh 分页输出的真实形状，跨页去重并按版本号而非分页位置排序。"""
    history = tmp_path / "pages.json"
    history.write_text(json.dumps([[_release("4.0.9")], [_release("4.0.12"), _release("4.0.9")]]), encoding="utf-8")
    assert read_release_versions(history) == (Version("4.0.12"), Version("4.0.9"))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        [None],
        [{"tag_name": "mineru-4.0.13-released"}],
        [_release("4.0.13", draft="false")],
        [_release("4.0.13", published_at=None)],
        [_release("invalid")],
        [_release("4.0.13", draft=True)],
    ],
)
def test_invalid_release_history_fails_without_writing(template_path: Path, payload: object) -> None:
    """空列表、API 错误响应及损坏记录不能修改模板或伪造版本。"""
    history = template_path.parent / "invalid-history.json"
    history.write_text(json.dumps(payload), encoding="utf-8")
    original = template_path.read_bytes()
    result = _run_sync(_wheel(template_path.parent), template_path, history=history)
    assert result.returncode == 1 and "error:" in result.stderr
    assert template_path.read_bytes() == original


@pytest.mark.parametrize("failure", ["invalid-json", "missing-file", "missing-release"])
def test_unavailable_release_data_fails_without_writing(template_path: Path, failure: str) -> None:
    """无效 JSON、读取失败或不包含当前发布的历史不能修改模板。"""
    history = template_path.parent / "history.json"
    if failure == "invalid-json":
        history.write_text("{", encoding="utf-8")
    elif failure == "missing-release":
        _history(history, BASELINE)
    original = template_path.read_bytes()
    assert _run_sync(_wheel(template_path.parent), template_path, history=history).returncode == 1
    assert template_path.read_bytes() == original


@pytest.mark.parametrize("template", TEMPLATES, ids=["en", "zh"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_update_keeps_five_versions_fallback_and_unrelated_text(tmp_path: Path, template: Path, newline: str) -> None:
    """两种语言均保留兜底、换行和无关文本，重复执行不触碰文件。"""
    target = tmp_path / template.name
    source = _baseline_template(template) + "# Unrelated version: 4.0.12\n"
    target.write_bytes(source.replace("\n", newline).encode("utf-8"))
    original = target.read_bytes()
    fallback = _options(target)[-1]
    wheel = _wheel(tmp_path)
    result = _run_sync(wheel, target)
    assert result.returncode == 0, result.stderr
    assert _options(target) == ["4.0.13", *BASELINE[:4], fallback]
    before = original.decode("utf-8").split("# BEGIN MINERU_RECENT_RELEASES")
    after = target.read_bytes().decode("utf-8").split("# BEGIN MINERU_RECENT_RELEASES")
    assert before[0] == after[0]
    assert before[1].split("# END MINERU_RECENT_RELEASES")[1] == after[1].split("# END MINERU_RECENT_RELEASES")[1]
    if newline == "\r\n":
        assert b"\n" not in target.read_bytes().replace(b"\r\n", b"")
    modified = target.stat().st_mtime_ns
    assert _run_sync(wheel, target).returncode == 0
    assert target.stat().st_mtime_ns == modified


def test_default_command_updates_both_forms_from_current_repository(tmp_path: Path) -> None:
    """不传模板参数时使用真实默认路径，同时更新两种语言。"""
    original_versions = _options(TEMPLATE)[:-1]
    latest = Version(original_versions[0])
    version = f"{latest.major}.{latest.minor}.{latest.micro + 1}"
    for template in TEMPLATES:
        target = tmp_path / template.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(template.read_bytes())
    history = _history(tmp_path / "history.json", (version, *original_versions))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--wheel",
            str(_wheel(tmp_path, f"Name: mineru\nVersion: {version}\n")),
            "--tag",
            f"mineru-{version}-released",
            "--releases-json",
            str(history),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    for template in TEMPLATES:
        assert _options(tmp_path / template.relative_to(ROOT))[:-1] == [version, *original_versions[:4]]


def test_old_release_replay_uses_latest_history_without_rollback(template_path: Path) -> None:
    """旧发布采用最新历史；陈旧快照则保留现有选择，不能回退任何选项。"""
    assert _run_sync(_wheel(template_path.parent), template_path).returncode == 0
    original = template_path.read_bytes()
    old_wheel = _wheel(template_path.parent, "Name: mineru\nVersion: 4.0.12\n")
    assert _run_sync(old_wheel, template_path, "mineru-4.0.12-released").returncode == 0
    assert template_path.read_bytes() == original
    stale = _history(template_path.parent / "stale.json", BASELINE)
    assert _run_sync(old_wheel, template_path, "mineru-4.0.12-released", stale).returncode == 0
    assert template_path.read_bytes() == original


@pytest.mark.parametrize(("version", "exit_code"), [("4.0.11", 0), ("4.0.12", 0), ("4.0.13", 1)])
def test_fixed_old_release_workflow_arguments_only_allow_safe_replay(tmp_path: Path, version: str, exit_code: int) -> None:
    """旧标签工作流缺少新增参数时只能安全重跑，不能无历史依据地推进版本列表。"""
    targets = []
    for template in TEMPLATES:
        target = tmp_path / template.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_baseline_template(template), encoding="utf-8")
        targets.append(target)
    originals = [target.read_bytes() for target in targets]
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--wheel",
            str(_wheel(tmp_path, f"Name: mineru\nVersion: {version}\n")),
            "--tag",
            f"mineru-{version}-released",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert [target.read_bytes() for target in targets] == originals


@pytest.mark.parametrize("version", ["5.0.0a1", "5.0.0rc1", "5.0.0.dev1", "4.0.13+local"])
def test_non_stable_publication_does_not_change_forms(template_path: Path, version: str) -> None:
    """预发布和本地版本不进入正式版本选择。"""
    original = template_path.read_bytes()
    result = _run_sync(
        _wheel(template_path.parent, f"Name: mineru\nVersion: {version}\n"), template_path, f"mineru-{version}-released"
    )
    assert result.returncode == 0, result.stderr
    assert template_path.read_bytes() == original


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("# END MINERU_RECENT_RELEASES", "# missing marker"),
        ("# BEGIN MINERU_RECENT_RELEASES", "# BEGIN MINERU_RECENT_RELEASES\n      # BEGIN MINERU_RECENT_RELEASES"),
        ("      # END MINERU_RECENT_RELEASES", "    # END MINERU_RECENT_RELEASES"),
        ('"4.0.12"', '"5.0.0rc1"'),
        ('"4.0.11"', '"4.0.12"'),
        ("id: software_version", "id: wrong_field"),
        ("  - type: dropdown\n    id: software_version", "  - type: textarea\n    id: software_version"),
        ("id: device_mode", "id: software_version"),
        ("      label: MinerU version", "      label: MinerU version\n      default: 0"),
        ("Other / development / unknown", "invalid fallback"),
        ("      default: 5", "      default: 0"),
        ("name: 🐛 Bug report (English)", "name: [invalid yaml"),
    ],
)
def test_invalid_template_fails_before_any_template_is_written(tmp_path: Path, old: str, new: str) -> None:
    """第二份模板损坏也必须在第一份写入前失败，避免只同步一种语言。"""
    first, second = tmp_path / "first.yml", tmp_path / "second.yml"
    first.write_text(_baseline_template(TEMPLATES[1]), encoding="utf-8")
    original = _baseline_template()
    assert old in original
    second.write_text(original.replace(old, new), encoding="utf-8")
    before = [path.read_bytes() for path in (first, second)]
    result = _run_sync(_wheel(tmp_path), [first, second])
    assert result.returncode == 1, result.stdout
    assert [path.read_bytes() for path in (first, second)] == before


def test_identical_options_in_wrong_field_cannot_claim_markers(template_path: Path) -> None:
    """即使另一个字段选项完全相同，源码位置也必须属于真正的版本字段。"""
    source = template_path.read_text(encoding="utf-8")
    start = source.index("  - type: dropdown\n    id: software_version")
    end = source.index("  - type: dropdown\n    id: tier", start)
    original_field = source[start:end]
    unmarked = original_field.replace("      # BEGIN MINERU_RECENT_RELEASES\n", "").replace(
        "      # END MINERU_RECENT_RELEASES\n", ""
    )
    source = source[:start] + unmarked + original_field.replace("id: software_version", "id: decoy") + source[end:]
    template_path.write_text(source, encoding="utf-8")
    before = template_path.read_bytes()
    assert _run_sync(_wheel(template_path.parent), template_path).returncode == 1
    assert template_path.read_bytes() == before


def test_issue_forms_are_equivalent_and_keep_unavailable_environment_fallbacks() -> None:
    """两种语言保留七项数据和一次确认，旧版、开发版及安装失败用户仍可如实反馈。"""
    structures = []
    for template in TEMPLATES:
        form = yaml.safe_load(template.read_text(encoding="utf-8"))
        fields = {field["id"]: field for field in form["body"] if "id" in field}
        assert list(fields) == [
            "precheck",
            "description",
            "reproduce",
            "usage_entry",
            "os_mode",
            "software_version",
            "tier",
            "device_mode",
        ]
        assert len([field for field in form["body"] if field["type"] == "checkboxes"]) == 1
        assert len(fields["precheck"]["attributes"]["options"]) == 1
        assert fields["precheck"]["attributes"]["options"][0]["required"] is True
        structure = []
        for field_id, field in fields.items():
            attrs = field["attributes"]
            required = field.get("validations", {}).get("required", False)
            if field_id != "precheck":
                assert required
            assert "value" not in attrs and "\n" not in attrs.get("description", "")
            if field["type"] == "dropdown":
                options = attrs["options"]
                assert len(options) == len(set(options)) and all(isinstance(value, str) for value in options)
                if field_id in {"software_version", "tier"}:
                    assert attrs["default"] == len(options) - 1
                else:
                    assert "default" not in attrs
            structure.append((field_id, field["type"], required))
        assert "render" not in fields["reproduce"]["attributes"]
        assert fields["software_version"]["type"] == "dropdown"
        assert len(fields["software_version"]["attributes"]["options"]) == 6
        assert fields["tier"]["attributes"]["options"][:4] == ["flash", "basic", "standard", "advanced"]
        # GitHub 对可选下拉自动添加 None，非空约束配合兜底默认值保留免填写体验。
        tier = fields["tier"]
        assert tier["attributes"]["options"][tier["attributes"]["default"]] == (
            "Not applicable / unknown" if template == TEMPLATE else "不适用或不清楚"
        )
        assert fields["device_mode"]["attributes"]["render"] == "text"
        assert "mineru-kit hardware info" in fields["device_mode"]["attributes"]["placeholder"]
        markdown = "\n".join(field["attributes"]["value"] for field in form["body"] if field["type"] == "markdown")
        assert "mineru-kit hardware info" in markdown and "Docker" in markdown and "WSL" in markdown
        if template == TEMPLATE:
            assert "Installation / build" in fields["usage_entry"]["attributes"]["options"]
            assert all(word in markdown for word in ("anonymized", "installation failed", "cannot access the server"))
        else:
            assert "安装或构建" in fields["usage_entry"]["attributes"]["options"]
            assert all(word in markdown for word in ("脱敏", "安装失败", "无法访问服务端"))
        structures.append(structure)
    assert structures[0] == structures[1]
    assert _options(TEMPLATES[0])[:-1] == _options(TEMPLATES[1])[:-1]


def test_all_issue_template_yaml_and_chooser_links_are_valid() -> None:
    """模板目录的配置与新增表单都需经过基础校验，不能只验证两个已知文件。"""
    names = []
    for path in sorted(TEMPLATE.parent.iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), path
        if path.name == "config.yml":
            assert isinstance(data["blank_issues_enabled"], bool)
            for link in data["contact_links"]:
                assert all(isinstance(link[key], str) and link[key].strip() for key in ("name", "about", "url"))
                assert link["url"].startswith("https://")
            continue
        assert all(isinstance(data[key], str) and data[key].strip() for key in ("name", "description"))
        names.append(data["name"])
        assert isinstance(data["body"], list) and data["body"]
        ids = []
        for field in data["body"]:
            assert field["type"] in {"markdown", "input", "textarea", "dropdown", "checkboxes", "upload"}
            attributes = field["attributes"]
            assert isinstance(attributes, dict)
            label = "value" if field["type"] == "markdown" else "label"
            assert isinstance(attributes[label], str) and attributes[label].strip()
            if "id" in field:
                assert re.fullmatch(r"[A-Za-z0-9_-]+", field["id"])
                ids.append(field["id"])
        assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))


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
    assert "gh api --paginate --slurp" in _workflow_script()
    assert "--jq" not in _workflow_script()


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
    for source in (*TEMPLATES, SCRIPT):
        target = checkout / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for template in TEMPLATES:
        (checkout / template.relative_to(ROOT)).write_text(_baseline_template(template), encoding="utf-8")
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
    gh = binaries / "gh"
    gh.write_text(
        '#!/bin/sh\nif [ "$SIMULATE_RELEASE_FETCH_FAILURE" = 1 ]; then exit 1; fi\ncat "$SYNC_RELEASE_HISTORY_FIXTURE"\n',
        encoding="utf-8",
    )
    gh.chmod(0o755)
    history = _history(tmp_path / "release-history-fixture.json")
    wheel = _wheel(tmp_path / "published-dist")
    return {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "RELEASE_HISTORY": str(tmp_path / "downloaded-history.json"),
        "SYNC_RELEASE_HISTORY_FIXTURE": str(history),
        "GITHUB_REPOSITORY": "test/MinerU",
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
    assert set(_git(release_checkout, "show", "--format=", "--name-only", "HEAD").splitlines()) == {
        str(template.relative_to(ROOT)) for template in TEMPLATES
    }
    for template in TEMPLATES:
        assert _options(release_checkout / template.relative_to(ROOT))[:-1] == ["4.0.13", *BASELINE[:4]]
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
        '  if [ -n "$CONCURRENT_RELEASE_HISTORY" ]; then cp "$CONCURRENT_RELEASE_HISTORY" "$SYNC_RELEASE_HISTORY_FIXTURE"; fi\n'
        "fi\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env.update(REAL_GIT=real_git, COMPETING_CHECKOUT=str(competing), SYNC_RACE_FLAG=str(tmp_path / "race-injected"))
    if newer_remote_version:
        env["CONCURRENT_RELEASE_HISTORY"] = str(_history(tmp_path / "new-history.json", ("4.0.14", "4.0.13", *BASELINE)))
    result = _run_workflow(release_checkout, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Push attempt 1 failed" in result.stdout
    updated = (release_checkout / TEMPLATE_RELATIVE).read_text(encoding="utf-8")
    expected_version = "4.0.14" if newer_remote_version else "4.0.13"
    for template in TEMPLATES:
        assert _options(release_checkout / template.relative_to(ROOT))[0] == expected_version
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
    assert _git(release_checkout, "show", f"origin/master:{TEMPLATE_RELATIVE}") == _baseline_template().strip()


def test_release_list_download_failure_does_not_commit(release_checkout: Path, tmp_path: Path) -> None:
    """GitHub 列表读取失败时终止同步，保留远端和本地两份模板。"""
    env = _workflow_environment(tmp_path)
    env["SIMULATE_RELEASE_FETCH_FAILURE"] = "1"
    head = _git(release_checkout, "rev-parse", "HEAD")
    result = _run_workflow(release_checkout, env)
    assert result.returncode != 0
    assert _git(release_checkout, "rev-parse", "HEAD") == head
    assert _git(release_checkout, "status", "--porcelain") == ""
