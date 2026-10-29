# Copyright (c) Opendatalab. All rights reserved.
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging", "PyYAML>=6,<7"]
# ///
"""依据已发布 wheel 和 GitHub Release 列表同步中英文 issue 表单的最近正式版本。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import yaml
from packaging.utils import canonicalize_name
from packaging.version import Version
from yaml.nodes import MappingNode, Node, SequenceNode

__all__ = ["main", "read_release_versions", "read_wheel_version", "sync_template_versions"]

DEFAULT_TEMPLATES = (
    Path(".github/ISSUE_TEMPLATE/bug_report.yml"),
    Path(".github/ISSUE_TEMPLATE/bug_report_zh.yml"),
)
FALLBACK_OPTIONS = ("Other / development / unknown", "其他 / 开发版 / 不清楚")
BEGIN_MARKER = "# BEGIN MINERU_RECENT_RELEASES"
END_MARKER = "# END MINERU_RECENT_RELEASES"
REGION_PATTERN = re.compile(
    rf"^ {{6}}{re.escape(BEGIN_MARKER)}\r?\n(?P<region>.*?)^ {{6}}{re.escape(END_MARKER)}\r?$",
    re.MULTILINE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"mineru-(.+)-released")
MAX_VERSIONS = 5


def read_wheel_version(wheel_path: Path) -> Version:
    """读取唯一 MinerU wheel 的 METADATA，不安装软件包或导入 MinerU。"""
    if wheel_path.is_dir():
        wheels = sorted(wheel_path.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected exactly one wheel in {wheel_path}, found {len(wheels)}.")
        wheel_path = wheels[0]
    if wheel_path.suffix != ".whl":
        raise ValueError(f"Expected a .whl file, got {wheel_path}.")
    with ZipFile(wheel_path) as wheel:
        metadata_files = [name for name in wheel.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
        if len(metadata_files) != 1:
            raise ValueError(f"Expected exactly one METADATA file in {wheel_path}.")
        metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(metadata_files[0]))
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if len(names) != 1 or canonicalize_name(str(names[0])) != "mineru":
        raise ValueError("Wheel METADATA must identify the mineru distribution exactly once.")
    if len(versions) != 1:
        raise ValueError("Wheel METADATA must contain exactly one Version field.")
    return Version(str(versions[0]))


def read_release_versions(path: Path) -> tuple[Version, ...]:
    """校验离线 Release 列表，按语义版本降序去重，排除草稿、预发布和无关标签。"""
    releases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(releases, list):
        raise ValueError("Release history must be a JSON array.")
    # gh --paginate --slurp 返回分页数组；单页 REST 响应也可直接用于本地验证。
    if all(isinstance(page, list) for page in releases):
        releases = [release for page in releases for release in page]
    versions: set[Version] = set()
    for release in releases:
        if (
            not isinstance(release, dict)
            or type(release.get("draft")) is not bool
            or type(release.get("prerelease")) is not bool
            or not isinstance(release.get("tag_name"), str)
        ):
            raise ValueError("Invalid GitHub Release record.")
        if release["draft"] or release["prerelease"]:
            continue
        if not isinstance(release.get("published_at"), str) or not release["published_at"]:
            raise ValueError("A public Release must have a publication timestamp.")
        tag = TAG_PATTERN.fullmatch(release["tag_name"])
        if tag is None:
            continue
        version = Version(tag[1])
        if not version.is_prerelease and version.local is None:
            versions.add(version)
    if not versions:
        raise ValueError("Release history contains no published stable MinerU version.")
    return tuple(sorted(versions, reverse=True))


def _mapping_entry(node: Node | None, key: str) -> tuple[Node, Node]:
    """定位唯一 YAML 键及其源码位置，拒绝重复键或不符合预期的结构。"""
    if not isinstance(node, MappingNode):
        raise ValueError(f"Expected a mapping containing {key!r}.")
    entries = [(name, value) for name, value in node.value if name.value == key]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one {key!r} key.")
    return entries[0]


def _prepare_template(source: str, versions: tuple[Version, ...] | None) -> tuple[str, tuple[Version, ...]]:
    """严格绑定版本下拉框的选项区块，并保留区块外文案、兜底语言与原换行符。"""
    if source.count(BEGIN_MARKER) != 1 or source.count(END_MARKER) != 1:
        raise ValueError("Template must contain exactly one pair of recent release markers.")
    region = REGION_PATTERN.search(source)
    if region is None or not region["region"].startswith("      options:"):
        raise ValueError("Recent release markers must surround the options mapping.")
    form = yaml.safe_load(source)
    if not isinstance(form, dict) or not isinstance(form.get("body"), list):
        raise ValueError("Template must be an issue form with a body array.")
    fields = [
        (index, field)
        for index, field in enumerate(form["body"])
        if isinstance(field, dict) and field.get("id") == "software_version"
    ]
    if len(fields) != 1 or fields[0][1].get("type") != "dropdown":
        raise ValueError("Template must contain exactly one software_version dropdown.")
    index, field = fields[0]
    attributes = field.get("attributes")
    validations = field.get("validations")
    if (
        not isinstance(attributes, dict)
        or attributes.get("multiple", False) is not False
        or not isinstance(validations, dict)
        or validations.get("required") is not True
    ):
        raise ValueError("Version selection must be required and single-select.")
    options = attributes.get("options")
    if (
        not isinstance(options, list)
        or not 2 <= len(options) <= MAX_VERSIONS + 1
        or not all(isinstance(option, str) for option in options)
        or options[-1] not in FALLBACK_OPTIONS
    ):
        raise ValueError("Expected up to five version strings followed by the localized fallback option.")
    # GitHub 必填单选框会自动选择首项，显式选择兜底项以免误报最新版。
    if type(attributes.get("default")) is not int or attributes["default"] != len(options) - 1:
        raise ValueError("Version selection must default to the fallback option, never a release.")
    current = tuple(Version(option) for option in options[:-1])
    if any(version.is_prerelease or version.local is not None for version in current):
        raise ValueError("Version choices must be public stable releases.")
    if tuple(sorted(set(current), reverse=True)) != current:
        raise ValueError("Version choices must be unique and sorted newest first.")
    body_node = _mapping_entry(yaml.compose(source), "body")[1]
    if not isinstance(body_node, SequenceNode):
        raise ValueError("Issue form body must be a sequence.")
    attributes_node = _mapping_entry(body_node.value[index], "attributes")[1]
    options_key, _ = _mapping_entry(attributes_node, "options")
    default_key, _ = _mapping_entry(attributes_node, "default")
    if options_key.start_mark.index != region.start("region") + 6:
        raise ValueError("Recent release markers must belong to software_version.attributes.options.")
    if not region.start("region") <= default_key.start_mark.index < region.end("region"):
        raise ValueError("The fallback default must belong to the marked version choices.")
    if yaml.safe_load(region["region"]) != {"options": options, "default": len(options) - 1}:
        raise ValueError("Recent release markers must contain only version options and their fallback default.")
    if versions is None or current == versions:
        return source, current
    newline = "\r\n" if "\r\n" in region[0] else "\n"
    values = [*(str(version) for version in versions), options[-1]]
    replacement = "      options:" + newline
    replacement += "".join(f"        - {json.dumps(value, ensure_ascii=False)}{newline}" for value in values)
    replacement += f"      default: {len(versions)}{newline}"
    return source[: region.start("region")] + replacement + source[region.end("region") :], current


def sync_template_versions(
    wheel_path: Path, template_paths: Sequence[Path], release_tag: str, releases_path: Path | None
) -> bool:
    """验证所有输入后统一更新两种语言；陈旧列表不得回退任何模板的版本选择。"""
    version = read_wheel_version(wheel_path)
    tag = TAG_PATTERN.fullmatch(release_tag)
    if tag is None or Version(tag[1]) != version:
        raise ValueError("Release tag must be mineru-<wheel version>-released.")
    released = read_release_versions(releases_path) if releases_path is not None else None
    if version.is_prerelease or version.local is not None:
        print(f"No template update needed for non-stable version {version}.")
        return False
    if released is not None and version not in released:
        raise ValueError("Published wheel version is missing from the Release history.")
    recent = released[:MAX_VERSIONS] if released is not None else None
    paths = list(template_paths)
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Provide distinct template paths.")
    updates: list[tuple[Path, str, str, tuple[Version, ...]]] = []
    for path in paths:
        source = path.read_bytes().decode("utf-8")
        updated, current = _prepare_template(source, recent)
        updates.append((path, source, updated, current))
    if recent is None:
        # 已发布标签固定使用旧工作流参数，但其同步 checkout 指向最新默认分支。
        # 缺少历史时只允许旧版本幂等退出，真正推进版本仍必须提供已发布列表。
        if all(version <= current[0] for _, _, _, current in updates):
            print("No template update needed when replaying an existing release without history.")
            return False
        raise ValueError("--releases-json is required to advance the version choices.")
    if any(recent < current for _, _, _, current in updates):
        print("No template update: Release history is older than the checked-out choices.")
        return False
    changed = False
    for path, source, updated, _ in updates:
        if updated != source:
            path.write_bytes(updated.encode("utf-8"))
            changed = True
            print(f"Updated recent release choices in {path}.")
    if not changed:
        print("Recent release choices are already current.")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """解析多模板同步参数，将 Release、产物及模板异常转换为明确的失败退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="Published wheel file or directory.")
    parser.add_argument("--tag", required=True, help="Release tag, in the form mineru-<version>-released.")
    parser.add_argument("--releases-json", type=Path, help="GitHub Releases JSON; required to advance version choices.")
    parser.add_argument("--template", type=Path, action="append", help="Repeat to override the default English/Chinese forms.")
    args = parser.parse_args(argv)
    try:
        sync_template_versions(args.wheel, args.template or DEFAULT_TEMPLATES, args.tag, args.releases_json)
    except (ValueError, OSError, BadZipFile, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
