# Copyright (c) Opendatalab. All rights reserved.
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging", "PyYAML>=6,<7"]
# ///
"""依据已发布 wheel 的元数据更新 issue 表单中的最新正式版本提示。"""

from __future__ import annotations

import argparse
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

__all__ = ["main", "read_wheel_version", "sync_template_version", "update_template_version"]

BEGIN_MARKER = "# BEGIN MINERU_LATEST_STABLE_VERSION"
END_MARKER = "# END MINERU_LATEST_STABLE_VERSION"
REGION_PATTERN = re.compile(
    rf"^ {{6}}{re.escape(BEGIN_MARKER)}\r?\n(?P<region>.*?)^ {{6}}{re.escape(END_MARKER)}\r?$",
    re.MULTILINE | re.DOTALL,
)
VERSION_LINE_PATTERN = re.compile(
    r"^ {8}Latest stable release: (?P<version>\S+) \| 最新正式版本：(?P=version)\r?$", re.MULTILINE
)
TAG_PATTERN = re.compile(r"mineru-(.+)-released")


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


def update_template_version(source: str, version: Version) -> str:
    """校验版本提示的归属与标记，只替换该区块的版本号并阻止提示回退。"""
    if source.count(BEGIN_MARKER) != 1 or source.count(END_MARKER) != 1:
        raise ValueError("Template must contain exactly one pair of latest stable version markers.")
    region = REGION_PATTERN.search(source)
    if region is None:
        raise ValueError("Latest stable version markers must delimit a software_version description block.")
    lines = list(VERSION_LINE_PATTERN.finditer(region["region"]))
    if len(lines) != 1:
        raise ValueError("Latest stable version block must contain exactly one matching bilingual version line.")
    form = yaml.safe_load(source)
    if not isinstance(form, dict) or not isinstance(form.get("body"), list):
        raise ValueError("Template must be an issue form with a body array.")
    fields = [field for field in form["body"] if isinstance(field, dict) and field.get("id") == "software_version"]
    if len(fields) != 1 or fields[0].get("type") not in ("input", "textarea"):
        raise ValueError("Template must contain exactly one software_version text field.")
    attributes = fields[0].get("attributes")
    description = attributes.get("description") if isinstance(attributes, dict) else None
    if not isinstance(description, str) or lines[0][0].strip() not in description.splitlines():
        raise ValueError("Latest stable version line must belong to software_version.description.")
    current = Version(lines[0]["version"])
    if current.is_prerelease or current.local is not None:
        raise ValueError("The current latest stable version hint must be a public stable version.")
    if version.is_prerelease or version.local is not None or version <= current:
        return source
    updated_region = VERSION_LINE_PATTERN.sub(
        lambda match: f"        Latest stable release: {version} | 最新正式版本：{version}"
        + ("\r" if match[0].endswith("\r") else ""),
        region["region"],
        count=1,
    )
    updated = source[: region.start("region")] + updated_region + source[region.end("region") :]
    yaml.safe_load(updated)
    return updated


def sync_template_version(wheel_path: Path, template_path: Path, release_tag: str) -> bool:
    """先验证产物版本与发版标签一致，完成全部校验后才写入模板。"""
    version = read_wheel_version(wheel_path)
    tag = TAG_PATTERN.fullmatch(release_tag)
    if tag is None:
        raise ValueError("Expected release tag mineru-<version>-released.")
    if Version(tag[1]) != version:
        raise ValueError(f"Wheel version {version} does not match release tag {release_tag!r}.")
    source = template_path.read_bytes().decode("utf-8")
    updated = update_template_version(source, version)
    if updated == source:
        print(f"No template update needed for {version} (unchanged, older, or non-stable version).")
        return False
    template_path.write_bytes(updated.encode("utf-8"))
    print(f"Updated latest stable release hint to {version} in {template_path}.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """解析发版同步参数，并将产物或模板异常转换为明确的失败退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="Released wheel file or directory containing one wheel.")
    parser.add_argument("--tag", required=True, help="Release tag, in the form mineru-<version>-released.")
    parser.add_argument("--template", type=Path, default=Path(".github/ISSUE_TEMPLATE/bug_report.yml"))
    args = parser.parse_args(argv)
    try:
        sync_template_version(args.wheel, args.template, args.tag)
    except (ValueError, OSError, BadZipFile, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
