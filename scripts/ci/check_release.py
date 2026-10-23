"""校验固定源码提交、release 标签和最终 wheel，输出可追溯的发布记录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def source_version(root: Path, tag: str) -> Version:
    """发版只验证已经提交的版本，拒绝用移动分支或临时改版本替代标签源码。"""
    source = (root / "mineru/version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    if match is None:
        raise ValueError("Missing __version__ assignment")
    version = Version(match.group(1))
    if tag:
        tag_match = re.fullmatch(r"mineru-(.+)-released", tag)
        if tag_match is None or Version(tag_match.group(1)) != version:
            raise ValueError(f"Release tag {tag!r} must match the committed version {version}; update it before tagging")
    return version


def verify_wheel(root: Path, directory: Path, version: Version) -> dict[str, str]:
    """检查实际发布产物的版本、Python 范围和 classifiers，记录完整 wheel 摘要。"""
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel, found {len(wheels)}")
    wheel = wheels[0]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    with ZipFile(wheel) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must have exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        for field in ("Name", "Version", "Requires-Python"):
            if len(metadata.get_all(field, [])) != 1:
                raise ValueError(f"Missing or duplicate wheel metadata: {field}")
        if metadata["Name"] != project["name"] or Version(metadata["Version"]) != version:
            raise ValueError("Wheel name/version does not match the release source")
        if SpecifierSet(metadata["Requires-Python"]) != SpecifierSet(project["requires-python"]):
            raise ValueError("Wheel Python requirement does not match pyproject.toml")
        if set(metadata.get_all("Classifier", [])) != set(project["classifiers"]):
            raise ValueError("Wheel classifiers do not match pyproject.toml")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"wheel": wheel.name, "wheel_sha256": digest}


def main() -> None:
    """在构建前校验标签，在构建后补充 wheel 身份，两个阶段都锁定相同源码提交。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or commit != args.commit:
        raise ValueError("Checkout must match the workflow commit SHA")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "mineru", "pyproject.toml"], check=True)
    version = source_version(root, args.tag)
    record = {"source_commit": commit, "tag": args.tag, "version": str(version)}
    if args.wheel_dir is not None:
        record.update(verify_wheel(root, args.wheel_dir, version))
    result = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
