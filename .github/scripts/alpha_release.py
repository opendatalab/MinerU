from __future__ import annotations

import argparse
import ast
import re
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path


ALPHA_TAG_PATTERN = re.compile(r"^v(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)a[1-9]\d*)$")


@dataclass(frozen=True)
class WheelMetadata:
    name: str
    version: str


def parse_alpha_tag(tag: str) -> str:
    match = ALPHA_TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"Invalid Alpha release tag {tag!r}; expected canonical format vX.Y.ZaN, for example v4.0.0a1.")
    return match.group("version")


def read_declared_version(version_file: str | Path) -> str:
    path = Path(version_file)
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declared_versions: list[str] = []

    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != "__version__":
            continue
        if not isinstance(statement.value, ast.Constant) or not isinstance(statement.value.value, str):
            raise ValueError(f"{path} must assign __version__ to a string literal.")
        declared_versions.append(statement.value.value)

    if len(declared_versions) != 1:
        raise ValueError(f"{path} must contain exactly one top-level __version__ assignment.")
    return declared_versions[0]


def validate_source(tag: str, version_file: str | Path) -> str:
    tag_version = parse_alpha_tag(tag)
    declared_version = read_declared_version(version_file)
    if declared_version != tag_version:
        raise ValueError(
            f"Release tag {tag!r} declares version {tag_version!r}, but {version_file} declares {declared_version!r}."
        )
    return tag_version


def read_wheel_metadata(wheel: str | Path) -> WheelMetadata:
    path = Path(wheel)
    with zipfile.ZipFile(path) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"{path} must contain exactly one .dist-info/METADATA file.")
        metadata_text = archive.read(metadata_files[0]).decode("utf-8")

    metadata = Parser().parsestr(metadata_text)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"{path} metadata must contain Name and Version fields.")
    return WheelMetadata(name=name, version=version)


def validate_wheel(wheel: str | Path, *, expected_name: str, expected_version: str) -> WheelMetadata:
    metadata = read_wheel_metadata(wheel)
    if metadata.name != expected_name:
        raise ValueError(f"Wheel project name is {metadata.name!r}; expected {expected_name!r}.")
    if metadata.version != expected_version:
        raise ValueError(f"Wheel version is {metadata.version!r}; expected {expected_version!r}.")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate MinerU Alpha release source and wheel metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("validate-source", help="Validate the release tag against mineru/version.py.")
    source_parser.add_argument("--tag", required=True)
    source_parser.add_argument("--version-file", required=True)

    wheel_parser = subparsers.add_parser("validate-wheel", help="Validate the built wheel name and version.")
    wheel_parser.add_argument("--wheel", required=True)
    wheel_parser.add_argument("--expected-name", default="mineru")
    wheel_parser.add_argument("--expected-version", required=True)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate-source":
        print(validate_source(args.tag, args.version_file))
        return 0

    if args.command == "validate-wheel":
        metadata = validate_wheel(
            args.wheel,
            expected_name=args.expected_name,
            expected_version=args.expected_version,
        )
        print(f"Validated {metadata.name} {metadata.version}.")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
