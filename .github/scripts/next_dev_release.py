from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_BASE_VERSION = "4.0.0"


@dataclass(frozen=True)
class ReleaseState:
    package: str
    version: str
    source_branch: str
    source_commit: str
    published_at: str


def build_version(base_version: str = DEFAULT_BASE_VERSION, now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return f"{base_version}.dev{current.strftime('%Y%m%d')}"


def read_state(state_file: str | Path) -> ReleaseState | None:
    path = Path(state_file)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReleaseState(
        package=payload["package"],
        version=payload["version"],
        source_branch=payload["source_branch"],
        source_commit=payload["source_commit"],
        published_at=payload["published_at"],
    )


def write_state(
    state_file: str | Path,
    *,
    package: str,
    version: str,
    source_branch: str,
    source_commit: str,
    published_at: str | None = None,
) -> None:
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package": package,
        "version": version,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "published_at": published_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    path.write_text(f"{json.dumps(payload, ensure_ascii=True, indent=2)}\n", encoding="utf-8")


def patch_pyproject_package_name(pyproject_path: str | Path, package_name: str) -> None:
    path = Path(pyproject_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    in_project_section = False
    replaced = False
    updated_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_section = stripped == "[project]"
        if in_project_section and stripped.startswith("name = "):
            updated_lines.append(f'name = "{package_name}"')
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        raise ValueError(f"Could not find [project].name in {path}")

    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def patch_version_file(version_file_path: str | Path, version: str) -> None:
    Path(version_file_path).write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def patch_workspace(
    *,
    pyproject_path: str | Path,
    version_file_path: str | Path,
    package_name: str,
    version: str,
) -> None:
    patch_pyproject_package_name(pyproject_path, package_name)
    patch_version_file(version_file_path, version)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper utilities for mineru-next-dev daily release workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Generate the next-dev package version.")
    version_parser.add_argument("--base-version", default=DEFAULT_BASE_VERSION)

    state_commit_parser = subparsers.add_parser("state-commit", help="Print the last published source commit from state.")
    state_commit_parser.add_argument("--state-file", required=True)

    write_state_parser = subparsers.add_parser("write-state", help="Write release state JSON.")
    write_state_parser.add_argument("--state-file", required=True)
    write_state_parser.add_argument("--package", required=True)
    write_state_parser.add_argument("--version", required=True)
    write_state_parser.add_argument("--source-branch", required=True)
    write_state_parser.add_argument("--source-commit", required=True)
    write_state_parser.add_argument("--published-at")

    patch_parser = subparsers.add_parser("patch-workspace", help="Patch package name and version in the workspace.")
    patch_parser.add_argument("--pyproject", required=True)
    patch_parser.add_argument("--version-file", required=True)
    patch_parser.add_argument("--package-name", required=True)
    patch_parser.add_argument("--version", required=True)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "version":
        print(build_version(base_version=args.base_version))
        return 0

    if args.command == "state-commit":
        state = read_state(args.state_file)
        print("" if state is None else state.source_commit)
        return 0

    if args.command == "write-state":
        write_state(
            args.state_file,
            package=args.package,
            version=args.version,
            source_branch=args.source_branch,
            source_commit=args.source_commit,
            published_at=args.published_at,
        )
        return 0

    if args.command == "patch-workspace":
        patch_workspace(
            pyproject_path=args.pyproject,
            version_file_path=args.version_file,
            package_name=args.package_name,
            version=args.version,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
