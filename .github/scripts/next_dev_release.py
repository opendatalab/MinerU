from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_BASE_VERSION = "4.0.0"


@dataclass(frozen=True)
class ReleaseRecord:
    version: str
    source_branch: str
    source_commit: str
    trigger: str
    published_at: str


@dataclass(frozen=True)
class ReleaseState:
    package: str
    version: str
    source_branch: str
    source_commit: str
    published_at: str
    last_auto_date: str | None = None
    daily_sequences: dict[str, int] = field(default_factory=dict)
    releases: list[ReleaseRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ReleasePlan:
    should_publish: bool
    version: str
    release_date: str
    sequence: int
    reason: str


def build_version(base_version: str = DEFAULT_BASE_VERSION, now: datetime | None = None, sequence: int = 0) -> str:
    current = now or datetime.now(UTC)
    return build_version_for_date(base_version, current.strftime("%Y%m%d"), sequence)


def build_version_for_date(base_version: str, release_date: str, sequence: int) -> str:
    if sequence < 0 or sequence > 99:
        raise ValueError("Daily release sequence must be between 0 and 99.")
    return f"{base_version}.dev{release_date}{sequence:02d}"


def read_state(state_file: str | Path) -> ReleaseState | None:
    path = Path(state_file)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    releases = [
        ReleaseRecord(
            version=release["version"],
            source_branch=release["source_branch"],
            source_commit=release["source_commit"],
            trigger=release.get("trigger", "unknown"),
            published_at=release["published_at"],
        )
        for release in payload.get("releases", [])
    ]
    return ReleaseState(
        package=payload["package"],
        version=payload["version"],
        source_branch=payload["source_branch"],
        source_commit=payload["source_commit"],
        published_at=payload["published_at"],
        last_auto_date=payload.get("last_auto_date"),
        daily_sequences={key: int(value) for key, value in payload.get("daily_sequences", {}).items()},
        releases=releases,
    )


def _known_published_commits(state: ReleaseState | None) -> set[str]:
    if state is None:
        return set()
    commits = {state.source_commit}
    commits.update(release.source_commit for release in state.releases)
    return commits


def plan_release(
    state: ReleaseState | None,
    *,
    base_version: str,
    source_commit: str,
    trigger: str,
    force_publish: bool = False,
    now: datetime | None = None,
) -> ReleasePlan:
    current = now or datetime.now(UTC)
    release_date = current.strftime("%Y%m%d")
    iso_date = current.date().isoformat()

    if trigger == "schedule" and state is not None and state.last_auto_date == iso_date:
        sequence = state.daily_sequences.get(release_date, 0)
        version = build_version_for_date(base_version, release_date, sequence)
        return ReleasePlan(False, version, release_date, sequence, f"automatic release already ran on {iso_date}")

    published_commits = _known_published_commits(state)
    if source_commit in published_commits and not force_publish:
        sequence = 0 if state is None else state.daily_sequences.get(release_date, 0)
        version = build_version_for_date(base_version, release_date, sequence)
        return ReleasePlan(False, version, release_date, sequence, "source commit was already published")

    previous_sequence = -1 if state is None else state.daily_sequences.get(release_date, -1)
    sequence = previous_sequence + 1
    version = build_version_for_date(base_version, release_date, sequence)
    return ReleasePlan(True, version, release_date, sequence, "publish requested")


def write_state(
    state_file: str | Path,
    *,
    package: str,
    version: str,
    source_branch: str,
    source_commit: str,
    trigger: str = "unknown",
    release_date: str | None = None,
    sequence: int | None = None,
    published_at: str | None = None,
) -> None:
    path = Path(state_file)
    previous = read_state(path)
    published_at_value = published_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    daily_sequences = dict(previous.daily_sequences) if previous is not None else {}
    if release_date is not None and sequence is not None:
        daily_sequences[release_date] = sequence

    releases = list(previous.releases) if previous is not None else []
    releases.append(
        ReleaseRecord(
            version=version,
            source_branch=source_branch,
            source_commit=source_commit,
            trigger=trigger,
            published_at=published_at_value,
        )
    )

    last_auto_date = previous.last_auto_date if previous is not None else None
    if trigger == "schedule":
        if release_date is not None:
            last_auto_date = f"{release_date[:4]}-{release_date[4:6]}-{release_date[6:]}"
        elif published_at_value.endswith("Z"):
            last_auto_date = published_at_value[:10]
        else:
            last_auto_date = datetime.now(UTC).date().isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package": package,
        "version": version,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "published_at": published_at_value,
        "last_auto_date": last_auto_date,
        "daily_sequences": daily_sequences,
        "releases": [
            {
                "version": release.version,
                "source_branch": release.source_branch,
                "source_commit": release.source_commit,
                "trigger": release.trigger,
                "published_at": release.published_at,
            }
            for release in releases
        ],
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
    version_parser.add_argument("--sequence", type=int, default=0)

    plan_parser = subparsers.add_parser("plan-github-output", help="Write the release plan as GitHub output lines.")
    plan_parser.add_argument("--state-file", required=True)
    plan_parser.add_argument("--base-version", default=DEFAULT_BASE_VERSION)
    plan_parser.add_argument("--source-commit", required=True)
    plan_parser.add_argument("--trigger", required=True)
    plan_parser.add_argument("--force-publish", choices=["true", "false"], default="false")

    state_commit_parser = subparsers.add_parser("state-commit", help="Print the last published source commit from state.")
    state_commit_parser.add_argument("--state-file", required=True)

    write_state_parser = subparsers.add_parser("write-state", help="Write release state JSON.")
    write_state_parser.add_argument("--state-file", required=True)
    write_state_parser.add_argument("--package", required=True)
    write_state_parser.add_argument("--version", required=True)
    write_state_parser.add_argument("--source-branch", required=True)
    write_state_parser.add_argument("--source-commit", required=True)
    write_state_parser.add_argument("--trigger", required=True)
    write_state_parser.add_argument("--release-date", required=True)
    write_state_parser.add_argument("--sequence", type=int, required=True)
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
        print(build_version(base_version=args.base_version, sequence=args.sequence))
        return 0

    if args.command == "plan-github-output":
        plan = plan_release(
            read_state(args.state_file),
            base_version=args.base_version,
            source_commit=args.source_commit,
            trigger=args.trigger,
            force_publish=args.force_publish == "true",
        )
        print(f"should_publish={str(plan.should_publish).lower()}")
        print(f"version={plan.version}")
        print(f"release_date={plan.release_date}")
        print(f"sequence={plan.sequence}")
        print(f"skip_reason={plan.reason}")
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
            trigger=args.trigger,
            release_date=args.release_date,
            sequence=args.sequence,
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
