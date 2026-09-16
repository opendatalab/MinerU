# Copyright (c) Opendatalab. All rights reserved.
import os
import re
import sys
from pathlib import Path

from packaging.version import Version

VERSION_FILE = Path(__file__).parent / "mineru" / "version.py"
TAG_PATTERN = re.compile(r"^mineru-(.+)-released$")
VERSION_PATTERN = re.compile(r'^__version__ = "(.+)"$', re.MULTILINE)


def get_tag_version() -> str | None:
    """Return the version carried by the tag that triggered the workflow, or None when not triggered by a release tag."""
    ref_name = os.environ.get("GITHUB_REF_NAME", "").strip()
    match = TAG_PATTERN.match(ref_name)
    if match:
        return match.group(1)
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        raise ValueError(f"Invalid release tag {ref_name!r}. Expected format is mineru-<version>-released.")
    return None


def get_file_version() -> str:
    match = VERSION_PATTERN.search(VERSION_FILE.read_text())
    if match is None:
        raise ValueError(f"No __version__ assignment found in {VERSION_FILE}.")
    return match.group(1)


def main() -> None:
    tag_version = get_tag_version()
    if tag_version is None:
        print("Not triggered by a release tag; leaving mineru/version.py unchanged.")
        return
    file_version = get_file_version()
    if Version(file_version) >= Version(tag_version):
        print(f"mineru/version.py ({file_version}) is not lower than the tag ({tag_version}); leaving it unchanged.")
        return
    VERSION_FILE.write_text(f'__version__ = "{tag_version}"\n')
    print(f"Updated mineru/version.py: {file_version} -> {tag_version}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
