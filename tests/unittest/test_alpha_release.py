from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_release_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "alpha_release.py"
    spec = importlib.util.spec_from_file_location("alpha_release", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, *, name: str = "mineru", version: str = "4.0.0a1") -> None:
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)


@pytest.mark.parametrize("tag", ["v4.0.0a1", "v4.0.0a12", "v10.20.30a4"])
def test_parse_alpha_tag_accepts_canonical_tags(tag: str) -> None:
    module = _load_release_module()

    assert module.parse_alpha_tag(tag) == tag.removeprefix("v")


@pytest.mark.parametrize(
    "tag",
    [
        "4.0.0a1",
        "v4.0.0",
        "v4.0.0a0",
        "v4.0.0b1",
        "v4.0a1",
        "v04.0.0a1",
        "v4.0.0-alpha.1",
        "v4.0.0a1-extra",
    ],
)
def test_parse_alpha_tag_rejects_noncanonical_tags(tag: str) -> None:
    module = _load_release_module()

    with pytest.raises(ValueError, match="expected canonical format"):
        module.parse_alpha_tag(tag)


def test_read_declared_version_uses_static_top_level_assignment(tmp_path: Path) -> None:
    module = _load_release_module()
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "4.0.0a1"\n', encoding="utf-8")

    assert module.read_declared_version(version_file) == "4.0.0a1"


@pytest.mark.parametrize(
    "contents",
    [
        "OTHER = '4.0.0a1'\n",
        "__version__ = get_version()\n",
        "__version__ = '4.0.0a1'\n__version__ = '4.0.0a2'\n",
    ],
)
def test_read_declared_version_rejects_ambiguous_files(tmp_path: Path, contents: str) -> None:
    module = _load_release_module()
    version_file = tmp_path / "version.py"
    version_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="__version__"):
        module.read_declared_version(version_file)


def test_validate_source_requires_tag_and_file_versions_to_match(tmp_path: Path) -> None:
    module = _load_release_module()
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "4.0.0a2"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="4.0.0a1.*4.0.0a2"):
        module.validate_source("v4.0.0a1", version_file)


def test_validate_wheel_accepts_expected_metadata(tmp_path: Path) -> None:
    module = _load_release_module()
    wheel = tmp_path / "mineru-4.0.0a1-py3-none-any.whl"
    _write_wheel(wheel)

    metadata = module.validate_wheel(wheel, expected_name="mineru", expected_version="4.0.0a1")

    assert metadata.name == "mineru"
    assert metadata.version == "4.0.0a1"


@pytest.mark.parametrize(
    ("name", "version", "message"),
    [
        ("mineru-alpha", "4.0.0a1", "project name"),
        ("mineru", "4.0.0a2", "Wheel version"),
    ],
)
def test_validate_wheel_rejects_unexpected_metadata(tmp_path: Path, name: str, version: str, message: str) -> None:
    module = _load_release_module()
    wheel = tmp_path / "release.whl"
    _write_wheel(wheel, name=name, version=version)

    with pytest.raises(ValueError, match=message):
        module.validate_wheel(wheel, expected_name="mineru", expected_version="4.0.0a1")
