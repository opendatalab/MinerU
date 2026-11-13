"""Common helpers for mineru-kit commands."""

from __future__ import annotations

import zipfile
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

from ..errors import InvalidRequestError
from ..filetypes import PARSEABLE_EXTENSIONS
from ..parser.base import ParseResult
from ..parser.writer import FileBasedDataWriter
from ..types import Tier
from ..utils.i18n import t

KitFormat = Literal["markdown", "middle_json", "zip"]
LocalTier: TypeAlias = Tier


PARSEABLE_SUFFIXES = frozenset(f".{ext}" for ext in PARSEABLE_EXTENSIONS)
OUTPUT_FILE_SUFFIXES = {
    "markdown": ".md",
    "middle_json": ".json",
    "zip": ".zip",
}


def expand_input_paths(inputs: list[str]) -> list[Path]:
    paths = [Path(raw).expanduser() for raw in inputs]
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() not in PARSEABLE_SUFFIXES:
                    continue
                try:
                    is_file = stat.S_ISREG(child.stat().st_mode)
                except (FileNotFoundError, PermissionError):
                    # 保留发现后消失或无法读取的候选，交给逐项校验报告，不能静默遗漏。
                    expanded.append(child)
                    continue
                if is_file:
                    expanded.append(child)
        else:
            expanded.append(path)
    return expanded


def resolve_single_output_path(source: Path, output: Path, format: KitFormat) -> Path:
    if output.exists() and output.is_dir():
        return output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
    if not output.exists() and output.suffix == "":
        return output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
    return output


def resolve_batch_output_paths(paths: list[Path], output: Path, format: KitFormat, *, batch: bool) -> dict[Path, Path]:
    """按原始输入模式统一验证和映射输出；批量目录名称不受后缀限制。"""
    if format not in OUTPUT_FILE_SUFFIXES:
        raise InvalidRequestError("unsupported_output_format", t("Unsupported format: {format}", format=format), "format")
    if any(path.parent == path for path in paths):
        raise ValueError(t("Invalid input path."))
    if batch and output.exists() and not output.is_dir():
        raise ValueError(t("When input is multiple files or directories, --output must be a directory path."))

    destinations: dict[Path, Path] = {}
    seen: dict[Path, Path] = {}
    sources = {source.resolve() for source in paths}
    for source in paths:
        dest = (
            resolve_single_output_path(source, output, format)
            if not batch
            else output / f"{source.stem}{OUTPUT_FILE_SUFFIXES[format]}"
        )
        resolved_dest = dest.resolve()
        if resolved_dest in sources:
            raise ValueError(t("Output path must not be the same as the input file: {path}", path=source))
        existing = seen.get(resolved_dest)
        if existing is not None:
            raise ValueError(
                t(
                    "Output name collision: {existing} and {source} both map to {dest}",
                    existing=existing.name,
                    source=source.name,
                    dest=dest,
                )
            )
        seen[resolved_dest] = source
        destinations[source] = dest
    return destinations


def build_remote_api_url(remote: bool, remote_url: str | None) -> str | None:
    if remote and remote_url:
        raise ValueError(t("--remote and --remote-url are mutually exclusive."))
    if remote_url:
        return remote_url
    if remote:
        return "https://mineru.net/api"
    return None


def save_parse_result(result: ParseResult, dest: Path, format: KitFormat) -> None:
    """保存解析结果，ZIP 使用独立临时目录并在成功或异常退出时清理。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if format == "markdown":
        _write_utf8_text(dest, result.markdown())
        return
    if format == "middle_json":
        _write_utf8_text(dest, result.to_json())
        return
    if format == "zip":
        # 临时素材沿用输出盘，避免大文档占满系统临时盘；只清理本次导出创建的独立目录。
        with TemporaryDirectory(prefix=".mineru-kit-", dir=dest.parent) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            writer = FileBasedDataWriter(tmp_dir_name)
            result.save(writer)
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for child in sorted(tmp_dir.rglob("*")):
                    if child.is_file():
                        zf.write(child, arcname=child.relative_to(tmp_dir).as_posix())
        return
    raise ValueError(t("Unsupported format: {format}", format=format))


def _write_utf8_text(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8", errors="replace"))


def parse_result_payload(path: Path, dest: Path, format: KitFormat) -> dict[str, str]:
    return {
        "input": str(path),
        "output": str(dest),
        "format": format,
    }
