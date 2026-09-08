"""检查 wheel 与 sdist 是否完整携带可离线运行的 PDF.js 预览资源。"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path


def verify_distribution(path: Path) -> None:
    """逐文件比较发布包与源码资源，防止递归目录或许可证在打包时遗漏。"""
    root = Path(__file__).resolve().parents[1]
    resources = root / "mineru/resources/pdf_preview"
    expected = [*resources.rglob("*"), root / "mineru/resources/gradio_pdf_preview.js"]
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist() if "mineru/resources/" in name}
    else:
        with tarfile.open(path) as archive:
            contents = {}
            for member in archive.getmembers():
                if member.isfile() and "mineru/resources/" in member.name:
                    stream = archive.extractfile(member)
                    if stream is not None:
                        contents[member.name.split("/", 1)[1]] = stream.read()
    for file in expected:
        # Finder 元数据不属于发行资源，setuptools 也不会将这些隐藏文件打包。
        if not file.is_file() or file.name == ".DS_Store":
            continue
        relative = file.relative_to(root).as_posix()
        if relative not in contents:
            raise ValueError(f"{path.name}: missing {relative}")
        if hashlib.sha256(contents[relative]).digest() != hashlib.sha256(file.read_bytes()).digest():
            raise ValueError(f"{path.name}: changed {relative}")
    print(f"{path.name}: all PDF.js resources and licenses verified")


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        verify_distribution(Path(filename))
