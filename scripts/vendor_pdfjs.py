"""下载并校验固定版本的 PDF.js，只保留浏览器预览所需的发行文件。"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

VERSION = "6.3.289"
TARBALL_URL = f"https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-{VERSION}.tgz"
INTEGRITY = "sha512-ZHjSVpDa3D6izMq8/04lvkhkATUmL9px6ChPaXc1k6nU2Mrhlg1/7F0bdUqCwUjw3NsPTfPZsMDUU6ZIcRaeQw=="
DESTINATION = Path(__file__).resolve().parents[1] / "mineru/resources/pdf_preview/vendor/pdfjs"
FILES = {
    "LICENSE",
    "legacy/build/pdf.min.mjs",
    "legacy/build/pdf.worker.min.mjs",
    "legacy/web/pdf_viewer.mjs",
    "legacy/web/pdf_viewer.css",
}
DIRECTORIES = ("legacy/web/images/", "cmaps/", "standard_fonts/", "wasm/")


def vendor_pdfjs() -> None:
    """校验上游压缩包与每个选中文件，生成可复核的资源清单。"""
    with urlopen(TARBALL_URL, timeout=60) as response:
        archive_bytes = response.read()
    actual = "sha512-" + base64.b64encode(hashlib.sha512(archive_bytes).digest()).decode("ascii")
    if actual != INTEGRITY:
        raise ValueError("PDF.js archive integrity mismatch")
    hashes: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith("package/"):
                continue
            relative = member.name.removeprefix("package/")
            if relative not in FILES and not relative.startswith(DIRECTORIES):
                continue
            # 不分发源码映射或仅供文档脚本执行使用的 QuickJS 引擎。
            if relative.endswith(".map") or "quickjs" in relative:
                continue
            parts = PurePosixPath(relative).parts
            if ".." in parts or PurePosixPath(relative).is_absolute():
                raise ValueError(f"Unsafe archive member: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive member: {relative}")
            content = stream.read()
            destination = DESTINATION.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
    manifest = {
        "package": "pdfjs-dist",
        "version": VERSION,
        "source": TARBALL_URL,
        "integrity": INTEGRITY,
        "license": "Apache-2.0; bundled fonts and WASM retain their accompanying licenses",
        "sha256": dict(sorted(hashes.items())),
    }
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored PDF.js {VERSION}: {len(hashes)} files")


if __name__ == "__main__":
    vendor_pdfjs()
