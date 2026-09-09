"""比较冻结基线和新版本产物，仅忽略明确声明的版本与容器时间字段。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree
from zipfile import ZipFile


def normalized_json(path: Path) -> object:
    """仅归一化根文档的引擎 producer 版本，保留全部语义和来源属性。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        producer = value.get("metadata", {}).get("producer")
        if isinstance(producer, dict) and producer.get("name") == "docvortex":
            producer["version"] = "<engine-version>"
    return value


def archive_members(path: Path) -> dict[str, bytes]:
    """比较容器内容，校验并归一化由 producer 版本派生的 EPUB 标识。"""
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if path.suffix == ".epub":
        for name in members:
            if name.endswith(".opf"):
                document = ElementTree.fromstring(members[name])
                namespaces = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
                metadata = document.find("opf:metadata", namespaces)
                identifier = metadata.find("dc:identifier", namespaces).text
                seed = {
                    "title": metadata.find("dc:title", namespaces).text,
                    "authors": tuple(item.text for item in metadata.findall("dc:creator", namespaces)),
                    "language": metadata.find("dc:language", namespaces).text,
                    "middle_json": json.loads((path.parent / "middle.json").read_text()),
                }
                expected = stable_identifier(seed)
                if identifier != expected:
                    raise ValueError(f"EPUB identifier is not derived from captured document: {path}")
                seed["middle_json"] = normalized_json(path.parent / "middle.json")
                members[name] = members[name].replace(identifier.encode(), stable_identifier(seed).encode())
                members[name] = re.sub(
                    rb'(<meta property="dcterms:modified">)[^<]*(</meta>)', rb"\1<timestamp>\2", members[name]
                )
    return members


def stable_identifier(seed: dict[str, object]) -> str:
    """独立复算现有 EPUB 的确定性标识，避免盲目忽略内容标识差异。"""
    payload = json.dumps(seed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"urn:uuid:{uuid5(NAMESPACE_URL, hashlib.sha256(payload).hexdigest())}"


def compare(before: Path, after: Path) -> dict[str, object]:
    """逐文档核对样本、协议、全部渲染和素材，输出机器可读差分证据。"""
    before_manifest = json.loads((before / "manifest.json").read_text())
    after_manifest = json.loads((after / "manifest.json").read_text())
    checks = []
    errors = []
    if len(before_manifest["documents"]) != len(after_manifest["documents"]):
        errors.append("document count differs")
    for old, new in zip(before_manifest["documents"], after_manifest["documents"]):
        for key in ("name", "sha256", "pages"):
            if old.get(key) != new.get(key):
                errors.append(f"{old['name']}: {key} differs")
        if "error" in old or "error" in new:
            errors.append(f"{old['name']}: capture failed")
        for engine in ("mineru", "docvortex"):
            directory = Path(Path(old["name"]).stem) / engine
            original_files = {path.relative_to(before) for path in (before / directory).rglob("*") if path.is_file()}
            current_files = {path.relative_to(after) for path in (after / directory).rglob("*") if path.is_file()}
            if original_files != current_files:
                errors.append(f"{directory}: file inventory differs")
            for relative in sorted(original_files & current_files):
                a, b = before / relative, after / relative
                identical = a.read_bytes() == b.read_bytes()
                if identical:
                    mode, equal = "bytes", True
                elif a.suffix == ".json":
                    mode, equal = "producer-version-only", normalized_json(a) == normalized_json(b)
                elif a.suffix in {".docx", ".epub"}:
                    mode, equal = "archive-timestamps-and-verified-producer-id", archive_members(a) == archive_members(b)
                else:
                    mode, equal = "bytes", False
                checks.append(
                    {
                        "path": str(relative),
                        "equal": equal,
                        "comparison": mode,
                        "sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
                    }
                )
                if not equal:
                    errors.append(str(relative))
        for outputs in (old["outputs"], new["outputs"]):
            for name, value in outputs.items():
                if isinstance(value, dict):
                    errors.append(f"{old['name']}/{name}: {value}")
    return {"documents": len(before_manifest["documents"]), "checks": checks, "errors": errors, "passed": not errors}


def main() -> None:
    """将差分报告保存到指定路径，发生未声明的变化时返回非零状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.before, args.after)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
