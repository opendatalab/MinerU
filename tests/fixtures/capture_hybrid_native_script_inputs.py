"""从 DocGale 原始语料再生 Hybrid 测试所需的最小字符输入。"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from docgale.document.pdf import PDFDocument
from docgale.document.pdf.document import get_lines_from_chars

_CONTROL_CHARS = {"\r", "\n", "\x02", "\ufffe", "\uffff"}


def extract_case(source_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    """只提取原断言实际使用的字符，保留原始索引、字体和对应几何。"""
    with PDFDocument(str(source_root / case["source"])) as document:
        geometry = document.get_page_chars_with_geometry(case["page_index"])
    probe = case["probe"]
    if case["selection"] == "contiguous":
        visible = [char for char in geometry.chars if str(char.get("char", "")) not in _CONTROL_CHARS]
        text = "".join(str(char["char"]) for char in visible)
        start = text.find(probe)
        if start < 0 or text.find(probe, start + 1) >= 0:
            raise ValueError(f"Probe is missing or ambiguous: {case['source']}, {probe}")
        groups = [visible[start : start + len(probe)]]
    elif case["selection"] == "line":
        groups = []
        for line in get_lines_from_chars(geometry.chars):
            text = "".join(span["text"] for span in line["spans"])
            text = "".join(char for char in text if char not in _CONTROL_CHARS)
            if probe in text:
                groups.append([char for span in line["spans"] for char in span.get("chars", [])])
    else:
        raise ValueError(f"Unknown selection: {case['selection']}")
    if len(groups) != 1:
        raise ValueError(f"Expected one fragment: {case['source']}, {probe}")
    fragments = []
    for chars in groups:
        indices = {int(char["char_idx"]) for char in chars}
        fragments.append(
            {
                "chars": chars,
                "tight_bboxes": {key: value for key, value in geometry.tight_bboxes.items() if key in indices},
                "origins": {key: value for key, value in geometry.origins.items() if key in indices},
            }
        )
    return {**case, "fragments": fragments}


def main() -> None:
    """校验源文件身份后检查或显式更新 fixture；日常 pytest 无需源文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="DocGale demo/pdfs 目录")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("hybrid_native_script_inputs.json"))
    parser.add_argument("--check", action="store_true", help="只比较，不写出")
    args = parser.parse_args()
    original = json.loads(args.fixture.read_text(encoding="utf-8"))
    for name, source in original["sources"].items():
        if hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
    updated = {
        **original,
        "producer": {"docgale": version("docgale"), "pypdfium2": version("pypdfium2")},
        "cases": [extract_case(args.source_root, case) for case in original["cases"]],
    }
    serialized = json.dumps(updated, ensure_ascii=False, indent=2, default=list) + "\n"
    if args.check:
        if json.loads(serialized) != original:
            raise ValueError("Fixture differs from the recorded extraction; inspect before updating")
        print(f"Verified {len(updated['cases'])} cases")
    else:
        args.fixture.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
