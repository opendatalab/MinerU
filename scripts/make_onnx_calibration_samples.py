"""Build the six bundled, original text-line fixtures; no font is bundled at runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXTS = (
    "文档解析与智能识别 2026",
    "文字、公式和表格转换为结构化内容，支持检索与分析。",
    "Document recognition and structured content",
    "The quick brown fox jumps over the lazy dog. Page 123 of 456.",
    "0123456789 9876543210 2026-10-18",
    "Invoice 12345678: 3 x 128.50 = 385.50; Total 1,234.56",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "mineru/resources/onnx_calibration"
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 32)
    rows = []
    for index, text in enumerate(TEXTS):
        bbox = font.getbbox(text)
        image = Image.new("RGB", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), "white")
        ImageDraw.Draw(image).text((6 - bbox[0], 6 - bbox[1]), text, font=font, fill="black")
        path = destination / f"line-{index + 1}.png"
        image.save(path, optimize=True)
        rows.append({"file": path.name, "text": text, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "version": 1,
        "source": "Original MinerU calibration sentences, rendered by scripts/make_onnx_calibration_samples.py",
        "font": args.font.name,
        "font_sha256": hashlib.sha256(args.font.read_bytes()).hexdigest(),
        "font_size": 32,
        "preprocessing": {"color": "BGR", "image_shape": [3, 48, 320], "batch": 6, "normalization": "production rec"},
        "samples": rows,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
