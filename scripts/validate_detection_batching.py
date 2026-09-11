# Copyright (c) Opendatalab. All rights reserved.
"""验证 ONNX 检测真批处理：真实概率图/几何对照和带调用记录的 PDF 解析。"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from mineru.model.runtime.onnx import ort_session
from .validate_onnx_models import overlay, parse_documents, write_json

__all__ = ["RecordingSession", "components", "integration"]


class RecordingSession:
    """只在验证脚本记录 ORT 的真实输入批次，不更改模型图或生产调度。"""

    def __init__(self, session: Any, name: str, *, retain_outputs: bool = False) -> None:
        """保存原会话；组件对照可额外保留概率图。"""
        self.session = session
        self.name = name
        self.retain_outputs = retain_outputs
        self.calls: list[dict[str, Any]] = []
        self.outputs: list[list[np.ndarray]] = []

    def get_inputs(self) -> list[Any]:
        """透传图输入签名。"""
        return self.session.get_inputs()

    def get_outputs(self) -> list[Any]:
        """透传图输出签名。"""
        return self.session.get_outputs()

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """记录送入 ORT 的形状和纯推理耗时。"""
        started = time.perf_counter()
        result = self.session.run(output_names, input_feed)
        self.calls.append(
            {"inputs": {key: list(value.shape) for key, value in input_feed.items()}, "seconds": time.perf_counter() - started}
        )
        if self.retain_outputs:
            self.outputs.append(result)
        return result


def components(args: argparse.Namespace) -> dict[str, Any]:
    """用不同页面、翻转页、空图和真实印章比较逐项与批量输出。"""
    from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX
    from mineru.model.ocr.pp_ocr_v6_onnx import TextDetectorONNX

    repo = args.models_dir / "MinerU-4_models_onnx"
    pages = [
        Image.open(args.samples_dir / name).convert("RGB") for name in ("demo1-page1.png", "demo2-page3.png", "seal-page.png")
    ]
    layout = PPDocLayoutV2LayoutModelONNX(
        str(repo / "Layout/PP-DocLayoutV2/inference.onnx"), config_path=str(repo / "Layout/PP-DocLayoutV2/inference.yml")
    )
    observer = RecordingSession(layout.session, "layout")
    layout.session = observer
    layout.predict(pages[0])
    observer.calls.clear()
    images = [*pages, Image.new("RGB", (800, 1000), "white")]
    started = time.perf_counter()
    single = layout.batch_predict(images, batch_size=1)
    single_seconds = time.perf_counter() - started
    single_calls = list(observer.calls)
    observer.calls.clear()
    started = time.perf_counter()
    batch = layout.batch_predict(images, batch_size=2)
    batch_seconds = time.perf_counter() - started
    checks = []
    for index, (left, right) in enumerate(zip(single, batch)):
        structure_match = [(x["label"], x["index"]) for x in left] == [(x["label"], x["index"]) for x in right]
        checks.append(
            {
                "index": index,
                "boxes_single": len(left),
                "boxes_batch": len(right),
                "labels_order_match": structure_match,
                "exact_match": left == right,
                "max_bbox_delta": max(
                    (float(np.abs(np.array(a["bbox"]) - b["bbox"]).max()) for a, b in zip(left, right)), default=0
                )
                if structure_match
                else None,
                "max_score_delta": max((abs(a["score"] - b["score"]) for a, b in zip(left, right)), default=0)
                if structure_match
                else None,
            }
        )
        overlay(
            np.array(images[index])[:, :, ::-1].copy(),
            [
                [
                    [x["bbox"][0], x["bbox"][1]],
                    [x["bbox"][2], x["bbox"][1]],
                    [x["bbox"][2], x["bbox"][3]],
                    [x["bbox"][0], x["bbox"][3]],
                ]
                for x in right
            ],
            args.output_dir / f"layout-{index}.png",
        )
    report: dict[str, Any] = {
        "layout": {
            "single_seconds": single_seconds,
            "batch_seconds": batch_seconds,
            "single_calls": single_calls,
            "batch_calls": observer.calls,
            "checks": checks,
        }
    }
    write_json(args.output_dir / "layout-single.json", single)
    write_json(args.output_dir / "layout-batch.json", batch)
    write_json(args.output_dir / "report.json", report)
    for kind in ("ocr_det", "seal"):
        seal = kind == "seal"
        detector = TextDetectorONNX(
            str(repo / "OCR/paddleocr" / ("seal_PP-OCRv4_det_infer.onnx" if seal else "ch_PP-OCRv6_tiny_det_infer.onnx")),
            **(
                {
                    "limit_side_len": 736,
                    "limit_type": "min",
                    "thresh": 0.2,
                    "box_thresh": 0.6,
                    "unclip_ratio": 0.5,
                    "box_type": "poly",
                }
                if seal
                else {}
            ),
        )
        originals = [np.array(page)[:, :, ::-1].copy() for page in ([pages[2]] if seal else pages[:2])]
        samples = [value for page in originals for value in (page, page[:, ::-1].copy())]
        samples.insert(1, np.empty((0, 0, 3), np.uint8))
        detector(originals[0])
        observer = RecordingSession(detector.session, kind, retain_outputs=True)
        detector.session = observer
        started = time.perf_counter()
        single = detector.batch_predict(samples, max_batch_size=1)
        single_seconds = time.perf_counter() - started
        single_calls = list(observer.calls)
        raw_single = [output[0][0] for output in observer.outputs]
        observer.calls.clear()
        observer.outputs.clear()
        started = time.perf_counter()
        batch = detector.batch_predict(samples, max_batch_size=16)
        batch_seconds = time.perf_counter() - started
        # 本样本按尺寸连续排列；按桶首见顺序处理尾批后，可直接对照概率图。
        raw_batch = [image for output in observer.outputs for image in output[0]]
        checks = []
        for index, (left, right) in enumerate(zip(single, batch)):
            a, b = left[0], right[0]
            equal = (
                a is None and b is None
                if a is None or b is None
                else (len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)))
            )
            checks.append(
                {
                    "index": index,
                    "boxes_match": bool(equal),
                    "single_count": 0 if a is None else len(a),
                    "batch_count": 0 if b is None else len(b),
                }
            )
            if b is not None:
                overlay(samples[index], b, args.output_dir / f"{kind}-{index}.png")
        probability = [
            {"max_abs": float(np.abs(a - b).max()), "allclose": bool(np.allclose(a, b, rtol=1e-4, atol=1e-5))}
            for a, b in zip(raw_single, raw_batch)
        ]
        report[kind] = {
            "single_seconds": single_seconds,
            "batch_seconds": batch_seconds,
            "single_calls": single_calls,
            "batch_calls": observer.calls,
            "checks": checks,
            "probabilities": probability,
        }
        write_json(args.output_dir / f"{kind}-single.json", [x[0] for x in single])
        write_json(args.output_dir / f"{kind}-batch.json", [x[0] for x in batch])
        write_json(args.output_dir / "report.json", report)
        assert all(x["boxes_match"] for x in checks) and all(x["allclose"] for x in probability)
    assert all(x["exact_match"] for x in report["layout"]["checks"])
    return report


def integration(args: argparse.Namespace) -> dict[str, Any]:
    """对真实 Parser 临时安装会话观察器，证明 PDF 调用链实际合批且不加载 Torch。"""
    observers = []

    def factory(model_path: str, device: str | None = None, intra_op_num_threads: int = 0) -> RecordingSession:
        """验证用会话工厂，生产会话配置原样透传。"""
        observer = RecordingSession(ort_session(model_path, device, intra_op_num_threads), Path(model_path).name)
        observers.append(observer)
        return observer

    with (
        patch("mineru.model.layout.pp_doclayout_v2_onnx.ort_session", factory),
        patch("mineru.model.ocr.pp_ocr_v6_onnx.ort_session", factory),
    ):
        report = parse_documents(args)
    report["sessions"] = [{"name": observer.name, "calls": observer.calls} for observer in observers]
    layout_calls = [call for observer in observers if observer.name == "inference.onnx" for call in observer.calls]
    det_calls = [call for observer in observers if "det_infer" in observer.name for call in observer.calls]
    assert any(call["inputs"]["image"][0] > 1 for call in layout_calls)
    assert any(next(iter(call["inputs"].values()))[0] > 1 for call in det_calls)
    return report


def main() -> None:
    """显式选择组件对照或真实 PDF 集成验证，并保存配置与内存指标。"""
    from mineru.config import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["components", "parse"])
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("output/model-migration/models"))
    parser.add_argument("--samples-dir", type=Path, default=Path("output/model-migration/samples"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", default="")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "ocr", "txt"])
    args = parser.parse_args()
    args.small_backend, args.tier, args.rounds = "light", "basic", 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.model.base_dir, config.model.source, config.model.small_backend = str(args.models_dir.resolve()), "local", "onnx"
    config.llm_aided.features.title_leveling = False
    config.llm_aided.features.cross_page_table_cell_merge = False
    report = components(args) if args.kind == "components" else integration(args)
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["environment"] = {"python": sys.version.split()[0], "onnxruntime": version("onnxruntime")}
    report["heavy_modules"] = [name for name in ("torch", "transformers") if name in sys.modules]
    assert not report["heavy_modules"]
    write_json(args.output_dir / "report.json", report)
    print("Saved", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
