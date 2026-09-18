"""测量真实模型或固定模型输出下的 PDF CPU gap，录制文件仅供本机可信回放。

在仓库根目录用 PYTHONPATH=. python tests/benchmarks/pdf_cpu_gaps.py 运行。
--record --tape FILE 保存一次模型与渲染输出；--tape FILE 回放相同输入。
省略 --tape 使用真实模型，可直接在 vLLM 服务器重复测量端到端耗时。
回放墙钟不包含真实渲染和模型推理，gap1 边界为 VLM 调用入口而非进度条创建。
"""

from __future__ import annotations
import argparse
import copy
import gc
import gzip
import hashlib
import json
import pickle
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


def _input_digest(value: object) -> str:
    """按值指纹化模型输入，覆盖数组像素、PIL 页图、顺序和标量参数。"""
    import numpy as np
    from PIL.Image import Image

    def normalize(item: object) -> object:
        """将模型参数变成稳定 JSON，未知类型显式报错而非忽略。"""
        if isinstance(item, np.ndarray):
            return ["array", str(item.dtype), item.shape, hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, Image):
            return ["image", item.mode, item.size, hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, bytes):
            return ["bytes", hashlib.sha256(item).hexdigest()]
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [normalize(part) for part in item]]
        if isinstance(item, (set, frozenset)):
            return [type(item).__name__, sorted((normalize(part) for part in item), key=lambda part: json.dumps(part))]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported model input: {type(item)}")

    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    """保存模型输出或回放已冻结输入，并产出完整协议与阶段耗时。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--tape", type=Path, help="本机录制的固定模型输出；省略时运行真实模型")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--runs", type=int, default=6, help="首次单独报告，其余运行用于预热后中位数")
    parser.add_argument("--compare", type=Path, help="需要逐字段一致的基线 output.json")
    parser.add_argument("--mode", choices=["txt", "ocr"], default="txt")
    parser.add_argument("--page-range", default="", help="复用 MinerU 的页范围语法，例如 1-20")
    parser.add_argument("--input-signatures", action="store_true", help="额外验证模型输入；该运行不用于计时")
    parser.add_argument("--compare-inputs", type=Path, help="需要一致的基线 input-signatures.json")
    args = parser.parse_args()
    if args.runs < 1 or (args.record and args.tape is None):
        parser.error("runs 必须为正整数，record 必须提供 tape")
    replay = args.tape is not None and not args.record
    live = not replay
    if args.record:
        args.runs = 1
    from mineru.backend.analysis.pdf import pipeline as p, window as w, ocr as o
    from mineru.backend.analysis.pdf.text import content as c
    from mineru.backend.analyze import _build_model_json
    from docvortex.schema import DocumentProperties
    from docvortex.postprocess.document import model_json_to_middle_json
    from mineru.render import render_content_list, render_content_list_v2, render_html
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    payload = args.pdf.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    page_index_map = None
    tape = defaultdict(list)
    if replay:
        with gzip.open(args.tape, "rb") as stream:
            tape = pickle.load(stream)
        assert tape["source_hash"] == source_hash
        selection = tape.get("selection")
        if selection is not None:
            assert args.page_range == selection["page_range"], "Page range differs from recording"
            payload, page_index_map = selection["payload"], selection["page_index_map"]
        else:
            assert not args.page_range, "Recording did not select a page range"
    elif args.page_range:
        from mineru.parser.mineru_parser import MinerUParser

        payload, page_index_map, _broken_pages = MinerUParser()._maybe_adjust_pdf_bytes(payload, "pdf", args.page_range)
        if args.record:
            # PDFium 重写会产生新的文件标识；回放保存的字节，避免标识差异污染固定输入校验。
            tape["selection"] = {"payload": payload, "page_index_map": page_index_map, "page_range": args.page_range}
    positions = defaultdict(int)
    events = {}
    timings = defaultdict(float)
    gaps = defaultdict(list)
    input_signatures = defaultdict(list)

    def wrapped(owner: object, method: str, name: str) -> None:
        """按调用顺序录制或重放模型/渲染结果，每次交付独立可修改对象。"""
        original = getattr(owner, method, None)

        def call(*values: object, **options: object) -> object:
            """记录真实阶段边界，避免把复制固定返回值计入 gap。"""
            events[name + "_start"] = time.perf_counter()
            if args.input_signatures or args.compare_inputs is not None:
                input_signatures[name].append(_input_digest((values, options)))
            if name == "layout":
                events.pop("det_end", None)
            if name == "vlm" and "layout_end" in events:
                gaps["gap1"].append(events["vlm_start"] - events["layout_end"])
            if name == "rec" and "det_end" in events:
                gaps["gap2"].append(events["rec_start"] - events.pop("det_end"))
            if live:
                result = original(*values, **options)
                if args.record:
                    tape[name].append(copy.deepcopy(result))
            else:
                index = positions[name]
                positions[name] += 1
                result = copy.deepcopy(tape[name][index])
            events[name + "_end"] = time.perf_counter()
            return result

        setattr(owner, method, call)

    def timed(owner: object, method: str) -> None:
        """聚合已有 CPU 阶段耗时，不改变阶段输入输出。"""
        original = getattr(owner, method)

        def call(*values: object, **options: object) -> object:
            """保留异常并累计真实执行时长。"""
            start = time.perf_counter()
            try:
                return original(*values, **options)
            finally:
                timings[method] += time.perf_counter() - start

        setattr(owner, method, call)

    setup_started = time.perf_counter()
    if live:
        model = p.HybridLocalModelContextSingleton().get_model()
        predictor, backend = p.get_vlm_predictor(None)
    else:
        model = SimpleNamespace(
            device="cpu",
            layout_model=SimpleNamespace(),
            table_orientation_cls_model=SimpleNamespace(),
            mfr_model=SimpleNamespace(),
            ocr_model=SimpleNamespace(text_detector=SimpleNamespace()),
            seal_model=SimpleNamespace(),
        )
        predictor = SimpleNamespace()
        backend = "replay"
        p.HybridLocalModelContextSingleton = lambda: SimpleNamespace(get_model=lambda: model)
    p.get_vlm_predictor = lambda config: (predictor, backend)
    model_setup_seconds = time.perf_counter() - setup_started
    for owner, method, name in [
        (model.layout_model, "batch_predict", "layout"),
        (model.table_orientation_cls_model, "batch_predict", "orientation"),
        (model.mfr_model, "batch_predict", "mfr"),
        (model.ocr_model.text_detector, "batch_predict", "det"),
        (model.ocr_model, "ocr", "rec"),
        (model.seal_model, "ocr", "seal"),
        (predictor, "batch_extract_with_layout", "vlm"),
        (w, "load_images_from_pdf_bytes_range", "render"),
    ]:
        if owner is not None:
            wrapped(owner, method, name)
    for owner, method in [
        (w, "_apply_table_orientations"),
        (w, "_apply_native_txt_table_priority"),
        (w, "_process_text_and_formulas"),
        (w, "_ocr_det"),
        (c, "prepare_text_evidence"),
        (c, "txt_spans_extract"),
        (o, "_normalize_batch_ocr_det_boxes"),
        (o, "_append_ocr_det_result"),
    ]:
        timed(owner, method)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    expected = None
    for run in range(args.runs):
        positions.clear()
        events.clear()
        timings.clear()
        gaps.clear()
        input_signatures.clear()
        start = time.perf_counter()
        cpu = time.process_time()
        result = p.analyze_pdf(payload, effort="high", parse_mode=args.mode)
        wall = time.perf_counter() - start
        cpu = time.process_time() - cpu
        model_json = _build_model_json(result, "pdf", page_index_map, DocumentProperties())
        middle = model_json_to_middle_json(model_json)
        parse_wall = time.perf_counter() - start
        output = {
            "model": model_json.model_dump(mode="json"),
            "middle": middle.model_dump(mode="json"),
            "content_list": render_content_list(middle),
            "content_list_v2": render_content_list_v2(middle),
            "html": render_html(middle),
        }
        encoded = json.dumps(output, ensure_ascii=False, sort_keys=True).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if replay and expected is not None:
            assert digest == expected
        if args.compare is not None:
            assert output == json.loads(args.compare.read_text(encoding="utf-8")), "Full output differs"
        if args.compare_inputs is not None:
            assert dict(input_signatures) == json.loads(args.compare_inputs.read_text(encoding="utf-8")), "Model inputs differ"
        expected = digest
        if run == 0:
            (args.out / "output.json").write_bytes(encoded)
            (args.out / "output.html").write_text(output["html"], encoding="utf-8")
            if input_signatures:
                (args.out / "input-signatures.json").write_text(json.dumps(input_signatures, indent=2), encoding="utf-8")
        if replay:
            for name, values in tape.items():
                if name not in {"source_hash", "selection"}:
                    assert positions[name] == len(values), (name, positions[name], len(values))
        report = {
            "run": run,
            "wall": wall,
            "cpu": cpu,
            "infer": result.elapsed,
            "parse_wall": parse_wall,
            "gap1": sum(gaps["gap1"]) if gaps["gap1"] else None,
            "gap2": sum(gaps["gap2"]) if gaps["gap2"] else None,
            "stages": dict(timings),
            "sha256": digest,
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
        del result, model_json, middle, output, encoded
        gc.collect()
    summary = {
        "mode": "replay" if replay else "record" if args.record else "live",
        "timing_eligible": not (args.input_signatures or args.compare_inputs is not None),
        "pdf_sha256": source_hash,
        "analyzed_pdf_sha256": hashlib.sha256(payload).hexdigest(),
        "model_setup_seconds": model_setup_seconds,
        "cold": reports[0],
        "warm_median": {
            key: statistics.median(values) if (values := [row[key] for row in reports[1:] if row[key] is not None]) else None
            for key in ("wall", "parse_wall", "cpu", "infer", "gap1", "gap2")
        }
        if len(reports) > 1
        else None,
        "peak_rss_bytes": max(row["peak_rss"] for row in reports) * (1 if sys.platform == "darwin" else 1024),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "timings.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if args.record:
        tape["source_hash"] = source_hash
        with gzip.open(args.tape, "wb", compresslevel=1) as stream:
            pickle.dump(dict(tape), stream, protocol=5)


if __name__ == "__main__":
    main()
