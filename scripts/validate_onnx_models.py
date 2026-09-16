# Copyright (c) Opendatalab. All rights reserved.
"""验证真实 ONNX/Torch 模型及 PDF 解析，报告保留样本、输出和运行指标。"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def json_value(value: Any) -> Any:
    """把 NumPy 标量与数组转换成 JSON 原生类型，不改变数值或文字。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    """逐阶段保存结果，长时间验证被中断后仍可复查已完成样本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value) + "\n", encoding="utf-8")


def overlay(image: np.ndarray, boxes: list[Any], path: Path) -> None:
    """输出检测轮廓与序号，供人工核对真实页面和印章裁剪。"""
    canvas = image.copy()
    for index, box in enumerate(boxes):
        points = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(canvas, [points], True, (0, 160, 0), 2)
        cv2.putText(canvas, str(index), tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def ocr_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """对照 CPU 概率图、几何、弯曲裁剪与完整 OCR 文本，印章采用严格误差门槛。"""
    import torch

    from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX
    from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
    from mineru.model.registry import MINERU_4_MODELS_ONNX

    torch.set_num_threads(2)
    lang = "seal" if args.kind == "seal" else "ch"
    started = time.perf_counter()
    reference = PytorchPaddleOCR(lang=lang, device="cpu")
    repo = MINERU_4_MODELS_ONNX
    detector = repo.seal_det if lang == "seal" else repo.ocr_det
    candidate = PPOCRv6ONNX(str(detector.local_path()), str(repo.ocr_rec.local_path()), lang=lang)
    report: dict[str, Any] = {"initialization_seconds": time.perf_counter() - started, "samples": []}
    float64_reference = None
    for sample_index, path in enumerate(args.inputs):
        source = cv2.imread(str(path))
        if source is None:
            raise ValueError(f"Cannot read image: {path}")
        # 印章覆盖原尺寸、放大和非方形边界，避免只验证随机输入或单一尺寸。
        images = [source]
        if lang == "seal":
            images += [
                cv2.resize(source, None, fx=1.5, fy=1.5),
                cv2.copyMakeBorder(source, 20, 0, 45, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
            ]
        for variant_index, image in enumerate(images):
            prefix = f"sample-{sample_index}-{variant_index}"
            full_input, _, _ = reference.text_detector._preprocess_det_image(image)
            light_input, _ = candidate.text_detector._preprocess(image)
            np.testing.assert_array_equal(full_input[np.newaxis, :], light_input)
            with torch.inference_mode():
                full_maps = reference.text_detector.net(torch.from_numpy(light_input))["maps"].numpy()
            light_maps = candidate.text_detector.session.run(None, {candidate.text_detector.input_name: light_input})[0]
            error = float(np.abs(full_maps - light_maps).max())
            probability_matches = bool(np.allclose(full_maps, light_maps, rtol=1e-4, atol=1e-5))
            numerical_diagnostics = None
            if lang == "seal" and not probability_matches:
                # 保留原门槛的失败结果，用 FP64 参考定位 FP32 舍入，不通过放宽容差掩盖差异。
                if float64_reference is None:
                    float64_reference = copy.deepcopy(reference.text_detector.net).double()
                with torch.inference_mode():
                    precise_maps = float64_reference(torch.from_numpy(light_input).double())["maps"].numpy()
                numerical_diagnostics = {
                    "strict_rtol": 1e-4,
                    "strict_atol": 1e-5,
                    "strict_violations": int((~np.isclose(full_maps, light_maps, rtol=1e-4, atol=1e-5)).sum()),
                    "torch_fp32_max_error_vs_fp64": float(np.abs(full_maps - precise_maps).max()),
                    "onnx_fp32_max_error_vs_fp64": float(np.abs(light_maps - precise_maps).max()),
                    "threshold_mask_changes": int(((full_maps > 0.2) != (light_maps > 0.2)).sum()),
                }
            durations = {}
            outputs = {}
            saved_crops = {}
            for name, model in (("torch", reference), ("onnx", candidate)):
                started = time.perf_counter()
                result = model.ocr(image)[0] or []
                durations[name] = time.perf_counter() - started
                outputs[name] = result
                overlay(image, [item[0] for item in result], args.output_dir / f"{prefix}-{name}.png")
                if lang == "seal":
                    polygons, _ = model.text_detector(image)
                    polygons = model._seal_sort_boxes(polygons)
                    crops = model._seal_crop_by_polys(image, polygons)
                    saved_crops[name] = crops
                    for crop_index, crop in enumerate(crops):
                        cv2.imwrite(str(args.output_dir / f"{prefix}-{name}-crop-{crop_index}.png"), crop)
            texts = {name: [item[1][0] for item in value] for name, value in outputs.items()}
            geometry_matches = [item[0] for item in outputs["torch"]] == [item[0] for item in outputs["onnx"]]
            entry = {
                "input": str(path),
                "variant": variant_index,
                "shape": image.shape,
                "max_probability_error": error,
                "probability_matches": probability_matches,
                "geometry_matches": geometry_matches,
                "text_matches": texts["torch"] == texts["onnx"],
                "texts": texts,
                "seconds": durations,
                "numerical_diagnostics": numerical_diagnostics,
            }
            if lang == "seal":
                entry["crops_match"] = len(saved_crops["torch"]) == len(saved_crops["onnx"]) and all(
                    np.array_equal(a, b) for a, b in zip(saved_crops["torch"], saved_crops["onnx"])
                )
            report["samples"].append(entry)
            write_json(args.output_dir / "report.json", report)
            write_json(args.output_dir / f"{prefix}-outputs.json", outputs)
            if lang == "seal" and (
                not entry["text_matches"]
                or not geometry_matches
                or not entry["crops_match"]
                or (numerical_diagnostics is not None and numerical_diagnostics["threshold_mask_changes"] != 0)
            ):
                raise AssertionError(f"Seal output differs for {prefix}; inspect saved outputs")
    return report


def formula_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """使用同一批公式裁图对照 Plus-M，两种执行路径分别保留完整结果与耗时。"""
    import torch

    from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer
    from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX
    from mineru.model.registry import MINERU_4_MODELS_TORCH, MINERU_4_MODELS_ONNX

    torch.set_num_threads(2)
    images = [np.asarray(Image.open(path).convert("RGB")) for path in args.inputs]
    detections = [[{"label": "display_formula", "bbox": [0, 0, image.shape[1], image.shape[0]]}] for image in images]
    timings = {}
    results = {}
    for stack in ("torch", "onnx"):
        started = time.perf_counter()
        if stack == "torch":
            model = FormulaRecognizer(str(MINERU_4_MODELS_TORCH.pp_formulanet_plus_m_weights.local_path().parent), "cpu")
        else:
            model = PPFormulaNetPlusMONNX(
                str(MINERU_4_MODELS_ONNX.pp_formulanet_plus_m_weights.local_path()),
                str(MINERU_4_MODELS_ONNX.pp_formulanet_plus_m_config.local_path()),
                intra_op_num_threads=2,
            )
        timings[f"{stack}_initialization"] = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.batch_predict(detections, images, batch_size=4)
        timings[f"{stack}_inference"] = time.perf_counter() - started
        results[stack] = [page[0]["latex"] for page in predictions]
        write_json(args.output_dir / f"{stack}.json", {"results": results[stack], "seconds": timings})
        del model
    samples = [
        {"input": str(path), "torch": full, "onnx": light, "match": full == light}
        for path, full, light in zip(args.inputs, results["torch"], results["onnx"])
    ]
    return {
        "count": len(samples),
        "matched": sum(sample["match"] for sample in samples),
        "samples": samples,
        "seconds": timings,
    }


def layout_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """保存两种版面模型的标签、框、顺序和叠图，批次与单页必须一致。"""
    import torch

    from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from mineru.model.registry import MINERU_4_MODELS_TORCH, MINERU_4_MODELS_ONNX

    torch.set_num_threads(2)
    images = [Image.open(path).convert("RGB") for path in args.inputs]
    outputs = {}
    durations = {}
    for stack in ("torch", "onnx"):
        started = time.perf_counter()
        if stack == "torch":
            model = PPDocLayoutV2LayoutModel(str(MINERU_4_MODELS_TORCH.pp_doclayout_v2.local_path()), "cpu")
        else:
            model = PPDocLayoutV2LayoutModelONNX(
                str(MINERU_4_MODELS_ONNX.pp_doclayout_v2.local_path()),
                config_path=str(MINERU_4_MODELS_ONNX.pp_doclayout_v2_config.local_path()),
                intra_op_num_threads=2,
            )
        durations[f"{stack}_initialization"] = time.perf_counter() - started
        started = time.perf_counter()
        outputs[stack] = model.batch_predict(images, batch_size=2)
        durations[f"{stack}_inference"] = time.perf_counter() - started
        if stack == "onnx":
            assert outputs[stack] == [model.predict(image) for image in images]
        for index, (image, predictions) in enumerate(zip(images, outputs[stack])):
            boxes = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in [item["bbox"] for item in predictions]]
            overlay(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), boxes, args.output_dir / f"page-{index}-{stack}.png")
        del model
    return {"outputs": outputs, "seconds": durations}


def parse_documents(args: argparse.Namespace) -> dict[str, Any]:
    """通过产品 Parser 解析真实文档，保存结果包、重复运行耗时及导入边界。"""
    from mineru.kit.common import save_parse_result
    from mineru.parser import parse
    from mineru.parser.api_server import _preload_server_models

    report: dict[str, Any] = {"stack": args.small_backend, "tier": args.tier, "ocr_mode": args.ocr_mode, "documents": []}
    started = time.perf_counter()
    preload = _preload_server_models(args.tier, language="ch")
    report["preload_seconds"] = time.perf_counter() - started
    report["engine"] = preload.engine
    for path in args.inputs:
        durations = []
        document_dir = args.output_dir / path.stem
        document_dir.mkdir(parents=True, exist_ok=True)
        for iteration in range(args.rounds):
            started = time.perf_counter()
            result = parse(path, tier=args.tier, ocr_mode=args.ocr_mode, page_range=args.pages)
            durations.append(time.perf_counter() - started)
            if iteration == 0:
                save_parse_result(result, document_dir / "result.zip", "zip")
                save_parse_result(result, document_dir / "markdown.md", "markdown")
                save_parse_result(result, document_dir / "middle_json.json", "middle_json")
                if result._model_output is not None:
                    write_json(document_dir / "model_output.json", result._model_output.to_dict())
        report["documents"].append(
            {
                "input": str(path),
                "pages": len(result.pages),
                "seconds": durations,
                "markdown_characters": len(result.markdown()),
            }
        )
        write_json(args.output_dir / "report.json", report)
    report["heavy_modules_imported"] = [name for name in ("torch", "transformers") if name in sys.modules]
    if args.small_backend == "light" and report["heavy_modules_imported"]:
        raise AssertionError(f"Light imported heavy modules: {report['heavy_modules_imported']}")
    return report


def main() -> None:
    """配置隔离的模型目录和本地来源，运行指定验证场景并写出报告。"""
    from mineru.config import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["ocr", "seal", "mfr", "layout", "parse"])
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--models-dir", type=Path, default=Path("output/model-migration/models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--tier", choices=["basic", "standard"], default="basic")
    parser.add_argument("--ocr-mode", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--pages", default="")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.model.base_dir = str(args.models_dir.resolve())
    config.model.source = "local"
    # 组件对照需要同时加载 Torch；完整 Light 解析则严格使用 Light 配置。
    config.model.small_backend = args.small_backend if args.kind == "parse" else "torch"
    config.llm_aided.features.title_leveling = False
    config.llm_aided.features.cross_page_table_cell_merge = False
    operations = {
        "ocr": ocr_comparison,
        "seal": ocr_comparison,
        "mfr": formula_comparison,
        "layout": layout_comparison,
        "parse": parse_documents,
    }
    report = operations[args.kind](args)
    import mineru

    report["environment"] = {
        "python": sys.version.split()[0],
        "onnxruntime": version("onnxruntime"),
        "mineru": version("mineru"),
        "implementation": str(Path(mineru.__file__).resolve()),
    }
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    write_json(args.output_dir / "report.json", report)
    print(f"Saved {args.kind} validation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
