"""隔离加载指定源码，记录真实档位输出、图像边界及性能。"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile


def main() -> None:
    """保护 spawn 子进程入口，仅主进程执行解析与模型初始化。"""
    startup_started = time.perf_counter()
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--root", type=Path, required=True)
    arg.add_argument("--out", type=Path, required=True)
    arg.add_argument("--tier", default="flash")
    arg.add_argument("--small-backend", default="onnx")
    arg.add_argument("--mode", default="txt")
    arg.add_argument("--source", type=Path)
    arg.add_argument("--pages")
    arg.add_argument("--repeat", type=int, default=1)
    arg.add_argument("--block-opencv", action="store_true")
    arg.add_argument("--no-input-trace", action="store_true")
    args = arg.parse_args()
    if args.repeat < 1:
        arg.error("--repeat must be positive")
    if not (args.root / "mineru").is_dir():
        arg.error("--root must contain the MinerU source package")
    sys.path.insert(0, str(args.root))
    if args.block_opencv:
        import importlib.abc

        class BlockOpenCV(importlib.abc.MetaPathFinder):
            """独立验收进程在真实模型加载前阻断直接及间接的 cv2 导入。"""

            def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
                if fullname == "cv2" or fullname.startswith("cv2."):
                    raise ModuleNotFoundError("OpenCV is blocked for acceptance", name=fullname)
                return None

        sys.meta_path.insert(0, BlockOpenCV())
    os.environ["MINERU_MODEL_SMALL_BACKEND"] = args.small_backend
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    os.environ["MINERU_TABLE_DEVICE"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MINERU_LANG"] = "en"
    args.out.mkdir(parents=True, exist_ok=True)
    from mineru.config import VlmConfig, get_config
    import mineru

    assert Path(mineru.__file__).resolve().parent == (args.root / "mineru").resolve()

    get_config().llm_aided.features.title_leveling = False
    get_config().llm_aided.features.cross_page_table_cell_merge = False

    def fingerprint(value: Any) -> Any:
        """完整记录模型输入数组及相关参数，不使用有损数值归一化。"""
        import numpy as np
        from PIL import Image

        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, Image.Image):
            return {"size": list(value.size), "mode": value.mode, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, (list, tuple)):
            return [fingerprint(v) for v in value]
        if isinstance(value, dict):
            return {str(k): fingerprint(v) for k, v in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if "torch" in sys.modules and isinstance(value, sys.modules["torch"].Tensor):
            return fingerprint(value.detach().cpu().numpy())
        raise TypeError(f"Unsupported model input fingerprint type: {type(value).__name__}")

    traces = {}
    if not args.no_input_trace and (args.tier != "flash" or args.mode == "ocr"):
        from mineru.model.runtime.hybrid import HybridLocalModelContextSingleton

        context = HybridLocalModelContextSingleton().get_model()
        for prop, methods in {
            "layout_model": ["batch_predict"],
            "ocr_model": ["ocr"],
            "table_cls_model": ["predict", "batch_predict"],
            "table_orientation_cls_model": ["batch_predict"],
            "mfr_model": ["batch_predict"],
        }.items():
            try:
                model = getattr(context, prop)
            except AttributeError:
                continue
            for method in methods:
                original = getattr(model, method, None)
                if not callable(original):
                    continue
                key = prop + "." + method

                def wrapped(*values: Any, _original: Callable[..., Any] = original, _key: str = key, **options: Any) -> Any:
                    """仅在验收脚本采集真实模型调用，不替换算法或结果。"""
                    traces.setdefault(_key, []).append({"input": fingerprint(values), "options": fingerprint(options)})
                    result = _original(*values, **options)
                    # OCR 返回值可能含运行耗时；最终协议另行完整比较。
                    return result

                setattr(model, method, wrapped)

    if not args.no_input_trace and args.tier != "flash":
        import onnxruntime

        original_ort_run = onnxruntime.InferenceSession.run

        def record_ort_input(session: Any, output_names: Any, input_feed: dict[str, Any], *values: Any, **options: Any) -> Any:
            """记录送入 ONNX Runtime 的真实张量，原样透传模型推理。"""
            key = "onnx." + Path(str(getattr(session, "_model_path", "unknown"))).name
            traces.setdefault(key, []).append(fingerprint(input_feed))
            return original_ort_run(session, output_names, input_feed, *values, **options)

        onnxruntime.InferenceSession.run = record_ort_input
        if args.small_backend == "torch":
            for name, model in (
                ("torch.layout", context.layout_model.model),
                ("torch.ocr_det", context.ocr_model.text_detector.net),
                ("torch.ocr_rec", context.ocr_model.text_recognizer.net),
                ("torch.mfr", context.mfr_model.net),
            ):
                original_forward = model.forward

                def record_forward(
                    *values: Any, _original: Callable[..., Any] = original_forward, _key: str = name, **options: Any
                ) -> Any:
                    """只在完整模型 forward 边界记录张量，避免遍历每一层。"""
                    traces.setdefault(_key, []).append({"args": fingerprint(values), "kwargs": fingerprint(options)})
                    return _original(*values, **options)

                model.forward = record_forward

    from importlib.metadata import PackageNotFoundError, version

    def installed_version(name: str) -> str | None:
        """无 OpenCV 的验收环境允许可选依赖元数据缺失。"""
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    from mineru.parser import parse
    from mineru.render import RenderFormat, render

    source = args.root / "demo/pdfs/demo2.pdf"
    source = args.source or source
    options = {"tier": args.tier, "ocr_mode": args.mode, "page_range": args.pages or ("all" if args.tier == "flash" else "1-2")}
    if args.tier in ("standard", "advanced"):
        options["vlm_config"] = VlmConfig(engine="llama-cpp", max_concurrency=1)
    from docvortex.analyzers.native import PdfModel
    from docvortex.document.pdf import PDFDocument

    native_parse_calls = 0
    native_pages_requested = 0
    original_predict = PdfModel.predict

    def counted_predict(model: PdfModel, document: PDFDocument) -> list[list[dict[str, Any]]]:
        """通过公开原生模型接口计数整本文档调用和页请求，不依赖私有诊断模块。"""
        nonlocal native_parse_calls, native_pages_requested
        native_parse_calls += 1
        native_pages_requested += document.page_count
        return original_predict(model, document)

    PdfModel.predict = counted_predict
    startup_seconds = time.perf_counter() - startup_started
    times = []
    for index in range(args.repeat):
        start = time.perf_counter()
        result = parse(source, **options)
        times.append(time.perf_counter() - start)
        if index:
            continue
        middle = result.middle_json.model_dump_json()
        (args.out / "middle.json").write_text(middle)
        digests = {}
        for output_format in RenderFormat:
            value = render(result.middle_json, output_format)
            if isinstance(value, bytes):
                (args.out / (output_format.value + ".bin")).write_bytes(value)
                if output_format in (RenderFormat.DOCX, RenderFormat.EPUB):
                    with ZipFile(BytesIO(value)) as archive:
                        entries = {}
                        for name in sorted(archive.namelist()):
                            payload = archive.read(name)
                            if name in ("docProps/core.xml", "OEBPS/content.opf", "EPUB/package.opf"):
                                payload = re.sub(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", b"NORMALIZED_CREATION_TIME", payload)
                            entries[name] = hashlib.sha256(payload).hexdigest()
                        digests[output_format.value] = entries
                elif output_format == RenderFormat.PDF:
                    from docvortex.document.pdf import PDFDocument

                    with PDFDocument(value) as pdf:
                        pages = []
                        for page in range(len(pdf)):
                            image = pdf.render_page(page, scale=1).pil_image
                            pages.append(fingerprint(image))
                            image.close()
                        digests["pdf"] = pages
                else:
                    digests[output_format.value] = hashlib.sha256(value).hexdigest()
            elif isinstance(value, str):
                (args.out / (output_format.value + ".txt")).write_text(value)
                digests[output_format.value] = hashlib.sha256(value.encode()).hexdigest()
            else:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
                (args.out / (output_format.value + ".json")).write_text(text)
                digests[output_format.value] = hashlib.sha256(text.encode()).hexdigest()
        (args.out / "digests.json").write_text(json.dumps(digests, indent=2))
        assert middle == result.middle_json.model_dump_json()
    (args.out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2))
    # 验收诊断记录实际扩展及其摘要，不以环境变量代替已选择的后端。
    from docvortex._compute_backend import backend_info

    (args.out / "performance.json").write_text(
        json.dumps(
            {
                "times": times,
                "startup_seconds": startup_seconds,
                "native_parse_calls": native_parse_calls,
                "native_pages_requested": native_pages_requested,
                "cv2_loaded": "cv2" in sys.modules,
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "options": str(options),
                "docvortex": version("docvortex"),
                "opencv_blocked": args.block_opencv,
                "compute_backend": backend_info(),
                "opencv": installed_version("opencv-python"),
                "python": sys.version,
                "numpy": version("numpy"),
                "pillow": version("pillow"),
            },
            indent=2,
        )
    )
    print(args.out, times, flush=True)


if __name__ == "__main__":
    main()
