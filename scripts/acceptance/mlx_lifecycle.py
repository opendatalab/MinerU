"""macOS MLX 真机验收：真实解析、在途取消排空和未知错误后的实例复用。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


async def accept(sample: Path, evidence: Path, report: dict[str, Any]) -> None:
    """所有注入只发生在独立验收进程，真实模型与产物始终可复核。"""
    from PIL import Image, ImageDraw

    from mineru.config import VlmConfig
    from mineru.model.vlm.client import get_vlm_predictor
    from mineru.parser import parse_async
    from mineru.parser.writer import FileBasedDataWriter

    settings = VlmConfig(engine="mlx", server_url="")
    for tier in ("standard", "advanced"):
        result = await parse_async(sample, tier=tier, vlm_config=settings)
        assert result.middle_json.pages
        result.save(FileBasedDataWriter(str(evidence / tier)))
        report[tier] = {
            "pages": len(result.middle_json.pages),
            "markdown_characters": len((evidence / tier / "markdown.md").read_text(encoding="utf-8")),
        }
    runtime, backend = get_vlm_predictor(settings)
    assert backend == "mlx-engine"
    client = runtime.predictor.client
    assert client.health_probe_mode == "none"
    original = client.generate
    entered, drained = threading.Event(), threading.Event()
    image = Image.new("RGB", (512, 128), "white")
    ImageDraw.Draw(image).text((20, 30), "2 + 2 = 4; MinerU lifecycle", fill="black", font_size=28)

    def observe_generate(*args: Any, **kwargs: Any) -> Any:
        """进入真实同步生成后记录开始，直到真实生成自然退出才记录完成。"""
        entered.set()
        try:
            return original(*args, **kwargs)
        finally:
            drained.set()

    def unknown_failure(*args: Any, **kwargs: Any) -> Any:
        """显式模拟未分类执行故障，不声称这是 MLX 自身的崩溃。"""
        raise RuntimeError("acceptance injected unknown MLX execution failure")

    try:
        client.generate = observe_generate
        task = asyncio.create_task(runtime.aio_batch_recognize_text([image]))
        assert await asyncio.to_thread(entered.wait, 30), "Real MLX generation did not start"
        assert not drained.is_set(), "Generation already finished before cancellation"
        start = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation was not propagated")
        assert drained.is_set()
        report["cancellation"] = {"drained": True, "return_seconds": time.monotonic() - start}
        client.generate = unknown_failure
        try:
            await runtime.aio_batch_recognize_text([image])
        except RuntimeError as error:
            assert "acceptance injected unknown" in str(error)
        else:
            raise AssertionError("Unknown execution failure was swallowed")
        runtime.require_ready()
        client.generate = original
        result = await runtime.aio_batch_recognize_text([image])
        assert len(result) == 1 and isinstance(result[0], str) and result[0].strip(), result
        report["unknown_failure"] = {"injected": True, "instance_reused": True, "next_real_inference": result}
        report["passed"] = True
    finally:
        client.generate = original
        image.close()


def main() -> int:
    """始终保存版本、输入哈希、错误与关闭状态，缺少实机环境时明确失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"passed": False, "python": sys.version, "platform": platform.platform()}
    try:
        if sys.platform != "darwin":
            raise RuntimeError("MLX acceptance requires macOS with Apple Silicon")
        report["versions"] = {
            name: importlib.metadata.version(name) for name in ("mineru", "mineru-vl-utils", "mlx-vlm", "mlx")
        }
        import mineru_vl_utils

        import mineru

        report["source_paths"] = {"mineru": mineru.__file__, "mineru_vl_utils": mineru_vl_utils.__file__}
        report["sample_sha256"] = hashlib.sha256(args.sample.read_bytes()).hexdigest()
        asyncio.run(accept(args.sample, args.evidence, report))
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        from mineru.model.vlm.runtime import ModelSingleton, shutdown_cached_models

        try:
            shutdown_cached_models()
            assert not ModelSingleton._models
            report["runtime_cache_released"] = True
        except Exception:
            report["passed"] = False
            report["cleanup_error"] = traceback.format_exc()
        (args.evidence / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
