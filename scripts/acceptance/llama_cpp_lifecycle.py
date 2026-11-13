"""llama.cpp 真机验收：实际文档、取消排空、明确关闭及宿主显式重建。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


async def accept(sample: Path, evidence: Path, report: dict[str, Any]) -> None:
    """真实引擎调用与测试注入分别记录；不把显式关闭冒充原始崩溃复现。"""
    from mineru_llama_cpp import EngineClosedError, EngineError
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError, ServerError
    from PIL import Image, ImageDraw

    from mineru.config import VlmConfig
    from mineru.model.runtime.ownership import RuntimeOwner
    from mineru.model.vlm.client import get_vlm_predictor
    from mineru.model.vlm.errors import EngineDeadError
    from mineru.model.vlm.runtime import shutdown_cached_models
    from mineru.parser import parse_async
    from mineru.parser.writer import FileBasedDataWriter

    settings = VlmConfig(engine="llama-cpp", server_url="")
    for tier in ("standard", "advanced"):
        result = await parse_async(sample, tier=tier, vlm_config=settings)
        assert result.middle_json.pages
        result.save(FileBasedDataWriter(str(evidence / tier)))
        report[tier] = {
            "pages": len(result.middle_json.pages),
            "markdown_characters": len((evidence / tier / "markdown.md").read_text(encoding="utf-8")),
        }
    runtime, backend = get_vlm_predictor(settings)
    assert backend == "llama-cpp-engine" and runtime.supports_health_probe
    engine = runtime.predictor.client.llama_cpp_engine
    runtime.probe_health()
    engine.check_health()
    image = Image.new("RGB", (512, 128), "white")
    ImageDraw.Draw(image).text((20, 30), "2 + 2 = 4; MinerU lifecycle", fill="black", font_size=28)
    original = engine.generate
    entered, drained = threading.Event(), threading.Event()

    def observe(*args: Any, **kwargs: Any) -> Any:
        """标记真实原生调用提交与返回，不声称该标记能证明 GPU 已生成 token。"""
        entered.set()
        try:
            return original(*args, **kwargs)
        finally:
            drained.set()

    def ordinary_error(*args: Any, **kwargs: Any) -> Any:
        """只注入普通服务端错误，验证其不会被扩大为实例死亡。"""
        raise EngineError("acceptance ordinary request failure", error_type="server_error", code=500)

    try:
        # 直接 SDK 边界，不能让 MinerU 外层线程排空掩盖底层取消漏洞。
        sdk = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=engine, use_tqdm=False)
        engine.generate = observe
        task = asyncio.create_task(sdk.client.aio_predict(image, "Read the text."))
        assert await asyncio.to_thread(entered.wait, 30)
        assert not drained.is_set()
        started = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation was not propagated")
        assert drained.is_set()
        report["cancellation"] = {"direct_sdk": True, "drained": True, "seconds": time.monotonic() - started}
        engine.generate = ordinary_error
        try:
            await sdk.client.aio_predict(image)
        except ServerError as error:
            assert not isinstance(error, EngineUnavailableError) and isinstance(error.__cause__, EngineError)
        else:
            raise AssertionError("Ordinary error was swallowed")
        engine.generate = original
        runtime.require_ready()
        outputs = await runtime.aio_batch_recognize_text([image])
        assert outputs[0].strip()
        report["ordinary_error"] = {"injected": True, "same_instance_next_real_inference": outputs}
        await sdk.aclose()
        engine.check_health()  # SDK 关闭不能关闭注入的引擎。
        owners = [RuntimeOwner(), RuntimeOwner()]
        runtime.owners.update(owners)
        engine.close()  # 无在途推理；真实显式关闭，与 native abort 独立记录。
        try:
            runtime.probe_health()
        except EngineUnavailableError as error:
            assert isinstance(error.__cause__, EngineClosedError)
            report["closed_engine"] = {
                "kind": "explicit_idle_close",
                "host_alive": True,
                "error_code": error.error_code,
                "cause": type(error.__cause__).__name__,
            }
        else:
            raise AssertionError("Closed engine remained healthy")
        assert all(owner.engine_error for owner in owners)
        try:
            get_vlm_predictor(settings)
        except EngineDeadError:
            pass
        else:
            raise AssertionError("Cache reused or silently replaced a dead engine")
        report["closed_engine"]["all_owners_notified"] = True
        shutdown_cached_models()
        restored, _ = get_vlm_predictor(settings)
        assert restored is not runtime
        output = await restored.aio_batch_recognize_text([image])
        assert output[0].strip()
        report["explicit_host_rebuild"] = {"next_real_inference": output}
        report["passed"] = True
    finally:
        engine.generate = original
        image.close()


def main() -> int:
    """保存候选版本、输入哈希与真实产物；失败退出并始终清理宿主缓存。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    args = parser.parse_args()
    if args.models_dir:
        os.environ["MINERU_MODEL_BASE_DIR"] = str(args.models_dir.resolve())
    args.evidence.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"passed": False, "python": sys.version, "platform": platform.platform()}
    try:
        import mineru
        import mineru_llama_cpp
        import mineru_vl_utils

        report["versions"] = {
            name: importlib.metadata.version(name) for name in ("mineru", "mineru-vl-utils", "mineru-llama-cpp")
        }
        report["source_paths"] = {
            "mineru": mineru.__file__,
            "native": mineru_llama_cpp.__file__,
            "vl_utils": mineru_vl_utils.__file__,
        }
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
