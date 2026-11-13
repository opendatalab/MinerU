"""Linux/WSL2 vLLM 真机验收，复用现有任务与托管进程验收流程。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections.abc import AsyncIterator, Callable
from copy import copy
from typing import Any


async def observe_generation(
    generate: Callable[..., AsyncIterator[Any]],
    active: set[str],
    metrics: dict[str, int],
    cumulative_output_kind: Any,
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """验收时观察非终态 token；只改变输出投递方式，不改生成参数或共享配置。"""
    sampling = copy(kwargs["sampling_params"])
    sampling.output_kind = cumulative_output_kind
    kwargs["sampling_params"] = sampling
    request_id = kwargs["request_id"]
    try:
        async for output in generate(*args, **kwargs):
            count = sum(len(choice.token_ids) for choice in output.outputs)
            if count and not output.finished:
                active.add(request_id)
                metrics["token_events"] += 1
            if output.finished:
                active.discard(request_id)
            yield output
    finally:
        active.discard(request_id)


def run_worker(args: argparse.Namespace) -> None:
    """仅在独立验收 worker 注入路由及观测，生产模块不包含故障注入。"""
    import uvicorn
    from lmdeploy_lifecycle import _versions
    from PIL import Image, ImageDraw
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.exceptions import EngineDeadError

    from mineru.config import VlmConfig
    from mineru.model.vlm.runtime import ModelSingleton
    from mineru.parser.api_server import create_app
    from mineru.parser.process_control import ManagedProcessControlWatcher

    log = open(args.evidence / f"worker-{os.getpid()}.log", "a", encoding="utf-8", buffering=1)
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    print(json.dumps({"versions": _versions("vllm")}), flush=True)
    application = create_app(
        upload_dir=str(args.uploads),
        tiers=("flash", "basic", "standard", "advanced"),
        concurrency=1,
        preload_models=True,
        vlm_config=VlmConfig(engine="vllm", server_url=""),
    )
    active: set[str] = set()
    metrics = {"token_events": 0}
    instrumented = False

    def handle() -> Any:
        """取得本应用已加载的唯一异步引擎，不为了观察而创建模型。"""
        pool = ModelSingleton()
        with pool._lock:
            found = [item for item in pool._models.values() if item.backend == "vllm-async-engine"]
        if len(found) != 1:
            raise RuntimeError(f"Expected one vLLM runtime, got {len(found)}")
        return found[0]

    async def instrument() -> None:
        """在引擎循环观察真实输出 token，避免把任务 running 当作 GPU 在途证据。"""
        nonlocal instrumented
        if instrumented:
            return
        engine = handle()._predictor.client.vllm_async_llm
        original = engine.generate

        async def generate(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            """真实客户端默认只接收终态，验收改为累计投递以观察尚未完成的 token。"""
            async for output in observe_generation(original, active, metrics, RequestOutputKind.CUMULATIVE, *args, **kwargs):
                yield output

        engine.generate = generate
        instrumented = True

    @application.get("/_acceptance/runtime")
    async def inspect_runtime() -> dict[str, Any]:
        """同时安装仅供验收使用的输出观测，并报告真实引擎身份。"""
        runtime = handle()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(instrument(), runtime._loop))
        return {"pid": os.getpid(), "backend": runtime.backend, "backend_config": "AsyncLLM"}

    @application.post("/_acceptance/infer")
    async def infer_crop() -> dict[str, Any]:
        """执行真实切片识别，健康检查通过不能代替推理验证。"""
        image = Image.new("RGB", (384, 96), "white")
        try:
            ImageDraw.Draw(image).text((20, 25), "2 + 2 = 4", fill="black", font_size=32)
            outputs = await handle().aio_batch_recognize_text([image])
            assert len(outputs) == 1 and isinstance(outputs[0], str) and outputs[0].strip(), outputs
            return {"recognized": outputs[0]}
        finally:
            image.close()

    @application.post("/_acceptance/delay-loop")
    async def delay_loop() -> dict[str, bool]:
        """只阻塞引擎循环，API 循环必须仍能给出暂不可用响应。"""

        def delay() -> None:
            """覆盖五秒周期与五秒期限，随后自然恢复。"""
            threading.Event().wait(12)

        handle()._loop.call_soon_threadsafe(delay)
        return {"scheduled": True}

    async def activity(inject: bool = False) -> dict[str, Any]:
        """在模型循环中原子观测输出与注入健康错误，避免查询后请求已经结束。"""
        snapshot = {"active_seqs": len(active), "token_events": metrics["token_events"], "injected": False}
        if inject and active:

            async def report_dead() -> None:
                """注入真实 vLLM 致命异常类型，仍允许本机 GPU 工作自然结束。"""
                raise EngineDeadError()

            # 适配器保存公开回调，替换回调是本验收唯一的合成死亡信号。
            handle()._predictor.client._health_probe._check_health = report_dead
            snapshot["injected"] = True
        return snapshot

    @application.get("/_acceptance/activity")
    async def inspect_activity() -> dict[str, Any]:
        """读取已产生真实 token 且尚未退出的推理数。"""
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(activity(), handle()._loop))

    @application.post("/_acceptance/signal-dead")
    async def signal_dead() -> dict[str, Any]:
        """在活跃推理期间注入健康故障；报告明确标注为信号注入。"""
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(activity(True), handle()._loop))

    @application.post("/_acceptance/stop-engine")
    async def stop_engine() -> dict[str, bool]:
        """在所属循环调用真实公开 shutdown，保留 API 进程观察空闲死亡。"""
        runtime = handle()
        runtime._loop.call_soon_threadsafe(runtime._predictor.client.vllm_async_llm.shutdown)
        return {"scheduled": True}

    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=args.port, log_level="info"))

    def request_shutdown() -> None:
        """复用宿主控制通道及既有关闭超时。"""
        server.should_exit = True

    watcher = ManagedProcessControlWatcher.from_environment(request_shutdown)
    if watcher is not None:
        watcher.start()
    try:
        server.run()
    finally:
        if watcher is not None:
            watcher.close()
        log.close()


if __name__ == "__main__":
    from lmdeploy_lifecycle import main

    raise SystemExit(main("vllm"))
