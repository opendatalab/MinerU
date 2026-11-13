"""Windows LMDeploy 真机验收：真实推理、暂不可用、后台停机与托管进程恢复。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    """声明验收输入及子进程参数，普通运行不隐式安装或更换推理依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="包含图像或公式的真实 PDF，建议至少数页")
    parser.add_argument(
        "--backend", choices=("turbomind",), default="turbomind", help="Windows 验收只覆盖受支持的 TurboMind 后端"
    )
    parser.add_argument("--models-dir", type=Path, help="可选 MinerU 模型根目录，默认沿用已安装环境配置")
    parser.add_argument("--evidence", type=Path, default=Path(f"lmdeploy-evidence-{time.strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--incident-log", type=Path, help="可选：保存原始事故完整日志，用于独立根因调查")
    parser.add_argument(
        "--active-stop",
        choices=("close", "signal"),
        default="signal",
        help="默认 signal 验证宿主存活；close 仅用于独立原生崩溃诊断，可能令 worker 直接退出",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--uploads", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _versions() -> dict[str, Any]:
    """收集版本和设备信息，不记录 API Key 或整个配置环境。"""
    versions: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for name in ("mineru", "mineru-vl-utils", "docvortex", "lmdeploy", "torch", "transformers"):
        versions[name] = importlib.metadata.version(name)
    import mineru
    import mineru_vl_utils
    from mineru.model.vlm.handles import RuntimeHandle
    from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient
    from packaging.specifiers import SpecifierSet

    versions["mineru_source"] = mineru.__file__
    versions["vl_utils_source"] = mineru_vl_utils.__file__
    if not callable(getattr(RuntimeHandle, "probe_health", None)) or not callable(
        getattr(LmdeployEngineVlmClient, "close_health_probe", None)
    ):
        raise RuntimeError("The candidate MinerU and mineru-vl-utils wheels are not active in this interpreter")
    if versions["lmdeploy"] not in SpecifierSet(">=0.17.0,<0.19"):
        raise RuntimeError("This acceptance covers LMDeploy 0.17 and 0.18 only")
    import torch

    versions["cuda_available"] = torch.cuda.is_available()
    versions["cuda_version"] = torch.version.cuda
    versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return versions


def _worker(args: argparse.Namespace) -> None:
    """启动正式 API app，仅在独立验收进程中增加可控故障注入路由。"""
    log = open(args.evidence / f"worker-{os.getpid()}.log", "a", encoding="utf-8", buffering=1)
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    print(json.dumps({"versions": _versions(), "backend": args.backend}, ensure_ascii=False), flush=True)

    import uvicorn
    from PIL import Image, ImageDraw

    from mineru.config import VlmConfig
    from mineru.model.vlm.runtime import ModelSingleton
    from mineru.parser.api_server import create_app
    from mineru.parser.process_control import ManagedProcessControlWatcher

    application = create_app(
        upload_dir=str(args.uploads),
        tiers=("flash", "basic", "standard", "advanced"),
        concurrency=1,
        preload_models=True,
        vlm_config=VlmConfig(engine="lmdeploy", server_url=""),
    )

    def handle() -> Any:
        """只取得本验收服务已经加载的 LMDeploy 实例，不额外构造模型。"""
        pool = ModelSingleton()
        with pool._lock:
            matches = [item for item in pool._models.values() if item.backend == "lmdeploy-engine"]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one loaded LMDeploy runtime, got {len(matches)}")
        return matches[0]

    @application.get("/_acceptance/runtime")
    async def inspect_runtime() -> dict[str, Any]:
        """核对实际引擎配置和进程，避免误把其他后端当作 LMDeploy 验收。"""
        runtime = handle()
        pipeline = runtime.predictor.client.lmdeploy_engine
        return {
            "pid": os.getpid(),
            "backend": runtime.backend,
            "backend_config": type(pipeline.backend_config).__name__,
        }

    @application.post("/_acceptance/infer")
    async def infer_crop() -> dict[str, Any]:
        """真实调用 VLM 识别合成切片，确保健康状态背后确实存在可工作的模型。"""
        image = Image.new("RGB", (384, 96), "white")
        ImageDraw.Draw(image).text((20, 25), "2 + 2 = 4", fill="black", font_size=32)
        outputs = await handle().aio_batch_recognize_text([image])
        if len(outputs) != 1 or not isinstance(outputs[0], str) or not outputs[0].strip():
            raise RuntimeError(f"Unexpected VLM crop result: {outputs!r}")
        return {"recognized": outputs[0]}

    @application.post("/_acceptance/delay-loop")
    async def delay_loop() -> dict[str, bool]:
        """暂缓 Pipeline 循环以触发有界健康超时，保留宿主服务循环响应能力。"""
        pipeline = handle().predictor.client.lmdeploy_engine

        def delay() -> None:
            """阻塞十二秒，覆盖一次五秒检查间隔及五秒期限，然后自然恢复。"""
            threading.Event().wait(12)

        pipeline.internal_thread.loop.call_soon_threadsafe(delay)
        return {"scheduled": True}

    @application.get("/_acceptance/activity")
    async def inspect_activity() -> dict[str, int]:
        """读取实际后端调度指标，区分 API 任务在途与模型推理在途。"""
        pipeline = handle().predictor.client.lmdeploy_engine
        future = asyncio.run_coroutine_threadsafe(
            pipeline.async_engine.engine.get_health_status(),
            pipeline.internal_thread.loop,
        )
        status = await asyncio.wait_for(asyncio.wrap_future(future), timeout=3)
        metrics = status["schedule_metrics"]
        return {"active_seqs": metrics.active_seqs, "waiting_seqs": metrics.waiting_seqs}

    @application.post("/_acceptance/signal-dead")
    async def signal_dead() -> dict[str, Any]:
        """活跃 GPU 推理期间只注入健康故障信号，允许本机计算自然收尾。"""
        pipeline = handle().predictor.client.lmdeploy_engine
        engine = pipeline.async_engine.engine

        async def inject() -> dict[str, Any]:
            """在所属循环确认真实活跃序列后，只替换验收进程的健康响应。"""
            get_health_status = engine.get_health_status
            status = await get_health_status()
            metrics = status["schedule_metrics"]
            if metrics.active_seqs == 0:
                return {"injected": False, "active_seqs": 0, "waiting_seqs": metrics.waiting_seqs}

            async def report_dead() -> dict[str, Any]:
                """用明确的注入标记模拟整体失效，不并发释放仍在运行的原生资源。"""
                return {"alive": False, "message": "Acceptance injected dead health signal during live inference"}

            engine.get_health_status = report_dead
            return {"injected": True, "active_seqs": metrics.active_seqs, "waiting_seqs": metrics.waiting_seqs}

        future = asyncio.run_coroutine_threadsafe(inject(), pipeline.internal_thread.loop)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=3)

    @application.post("/_acceptance/stop-engine")
    async def stop_engine() -> dict[str, bool]:
        """只停止验收进程内的后端，保留 Pipeline 与 API 进程观察故障传播。"""
        pipeline = handle().predictor.client.lmdeploy_engine

        def stop() -> None:
            """使用后端明确的停止接口，不操作私有任务或生产进程。"""
            pipeline.async_engine.engine.close()

        pipeline.internal_thread.loop.call_soon_threadsafe(stop)
        return {"scheduled": True}

    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=args.port, log_level="info"))

    def request_shutdown() -> None:
        """复用正式托管控制通道，请求优雅退出后由父管理者负责有界强制停止。"""
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


async def _wait_health(client: Any, base: str, code: str | None, timeout: float = 30) -> dict[str, Any]:
    """按公开健康接口等待目标状态，保留明确错误码。"""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        response = await client.get(base + "/v1/health")
        last = response.json()
        if code is None and response.status_code == 200:
            return last
        if code is not None and response.status_code == 503 and last.get("error", {}).get("code") == code:
            return last
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Health did not reach {code or 'healthy'}: {last}")


async def _submit(client: Any, base: str, data: bytes, name: str, tier: str) -> dict[str, Any]:
    """走正式 Tasks 清单和上传协议，不调用测试专用解析入口。"""
    response = await client.post(
        base + "/v1/tasks",
        json={
            "tier": tier,
            "files": [{"file_name": name, "bytes": len(data), "sha256sum": hashlib.sha256(data).hexdigest()}],
        },
    )
    response.raise_for_status()
    task = response.json()
    for item in task["files"]:
        upload = item.get("upload")
        if upload:
            response = await client.put(upload["url"], content=data, headers=upload["headers"])
            response.raise_for_status()
    return task


async def _terminal(client: Any, base: str, task: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    """等待任务终态，超时保留最后一次完整状态供诊断。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"{base}/v1/tasks/{task['task_id']}")
        response.raise_for_status()
        task = response.json()
        if task["status"] in {"completed", "partial", "failed", "canceled"}:
            return task
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Task did not finish: {task}")


async def _convert(client: Any, base: str, data: bytes, tier: str, evidence: Path, name: str) -> dict[str, Any]:
    """真实转换并检查产物协议，保存 ZIP 供人工核对 Markdown 与版面。"""
    task = await _terminal(client, base, await _submit(client, base, data, name + ".pdf", tier))
    if task["status"] != "completed":
        raise RuntimeError(f"Healthy conversion failed: {task}")
    response = await client.get(task["files"][0]["result_url"])
    response.raise_for_status()
    (evidence / f"{name}.zip").write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member = next(item for item in archive.namelist() if item.endswith("middle_json.json"))
        middle = json.loads(archive.read(member))
        if not middle["pages"] or middle["extensions"]["mineru"]["tier"] != tier:
            raise RuntimeError("Result does not match requested tier or contains no pages")
    return task


async def _accept(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """通过正式 WorkerPool 管理真实进程，故障注入仅存在于验收子进程。"""
    import httpx
    from mineru.kit.router.workers import ManagedLocalWorker, RouterSettings, WorkerPool, WorkerState

    class AcceptanceWorker(ManagedLocalWorker):
        """只替换验收子进程命令，保留正式启停、超时与控制通道实现。"""

        def command(self, port: int, upload_dir: Path) -> list[str]:
            """启动带回环验收路由的正式 API app。"""
            return [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--port",
                str(port),
                "--uploads",
                str(upload_dir),
                "--backend",
                args.backend,
                "--evidence",
                str(args.evidence),
            ]

    settings = RouterSettings(local_gpus="none", worker_tier="standard", preload_models=True)
    local = AcceptanceWorker("acceptance", "127.0.0.1", None, settings)
    state = WorkerState("acceptance", "", "local", local_worker=local)
    pool = WorkerPool(settings)
    pool._workers[state.worker_id] = state
    data = args.sample.read_bytes()

    async def recover() -> None:
        """让现有管理者根据 engine_dead 剔除和替换进程，核对 PID 与 generation。"""
        old_process = local.process
        old_generation = state.generation
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            await pool.refresh(state)
            if state.generation > old_generation and state.healthy:
                assert old_process.poll() is not None
                assert local.process.pid != old_process.pid
                report["recoveries"].append(
                    {"old_pid": old_process.pid, "new_pid": local.process.pid, "generation": state.generation}
                )
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("Managed worker did not recover")

    try:
        await pool.refresh(state)
        if not state.healthy:
            raise RuntimeError(f"Worker failed startup: {state.last_error}")
        async with httpx.AsyncClient(timeout=120) as client:
            report["runtime"] = (await client.get(state.base_url + "/_acceptance/runtime")).json()
            assert report["runtime"]["backend_config"] == "TurbomindEngineConfig", report["runtime"]
            response = await client.post(state.base_url + "/_acceptance/infer")
            response.raise_for_status()
            report["real_inference"] = response.json()
            for tier in ("standard", "advanced"):
                await _convert(client, state.base_url, data, tier, args.evidence, f"before-{tier}")
            report["healthy_conversions"] = True
            response = await client.post(state.base_url + "/_acceptance/delay-loop")
            response.raise_for_status()
            report["transient"] = await _wait_health(client, state.base_url, "engine_unavailable")
            response = await client.post(
                state.base_url + "/v1/tasks",
                json={
                    "tier": "standard",
                    "files": [{"file_name": "unavailable.pdf", "bytes": 3}],
                },
            )
            assert response.status_code == 503 and response.json()["error"]["code"] == "engine_unavailable"
            await _wait_health(client, state.base_url, None)
            report["transient_recovered"] = True
            response = await client.post(state.base_url + "/_acceptance/stop-engine")
            response.raise_for_status()
            report["idle_death"] = await _wait_health(client, state.base_url, "engine_dead")
            await recover()
            await _convert(client, state.base_url, data, "standard", args.evidence, "after-idle-restart")
            jobs = [await _submit(client, state.base_url, data, f"inflight-{i}.pdf", "advanced") for i in range(4)]
            interrupted_runtime = (await client.get(state.base_url + "/_acceptance/runtime")).json()
            interrupted_process = local.process
            interrupted_dir = Path(local.temp_dir.name)
            report["inflight_runtime"] = interrupted_runtime
            report["inflight_stop_mode"] = args.active_stop
            async with asyncio.timeout(60):
                while True:
                    snapshots = [(await client.get(f"{state.base_url}/v1/tasks/{job['task_id']}")).json() for job in jobs]
                    if args.active_stop == "signal":
                        activity = await client.post(state.base_url + "/_acceptance/signal-dead")
                    else:
                        activity = await client.get(state.base_url + "/_acceptance/activity")
                    activity.raise_for_status()
                    activity_snapshot = activity.json()
                    if activity_snapshot["active_seqs"] > 0:
                        if args.active_stop == "signal":
                            assert activity_snapshot["injected"]
                        report["activity_before_inflight_stop"] = activity_snapshot
                        report["tasks_before_inflight_stop"] = [item["status"] for item in snapshots]
                        break
                    if all(item["status"] in {"completed", "partial", "failed", "canceled"} for item in snapshots):
                        raise RuntimeError("No active LMDeploy sequences observed; use a longer PDF sample")
                    await asyncio.sleep(0.05)
            if args.active_stop == "close":
                response = await client.post(state.base_url + "/_acceptance/stop-engine")
                response.raise_for_status()
            try:
                report["inflight_health"] = await _wait_health(client, state.base_url, "engine_dead")
            except httpx.TransportError:
                if args.active_stop != "close":
                    raise
                # 原生崩溃可能先于 HTTP 响应终止宿主，单独保留进程恢复证据。
                report["inflight_transport_error"] = traceback.format_exc()
                exit_code = await asyncio.to_thread(interrupted_process.wait, 15)
                if exit_code == 0:
                    raise RuntimeError("Worker exited without an error after losing the HTTP connection")
                report["inflight_process_exit"] = {"code": exit_code, "hex": f"0x{exit_code & 0xFFFFFFFF:08x}"}
                report["inflight_task_terminal_states_observed"] = False
            else:
                report["inflight_results"] = [await _terminal(client, state.base_url, job, 30) for job in jobs]
                failures = [file for task in report["inflight_results"] for file in task["files"] if file["status"] == "failed"]
                assert len(failures) == len(jobs) and all(file["error"]["code"] == "engine_dead" for file in failures)
                assert all(file.get("result_url") is None for file in failures)
                report["inflight_task_terminal_states_observed"] = True
                assert interrupted_process.poll() is None
                report["host_alive_after_task_cleanup"] = True
            await recover()
            assert not interrupted_dir.exists(), interrupted_dir
            report["interrupted_upload_directory_removed"] = True
            replacement_runtime = (await client.get(state.base_url + "/_acceptance/runtime")).json()
            assert replacement_runtime["pid"] != interrupted_runtime["pid"]
            report["replacement_runtime"] = replacement_runtime
            await _convert(client, state.base_url, data, "advanced", args.evidence, "after-inflight-restart")
            report["recovered_after_active_stop"] = True
            report["passed"] = report["inflight_task_terminal_states_observed"]
    finally:
        await pool.close()


def main() -> int:
    """保存每次验收的完整报告，失败时也保留 traceback 和所有 worker 日志。"""
    args = _arguments()
    args.evidence = args.evidence.resolve()
    args.evidence.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["MINERU_LMDEPLOY_BACKEND"] = args.backend
    if args.models_dir:
        os.environ["MINERU_MODEL_BASE_DIR"] = str(args.models_dir.resolve())
    if args.worker:
        _worker(args)
        return 0
    report: dict[str, Any] = {
        "passed": False,
        "backend": args.backend,
        "recoveries": [],
        "original_incident_root_cause": "not established; lifecycle fault injection only",
    }
    try:
        if sys.platform != "win32":
            raise RuntimeError("Run this release acceptance on Windows with a configured CUDA MinerU environment")
        if args.sample is None or not args.sample.is_file() or args.sample.suffix.lower() != ".pdf":
            raise ValueError("--sample must point to an existing PDF")
        report["versions"] = _versions()
        report["sample_sha256"] = hashlib.sha256(args.sample.read_bytes()).hexdigest()
        if args.incident_log:
            (args.evidence / "original-incident.log").write_bytes(args.incident_log.read_bytes())
        asyncio.run(_accept(args, report))
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        (args.evidence / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evidence: {args.evidence}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
