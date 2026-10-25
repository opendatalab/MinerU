"""隔离子进程中的 MPS det 服务实验；不改变 MinerU 的默认设备和清理策略。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref
import zipfile
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = {
    "tender": Path("/Volumes/2TB/pdf-test/表格困难户/跨页合并表/跨页单元格合并列")
    / "中国科协创新战略研究院科技智库大模型迭代建设与运维项目招标文件-最终稿.pdf",
    "python": Path("/Volumes/2TB/pdf-test/新的一批badcase/来自mac/Python.pdf"),
    "114": Path("/Volumes/2TB/pdf-test/新的一批badcase/来自mac/_114.pdf"),
}
DEFAULT_OUTPUT = ROOT / "tmp/pdfs/issue5632-cross-document-20261012"


def write_json(path: Path, value: Any) -> None:
    """原子保存进度，进程异常退出时也保留此前完成的请求。"""
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Observer:
    """记录普通文字 det、清理边界和显存，不持有模型或页面图像。"""

    def __init__(self, policy: str, output: Path) -> None:
        """建立当前实验进程的取证目录和后台内存采样线程。"""
        import psutil
        import torch

        self.torch = torch
        self.process = psutil.Process()
        self.policy = policy
        self.output = output
        self.current: dict[str, Any] | None = None
        self.detectors: list[weakref.ReferenceType[Any]] = []
        self.seen_shapes: set[tuple[int, ...]] = set()
        self.seen_crops: set[str] = set()
        self.cleanups: list[dict[str, Any]] = []
        self.initialized: list[dict[str, Any]] = []
        self.trace: Any = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        (output / "crops").mkdir(parents=True, exist_ok=True)
        self.sampler = threading.Thread(target=self.sample_loop, daemon=True)
        self.sampler.start()

    def memory(self) -> dict[str, int]:
        """采集进程内存和 MPS allocator/driver 的当前值。"""
        return {
            "rss": self.process.memory_info().rss,
            "mps_allocated": self.torch.mps.current_allocated_memory(),
            "mps_driver": self.torch.mps.driver_allocated_memory(),
        }

    def sample_loop(self) -> None:
        """以半秒间隔记录高水位，JSONL 不保存在内存中。"""
        with (self.output / "memory-samples.jsonl").open("w", encoding="utf-8") as stream:
            while not self.stop_event.wait(0.5):
                sample = {"time": time.time(), **self.memory()}
                with self.lock:
                    if self.current is not None:
                        sample["document"] = self.current["label"]
                        for key in ("rss", "mps_allocated", "mps_driver"):
                            self.current["peak_memory"][key] = max(self.current["peak_memory"][key], sample[key])
                stream.write(json.dumps(sample) + "\n")
                stream.flush()

    def begin(self, label: str) -> dict[str, Any]:
        """开始一次 API 请求的取证，模型和跨文档形状集合继续复用。"""
        assert self.current is None
        directory = self.output / label
        directory.mkdir(exist_ok=True)
        self.trace = (directory / "det-trace.jsonl").open("w", encoding="utf-8")
        memory = self.memory()
        self.current = {
            "label": label,
            "pid": os.getpid(),
            "stages": [],
            "cleanups": [],
            "preprocess_seconds": 0.0,
            "batch_seconds": 0.0,
            "forward_seconds": 0.0,
            "capture_seconds": 0.0,
            "trace_seconds": 0.0,
            "ordinary_det_calls": 0,
            "crops": 0,
            "boxes": 0,
            "new_shapes": 0,
            "reused_shapes": 0,
            "shapes": Counter(),
            "memory_before": memory,
            "peak_memory": dict(memory),
        }
        return self.current

    def finish(self) -> dict[str, Any]:
        """请求结束后保存统计，释放取证句柄而不影响模型缓存。"""
        self.torch.mps.synchronize()
        with self.lock:
            assert self.current is not None
            result = self.current
            result["memory_after"] = self.memory()
            result["det_seconds"] = result["preprocess_seconds"] + result["batch_seconds"]
            result["shape_counts"] = [{"shape": list(shape), "calls": count} for shape, count in result.pop("shapes").items()]
            result["models"] = [
                {
                    "detector_id": id(detector),
                    "net_id": id(detector.net),
                    "device": detector.device,
                    "dtype": str(detector.ocr_inference_dtype),
                    "box_thresh": detector.args.det_db_box_thresh,
                    "unclip_ratio": detector.args.det_db_unclip_ratio,
                }
                for ref in self.detectors
                if (detector := ref()) is not None
            ]
            self.trace.close()
            self.trace = None
            self.current = None
        assert result["ordinary_det_calls"] > 0, "本次请求未执行真实 det"
        write_json(self.output / result["label"] / "observation.json", result)
        return result

    def log_stage(self, message: Any) -> None:
        """读取生产 stage_timer 的日志，不替换实际阶段编排。"""
        match = re.search(r"MinerU stage: name=([^,]+), elapsed_s=([0-9.]+)", message.record["message"])
        if match is not None and self.current is not None:
            self.current["stages"].append({"name": match[1], "seconds": float(match[2])})

    def cleanup(self, device: str, original: Any) -> None:
        """只在独立实验进程切换文档清理策略，记录真实调用边界。"""
        started = time.perf_counter()
        before = self.memory()
        if self.policy == "C" and device.startswith("mps"):
            gc.collect()
        else:
            original(device)
            if device.startswith("mps"):
                self.seen_shapes.clear()
        self.torch.mps.synchronize()
        event = {
            "device": device,
            "seconds": time.perf_counter() - started,
            "before": before,
            "after": self.memory(),
            "empty_cache": self.policy != "C",
        }
        self.cleanups.append(event)
        if self.current is not None:
            self.current["cleanups"].append(event)

    def close(self) -> None:
        """停止采样并关闭异常请求尚未释放的文件句柄。"""
        self.stop_event.set()
        self.sampler.join()
        if self.trace is not None:
            self.trace.close()


def install_det_observer(observer: Observer) -> None:
    """在实验子进程用计时子类替换构造器，覆盖所有普通文字 det 实例。"""
    import numpy as np
    import torch
    from PIL import Image

    from mineru.model._internal.pytorchocr.infer import predict_cls, predict_det, predict_rec, predict_system

    class ExperimentalDetector(predict_det.TextDetector):
        """保留生产前后处理，只显式选择普通文字 det 设备和记录数据。"""

        def __init__(self, args: Any, **kwargs: Any) -> None:
            """复用权重和 FP32 初始化，印章 det 维持原有行为。"""
            self.ordinary = getattr(args, "lang", "ch") not in {"seal", "seal_lite"}
            self.prepared: dict[int, str] = {}
            self.forward_started = 0.0
            super().__init__(args, **kwargs)
            if self.ordinary:
                self.device = "cpu" if observer.policy == "A" else "mps"
                self._apply_inference_precision(self.device, precision_override="fp32")
                assert next(self.net.parameters()).dtype == torch.float32
                observer.detectors.append(weakref.ref(self))
                observer.initialized.append({"id": id(self), "device": self.device, "box_thresh": args.det_db_box_thresh})
                self.net.register_forward_pre_hook(self.before_forward)
                self.net.register_forward_hook(self.after_forward)

        def before_forward(self, module: Any, inputs: Any) -> None:
            """同步实际执行起点，并记录 NCHW 形状是否在当前缓存周期出现过。"""
            if self.device == "mps":
                torch.mps.synchronize()
            self.forward_started = time.perf_counter()
            current = observer.current
            if current is not None:
                shape = tuple(inputs[0].shape)
                current["shapes"][shape] += 1
                current["new_shapes" if shape not in observer.seen_shapes else "reused_shapes"] += 1
                observer.seen_shapes.add(shape)

        def after_forward(self, module: Any, inputs: Any, outputs: Any) -> None:
            """同步 GPU 完成时间，避免把异步提交时间当作网络执行时间。"""
            if self.device == "mps":
                torch.mps.synchronize()
            if observer.current is not None:
                observer.current["forward_seconds"] += time.perf_counter() - self.forward_started

        def _preprocess_det_image(self, img: Any) -> Any:
            """计时生产预处理，并保存可复查的原始裁剪，取证耗时单独统计。"""
            started = time.perf_counter()
            result = super()._preprocess_det_image(img)
            elapsed = time.perf_counter() - started
            current = observer.current
            if self.ordinary and current is not None and result is not None:
                current["preprocess_seconds"] += elapsed
                capture_started = time.perf_counter()
                crop_hash = hashlib.sha256(str(img.shape).encode() + np.ascontiguousarray(img).tobytes()).hexdigest()
                if crop_hash not in observer.seen_crops:
                    Image.fromarray(img[:, :, ::-1]).save(observer.output / "crops" / f"{crop_hash}.png")
                    observer.seen_crops.add(crop_hash)
                self.prepared[id(result[0])] = crop_hash
                current["capture_seconds"] += time.perf_counter() - capture_started
            return result

        def _batch_process_preprocessed(self, batch_items: Any) -> Any:
            """计时生产批处理，保存输入对应的最终框而不持有原始数组。"""
            started = time.perf_counter()
            results, elapsed = super()._batch_process_preprocessed(batch_items)
            batch_seconds = time.perf_counter() - started
            current = observer.current
            if self.ordinary and current is not None:
                current["batch_seconds"] += batch_seconds
                current["ordinary_det_calls"] += 1
                current["crops"] += len(batch_items)
                trace_started = time.perf_counter()
                record = {
                    "detector_id": id(self),
                    "box_thresh": self.args.det_db_box_thresh,
                    "unclip_ratio": self.args.det_db_unclip_ratio,
                    "shape": [len(batch_items), *batch_items[0][1].shape],
                    "seconds": batch_seconds,
                    "items": [],
                }
                for item, (boxes, _) in zip(batch_items, results):
                    points = boxes.tolist() if hasattr(boxes, "tolist") else boxes
                    record["items"].append(
                        {"crop": self.prepared.pop(id(item[1])), "original_shape": list(item[3]), "boxes": points}
                    )
                    current["boxes"] += len(points) if points is not None else 0
                observer.trace.write(json.dumps(record) + "\n")
                current["trace_seconds"] += time.perf_counter() - trace_started
            return results, elapsed

    def initialize_system(system: Any, args: Any, **kwargs: Any) -> None:
        """在调用侧替换 det 构造器，保留原定义模块中显式 super 的类身份。"""
        system.text_detector = ExperimentalDetector(args, **kwargs)
        system.text_recognizer = predict_rec.TextRecognizer(args, **kwargs)
        system.use_angle_cls = args.use_angle_cls
        system.drop_score = args.drop_score
        if system.use_angle_cls:
            system.text_classifier = predict_cls.TextClassifier(args, **kwargs)

    predict_system.TextSystem.__init__ = initialize_system


def serve(args: argparse.Namespace) -> None:
    """启动只有本轮实验使用的本地 API，策略不传播到其他进程。"""
    os.environ["MINERU_DEVICE_MODE"] = "mps"
    os.environ["MINERU_PROFILE_STAGES"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import uvicorn
    from loguru import logger
    from mineru.config import get_config
    from mineru.backend.analysis.pdf import pipeline
    from mineru.model.runtime.hybrid import AtomModelSingleton, HybridLocalModelContextSingleton
    from mineru.model.runtime.memory import clean_memory
    from mineru.parser.api_server import create_app

    get_config().model.small_backend = "torch"
    output = args.output / args.policy
    output.mkdir(parents=True, exist_ok=True)
    observer = Observer(args.policy, output)
    install_det_observer(observer)
    logger.add(observer.log_stage, level="INFO", format="{message}")

    def document_cleanup(device: str) -> None:
        """保留生产 release_document 边界，仅注入本进程的回收回调。"""
        observer.cleanup(device, clean_memory)

    pipeline.clean_memory = document_cleanup
    app = create_app(
        tier="basic",
        concurrency=1,
        allow_local_source=True,
        image_analysis=False,
        preload_models=False,
        retention_seconds=0,
        upload_dir=str(output / "service-storage"),
    )
    production_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Any) -> Any:
        """先完整执行生产服务关闭，再释放本进程的本地模型和 MPS 缓存。"""
        try:
            async with production_lifespan(application):
                yield
        finally:
            before = observer.memory()
            HybridLocalModelContextSingleton._models.clear()
            AtomModelSingleton._models.clear()
            gc.collect()
            clean_memory("mps")
            observer.torch.mps.synchronize()
            write_json(
                output / "shutdown.json",
                {
                    "before": before,
                    "after": observer.memory(),
                    "initialized": observer.initialized,
                    "cleanups": observer.cleanups,
                },
            )
            observer.close()

    app.router.lifespan_context = lifespan

    @app.post("/_benchmark/begin")
    async def begin(body: dict[str, str]) -> dict[str, Any]:
        """控制器在提交单个真实 API 任务之前开启观察窗口。"""
        return observer.begin(body["label"])

    @app.post("/_benchmark/end")
    async def end() -> dict[str, Any]:
        """控制器确认任务终态后，保存此次请求的阶段与内存统计。"""
        return observer.finish()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


def reserve_port() -> int:
    """申请临时本地端口，避免打扰已经运行的用户服务。"""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_with_retry(client: Any, url: str) -> Any:
    """仅重试幂等 GET 的短暂连接重置，绝不重新提交正在执行的解析任务。"""
    import httpx

    for attempt in range(3):
        try:
            return client.get(url)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    raise AssertionError("不可达的重试分支")


def run_policy(args: argparse.Namespace, policy: str) -> list[dict[str, Any]]:
    """启动独立服务进程，逐个提交新任务，并完整关闭服务。"""
    import httpx

    output = args.output / policy
    if (output / "requests.json").exists():
        raise FileExistsError(f"已有结果，需使用新输出目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    port = reserve_port()
    records: list[dict[str, Any]] = []
    with (output / "server.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                "--policy",
                policy,
                "--output",
                str(args.output),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False) as client:
                deadline = time.monotonic() + 180
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f"服务启动失败：{output / 'server.log'}")
                    try:
                        if client.get("/v1/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("等待服务启动超时")
                    time.sleep(0.5)
                sequence = args.sequence.split(",")
                for position, sample in enumerate(sequence):
                    label = f"{position + 1:02d}-{sample}"
                    response = client.post("/_benchmark/begin", json={"label": label})
                    response.raise_for_status()
                    payload = {
                        "tier": "basic",
                        "ocr_mode": "ocr",
                        "output_formats": ["markdown", "middle_json", "zip"],
                        "files": [{"source": {"type": "local", "path": str(SAMPLES[sample])}, "page_range": args.pages}],
                    }
                    started = time.perf_counter()
                    response = client.post("/v1/parse/jobs", json=payload)
                    response.raise_for_status()
                    job = response.json()
                    job_id = job["job_id"]
                    announced = time.monotonic()
                    while job["status"] not in {"completed", "partial", "failed", "canceled"}:
                        time.sleep(0.25)
                        response = get_with_retry(client, f"/v1/parse/jobs/{job_id}")
                        response.raise_for_status()
                        job = response.json()
                        if time.monotonic() - announced >= 30:
                            print(
                                f"RUN policy={policy} document={label} elapsed={time.perf_counter() - started:.1f}s", flush=True
                            )
                            announced = time.monotonic()
                    request_seconds = time.perf_counter() - started
                    directory = output / label
                    write_json(directory / "job.json", job)
                    if job["status"] != "completed":
                        raise RuntimeError(f"任务未完整完成：{directory / 'job.json'}")
                    observation = client.post("/_benchmark/end")
                    observation.raise_for_status()
                    record = observation.json()
                    record.update(
                        {
                            "sample": sample,
                            "job_id": job_id,
                            "request_seconds": request_seconds,
                            "server_seconds": job["files"][0]["parse"]["duration_ms"] / 1000,
                        }
                    )
                    for kind, reference in job["files"][0]["output_files"].items():
                        if reference is None:
                            continue
                        response = get_with_retry(client, f"/v1/files/{reference['file_id']}/content")
                        response.raise_for_status()
                        suffix = {"markdown": "md", "middle_json": "json", "zip": "zip"}[kind]
                        (directory / f"{kind}.{suffix}").write_bytes(response.content)
                        if kind == "zip":
                            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                                archive.extractall(directory / "bundle")
                    records.append(record)
                    write_json(output / "requests.json", records)
                    if policy == "C" and len(records) == 6:
                        first = records[2]["memory_after"]
                        second = records[5]["memory_after"]
                        if any(second[key] - first[key] > max(256 * 1024**2, first[key] * 0.1) for key in first):
                            sequence.extend(sequence[:3])
                            print("EXTEND C: 内存增长超过门槛，在同一服务进程追加第三遍三本文档", flush=True)
                    print(
                        f"DONE policy={policy} document={label} request={request_seconds:.2f}s "
                        f"det={record['det_seconds']:.2f}s "
                        f"crops={record['crops']} memory={record['memory_after']}",
                        flush=True,
                    )
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=30)
            if child.returncode != 0:
                print(f"服务退出码={child.returncode}，日志：{output / 'server.log'}", flush=True)
    return records


def main() -> None:
    """提供服务子进程与串行控制器入口，默认执行三种策略的两遍文档。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", choices=("A", "B", "C"), default="A")
    parser.add_argument("--policies", default="A,B,C")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--sequence", default="tender,python,114,tender,python,114")
    parser.add_argument("--pages", default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "serve":
        serve(args)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "protocol.json",
            {
                "sequence": args.sequence,
                "pages": args.pages,
                "policies": args.policies,
                "samples": {
                    key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for key, path in SAMPLES.items()
                },
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            },
        )
        for policy in args.policies.split(","):
            run_policy(args, policy)


if __name__ == "__main__":
    main()
