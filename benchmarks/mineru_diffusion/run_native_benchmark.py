from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def wait_for_health(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    session = requests.Session()
    session.trust_env = False
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = session.get(url, timeout=2)
            response.raise_for_status()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="local path or Hugging Face cache path for MinerU-Diffusion weights",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="image URL passed to the OpenAI-compatible benchmark client",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/mineru_diffusion/native_direct"),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=360.0)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument(
        "--task-max-tokens",
        default="",
        help="comma-separated task=tokens overrides for generated default cases",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="optional baseline results JSONL for post-benchmark quality gate",
    )
    parser.add_argument("--compare-output", type=Path, default=None)
    parser.add_argument("--similarity-cases", default="")
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--max-control-repeat", type=int, default=None)
    parser.add_argument("--max-control-token-ratio", type=float, default=None)
    return parser.parse_args()


def latest_results_file(output_dir: Path) -> Path:
    latest = output_dir / "latest_results.jsonl"
    if latest.exists():
        return latest

    def sort_key(path: Path) -> tuple[int, float, str]:
        suffix = path.stem.removeprefix("results_")
        try:
            sequence = int(suffix)
        except ValueError:
            sequence = -1
        return sequence, path.stat().st_mtime, path.name

    results = sorted(output_dir.glob("results_*.jsonl"), key=sort_key)
    if not results:
        raise FileNotFoundError(f"no benchmark results found under {output_dir}")
    return results[-1]


def run_benchmark(args: argparse.Namespace) -> None:
    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    health_url = f"http://{args.host}:{args.port}/health"
    server_log = args.server_log
    if server_log is None:
        server_log = args.output_dir / "native_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    server_cmd = [
        sys.executable,
        "-m",
        "benchmarks.mineru_diffusion.native_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    bench_cmd = [
        sys.executable,
        "-m",
        "benchmarks.mineru_diffusion.bench_hf_server",
        "--endpoint",
        endpoint,
        "--image",
        args.image,
        "--output-dir",
        str(args.output_dir),
        "--timeout",
        str(args.timeout),
    ]
    if args.task_max_tokens:
        bench_cmd.extend(["--task-max-tokens", args.task_max_tokens])

    env = None
    if args.cuda_visible_devices is not None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with server_log.open("w", encoding="utf-8") as log_file:
        server = subprocess.Popen(
            server_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            wait_for_health(health_url, args.startup_timeout)
            subprocess.run(bench_cmd, check=True, env=env)
            if args.baseline_results is not None:
                compare_cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.mineru_diffusion.compare_results",
                    "--baseline",
                    str(args.baseline_results),
                    "--candidate",
                    str(latest_results_file(args.output_dir)),
                ]
                if args.compare_output is not None:
                    compare_cmd.extend(["--output", str(args.compare_output)])
                if args.similarity_cases:
                    compare_cmd.extend(["--similarity-cases", args.similarity_cases])
                if args.min_similarity is not None:
                    compare_cmd.extend(["--min-similarity", str(args.min_similarity)])
                if args.max_control_repeat is not None:
                    compare_cmd.extend([
                        "--max-control-repeat",
                        str(args.max_control_repeat),
                    ])
                if args.max_control_token_ratio is not None:
                    compare_cmd.extend([
                        "--max-control-token-ratio",
                        str(args.max_control_token_ratio),
                    ])
                subprocess.run(compare_cmd, check=True, env=env)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
