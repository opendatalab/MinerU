from __future__ import annotations

import argparse
import os
import sys

from ...model.registry import MINERU_2_5_PRO_2605_1_2B


def _run_native(args: list[str], prog_name: str, standalone_mode: bool) -> None:
    """调用原生入口并恢复参数，保持 Typer 与独立命令的退出语义。"""
    from mlx_vlm.server import main as native_main

    original_argv = sys.argv
    sys.argv = [prog_name, *args]
    try:
        native_main()
    except SystemExit as exc:
        if standalone_mode or exc.code:
            raise
    finally:
        sys.argv = original_argv


def main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
    """准备 MinerU 模型后启动原生 MLX 服务，生命周期覆盖整个服务进程。"""
    if "--help" in args or "-h" in args:
        _run_native(args, prog_name, standalone_mode)
        return

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    # CLI 优先于环境变量；未配置时将活跃并发与连续 batch 上限统一为八。
    parser.add_argument("--max-num-seqs", type=int, default=os.environ.get("MLX_VLM_MAX_NUM_SEQS") or "8")
    options, forwarded = parser.parse_known_args(args)

    from mineru_vl_utils.mlx_compat import prepare_mlx_model_path

    model = options.model or str(MINERU_2_5_PRO_2605_1_2B.ensure())
    with prepare_mlx_model_path(model) as prepared_path:
        _run_native(
            [
                "--model",
                str(prepared_path),
                "--host",
                options.host,
                "--port",
                str(options.port),
                "--max-num-seqs",
                str(options.max_num_seqs),
                *forwarded,
            ],
            prog_name,
            standalone_mode,
        )


__all__ = ["main"]
