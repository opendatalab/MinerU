# Copyright (c) Opendatalab. All rights reserved.
"""以薄包装启动 mineru-llama-cpp 分发的 llama-server（OpenAI 兼容）。

llama-server 的全部服务能力由 llama.cpp 提供，本模块只补默认参数并以
进程替换方式启动，llama.cpp 升级不需要改动这里。

默认参数对齐 EngineCore::EngineCore（进程内 llama-cpp-engine）在 C++ 侧
写死的 common_params 值，使 server 模式与解析模式的上下文容量、KV 分区、
并发与日志行为保持一致；用户显式传入的参数逐项让位。
"""

from __future__ import annotations

import os
import struct
import sys
import typing
from pathlib import Path

from ...model.registry import vlm_model_repo

DEFAULT_PORT = "30000"
DEFAULT_N_GPU_LAYERS = 99
DEFAULT_N_PARALLEL = 4
# 日志级别不注入：保持 llama-server 自身默认（INFO），否则启动横幅里的
# 监听地址/端口等关键信息会被 WARN 阈值压掉。
# 与 mineru_llama_cpp.engine._VALID_UNICODE_GRAMMAR 同款：作为服务端默认 grammar，
# 兜住 llama.cpp sampler 对 Q8_0 高分辨率页面可能采出的坏多字节 token；
# 请求级 grammar 仍可覆盖服务端默认。
VALID_UNICODE_GRAMMAR = "root ::= .*"

_GGUF_MAGIC = b"GGUF"
_GGUF_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT32 = 4

# llama-server 同一选项的短别名与等价选择器（清单来自 llama-server --help）。
# 追加默认参数前必须把整组别名视为「用户已显式提供」：llama.cpp 参数解析中
# 后出现的同名项生效，漏认别名会让追加的默认值覆盖用户先传入的值。
GRAMMAR_SELECTOR_FLAGS = ("--grammar", "--grammar-file", "-j", "--json-schema", "-jf", "--json-schema-file")
MODEL_SELECTOR_FLAGS = ("-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url")
PROJECTOR_SELECTOR_FLAGS = (
    "-mm",
    "--mmproj",
    "-mmu",
    "--mmproj-url",
    "-hfv",
    "-hfrv",
    "--hf-repo-v",
    "--mmproj-auto",
    "--no-mmproj",
    "--no-mmproj-auto",
)
PARALLEL_FLAGS = ("-np", "--parallel")
KV_UNIFIED_FLAGS = ("-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified")
SPECIAL_FLAGS = ("-sp", "--special")
# 值为凭据的旗标，启动横幅打印前脱敏；--api-key-file 等是路径不是凭据，不脱敏。
SECRET_VALUE_FLAGS = ("--api-key", "-hft", "--hf-token")


def llama_server_binary() -> Path:
    """定位 mineru-llama-cpp 包内分发的 llama-server 可执行文件。"""
    import mineru_llama_cpp

    # Windows wheel 分发的是 llama-server.exe，其余平台无后缀。
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return Path(mineru_llama_cpp.__file__).resolve().parent / "bin" / binary_name


def _has_arg(args: list[str], *flags: str) -> bool:
    return any(arg in flags or arg.startswith(f"{flag}=") for arg in args for flag in flags)


def _flag_value(args: list[str], *flags: str) -> str | None:
    for index, arg in enumerate(args):
        for flag in flags:
            if arg == flag and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
    return None


def _redact_secrets(args: list[str]) -> list[str]:
    """启动横幅脱敏：SECRET_VALUE_FLAGS 的值替换为 ***，覆盖 flag value 与 flag=value 两种形式。"""
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        flag = next((name for name in SECRET_VALUE_FLAGS if arg == name or arg.startswith(f"{name}=")), None)
        if flag is None:
            redacted.append(arg)
        elif arg == flag:
            redacted.append(arg)
            redact_next = True
        else:
            redacted.append(f"{flag}=***")
    return redacted


def _skip_gguf_value(f: typing.BinaryIO, value_type: int) -> None:
    """按 gguf.h 的类型布局跳过一个 kv 值。"""
    if value_type == _GGUF_STRING:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(length, os.SEEK_CUR)
    elif value_type == _GGUF_ARRAY:
        (element_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if element_type == _GGUF_STRING:
            for _ in range(count):
                (length,) = struct.unpack("<Q", f.read(8))
                f.seek(length, os.SEEK_CUR)
        else:
            f.seek(count * _GGUF_SCALAR_SIZE.get(element_type, 0), os.SEEK_CUR)
    else:
        f.seek(_GGUF_SCALAR_SIZE.get(value_type, 0), os.SEEK_CUR)


def _read_gguf_context_length(model_path: Path) -> int:
    """读取 GGUF 的 `<architecture>.context_length`，失败返回 0。

    对齐 EngineCore 的 read_n_ctx_train_from_gguf：server 的 --ctx-size
    只接受总量，而进程内引擎按单 slot 训练上下文 × n_parallel 配 KV，
    所以这里必须自己从模型元数据取训练上下文。

    GGUF 不保证 metadata key 顺序，`<arch>.context_length` 可能出现在
    general.architecture 之前：先按前缀收齐全部候选，读完后按 arch 查表。
    """
    context_by_architecture: dict[str, int] = {}
    architecture = ""
    try:
        with model_path.open("rb") as f:
            header = f.read(24)
            if len(header) < 24 or header[:4] != _GGUF_MAGIC:
                return 0
            version, _tensor_count, kv_count = struct.unpack_from("<IQQ", header, 4)
            if version < 2:
                return 0
            for _ in range(kv_count):
                (key_length,) = struct.unpack("<Q", f.read(8))
                key = f.read(key_length).decode("utf-8", errors="replace")
                (value_type,) = struct.unpack("<I", f.read(4))
                if key == "general.architecture" and value_type == _GGUF_STRING:
                    (value_length,) = struct.unpack("<Q", f.read(8))
                    architecture = f.read(value_length).decode("utf-8", errors="replace")
                elif key.endswith(".context_length") and value_type == _GGUF_UINT32:
                    (value,) = struct.unpack("<I", f.read(4))
                    context_by_architecture[key.removesuffix(".context_length")] = int(value)
                else:
                    _skip_gguf_value(f, value_type)
    except (OSError, struct.error):
        return 0
    return context_by_architecture.get(architecture, 0)


def main() -> None:
    args = sys.argv[1:]

    if not _has_arg(args, "--port"):
        args.extend(["--port", DEFAULT_PORT])
    if not _has_arg(args, *GRAMMAR_SELECTOR_FLAGS):
        args.extend(["--grammar", VALID_UNICODE_GRAMMAR])
    # params_.special=true：MinerU 输出里的 <|box_start|> 等格式标记注册为
    # special tokens，server 默认（false）会在输出 detokenize 时吞掉它们，
    # 解析格式依赖这些标记，必须显式开启。
    if not _has_arg(args, *SPECIAL_FLAGS):
        args.append("--special")

    # 模型与 mmproj 必须成对来自同一模型；用户通过任一原生选择器（-m/-hf/
    # --model-url，或 -mm/--mmproj 等投影选择器）自定义时不再补默认，
    # 避免拼出「自定义主模型 + 官方 mmproj」的不匹配组合。
    model_path: Path | None = None
    if _has_arg(args, "-m", "--model"):
        value = _flag_value(args, "-m", "--model")
        model_path = Path(value) if value else None
    elif not _has_arg(args, *MODEL_SELECTOR_FLAGS, *PROJECTOR_SELECTOR_FLAGS):
        repo = vlm_model_repo("llama-cpp")
        model_dir = repo.ensure()
        model_path = model_dir / repo.paths["main"]
        args.extend(["-m", str(model_path), "--mmproj", str(model_dir / repo.paths["mmproj"])])

    # ---- 对齐 EngineCore::EngineCore 写死的 common_params（用户显式传入时逐项让位）----
    user_defined_parallel = _has_arg(args, *PARALLEL_FLAGS)
    user_defined_ctx = _has_arg(args, "-c", "--ctx-size")
    if not user_defined_parallel:
        args.extend(["--parallel", str(DEFAULT_N_PARALLEL)])
    if not _has_arg(args, "-ngl", "--gpu-layers", "--n-gpu-layers"):
        args.extend(["--n-gpu-layers", str(DEFAULT_N_GPU_LAYERS)])
    # mmproj_use_gpu = (n_gpu_layers > 0)：纯 CPU（ngl=0）时 mmproj 也不上 GPU；
    # server 的 --mmproj-offload 默认无条件 enabled、不联动 ngl，需要显式对齐。
    if not _has_arg(args, "--mmproj-offload", "--no-mmproj-offload"):
        try:
            mmproj_use_gpu = int(_flag_value(args, "-ngl", "--gpu-layers", "--n-gpu-layers") or DEFAULT_N_GPU_LAYERS) > 0
        except ValueError:
            mmproj_use_gpu = True
        if not mmproj_use_gpu:
            args.append("--no-mmproj-offload")
    # EngineCore 用 kv_unified=false 的硬分区 KV（每 slot 私有 n_ctx_seq，互不侵占，
    # 见 engine_core.cpp 的注释）；llama-server 默认 unified=enabled，需显式关闭，
    # 并同步关闭 idle-slot RAM 缓存（EngineCore 的 cache_idle_slots=false / cache_ram_mib=0）。
    if not _has_arg(args, *KV_UNIFIED_FLAGS):
        args.append("--no-kv-unified")
    if not _has_arg(args, "--cache-ram", "-cram"):
        args.extend(["--cache-ram", "0"])

    # 总上下文 = n_ctx_seq(默认 0 → 训练上下文，读 GGUF) × n_parallel，复刻
    # EngineCore 的 params_.n_ctx = n_ctx_seq * n_parallel；server 的 --ctx-size
    # 是总量且默认 0 只含一份训练上下文，分区后每 slot 会缩水成 1/N。
    # 用户自定义并发或上下文任一项时不再换算，容量交由用户决定。
    if model_path is not None and not user_defined_ctx and not user_defined_parallel:
        n_ctx_train = _read_gguf_context_length(model_path)
        if n_ctx_train > 0:
            args.extend(["--ctx-size", str(n_ctx_train * DEFAULT_N_PARALLEL)])

    binary = llama_server_binary()
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server binary not found: {binary}")

    print(f"start llama.cpp server: {binary} {' '.join(_redact_secrets(args))}")
    os.execv(str(binary), [str(binary), *args])


if __name__ == "__main__":
    main()
