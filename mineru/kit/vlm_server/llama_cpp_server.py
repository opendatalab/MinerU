# Copyright (c) Opendatalab. All rights reserved.
"""以薄包装启动 mineru-llama-cpp 分发的 llama-server（OpenAI 兼容）。

llama-server 的全部服务能力由 llama.cpp 提供，本模块只补默认参数并以
进程替换方式启动，llama.cpp 升级不需要改动这里。

默认参数对齐 EngineCore::EngineCore（进程内 llama-cpp-engine）在 C++ 侧
写死的 common_params 值，使 server 模式与解析模式的上下文容量、KV 分区、
并发与日志行为保持一致；用户以 CLI 旗标或 llama-server 环境变量设定时逐项让位。
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
MODEL_SELECTOR_FLAGS = ("-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url", "-dr", "--docker-repo")
# 主模型来源对应的环境变量（llama-server --help 的 env: 名单）：CLI 缺省时
# llama-server 按环境变量选模型，而追加的 -m 会以 CLI 优先级压掉环境配置。
MODEL_SOURCE_ENV = (
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
    "LLAMA_ARG_MODEL_URL",
    "LLAMA_ARG_DOCKER_REPO",
)
# 投影「指定符」：给出投影文件/URL 即视为自带整套模型，不再补默认主模型，
# 避免拼出「官方主模型 + 自定义 mmproj」的不匹配组合（环境变量同理）。
# 投影开关 --mmproj-auto/--no-mmproj/--no-mmproj-auto 与 vocoder 仓库
# -hfv/-hfrv/--hf-repo-v 只作用于投影/vocoder，不构成模型选择、不参与判断。
PROJECTOR_SELECTOR_FLAGS = ("-mm", "--mmproj", "-mmu", "--mmproj-url")
PROJECTOR_SOURCE_ENV = ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
PARALLEL_FLAGS = ("-np", "--parallel")
# router 模式的模型来源（llama-server 未指定模型时按目录/预设自组路由）：
# 视为用户已提供模型，注入默认 -m 会关掉 router 模式并强制下载官方模型。
ROUTER_SOURCE_FLAGS = ("--models-dir", "--models-preset")
ROUTER_SOURCE_ENV = ("LLAMA_ARG_MODELS_DIR", "LLAMA_ARG_MODELS_PRESET")
KV_UNIFIED_FLAGS = ("-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified")
SPECIAL_FLAGS = ("-sp", "--special")
# 值为凭据的旗标，启动横幅打印前脱敏；--api-key-file 等是路径不是凭据，不脱敏。
SECRET_VALUE_FLAGS = ("--api-key", "-hft", "--hf-token")
# 打印信息即退出的选项（llama-server --help）：无需模型与默认参数，原样转发；
# 先解析默认模型会让 --version 这类命令在全新/离线环境先触发下载或失败。
INFO_EXIT_FLAGS = ("-h", "--help", "--usage", "--version", "-cl", "--cache-list", "--completion-bash", "--list-devices")


def llama_server_binary() -> Path:
    """定位 mineru-llama-cpp 包内分发的 llama-server 可执行文件。"""
    import mineru_llama_cpp

    # Windows wheel 分发的是 llama-server.exe，其余平台无后缀。
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return Path(mineru_llama_cpp.__file__).resolve().parent / "bin" / binary_name


def _has_arg(args: list[str], *flags: str) -> bool:
    return any(arg in flags or arg.startswith(f"{flag}=") for arg in args for flag in flags)


def _has_env(*names: str) -> bool:
    # 只认非空值：部署清单里 optional 的空环境变量（LLAMA_ARG_MODEL=""）
    # 视为未设置，否则默认模型对被跳过、server 落到零模型 router 模式。
    return any(os.environ.get(name) for name in names)


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


def _exec_server(args: list[str]) -> typing.NoReturn:
    """定位二进制并以进程替换方式转发参数。"""
    binary = llama_server_binary()
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server binary not found: {binary}")

    print(f"start llama.cpp server: {binary} {' '.join(_redact_secrets(args))}")
    os.execv(str(binary), [str(binary), *args])


def main() -> None:
    args = sys.argv[1:]

    # --version 等信息命令直接转发，不解析模型也不补默认参数。
    if _has_arg(args, *INFO_EXIT_FLAGS):
        _exec_server(args)
        return

    # 注入的每个默认值都让位两项：用户 CLI 旗标与对应环境变量（--help 的
    # env: 名单）；llama.cpp 里 CLI 优先级高于环境变量，漏认 env 会让追加的
    # 默认值静默压掉操作员的环境配置。
    if not _has_arg(args, "--port") and not _has_env("LLAMA_ARG_PORT"):
        args.extend(["--port", DEFAULT_PORT])
    if not _has_arg(args, *GRAMMAR_SELECTOR_FLAGS):
        args.extend(["--grammar", VALID_UNICODE_GRAMMAR])
    # params_.special=true：MinerU 输出里的 <|box_start|> 等格式标记注册为
    # special tokens，server 默认（false）会在输出 detokenize 时吞掉它们，
    # 解析格式依赖这些标记，必须显式开启。
    if not _has_arg(args, *SPECIAL_FLAGS):
        args.append("--special")

    # 模型与 mmproj 必须成对来自同一模型；用户通过任一原生选择器（-m/-hf/
    # --model-url）、router 来源（--models-dir/--models-preset）或对应环境变量
    # 指定模型时不再补默认：追加的 -m 会以 CLI 优先级压掉环境配置、关掉
    # router 模式，或拼出「官方主模型 + 自定义 mmproj」的不匹配组合。
    # 投影开关（--mmproj-auto/--no-mmproj 等）与 vocoder 开关不影响主模型，
    # 官方 -m 照常补齐；--no-mmproj 显式关投影时才不追加官方 mmproj。
    user_model_specified = _has_arg(args, *MODEL_SELECTOR_FLAGS, *ROUTER_SOURCE_FLAGS) or _has_env(
        *MODEL_SOURCE_ENV, *ROUTER_SOURCE_ENV
    )
    user_projector_specified = _has_arg(args, *PROJECTOR_SELECTOR_FLAGS) or _has_env(*PROJECTOR_SOURCE_ENV)
    model_path: Path | None = None
    if _has_arg(args, "-m", "--model"):
        value = _flag_value(args, "-m", "--model")
        model_path = Path(value) if value else None
    elif not user_model_specified and not user_projector_specified:
        repo = vlm_model_repo("llama-cpp")
        model_dir = repo.ensure()
        model_path = model_dir / repo.paths["main"]
        args.extend(["-m", str(model_path)])
        if not _has_arg(args, "--no-mmproj"):
            args.extend(["--mmproj", str(model_dir / repo.paths["mmproj"])])
        # /v1/models 与请求体里的 model 字段用 registry 的模型名，而不是 GGUF
        # 文件路径；用户自定义模型时不代设。
        if not _has_arg(args, "-a", "--alias") and not _has_env("LLAMA_ARG_ALIAS"):
            args.extend(["--alias", repo.name])

    # ---- 对齐 EngineCore::EngineCore 写死的 common_params（用户 CLI 或环境变量设定时逐项让位）----
    user_defined_parallel = _has_arg(args, *PARALLEL_FLAGS) or _has_env("LLAMA_ARG_N_PARALLEL")
    user_defined_ctx = _has_arg(args, "-c", "--ctx-size") or _has_env("LLAMA_ARG_CTX_SIZE")
    if not user_defined_parallel:
        args.extend(["--parallel", str(DEFAULT_N_PARALLEL)])
    if not _has_arg(args, "-ngl", "--gpu-layers", "--n-gpu-layers") and not _has_env("LLAMA_ARG_N_GPU_LAYERS"):
        args.extend(["--n-gpu-layers", str(DEFAULT_N_GPU_LAYERS)])
    # mmproj_use_gpu = (n_gpu_layers > 0)：纯 CPU（ngl=0）时 mmproj 也不上 GPU；
    # server 的 --mmproj-offload 默认无条件 enabled、不联动 ngl，需要显式对齐。
    if not _has_arg(args, "--mmproj-offload", "--no-mmproj-offload") and not _has_env("LLAMA_ARG_MMPROJ_OFFLOAD"):
        try:
            ngl_value = _flag_value(args, "-ngl", "--gpu-layers", "--n-gpu-layers") or os.environ.get("LLAMA_ARG_N_GPU_LAYERS")
            mmproj_use_gpu = int(ngl_value or DEFAULT_N_GPU_LAYERS) > 0
        except ValueError:
            mmproj_use_gpu = True
        if not mmproj_use_gpu:
            args.append("--no-mmproj-offload")
    # EngineCore 用 kv_unified=false 的硬分区 KV（每 slot 私有 n_ctx_seq，互不侵占，
    # 见 engine_core.cpp 的注释）；llama-server 默认 unified=enabled，需显式关闭，
    # 并同步关闭 idle-slot RAM 缓存（EngineCore 的 cache_idle_slots=false / cache_ram_mib=0）。
    if not _has_arg(args, *KV_UNIFIED_FLAGS) and not _has_env("LLAMA_ARG_KV_UNIFIED"):
        args.append("--no-kv-unified")
    if not _has_arg(args, "--cache-ram", "-cram") and not _has_env("LLAMA_ARG_CACHE_RAM"):
        args.extend(["--cache-ram", "0"])

    # 总上下文 = n_ctx_seq(默认 0 → 训练上下文，读 GGUF) × n_parallel，复刻
    # EngineCore 的 params_.n_ctx = n_ctx_seq * n_parallel；server 的 --ctx-size
    # 是总量且默认 0 只含一份训练上下文，分区后每 slot 会缩水成 1/N。
    # 用户自定义并发或上下文任一项时不再换算，容量交由用户决定。
    if model_path is not None and not user_defined_ctx and not user_defined_parallel:
        n_ctx_train = _read_gguf_context_length(model_path)
        if n_ctx_train > 0:
            args.extend(["--ctx-size", str(n_ctx_train * DEFAULT_N_PARALLEL)])

    _exec_server(args)


if __name__ == "__main__":
    main()
