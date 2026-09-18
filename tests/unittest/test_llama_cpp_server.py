from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mineru.kit.vlm_server import llama_cpp_server


@pytest.mark.parametrize(
    ("platform", "expected_name"),
    [("win32", "llama-server.exe"), ("darwin", "llama-server"), ("linux", "llama-server")],
)
def test_llama_server_binary_matches_platform_executable_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, expected_name: str
) -> None:
    """Windows wheel 分发 llama-server.exe，其余平台无后缀。"""
    fake_pkg = ModuleType("mineru_llama_cpp")
    fake_pkg.__file__ = str(tmp_path / "mineru_llama_cpp" / "__init__.py")
    monkeypatch.setitem(sys.modules, "mineru_llama_cpp", fake_pkg)
    monkeypatch.setattr(sys, "platform", platform)

    binary = llama_cpp_server.llama_server_binary()

    assert binary == tmp_path / "mineru_llama_cpp" / "bin" / expected_name


def _gguf_string_kv(key: str, value: str) -> bytes:
    key_bytes, value_bytes = key.encode(), value.encode()
    return (
        struct.pack("<Q", len(key_bytes))
        + key_bytes
        + struct.pack("<I", llama_cpp_server._GGUF_STRING)
        + struct.pack("<Q", len(value_bytes))
        + value_bytes
    )


def _gguf_uint32_kv(key: str, value: int) -> bytes:
    key_bytes = key.encode()
    return (
        struct.pack("<Q", len(key_bytes))
        + key_bytes
        + struct.pack("<I", llama_cpp_server._GGUF_UINT32)
        + struct.pack("<I", value)
    )


def _write_gguf(path: Path, kv_entries: list[bytes]) -> Path:
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv_entries)) + b"".join(kv_entries)
    )
    return path


def test_gguf_context_length_is_key_order_independent(tmp_path: Path) -> None:
    """GGUF 不保证 metadata key 顺序，context_length 在 architecture 前也必须读出。"""
    architecture = _gguf_string_kv("general.architecture", "qwen2vl")
    context = _gguf_uint32_kv("qwen2vl.context_length", 40960)

    assert (
        llama_cpp_server._read_gguf_context_length(_write_gguf(tmp_path / "arch_first.gguf", [architecture, context])) == 40960
    )
    assert (
        llama_cpp_server._read_gguf_context_length(_write_gguf(tmp_path / "ctx_first.gguf", [context, architecture])) == 40960
    )


def test_gguf_context_length_returns_zero_without_matching_architecture(tmp_path: Path) -> None:
    """没有与 architecture 匹配的 context_length 时返回 0，交给 llama-server 默认。"""
    mismatch = [
        _gguf_string_kv("general.architecture", "llama"),
        _gguf_uint32_kv("qwen2vl.context_length", 40960),
    ]

    assert llama_cpp_server._read_gguf_context_length(_write_gguf(tmp_path / "mismatch.gguf", mismatch)) == 0


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    user_args: list[str],
    *,
    repo: SimpleNamespace | None = None,
) -> tuple[list[str], str, Path]:
    """执行 main() 并捕获转交 llama-server 的 argv 与横幅输出，不真正启动进程。"""
    model_dir = tmp_path / "repo"
    model_dir.mkdir()
    # 故意用 context_length 在前的 GGUF，端到端覆盖 key 顺序无关的读取路径。
    gguf = _write_gguf(
        model_dir / "main.gguf",
        [_gguf_uint32_kv("qwen2vl.context_length", 40960), _gguf_string_kv("general.architecture", "qwen2vl")],
    )
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"")
    executed: list[list[str]] = []
    monkeypatch.setattr(llama_cpp_server, "llama_server_binary", lambda: binary)
    fake_repo = (
        repo
        if repo is not None
        else SimpleNamespace(
            name="mineru-test-model",
            ensure=lambda source=None: model_dir,
            paths={"main": gguf.name, "mmproj": "mmproj.gguf"},
        )
    )
    monkeypatch.setattr(llama_cpp_server, "vlm_model_repo", lambda name: fake_repo)
    monkeypatch.setattr("os.execv", lambda path, argv: executed.append(argv))
    monkeypatch.setattr(sys, "argv", ["llama_cpp_server", *user_args])

    llama_cpp_server.main()

    captured = capsys.readouterr()
    return executed[0], captured.out, model_dir


def test_main_appends_engine_aligned_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, out, model_dir = _run_main(monkeypatch, tmp_path, capsys, [])

    assert argv[argv.index("--port") + 1] == llama_cpp_server.DEFAULT_PORT
    assert argv[argv.index("--grammar") + 1] == llama_cpp_server.VALID_UNICODE_GRAMMAR
    assert "--special" in argv
    assert argv[argv.index("-m") + 1] == str(model_dir / "main.gguf")
    assert argv[argv.index("--mmproj") + 1] == str(model_dir / "mmproj.gguf")
    assert argv[argv.index("--alias") + 1] == "mineru-test-model"
    assert argv[argv.index("--parallel") + 1] == str(llama_cpp_server.DEFAULT_N_PARALLEL)
    assert argv[argv.index("--n-gpu-layers") + 1] == str(llama_cpp_server.DEFAULT_N_GPU_LAYERS)
    assert "--no-kv-unified" in argv
    assert argv[argv.index("--cache-ram") + 1] == "0"
    # 训练上下文 40960 × 默认并发 4，且来自 key 顺序颠倒的 GGUF。
    assert argv[argv.index("--ctx-size") + 1] == "163840"
    assert "start llama.cpp server:" in out


@pytest.mark.parametrize(
    "parallel_args",
    [["-np", "8"], ["--parallel", "6"], ["--parallel=6"]],
    ids=["short-alias", "long-separated", "long-inline"],
)
def test_main_respects_user_parallel_and_skips_ctx_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], parallel_args: list[str]
) -> None:
    """-np 是 --parallel 的别名：用户给过后不再追加默认值，也不做 ×4 换算。"""
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, parallel_args)

    joined = " ".join(argv)
    assert f"--parallel {llama_cpp_server.DEFAULT_N_PARALLEL}" not in joined
    assert "--ctx-size" not in argv


@pytest.mark.parametrize(
    "grammar_args",
    [
        ["--grammar-file", "g.gbnf"],
        ["-j", '{"type": "object"}'],
        ["--json-schema", '{"type": "object"}'],
        ["--json-schema={}"],
        ["-jf", "schema.json"],
        ["--json-schema-file", "schema.json"],
        ["--grammar", "root ::= A"],
    ],
    ids=[
        "grammar-file",
        "json-schema-short",
        "json-schema",
        "json-schema-inline",
        "json-schema-file-short",
        "json-schema-file",
        "grammar",
    ],
)
def test_main_skips_default_grammar_when_alternate_selector_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], grammar_args: list[str]
) -> None:
    """任一 grammar 等价选择器都视为用户已约束生成，不再注入兜底 grammar。"""
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, grammar_args)

    assert llama_cpp_server.VALID_UNICODE_GRAMMAR not in argv
    if grammar_args[0] == "--grammar":
        assert argv.count("--grammar") == 1
    else:
        assert "--grammar" not in argv


@pytest.mark.parametrize(
    "model_args",
    [
        ["-hf", "org/model:Q8_0"],
        ["-hfr", "org/model"],
        ["--hf-repo", "org/model"],
        ["-mu", "https://example.com/m.gguf"],
        ["--model-url", "https://example.com/m.gguf"],
        ["-dr", "ai/model:Q8_0"],
        ["--docker-repo", "ai/model"],
        ["-hf", "org/model:Q8_0", "-mm", "p.gguf"],
        ["-hf", "org/model:Q8_0", "--mmproj", "p.gguf"],
        ["-hf", "org/model:Q8_0", "-mmu", "https://example.com/p.gguf"],
        ["-hf", "org/model:Q8_0", "--mmproj-url", "https://example.com/p.gguf"],
        ["--models-dir", "models"],
        ["--models-preset", "presets.ini"],
    ],
    ids=[
        "hf",
        "hf-repo-short2",
        "hf-repo",
        "model-url-short",
        "model-url",
        "docker-repo-short",
        "docker-repo",
        "mmproj-with-hf",
        "mmproj-long-with-hf",
        "mmproj-url-short-with-hf",
        "mmproj-url-with-hf",
        "router-models-dir",
        "router-models-preset",
    ],
)
def test_main_skips_default_model_pair_when_native_selector_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], model_args: list[str]
) -> None:
    """主模型选择器（含搭配投影指定符）与 router 来源视为用户已提供模型，不再注入官方 -m + --mmproj 组合。"""
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, model_args)

    assert str(model_dir / "main.gguf") not in argv
    assert str(model_dir / "mmproj.gguf") not in argv


@pytest.mark.parametrize(
    ("modifier_args", "expect_mmproj"),
    [(["--no-mmproj"], False), (["--mmproj-auto"], True), (["--no-mmproj-auto"], True)],
    ids=["no-mmproj", "mmproj-auto", "no-mmproj-auto"],
)
def test_main_projector_modifier_keeps_default_main_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    modifier_args: list[str],
    expect_mmproj: bool,
) -> None:
    """投影开关只作用于投影：官方主模型仍补齐，--no-mmproj 时不追加官方 mmproj。"""
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, modifier_args)

    assert str(model_dir / "main.gguf") in argv
    assert (str(model_dir / "mmproj.gguf") in argv) is expect_mmproj


@pytest.mark.parametrize(
    "vocoder_args",
    [["-hfv", "org/vocoder"], ["-hfrv", "org/vocoder"], ["--hf-repo-v", "org/vocoder"]],
    ids=["short", "short2", "long"],
)
def test_main_vocoder_selector_keeps_default_model_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], vocoder_args: list[str]
) -> None:
    """vocoder 仓库独立于主模型：官方 -m + --mmproj 组合照常补齐。"""
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, vocoder_args)

    assert str(model_dir / "main.gguf") in argv
    assert str(model_dir / "mmproj.gguf") in argv


@pytest.mark.parametrize(
    "env_name",
    [
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_DOCKER_REPO",
        "LLAMA_ARG_MODELS_DIR",
        "LLAMA_ARG_MODELS_PRESET",
    ],
)
def test_main_env_model_source_suppresses_default_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], env_name: str
) -> None:
    """llama-server 按环境变量选模型时，追加的 -m 会以 CLI 优先级压掉配置。"""
    monkeypatch.setenv(env_name, "/env/model.gguf")
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, [])

    assert str(model_dir / "main.gguf") not in argv
    assert str(model_dir / "mmproj.gguf") not in argv


def test_main_env_hf_file_alone_keeps_default_model_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--hf-file 只覆盖 --hf-repo 内的量化文件：单独出现不构成主模型来源，官方模型对照常注入。"""
    monkeypatch.setenv("LLAMA_ARG_HF_FILE", "model-Q8_0.gguf")
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, [])

    assert str(model_dir / "main.gguf") in argv
    assert str(model_dir / "mmproj.gguf") in argv


@pytest.mark.parametrize(
    "projector_args",
    [
        ["-mm", "p.gguf"],
        ["--mmproj", "p.gguf"],
        ["-mmu", "https://example.com/p.gguf"],
        ["--mmproj-url", "https://example.com/p.gguf"],
    ],
    ids=["short", "long", "url-short", "url"],
)
def test_main_projector_without_main_model_source_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], projector_args: list[str]
) -> None:
    """只给投影而无主模型来源直接报错：静默跳过默认会让 server 进零模型 router 模式。"""
    with pytest.raises(SystemExit, match="主模型"):
        _run_main(monkeypatch, tmp_path, capsys, projector_args)


@pytest.mark.parametrize("env_name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
def test_main_env_projector_without_main_model_source_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], env_name: str
) -> None:
    """环境变量只指定投影同样直接报错，不进零模型 router 模式。"""
    monkeypatch.setenv(env_name, "/env/p.gguf")

    with pytest.raises(SystemExit, match="主模型"):
        _run_main(monkeypatch, tmp_path, capsys, [])


@pytest.mark.parametrize("env_name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
def test_main_env_projector_source_suppresses_default_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], env_name: str
) -> None:
    """环境变量指定投影与 -mm/--mmproj 同义：搭配主模型来源时视为自带整套模型，不再补默认对。"""
    monkeypatch.setenv(env_name, "/env/p.gguf")
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, ["-hf", "org/model:Q8_0"])

    assert str(model_dir / "main.gguf") not in argv
    assert str(model_dir / "mmproj.gguf") not in argv


@pytest.mark.parametrize(
    "model_args",
    [
        ["-hf", "org/model:Q8_0"],
        ["--model-url", "https://example.com/m.gguf"],
        ["-dr", "ai/model"],
        ["--models-dir", "models"],
    ],
    ids=["hf", "model-url", "docker-repo", "router"],
)
def test_main_non_file_model_source_skips_partitioned_parallel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], model_args: list[str]
) -> None:
    """读不到 GGUF 的模型来源不注入硬分区并发：--ctx-size 0 只装一份训练上下文，×4 分区每 slot 只剩 1/4。"""
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, model_args)

    assert "--parallel" not in argv
    assert "--ctx-size" not in argv


def test_main_env_model_source_skips_partitioned_parallel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """环境变量选模型同样读不到 GGUF：不注入默认并发，保持单 slot 全量上下文。"""
    monkeypatch.setenv("LLAMA_ARG_MODEL", "/env/m.gguf")
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, [])

    assert "--parallel" not in argv
    assert "--ctx-size" not in argv


@pytest.mark.parametrize(
    ("offline_args", "offline_env"),
    [(["--offline"], ""), ([], "1")],
    ids=["flag", "env"],
)
def test_main_offline_uses_local_verification_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    offline_args: list[str],
    offline_env: str,
) -> None:
    """--offline 禁止网络访问：默认模型只做本地校验（ensure(source="local")），不走远端下载。"""
    ensure_calls: list[dict[str, object]] = []

    def _ensure(**kwargs: object) -> Path:
        ensure_calls.append(kwargs)
        return tmp_path / "repo"

    repo = SimpleNamespace(name="mineru-test-model", ensure=_ensure, paths={"main": "main.gguf", "mmproj": "mmproj.gguf"})
    if offline_env:
        monkeypatch.setenv("LLAMA_ARG_OFFLINE", offline_env)

    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, offline_args, repo=repo)

    assert ensure_calls == [{"source": "local"}]
    assert argv[argv.index("-m") + 1] == str(tmp_path / "repo" / "main.gguf")


def test_main_offline_without_cached_model_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--offline 且默认模型未缓存：直接报错并给出修法，不碰网络。"""

    def _ensure(**kwargs: object) -> Path:
        if kwargs.get("source") == "local":
            raise FileNotFoundError("Model repo mineru-test-model is not ready; missing: main.gguf")
        return tmp_path / "repo"

    repo = SimpleNamespace(name="mineru-test-model", ensure=_ensure, paths={"main": "main.gguf", "mmproj": "mmproj.gguf"})

    with pytest.raises(SystemExit, match="--offline"):
        _run_main(monkeypatch, tmp_path, capsys, ["--offline"], repo=repo)


def test_main_env_parallel_suppresses_default_and_ctx_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """env 设定并发视为用户已定义：不追加默认并发，也不做训练上下文 ×N 换算。"""
    monkeypatch.setenv("LLAMA_ARG_N_PARALLEL", "8")
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, [])

    assert "--parallel" not in argv
    assert "--ctx-size" not in argv


def test_main_env_ctx_suppresses_ctx_conversion_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """env 只设定上下文：换算跳过，默认并发照常补齐。"""
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "32768")
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, [])

    assert "--ctx-size" not in argv
    assert "--parallel" in argv


@pytest.mark.parametrize(
    ("ngl_env", "expect_no_mmproj_offload"),
    [("0", True), ("2", False)],
    ids=["cpu-only", "gpu"],
)
def test_main_env_ngl_suppresses_default_and_drives_mmproj_offload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    ngl_env: str,
    expect_no_mmproj_offload: bool,
) -> None:
    """env 设定 GPU 层数时不追加默认 99；ngl=0 的纯 CPU 场景 mmproj 同步离卡。"""
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", ngl_env)
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, [])

    assert "--n-gpu-layers" not in argv
    assert ("--no-mmproj-offload" in argv) is expect_no_mmproj_offload


@pytest.mark.parametrize(
    ("env_name", "suppressed_flag"),
    [
        ("LLAMA_ARG_PORT", "--port"),
        ("LLAMA_ARG_KV_UNIFIED", "--no-kv-unified"),
        ("LLAMA_ARG_CACHE_RAM", "--cache-ram"),
    ],
    ids=["port", "kv-unified", "cache-ram"],
)
def test_main_env_setting_suppresses_matching_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], env_name: str, suppressed_flag: str
) -> None:
    """env 已配置的项不再追加默认值（CLI 追加会以更高优先级压掉环境配置）。"""
    monkeypatch.setenv(env_name, "1")
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, [])

    assert suppressed_flag not in argv


def test_main_env_alias_suppresses_registry_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """env 设定 alias 时不代设 registry 模型名，默认模型对仍正常补齐。"""
    monkeypatch.setenv("LLAMA_ARG_ALIAS", "env-alias")
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, [])

    assert "mineru-test-model" not in argv
    assert str(model_dir / "main.gguf") in argv


def test_main_empty_env_values_do_not_suppress_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """部署清单里 optional 的空环境变量视为未设置：默认模型对与默认参数照常注入。"""
    for name in ("LLAMA_ARG_MODEL", "LLAMA_ARG_N_PARALLEL"):
        monkeypatch.setenv(name, "")
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, [])

    assert str(model_dir / "main.gguf") in argv
    assert str(model_dir / "mmproj.gguf") in argv
    assert "--parallel" in argv


@pytest.mark.parametrize(
    "info_args",
    [
        ["--version"],
        ["-h"],
        ["--help"],
        ["--usage"],
        ["--cache-list"],
        ["-cl"],
        ["--completion-bash"],
        ["--list-devices"],
    ],
    ids=["version", "help-short", "help", "usage", "cache-list", "cache-list-short", "completion", "list-devices"],
)
def test_main_info_options_exec_without_model_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], info_args: list[str]
) -> None:
    """--version 等信息命令原样转发：不解析默认模型（全新/离线环境不下模型），不追加任何默认参数。"""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"")
    executed: list[list[str]] = []
    monkeypatch.setattr(llama_cpp_server, "llama_server_binary", lambda: binary)

    def _no_repo(name: str) -> None:
        raise AssertionError("info command must not resolve the model repository")

    monkeypatch.setattr(llama_cpp_server, "vlm_model_repo", _no_repo)
    monkeypatch.setattr("os.execv", lambda path, argv: executed.append(argv))
    monkeypatch.setattr(sys, "argv", ["llama_cpp_server", *info_args])

    llama_cpp_server.main()

    assert executed[0] == [str(binary), *info_args]


@pytest.mark.parametrize(
    "alias_args",
    [["-a", "my-model"], ["--alias", "my-model"], ["--alias=my-model"]],
    ids=["short", "long", "inline"],
)
def test_main_respects_user_alias_and_keeps_default_model_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], alias_args: list[str]
) -> None:
    """用户显式给 -a/--alias 时不注入 registry 模型名，默认模型对仍正常补齐。"""
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, alias_args)

    assert "mineru-test-model" not in argv
    assert argv.count("--alias") == (1 if alias_args[0] == "--alias" else 0)
    assert str(model_dir / "main.gguf") in argv


@pytest.mark.parametrize(
    "kv_args",
    [["-kvu"], ["--kv-unified"], ["-no-kvu"], ["--no-kv-unified"]],
    ids=["short", "long", "no-short", "no-long"],
)
def test_main_respects_user_kv_unified_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], kv_args: list[str]
) -> None:
    argv, _, _ = _run_main(monkeypatch, tmp_path, capsys, kv_args)

    assert argv.count("--no-kv-unified") == (1 if "--no-kv-unified" in kv_args else 0)


def test_main_redacts_secrets_in_startup_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """横幅对 --api-key/-hft/--hf-token 的值脱敏，转交进程的参数保持原值。"""
    argv, out, _ = _run_main(
        monkeypatch, tmp_path, capsys, ["--api-key", "sk-plain-secret", "-hft=tok-inline", "--hf-token", "tok-separated"]
    )

    assert "sk-plain-secret" not in out
    assert "tok-inline" not in out
    assert "tok-separated" not in out
    assert out.count("***") == 3
    assert "sk-plain-secret" in argv
    assert "-hft=tok-inline" in argv
    assert "tok-separated" in argv
