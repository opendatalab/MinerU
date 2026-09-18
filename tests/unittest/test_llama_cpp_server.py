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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], user_args: list[str]
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
    monkeypatch.setattr(
        llama_cpp_server,
        "vlm_model_repo",
        lambda name: SimpleNamespace(ensure=lambda: model_dir, paths={"main": gguf.name, "mmproj": "mmproj.gguf"}),
    )
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
        ["-mm", "p.gguf"],
        ["--mmproj", "p.gguf"],
        ["-mmu", "https://example.com/p.gguf"],
        ["--mmproj-url", "https://example.com/p.gguf"],
    ],
    ids=[
        "hf",
        "hf-repo-short2",
        "hf-repo",
        "model-url-short",
        "model-url",
        "mmproj-short",
        "mmproj",
        "mmproj-url-short",
        "mmproj-url",
    ],
)
def test_main_skips_default_model_pair_when_native_selector_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], model_args: list[str]
) -> None:
    """原生模型/投影选择器视为用户已提供模型，不再注入官方 -m + --mmproj 组合。"""
    argv, _, model_dir = _run_main(monkeypatch, tmp_path, capsys, model_args)

    assert str(model_dir / "main.gguf") not in argv
    assert str(model_dir / "mmproj.gguf") not in argv


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
