import json
import subprocess
from argparse import Namespace
from pathlib import Path

from benchmarks.mineru_diffusion.harness import (
    BenchmarkCase,
    BenchmarkResult,
    TASK_PROMPTS,
    extract_openai_image_and_prompt,
    summarize_results,
    write_default_cases,
)
from benchmarks.mineru_diffusion.bench_hf_server import create_session
from benchmarks.mineru_diffusion.bench_hf_server import build_request_payload
from benchmarks.mineru_diffusion.bench_hf_server import parse_task_max_tokens
from benchmarks.mineru_diffusion.compare_results import (
    QualityGateError,
    QualityThresholds,
    assert_quality_thresholds,
    compare_case,
    compare_results,
    control_token_ratio,
    max_consecutive_control_repeat,
    summarize_comparisons,
)
from benchmarks.mineru_diffusion.run_native_benchmark import run_benchmark


def test_extract_openai_image_and_prompt_from_multimodal_message():
    payload = {
        "messages": [
            {"role": "system", "content": "ignored"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "/tmp/page.png"}},
                    {"type": "text", "text": "\nText Recognition:"},
                ],
            },
        ]
    }

    image, prompt = extract_openai_image_and_prompt(payload)

    assert image == "/tmp/page.png"
    assert prompt == "\nText Recognition:"


def test_extract_openai_image_and_prompt_accepts_plain_string_prompt():
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    image, prompt = extract_openai_image_and_prompt(payload)

    assert image is None
    assert prompt == "hello"


def test_summarize_results_reports_latency_throughput_and_failures():
    results = [
        BenchmarkResult(
            case_id="a",
            ok=True,
            latency_s=2.0,
            output_text="abcd",
            error=None,
        ),
        BenchmarkResult(
            case_id="b",
            ok=True,
            latency_s=1.0,
            output_text="abcdef",
            error=None,
        ),
        BenchmarkResult(
            case_id="c",
            ok=False,
            latency_s=0.5,
            output_text="",
            error="boom",
        ),
    ]

    summary = summarize_results(results)

    assert summary["num_requests"] == 3
    assert summary["num_ok"] == 2
    assert summary["num_failed"] == 1
    assert summary["mean_latency_s"] == 1.5
    assert summary["output_chars_per_s"] == 10 / 3.0


def test_benchmark_case_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    case = BenchmarkCase(
        case_id="sample",
        image="/data/page.png",
        prompt="\nLayout Detection:",
    )
    path.write_text(json.dumps(case.to_payload()) + "\n", encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["case_id"] == "sample"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["type"] == "image_url"


def test_default_layout_prompt_matches_openai_layout_baseline():
    assert TASK_PROMPTS["layout"] == "\nLayout Analysis:"


def test_write_default_cases_accepts_task_max_tokens(tmp_path: Path):
    path = tmp_path / "cases.jsonl"

    write_default_cases(
        path,
        "/data/page.png",
        task_max_tokens={"text": 128, "layout": 64},
    )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["case_id"]: row for row in rows}

    assert by_id["text"]["max_tokens"] == 128
    assert by_id["layout"]["max_tokens"] == 64
    assert "max_tokens" not in by_id["table"]


def test_parse_task_max_tokens_accepts_comma_separated_pairs():
    assert parse_task_max_tokens("text=512,layout=128") == {
        "text": 512,
        "layout": 128,
    }


def test_benchmark_client_ignores_environment_proxies():
    assert create_session().trust_env is False


def test_benchmark_payload_passes_mineru_args_through_vllm_xargs():
    payload = build_request_payload(
        {
            "case_id": "text",
            "model": "mineru-diffusion",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "temperature": 0.0,
            "block_size": 16,
            "dynamic_threshold": 0.73,
            "max_denoising_steps": 8,
        }
    )

    assert payload["block_size"] == 16
    assert payload["dynamic_threshold"] == 0.73
    assert payload["max_denoising_steps"] == 8
    assert payload["vllm_xargs"] == {
        "block_size": 16,
        "dynamic_threshold": 0.73,
        "max_denoising_steps": 8,
    }


def test_benchmark_payload_sets_mineru_stop_strings_by_default():
    payload = build_request_payload(
        {
            "case_id": "text",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert payload["stop"] == ["<|endoftext|>", "<|im_end|>"]


def test_run_native_benchmark_starts_client_and_stops_server(
    monkeypatch, tmp_path: Path
):
    events = []

    class FakeServer:
        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append(("wait", timeout))

    def fake_popen(cmd, stdout, stderr, env):
        events.append(("popen", cmd, env))
        assert stdout.writable()
        assert stderr == subprocess.STDOUT
        return FakeServer()

    def fake_wait_for_health(url, timeout_s):
        events.append(("health", url, timeout_s))

    def fake_run(cmd, check, env):
        events.append(("run", cmd, check, env))
        if "benchmarks.mineru_diffusion.bench_hf_server" in cmd:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "results_2.jsonl").write_text("", encoding="utf-8")
            (args.output_dir / "results_1.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "benchmarks.mineru_diffusion.run_native_benchmark.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "benchmarks.mineru_diffusion.run_native_benchmark.wait_for_health",
        fake_wait_for_health,
    )
    monkeypatch.setattr(
        "benchmarks.mineru_diffusion.run_native_benchmark.subprocess.run",
        fake_run,
    )

    args = Namespace(
        model_path="/model",
        image="file:///page.png",
        host="127.0.0.1",
        port=18083,
        device="cuda:0",
        dtype="bfloat16",
        cuda_visible_devices="1",
        output_dir=tmp_path / "out",
        timeout=123.0,
        startup_timeout=45.0,
        server_log=None,
        task_max_tokens="text=128,layout=64",
        baseline_results=tmp_path / "baseline.jsonl",
        min_similarity=0.95,
        similarity_cases="text,table",
        max_control_repeat=8,
        max_control_token_ratio=None,
        compare_output=None,
    )

    run_benchmark(args)

    assert events[0][0] == "popen"
    assert events[0][2]["CUDA_VISIBLE_DEVICES"] == "1"
    assert events[1] == ("health", "http://127.0.0.1:18083/health", 45.0)
    assert events[2][0] == "run"
    assert events[2][1][1:] == [
        "-m",
        "benchmarks.mineru_diffusion.bench_hf_server",
        "--endpoint",
        "http://127.0.0.1:18083/v1/chat/completions",
        "--image",
        "file:///page.png",
        "--output-dir",
        str(tmp_path / "out"),
        "--timeout",
        "123.0",
        "--task-max-tokens",
        "text=128,layout=64",
    ]
    assert events[3][0] == "run"
    assert events[3][1][1:] == [
        "-m",
        "benchmarks.mineru_diffusion.compare_results",
        "--baseline",
        str(tmp_path / "baseline.jsonl"),
        "--candidate",
        str(tmp_path / "out" / "results_2.jsonl"),
        "--similarity-cases",
        "text,table",
        "--min-similarity",
        "0.95",
        "--max-control-repeat",
        "8",
    ]
    assert events[-2:] == ["terminate", ("wait", 30)]


def test_compare_results_reports_similarity_and_control_repeats():
    baseline = {
        "case_id": "table",
        "ok": True,
        "output_text": "<fcel>Abstract<nl>\n<fcel>Optimizing complex systems",
    }
    candidate = {
        "case_id": "table",
        "ok": True,
        "output_text": "<fcel>Abstract<nl>\n<fcel>Optimizing complex systems",
    }

    comparison = compare_case("table", baseline, candidate)

    assert comparison.baseline_ok is True
    assert comparison.candidate_ok is True
    assert comparison.similarity == 1.0
    assert comparison.char_ratio == 1.0
    assert comparison.candidate_max_control_repeat == 1


def test_compare_results_reports_layout_box_f1_for_layout_cases():
    baseline = {
        "case_id": "layout",
        "ok": True,
        "output_text": (
            "<|box_start|>000 000 028 031<|box_end|>"
            "<|ref_start|>title<|ref_end|><|rotate_up|>\n"
            "<|box_start|>000 086 999 999<|box_end|>"
            "<|ref_start|>table<|ref_end|><|rotate_up|>"
        ),
    }
    candidate = {
        "case_id": "layout",
        "ok": True,
        "output_text": (
            "<|box_start|>000 000 026 030<|box_end|>"
            "<|ref_start|>title<|ref_end|><|rotate_up|>\n"
            "<|box_start|>004 090 995 998<|box_end|>"
            "<|ref_start|>table<|ref_end|><|rotate_up|>\n"
            "<|box_start|>100 100 120 120<|box_end|>"
            "<|ref_start|>text<|ref_end|><|rotate_up|>"
        ),
    }

    comparison = compare_case("layout", baseline, candidate)

    assert comparison.layout_baseline_boxes == 2
    assert comparison.layout_candidate_boxes == 3
    assert comparison.layout_matched_boxes == 2
    assert comparison.layout_precision == 2 / 3
    assert comparison.layout_recall == 1.0
    assert comparison.layout_f1 == 0.8


def test_control_token_repeat_detects_broken_vllm_output():
    text = "<fcel><lcel><lcel><lcel><lcel>payload"

    assert control_token_ratio(text) > 0.5
    assert max_consecutive_control_repeat(text) == 4


def test_summarize_comparisons_aggregates_case_metrics():
    comparisons = compare_results(
        {
            "text": {
                "case_id": "text",
                "ok": True,
                "output_text": "Abstract optimization",
            },
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel>Abstract",
            },
        },
        {
            "text": {
                "case_id": "text",
                "ok": True,
                "output_text": "Abstract optimization",
            },
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel><lcel><lcel>",
            },
        },
    )

    summary = summarize_comparisons(comparisons)

    assert summary["num_cases"] == 2
    assert summary["num_candidate_ok"] == 2
    assert summary["mean_similarity"] < 1.0
    assert summary["max_control_repeat"] == 2


def test_quality_gate_passes_when_required_cases_meet_thresholds():
    comparisons = compare_results(
        {
            "text": {
                "case_id": "text",
                "ok": True,
                "output_text": "Abstract optimization",
            },
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel>Abstract",
            },
            "formula": {
                "case_id": "formula",
                "ok": True,
                "output_text": "noisy formula reference",
            },
        },
        {
            "text": {
                "case_id": "text",
                "ok": True,
                "output_text": "Abstract optimization",
            },
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel>Abstract",
            },
            "formula": {
                "case_id": "formula",
                "ok": True,
                "output_text": "different noisy formula",
            },
        },
    )

    assert_quality_thresholds(
        comparisons,
        QualityThresholds(
            min_similarity=0.95,
            similarity_cases=("text", "table"),
            max_control_repeat=2,
            max_control_token_ratio=0.9,
        ),
    )


def test_quality_gate_fails_on_repeated_control_tokens():
    comparisons = compare_results(
        {
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel>Abstract",
            },
        },
        {
            "table": {
                "case_id": "table",
                "ok": True,
                "output_text": "<fcel><lcel><lcel><lcel>",
            },
        },
    )

    try:
        assert_quality_thresholds(
            comparisons,
            QualityThresholds(max_control_repeat=2),
        )
    except QualityGateError as exc:
        assert "max_control_repeat" in str(exc)
    else:
        raise AssertionError("expected quality gate failure")
