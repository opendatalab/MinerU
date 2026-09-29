import json
import threading
from pathlib import Path

from PIL import Image

from benchmarks.mineru_diffusion.end2end_openai import (
    End2EndConfig,
    End2EndOpenAIClient,
    crop_block_image,
    parse_layout_output,
    run_end2end_batch,
    run_end2end,
)


def test_parse_layout_output_normalizes_boxes_and_angles():
    blocks = parse_layout_output(
        "<|box_start|>900 800 100 200<|box_end|>"
        "<|ref_start|>table<|ref_end|><|rotate_right|>\n"
        "<|box_start|>000 000 000 010<|box_end|>"
        "<|ref_start|>text<|ref_end|><|rotate_up|>"
    )

    assert len(blocks) == 1
    assert blocks[0].type == "table"
    assert blocks[0].bbox == [0.1, 0.2, 0.9, 0.8]
    assert blocks[0].angle == 90


def test_crop_block_image_uses_normalized_layout_coordinates():
    image = Image.new("RGB", (200, 100), "white")
    block = parse_layout_output(
        "<|box_start|>250 250 750 750<|box_end|>"
        "<|ref_start|>text<|ref_end|><|rotate_up|>"
    )[0]

    cropped = crop_block_image(image, block)

    assert cropped.size == (100, 50)


def test_run_end2end_calls_layout_then_matching_block_prompts(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    calls = []

    class FakeClient:
        def infer(self, image_path, prompt, max_tokens):
            calls.append((Path(image_path).name, prompt, max_tokens))
            if prompt == "\nLayout Detection:":
                return (
                    "<|box_start|>000 000 500 500<|box_end|>"
                    "<|ref_start|>text<|ref_end|><|rotate_up|>\n"
                    "<|box_start|>500 000 1000 500<|box_end|>"
                    "<|ref_start|>table<|ref_end|><|rotate_up|>"
                )
            if prompt == "\nTable Recognition:":
                return "<fcel>A<nl>"
            return "plain text"

    result = run_end2end(
        End2EndConfig(
            image_path=image_path,
            output_dir=tmp_path / "out",
            endpoint="http://unused/v1/chat/completions",
            layout_max_tokens=64,
            content_max_tokens=128,
            table_max_tokens=256,
            formula_max_tokens=512,
        ),
        client=FakeClient(),
    )

    assert [call[1:] for call in calls] == [
        ("\nLayout Detection:", 64),
        ("\nText Recognition:", 128),
        ("\nTable Recognition:", 256),
    ]
    assert result.metrics["num_blocks"] == 2
    assert result.metrics["num_extracted_blocks"] == 2
    assert [block.type for block in result.blocks] == ["text", "table"]
    assert result.blocks[0].content == "plain text"
    assert result.blocks[1].content == "<table><tr><td>A</td></tr></table>"
    assert "plain text" in result.markdown
    assert "<table>" in result.markdown

    latest = json.loads((tmp_path / "out" / "latest_result.json").read_text())
    assert latest["metrics"]["num_blocks"] == 2
    assert latest["blocks"][1]["type"] == "table"


def test_run_end2end_can_extract_blocks_concurrently(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    first_content_started = threading.Event()
    second_content_started = threading.Event()
    release_content = threading.Event()
    content_prompts = []
    content_lock = threading.Lock()

    class FakeClient:
        def infer(self, image_path, prompt, max_tokens):
            if prompt == "\nLayout Detection:":
                return (
                    "<|box_start|>000 000 500 500<|box_end|>"
                    "<|ref_start|>text<|ref_end|><|rotate_up|>\n"
                    "<|box_start|>500 000 1000 500<|box_end|>"
                    "<|ref_start|>text<|ref_end|><|rotate_up|>"
                )
            with content_lock:
                content_prompts.append(prompt)
                call_index = len(content_prompts)
            if call_index == 1:
                first_content_started.set()
                assert second_content_started.wait(timeout=1)
                release_content.wait(timeout=1)
                return "first"
            second_content_started.set()
            assert first_content_started.is_set()
            release_content.set()
            return "second"

    result = run_end2end(
        End2EndConfig(
            image_path=image_path,
            output_dir=tmp_path / "out",
            endpoint="http://unused/v1/chat/completions",
            content_concurrency=2,
        ),
        client=FakeClient(),
    )

    assert sorted(block.content for block in result.blocks) == ["first", "second"]
    assert result.metrics["content_concurrency"] == 2


def test_run_end2end_batch_flattens_extract_requests_across_pages(tmp_path: Path):
    image_paths = []
    for index in range(2):
        image_path = tmp_path / f"page_{index}.png"
        Image.new("RGB", (100, 100), "white").save(image_path)
        image_paths.append(image_path)

    layout_calls = []
    content_calls = []

    class FakeClient:
        def infer(self, image_path, prompt, max_tokens):
            if prompt == "\nLayout Detection:":
                layout_calls.append(Path(image_path))
                return (
                    "<|box_start|>000 000 500 500<|box_end|>"
                    "<|ref_start|>text<|ref_end|><|rotate_up|>"
                )
            assert len(layout_calls) == 2
            content_calls.append(Path(image_path))
            return f"content-{Path(image_path).parent.parent.stem}"

    configs = [
        End2EndConfig(
            image_path=image_path,
            output_dir=tmp_path / "out" / image_path.stem,
            endpoint="http://unused/v1/chat/completions",
            content_concurrency=2,
        )
        for image_path in image_paths
    ]

    results = run_end2end_batch(
        configs,
        client=FakeClient(),
        layout_concurrency=2,
        content_concurrency=2,
    )

    assert len(results) == 2
    assert len(layout_calls) == 2
    assert len(content_calls) == 2
    assert all(result.metrics["throughput_batch_size"] == 2 for result in results)
    assert all(result.metrics["layout_concurrency"] == 2 for result in results)
    assert all(result.metrics["content_concurrency"] == 2 for result in results)
    assert [result.blocks[0].content for result in results] == [
        "content-page_0",
        "content-page_1",
    ]


def test_openai_client_builds_mineru_request_payload(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "crop.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeSession:
        trust_env = True

        def post(self, endpoint, json, timeout):
            captured["endpoint"] = endpoint
            captured["json"] = json
            captured["timeout"] = timeout
            captured["trust_env"] = self.trust_env
            return FakeResponse()

    monkeypatch.setattr(
        "benchmarks.mineru_diffusion.end2end_openai.requests.Session",
        FakeSession,
    )

    client = End2EndOpenAIClient(
        endpoint="http://host/v1/chat/completions",
        model="mineru-diffusion",
        timeout=12,
        block_size=32,
        dynamic_threshold=0.95,
        max_denoising_steps=16,
    )

    assert client.infer(image_path, "\nText Recognition:", 128) == "ok"
    assert captured["endpoint"] == "http://host/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["trust_env"] is False
    assert captured["json"]["max_tokens"] == 128
    assert captured["json"]["max_denoising_steps"] == 16
    assert captured["json"]["vllm_xargs"] == {
        "block_size": 32,
        "dynamic_threshold": 0.95,
        "max_denoising_steps": 16,
    }
    assert captured["json"]["messages"][1]["content"][0]["image_url"]["url"] == (
        f"file://{image_path}"
    )
