from __future__ import annotations

import base64
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TASK_PROMPTS = {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
    "layout": "\nLayout Analysis:",
}

STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    image: str
    prompt: str
    max_tokens: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    ok: bool
    latency_s: float
    output_text: str
    error: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _image_url_to_path_or_data(url: str) -> str:
    if url.startswith("data:"):
        header, encoded = url.split(",", 1)
        suffix = ".png"
        if "jpeg" in header or "jpg" in header:
            suffix = ".jpg"
        data = base64.b64decode(encoded)
        tmp = Path("/tmp") / f"mineru_diffusion_{abs(hash(url))}{suffix}"
        tmp.write_bytes(data)
        return str(tmp)
    if url.startswith("file://"):
        return url[len("file://") :]
    return url


def extract_openai_image_and_prompt(payload: dict[str, Any]) -> tuple[str | None, str]:
    image: str | None = None
    text_parts: list[str] = []

    for message in payload.get("messages", []):
        if message.get("role") not in {"user", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") == "user":
                text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif item_type in {"image_url", "input_image"}:
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                else:
                    url = image_url or item.get("image")
                if isinstance(url, str):
                    image = _image_url_to_path_or_data(url)

    return image, "".join(text_parts)


def read_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def write_default_cases(
    output: Path,
    image: str,
    *,
    task_max_tokens: Mapping[str, int] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    task_max_tokens = task_max_tokens or {}
    cases = [
        BenchmarkCase(
            case_id=task,
            image=image,
            prompt=prompt,
            max_tokens=task_max_tokens.get(task),
        ).to_payload()
        for task, prompt in TASK_PROMPTS.items()
    ]
    output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def summarize_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    ok_results = [result for result in results if result.ok]
    total_latency = sum(result.latency_s for result in ok_results)
    total_chars = sum(len(result.output_text) for result in ok_results)
    latencies = [result.latency_s for result in ok_results]

    return {
        "num_requests": len(results),
        "num_ok": len(ok_results),
        "num_failed": len(results) - len(ok_results),
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "p50_latency_s": statistics.median(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "output_chars": total_chars,
        "output_chars_per_s": total_chars / total_latency if total_latency else 0.0,
    }
