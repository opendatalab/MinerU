#!/usr/bin/env python3
"""同步 HTTP 包装示例：提前记录请求 ID，成功保存附件，超时返回查询链接。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx


def main() -> int:
    """演示标准 multipart 列表编码及二进制成功响应与 JSON 状态响应的区分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-format", action="append", choices=("default", "docx", "html", "latex"))
    parser.add_argument("--request-id", default=uuid.uuid4().hex)
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("--tier", choices=("flash", "basic", "standard", "advanced"))
    parser.add_argument("--ocr-mode", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--page-range", default="")
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error("Input file does not exist")
    formats = list(dict.fromkeys(args.output_format or ["default"]))
    extension = formats[0] if formats in (["docx"], ["html"]) else "zip"
    output = args.output or Path("results") / f"{args.file.stem}.{extension}"
    if output.exists():
        parser.error("Output already exists; choose another --output path")
    print(f"request_id={args.request_id}", flush=True)
    print(f"status_url={args.base_url.rstrip('/')}/v1/file_parse/{args.request_id}", flush=True)
    fields = [
        ("request_id", (None, args.request_id)),
        ("wait_timeout", (None, str(args.wait_timeout))),
        ("ocr_mode", (None, args.ocr_mode)),
        ("page_range", (None, args.page_range)),
        *(("output_formats", (None, value)) for value in formats),
    ]
    if args.tier is not None:
        fields.append(("tier", (None, args.tier)))
    temporary: Path | None = None
    try:
        with args.file.open("rb") as source, httpx.Client(timeout=httpx.Timeout(7200, connect=10, write=600)) as client:
            with client.stream(
                "POST", args.base_url.rstrip("/") + "/v1/file_parse", files=[("file", (args.file.name, source)), *fields]
            ) as response:
                if response.status_code != 200:
                    payload = response.read()
                    try:
                        print(json.dumps(json.loads(payload), ensure_ascii=False, indent=2))
                    except ValueError:
                        print(f"HTTP {response.status_code}: {payload[:4096].decode(errors='replace')}")
                    return 124 if response.status_code == 202 else 1
                output.parent.mkdir(parents=True, exist_ok=True)
                descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".part", dir=output.parent)
                temporary = Path(name)
                with os.fdopen(descriptor, "wb") as destination:
                    for chunk in response.iter_bytes(1024 * 1024):
                        destination.write(chunk)
                temporary.replace(output)
                print(f"Saved {output}")
                return 0
    except (httpx.HTTPError, OSError) as exc:
        print(f"Transfer failed ({type(exc).__name__}); use the printed status_url to recover")
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
