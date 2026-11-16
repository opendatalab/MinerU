"""仅供测试的独立 API 入口，在解析调用阶段注入一次可确认的停滞。"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mineru.parser import api_server


def main() -> None:
    """第一代报告真实到达的调用阶段并阻塞；第二代使用完整的 Flash PDF 解析。"""
    marker, failure, port, upload_dir = sys.argv[1:]
    first = not Path(marker).exists()
    original = api_server.parse_async

    async def parse(path: str, **kwargs: object) -> object:
        """只阻塞第一代测试进程，恢复后的请求完整经过实际解析及产物打包。"""
        if not first:
            return await original(path, **kwargs)
        Path(marker).write_text(json.dumps({"pid": os.getpid(), "phase": failure}))
        if failure == "event_loop":
            time.sleep(90)
        try:
            await asyncio.Future()
        finally:
            if failure == "cleanup":
                while True:
                    try:
                        await asyncio.sleep(90)
                    except asyncio.CancelledError:
                        pass

    api_server.parse_async = parse
    api_server.main.main(
        args=[
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--tier",
            "flash",
            "--task-timeout",
            "1" if first else "30",
            "--upload-dir",
            upload_dir,
        ],
        standalone_mode=False,
    )


if __name__ == "__main__":
    main()
