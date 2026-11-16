"""创建真实托管 Router/worker/模型树，供父进程先退出和单 worker 回收验收。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx

from mineru.kit.router.workers import ManagedLocalWorker, RouterSettings


async def main() -> None:
    """复用正式 worker 启动逻辑，仅用轻量子进程替代模型和 HTTP 就绪响应。"""
    marker, gate = map(Path, sys.argv[1:])
    code = (
        "import subprocess,sys,time,os,json,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)']); "
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(),p.pid]));time.sleep(90)"
    )

    class Worker(ManagedLocalWorker):
        """保留进程组和临时目录管理，只替换解析进程的工作内容。"""

        def command(self, port: int, upload_dir: Path) -> list[str]:
            """让 worker 及模型保持存活，模拟后台清理迟迟不能结束。"""
            return [sys.executable, "-c", code]

    worker = Worker("owned", "127.0.0.1", None, RouterSettings(managed_webui=True, local_gpus="inherit"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        await worker.start(client)
        while not gate.exists():
            await asyncio.sleep(0.01)
        if gate.read_text() == "exit":
            # 只移除测试控制端点，故意让 worker 和模型留在原地，验证外层的进程所有权。
            worker.control.close()
            os._exit(0)
        await worker.stop(timeout_seconds=2)
        gate.write_text("stopped")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
