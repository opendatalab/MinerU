"""验证 Gradio 组装循环的归属、异常清理以及实际服务循环中的队列执行。"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import gradio as gr
import pytest

from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio.client import V1ServerCapabilities


@pytest.fixture
def setup_loops(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[asyncio.AbstractEventLoop]]:
    """记录组装时创建的循环，并在断言失败时关闭残留资源以隔离其它测试。"""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    created: list[asyncio.AbstractEventLoop] = []
    factory = asyncio.new_event_loop

    def record_loop() -> asyncio.AbstractEventLoop:
        """保留真实循环，既检测遗漏关闭，也检测无意义的额外创建。"""
        loop = factory()
        created.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", record_loop)
    yield created
    for loop in created:
        if not loop.is_closed() and not loop.is_running():
            loop.close()


def _build(output_root: Path) -> gr.Blocks:
    """构建真实 UI 和队列，不建立外部解析连接。"""
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("default",), ("upload",))
    return gradio_app.build_gradio_app(Mock(), capabilities, output_root=output_root, enable_example=False)


@pytest.mark.parametrize("fail", [False, True])
def test_sync_build_preserves_host_loop_and_closes_setup_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_loops: list[asyncio.AbstractEventLoop], fail: bool
) -> None:
    """成功和组件构造失败都只关闭自建循环，不替换或关闭宿主尚未运行的循环。"""
    host = asyncio.new_event_loop()
    setup_loops.clear()
    asyncio.set_event_loop(host)
    pending: list[asyncio.Task[None]] = []

    async def wait_forever() -> None:
        """模拟未来 Gradio 组件在构造阶段提交的后台任务。"""
        await asyncio.Event().wait()

    def fail_header() -> str:
        """在 Blocks 已构造后失败，同时验证 Runner 会排空构造期任务。"""
        pending.append(asyncio.get_running_loop().create_task(wait_forever()))
        raise ValueError("component construction failed")

    try:
        if fail:
            monkeypatch.setattr(gradio_app, "_render_header", fail_header)
            with pytest.raises(ValueError, match="component construction failed"):
                _build(tmp_path)
            assert len(pending) == 1 and pending[0].cancelled()
        else:
            _build(tmp_path)
        assert len(setup_loops) == 1
        assert setup_loops[0].is_closed()
        assert asyncio.get_event_loop() is host
        assert not host.is_closed()
    finally:
        asyncio.set_event_loop(None)
        host.close()


def test_build_reuses_running_host_loop(tmp_path: Path, setup_loops: list[asyncio.AbstractEventLoop]) -> None:
    """外部异步宿主构建 UI 时不创建嵌套循环，构建结束后宿主任务仍能继续运行。"""
    host = asyncio.new_event_loop()
    setup_loops.clear()

    async def build_in_host() -> None:
        """在真实运行的宿主循环中构建并继续调度后续任务。"""
        assert asyncio.get_running_loop() is host
        demo = _build(tmp_path)
        await asyncio.sleep(0)
        assert demo._queue.default_concurrency_limit == 1
        assert asyncio.get_running_loop() is host

    try:
        host.run_until_complete(build_in_host())
        assert setup_loops == []
        assert not host.is_closed()
    finally:
        host.close()


def test_worker_thread_repeated_builds_close_their_loops(tmp_path: Path, setup_loops: list[asyncio.AbstractEventLoop]) -> None:
    """重载线程没有默认循环时也能重复构建，线程内不留下已关闭或无主的当前循环。"""

    def build_twice() -> None:
        """复用同一工作线程，暴露前一次构建遗留的循环状态。"""
        for _ in range(2):
            _build(tmp_path)
            with pytest.raises(RuntimeError, match="no current event loop"):
                asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(build_twice).result(timeout=15)
    assert len(setup_loops) == 2
    assert all(loop.is_closed() for loop in setup_loops)


@pytest.mark.parametrize("fail", [False, True])
def test_launch_setup_closes_its_loop(tmp_path: Path, setup_loops: list[asyncio.AbstractEventLoop], fail: bool) -> None:
    """启动阶段重新创建 App 的同步路径也具有独立清理边界。"""
    observed: list[asyncio.AbstractEventLoop] = []

    def launch(**kwargs: object) -> None:
        """模拟启动组装并记录调用时运行的循环，同时保留原有启动参数。"""
        observed.append(asyncio.get_running_loop())
        assert kwargs == {"server_name": "127.0.0.1", "server_port": None, "allowed_paths": [str(tmp_path)]}
        if fail:
            raise ValueError("launch failed")

    demo = Mock(launch=launch, _mineru_kit_launch_kwargs={})
    if fail:
        with pytest.raises(ValueError, match="launch failed"):
            gradio_app._launch_gradio_app(demo, server_name="127.0.0.1", server_port=None, output_root=tmp_path)
    else:
        gradio_app._launch_gradio_app(demo, server_name="127.0.0.1", server_port=None, output_root=tmp_path)
    assert setup_loops == observed
    assert len(setup_loops) == 1 and setup_loops[0].is_closed()


def test_real_launch_queue_and_first_interrupt(tmp_path: Path) -> None:
    """独立进程实测上传、并发排队、重新构建和首次 Ctrl+C，避免失败后遗留监听线程。"""
    script = r'''
        import asyncio
        import os
        import signal
        import sys
        import threading
        from pathlib import Path

        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["MINERU_LLM_AIDED_FEATURES_TITLE_LEVELING"] = "false"
        os.environ["MINERU_LLM_AIDED_FEATURES_CROSS_PAGE_TABLE_CELL_MERGE"] = "false"

        import httpx
        from gradio_client import Client, handle_file
        from mineru.kit.gradio.app import build_gradio_app, _launch_gradio_app
        from mineru.kit.gradio.client import V1ServerCapabilities
        from mineru.parser import parse

        class LocalClient:
            """只使用原生 HTML 解析，真实经过转换服务和 Gradio 队列。"""

            async def parse_file(self, path, *, tier, page_range, ocr_mode, status_callback):
                """让两次请求重叠，以验证队列锁在服务循环中正常绑定。"""
                await asyncio.sleep(0.02)
                return parse(path, tier=tier)

        root = Path(sys.argv[1])
        source = root / "loop.html"
        source.write_text("<html><body><h1>Loop lifecycle</h1><p>Queue conversion evidence.</p></body></html>")
        capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("default",), ("upload",))
        demo = build_gradio_app(LocalClient(), capabilities, output_root=root, enable_example=False)
        # 服务已经启动，替代阻塞等待的回调在主线程进行实际 HTTP 验证。
        def exercise_queue():
            """用两个独立会话提交真实上传任务，避免触发同会话重新提交的取消语义。"""
            with httpx.Client(trust_env=False) as http:
                assert http.get(demo.local_url + "config").status_code == 200
            clients = [Client(demo.local_url, verbose=False) for _ in range(2)]
            try:
                jobs = [
                    client.submit(handle_file(str(source)), 0, "", False, api_name="/convert_handler")
                    for client in clients
                ]
                for job in jobs:
                    values = job.result(timeout=20)
                    assert "Queue conversion evidence" in values[1], values
            finally:
                for client in clients:
                    client.close()

        demo.block_thread = exercise_queue
        try:
            _launch_gradio_app(demo, server_name="127.0.0.1", server_port=None, output_root=root)
        finally:
            demo.close()
        print("QUEUE_PASSED", flush=True)

        if os.name != "nt":
            # 用真实 block_thread 接收一次 SIGINT；Runner.run 会吞下首次中断，因此必须覆盖。
            demo = build_gradio_app(LocalClient(), capabilities, output_root=root, enable_example=False)
            reached = threading.Event()
            blocking = threading.Event()
            original_block_thread = demo.block_thread
            def block_until_interrupted():
                """确认启动已完成后进入 Gradio 自己的阻塞等待。"""
                blocking.set()
                original_block_thread()
            demo.block_thread = block_until_interrupted
            def interrupt_once():
                """等启动流程进入阻塞等待后，仅向主进程发送一次中断。"""
                assert blocking.wait(timeout=20), "service did not reach block_thread"
                reached.wait(0.1)
                os.kill(os.getpid(), signal.SIGINT)
            interrupter = threading.Thread(target=interrupt_once, daemon=True)
            interrupter.start()
            try:
                _launch_gradio_app(demo, server_name="127.0.0.1", server_port=None, output_root=root)
            finally:
                demo.close()
                reached.set()
                interrupter.join(timeout=2)
            assert not interrupter.is_alive()
            print("FIRST_INTERRUPT_PASSED", flush=True)
    '''
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "QUEUE_PASSED" in completed.stdout
    if sys.platform != "win32":
        assert "FIRST_INTERRUPT_PASSED" in completed.stdout
    assert "BaseEventLoop.__del__" not in completed.stderr
    assert "Task was destroyed" not in completed.stderr
