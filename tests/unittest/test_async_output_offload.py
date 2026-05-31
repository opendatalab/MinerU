import asyncio

from mineru.cli import common
from mineru.utils.enum_class import MakeMode


def _output_recorder(calls):
    def fake_process_output(*args, **kwargs):
        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )

    return fake_process_output


def _to_thread_recorder(calls):
    async def fake_to_thread(func, *args, **kwargs):
        calls.append(
            {
                "func": func,
                "args": args,
                "kwargs": kwargs,
            }
        )
        await asyncio.sleep(0.01)
        return func(*args, **kwargs)

    return fake_to_thread


def test_async_process_output_offloads_blocking_work(monkeypatch):
    output_calls = []
    to_thread_calls = []
    monkeypatch.setattr(common, "_process_output", _output_recorder(output_calls))
    monkeypatch.setattr(common.asyncio, "to_thread", _to_thread_recorder(to_thread_calls))

    async def run():
        task = asyncio.create_task(common._async_process_output("pdf-info", process_mode="vlm"))

        await asyncio.sleep(0)

        assert not task.done()
        await task
        assert to_thread_calls[0]["func"] is common._process_output
        assert to_thread_calls[0]["args"] == ("pdf-info",)
        assert to_thread_calls[0]["kwargs"] == {"process_mode": "vlm"}
        assert output_calls[0]["args"] == ("pdf-info",)
        assert output_calls[0]["kwargs"] == {"process_mode": "vlm"}

    asyncio.run(run())


def test_async_process_vlm_offloads_output(monkeypatch, tmp_path):
    output_calls = []
    to_thread_calls = []
    details_vlm_config = object()

    async def fake_aio_vlm_doc_analyze(*args, **kwargs):
        return {"pdf_info": []}, {"model": "output"}

    monkeypatch.setattr(common, "aio_vlm_doc_analyze", fake_aio_vlm_doc_analyze)
    monkeypatch.setattr(common, "_process_output", _output_recorder(output_calls))
    monkeypatch.setattr(common.asyncio, "to_thread", _to_thread_recorder(to_thread_calls))

    async def run():
        task = asyncio.create_task(
            common._async_process_vlm(
                tmp_path,
                ["sample"],
                [b"%PDF-1.7"],
                "vllm-async-engine",
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                MakeMode.MM_MD,
                details_vlm_config=details_vlm_config,
            )
        )

        await asyncio.sleep(0)

        assert not task.done()
        await task
        assert to_thread_calls[0]["func"] is common._process_output
        assert output_calls[0]["kwargs"]["process_mode"] == "vlm"
        assert output_calls[0]["kwargs"]["details_vlm_config"] is details_vlm_config

    asyncio.run(run())


def test_async_process_hybrid_offloads_output(monkeypatch, tmp_path):
    output_calls = []
    to_thread_calls = []
    details_vlm_config = object()

    async def fake_aio_hybrid_doc_analyze(*args, **kwargs):
        return {"pdf_info": []}, {"model": "output"}, False

    monkeypatch.setattr(
        common,
        "_load_hybrid_analyze_entrypoint",
        lambda entrypoint, backend: fake_aio_hybrid_doc_analyze,
    )
    monkeypatch.setattr(common, "_process_output", _output_recorder(output_calls))
    monkeypatch.setattr(common.asyncio, "to_thread", _to_thread_recorder(to_thread_calls))

    async def run():
        task = asyncio.create_task(
            common._async_process_hybrid(
                tmp_path,
                ["sample"],
                [b"%PDF-1.7"],
                ["en"],
                "auto",
                True,
                "vllm-async-engine",
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                MakeMode.MM_MD,
                details_vlm_config=details_vlm_config,
            )
        )

        await asyncio.sleep(0)

        assert not task.done()
        await task
        assert to_thread_calls[0]["func"] is common._process_output
        assert output_calls[0]["kwargs"]["process_mode"] == "vlm"
        assert output_calls[0]["kwargs"]["details_vlm_config"] is details_vlm_config

    asyncio.run(run())
