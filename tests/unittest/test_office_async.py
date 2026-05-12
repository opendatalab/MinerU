import asyncio
import threading

from mineru.cli import common


def test_aio_do_parse_runs_office_preprocessing_off_event_loop(monkeypatch, tmp_path):
    event_loop_thread_id = threading.get_ident()
    worker_thread_id = None

    def fake_process_office_doc(*args, **kwargs):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        return [0]

    monkeypatch.setattr(common, "_process_office_doc", fake_process_office_doc)

    asyncio.run(
        common.aio_do_parse(
            output_dir=str(tmp_path),
            pdf_file_names=["doc"],
            pdf_bytes_list=[b"docx"],
            p_lang_list=["ch"],
            backend="vlm-http-client",
        )
    )

    assert worker_thread_id is not None
    assert worker_thread_id != event_loop_thread_id
