# Copyright (c) Opendatalab. All rights reserved.
"""VLM后端两阶段解耦的真模型集成测试（在有算力的机器上运行）。

默认跳过。启用方式（在有GPU/远端VLM服务的机器上）：
  export MINERU_TEST_VLM_BACKEND=transformers          # 或 http-client 等
  export MINERU_TEST_VLM_SERVER_URL=http://host:30000  # http-client 时必填
  pytest tests/unittest/test_vlm_stages_e2e.py -o addopts="" -v

可选：MINERU_TEST_BASELINE_DIR 指向改动前生成的输出目录时，
test_default_path_unchanged 会做逐字节回归对比。
"""
import json
import os
from pathlib import Path

import pytest

VLM_BACKEND = os.getenv("MINERU_TEST_VLM_BACKEND")
VLM_SERVER_URL = os.getenv("MINERU_TEST_VLM_SERVER_URL")
BASELINE_DIR = os.getenv("MINERU_TEST_BASELINE_DIR")

pytestmark = pytest.mark.skipif(
    not VLM_BACKEND,
    reason="set MINERU_TEST_VLM_BACKEND to run real-model e2e tests",
)

DEMO_PDF = Path(__file__).parent.parent.parent / "demo" / "pdfs" / "small_ocr.pdf"


def _vlm_kwargs():
    return dict(backend=VLM_BACKEND, server_url=VLM_SERVER_URL)


def _pdf_bytes():
    from mineru.cli.common import convert_pdf_bytes_to_bytes, read_fn
    return convert_pdf_bytes_to_bytes(read_fn(DEMO_PDF))


def _make_markdown(middle_json):
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
    from mineru.utils.enum_class import MakeMode
    return union_make(middle_json["pdf_info"], MakeMode.MM_MD, "images")


def _strip_version(middle_json):
    data = json.loads(json.dumps(middle_json, ensure_ascii=False))
    data.pop("_version_name", None)
    return data


@pytest.fixture(scope="module")
def image_writer(tmp_path_factory):
    from mineru.data.data_reader_writer import FileBasedDataWriter
    return FileBasedDataWriter(str(tmp_path_factory.mktemp("images")))


def test_default_path_unchanged(image_writer, tmp_path):
    """默认路径（无注入）产出结构合法；提供基线目录时做逐字节回归对比。"""
    from mineru.backend.vlm import vlm_analyze

    middle_json, results = vlm_analyze.doc_analyze(_pdf_bytes(), image_writer, **_vlm_kwargs())
    md = _make_markdown(middle_json)
    assert middle_json["pdf_info"]
    assert len(md.strip()) > 50

    out_md = tmp_path / "default.md"
    out_middle = tmp_path / "default_middle.json"
    out_md.write_text(md, encoding="utf-8")
    out_middle.write_text(
        json.dumps(_strip_version(middle_json), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"default outputs saved to {tmp_path} (可作为下一次回归的基线)")

    if BASELINE_DIR:
        baseline_md = Path(BASELINE_DIR) / "default.md"
        baseline_middle = Path(BASELINE_DIR) / "default_middle.json"
        assert baseline_md.read_text(encoding="utf-8") == md
        assert json.loads(baseline_middle.read_text(encoding="utf-8")) == _strip_version(middle_json)


def test_mode_b_pipeline_layout_plus_vlm(image_writer):
    """模式b：PP-DocLayoutV2出layout + VLM识别，产出可用的结构与内容。"""
    from mineru.backend.vlm import vlm_analyze
    from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector

    middle_json, results = vlm_analyze.doc_analyze(
        _pdf_bytes(),
        image_writer,
        layout_detector=PipelineLayoutDetector(),
        **_vlm_kwargs(),
    )
    md = _make_markdown(middle_json)
    assert len(md.strip()) > 50
    block_types = {
        block.get("type")
        for page in middle_json["pdf_info"]
        for block in page.get("para_blocks", []) or page.get("preproc_blocks", [])
    }
    assert block_types, "expect non-empty blocks from decoupled path"


def test_mode_c_split_run_matches_mode_b(image_writer, tmp_path):
    """模式c：layout.json两阶段分离运行，markdown与模式b单次运行一致。"""
    from mineru.backend.vlm import vlm_analyze
    from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector
    from mineru.backend.vlm.stages import DEFAULT_LAYOUT_DOC_FILENAME, PrecomputedLayoutDetector
    from mineru.data.data_reader_writer import FileBasedDataWriter

    pdf_bytes = _pdf_bytes()
    detector = PipelineLayoutDetector()
    layout_writer = FileBasedDataWriter(str(tmp_path))

    # 单次运行（模式b），同时落盘layout.json
    middle_json_b, _ = vlm_analyze.doc_analyze(
        pdf_bytes,
        image_writer,
        layout_detector=detector,
        layout_writer=layout_writer,
        **_vlm_kwargs(),
    )

    # 分离运行（模式c）：从layout.json恢复
    layout_json_path = str(tmp_path / DEFAULT_LAYOUT_DOC_FILENAME)
    middle_json_c, _ = vlm_analyze.doc_analyze(
        pdf_bytes,
        image_writer,
        layout_detector=PrecomputedLayoutDetector(layout_json_path),
        **_vlm_kwargs(),
    )
    assert _make_markdown(middle_json_c) == _make_markdown(middle_json_b)


def test_aio_mode_b_parity(image_writer):
    """异步路径与同步路径在模式b下产出一致的markdown。"""
    import asyncio

    from mineru.backend.vlm import vlm_analyze
    from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector

    pdf_bytes = _pdf_bytes()
    detector = PipelineLayoutDetector()

    middle_sync, _ = vlm_analyze.doc_analyze(
        pdf_bytes, image_writer, layout_detector=detector, **_vlm_kwargs()
    )
    middle_async, _ = asyncio.run(
        vlm_analyze.aio_doc_analyze(
            pdf_bytes, image_writer, layout_detector=detector, **_vlm_kwargs()
        )
    )
    assert _make_markdown(middle_async) == _make_markdown(middle_sync)
