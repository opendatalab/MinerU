import os

from mineru.backend.vlm import visual_details_enrichment as enrichment
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.cli import common
from mineru.cli.api_client import build_parse_request_form_data
from mineru.utils.enum_class import BlockType, ContentType, MakeMode


def _visual_para(para_type, body_type, span_type, image_path, content):
    return {
        "type": para_type,
        "blocks": [
            {
                "type": body_type,
                "lines": [
                    {
                        "spans": [
                            {
                                "type": span_type,
                                "image_path": image_path,
                                "content": content,
                            }
                        ]
                    }
                ],
            }
        ],
    }


def _text_block(block_type, text, *, index=None):
    block = {
        "type": block_type,
        "lines": [{"spans": [{"type": ContentType.TEXT, "content": text}]}],
    }
    if index is not None:
        block["index"] = index
    return block


def _visual_para_with_context(
    para_type,
    body_type,
    span_type,
    image_path,
    content,
    *,
    caption_type=None,
    caption=None,
    footnote_type=None,
    footnote=None,
    index=None,
):
    para = _visual_para(para_type, body_type, span_type, image_path, content)
    if index is not None:
        para["index"] = index
    if caption_type and caption:
        para["blocks"].append(_text_block(caption_type, caption))
    if footnote_type and footnote:
        para["blocks"].append(_text_block(footnote_type, footnote))
    return para


def test_vlm_markdown_renders_referenced_image_details():
    pdf_info = [
        {
            "page_idx": 0,
            "page_size": [100, 100],
            "para_blocks": [
                _visual_para(
                    BlockType.IMAGE,
                    BlockType.IMAGE_BODY,
                    ContentType.IMAGE,
                    "image.jpg",
                    "A sparse description.",
                )
            ],
            "discarded_blocks": [],
        }
    ]

    markdown = union_make(pdf_info, MakeMode.MM_MD, "images")

    assert "![](images/image.jpg)" in markdown
    assert "<details>" in markdown
    assert "<summary>image content</summary>" in markdown
    assert "A sparse description." in markdown


def test_enrichment_selects_only_referenced_visual_details(tmp_path, monkeypatch):
    (tmp_path / "image.jpg").write_bytes(b"image")
    (tmp_path / "chart.jpg").write_bytes(b"chart")
    (tmp_path / "table.jpg").write_bytes(b"table")
    (tmp_path / "empty.jpg").write_bytes(b"empty image")
    pdf_info = [
        {
            "page_idx": 0,
            "page_size": [100, 100],
            "para_blocks": [
                _visual_para(BlockType.IMAGE, BlockType.IMAGE_BODY, ContentType.IMAGE, "image.jpg", "old image"),
                _visual_para(BlockType.CHART, BlockType.CHART_BODY, ContentType.CHART, "chart.jpg", "old chart"),
                _visual_para(BlockType.TABLE, BlockType.TABLE_BODY, ContentType.TABLE, "table.jpg", "old table"),
                _visual_para(BlockType.IMAGE, BlockType.IMAGE_BODY, ContentType.IMAGE, "missing.jpg", "old missing"),
                _visual_para(BlockType.IMAGE, BlockType.IMAGE_BODY, ContentType.IMAGE, "empty.jpg", ""),
            ],
            "discarded_blocks": [],
        }
    ]
    calls = []

    def fake_request(target, config):
        calls.append((target["span_type"], target["image_path"], config.model))
        return f"### Didactic interpretation\nenriched {target['image_path']}"

    monkeypatch.setattr(enrichment, "_request_visual_description", fake_request)
    config = enrichment.build_details_vlm_config(
        details_image_analysis=True,
        details_vlm_url="http://localhost:11434/v1",
        details_vlm_model="qwen",
    )

    count = enrichment.enrich_visual_details(pdf_info, os.fspath(tmp_path), config)

    assert count == 2
    assert calls == [
        (ContentType.IMAGE, "image.jpg", "qwen"),
        (ContentType.CHART, "chart.jpg", "qwen"),
    ]
    assert pdf_info[0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["content"] == (
        "old image\n\n### Didactic interpretation\nenriched image.jpg"
    )
    assert pdf_info[0]["para_blocks"][1]["blocks"][0]["lines"][0]["spans"][0]["content"] == (
        "old chart\n\n### Didactic interpretation\nenriched chart.jpg"
    )
    assert pdf_info[0]["para_blocks"][2]["blocks"][0]["lines"][0]["spans"][0]["content"] == "old table"
    assert pdf_info[0]["para_blocks"][4]["blocks"][0]["lines"][0]["spans"][0]["content"] == ""
    assert pdf_info[0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["details_source"] == "external_vlm"
    assert pdf_info[0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["details_model"] == "qwen"


def test_enrichment_collects_same_visual_caption_footnote_and_nearby_context(tmp_path, monkeypatch):
    (tmp_path / "chart.jpg").write_bytes(b"chart")
    pdf_info = [
        {
            "page_idx": 0,
            "page_size": [100, 100],
            "para_blocks": [
                _text_block(BlockType.TEXT, "The preceding paragraph explains drawdown risk.", index=1),
                _text_block(BlockType.TABLE, "This table must not be used.", index=2),
                _visual_para_with_context(
                    BlockType.CHART,
                    BlockType.CHART_BODY,
                    ContentType.CHART,
                    "chart.jpg",
                    "old chart",
                    caption_type=BlockType.CHART_CAPTION,
                    caption="FIGURE 2.1 Drawdown, maximum drawdown, and duration.",
                    footnote_type=BlockType.CHART_FOOTNOTE,
                    footnote="Equity is measured in dollars.",
                    index=3,
                ),
                _text_block(BlockType.CODE, "print('excluded')", index=4),
                _text_block(BlockType.REF_TEXT, "The following paragraph interprets the figure.", index=5),
            ],
            "discarded_blocks": [],
        },
        {
            "page_idx": 1,
            "page_size": [100, 100],
            "para_blocks": [
                _text_block(BlockType.TEXT, "Text from another page must not be used.", index=1),
            ],
            "discarded_blocks": [],
        },
    ]
    calls = []

    def fake_request(target, config):
        calls.append(target)
        return "enriched chart"

    monkeypatch.setattr(enrichment, "_request_visual_description", fake_request)
    config = enrichment.build_details_vlm_config(
        details_image_analysis=True,
        details_vlm_url="http://localhost:11434/v1",
        details_vlm_model="qwen",
    )

    assert enrichment.enrich_visual_details(pdf_info, os.fspath(tmp_path), config) == 1

    target = calls[0]
    assert target["caption"] == "FIGURE 2.1 Drawdown, maximum drawdown, and duration."
    assert target["footnote"] == "Equity is measured in dollars."
    assert target["preceding_context"] == "The preceding paragraph explains drawdown risk."
    assert target["following_context"] == "The following paragraph interprets the figure."
    assert "another page" not in target["preceding_context"]
    assert "another page" not in target["following_context"]


def test_nearby_context_uses_short_lists_and_truncates_to_limit(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"image")
    preceding = "prefix " + ("A" * 900)
    following = ("B" * 900) + " suffix"
    page = {
        "page_idx": 0,
        "page_size": [100, 100],
        "para_blocks": [
            _text_block(BlockType.LIST, "x" * 850, index=1),
            _text_block(BlockType.LIST, "Short list context", index=2),
            _text_block(BlockType.TEXT, preceding, index=3),
            _visual_para_with_context(
                BlockType.IMAGE,
                BlockType.IMAGE_BODY,
                ContentType.IMAGE,
                "image.jpg",
                "old image",
                index=4,
            ),
            _text_block(BlockType.TEXT, following, index=5),
            _text_block(BlockType.TEXT, "Later text should not be used.", index=6),
        ],
        "discarded_blocks": [],
    }

    target = next(enrichment._iter_referenced_visual_detail_targets([page], os.fspath(tmp_path)))

    assert len(target["preceding_context"]) == enrichment.NEARBY_CONTEXT_CHAR_LIMIT
    assert target["preceding_context"] == preceding[-enrichment.NEARBY_CONTEXT_CHAR_LIMIT:]
    assert len(target["following_context"]) == enrichment.NEARBY_CONTEXT_CHAR_LIMIT
    assert target["following_context"] == following[:enrichment.NEARBY_CONTEXT_CHAR_LIMIT]
    assert "Later text" not in target["following_context"]


def test_prompt_contains_visual_context_sections():
    target = {
        "span_type": ContentType.CHART,
        "original_content": "Sparse MinerU chart text.",
        "caption": "FIGURE 2.1 Drawdown and maximum drawdown.",
        "footnote": "Equity values are illustrative.",
        "preceding_context": "The text before introduces downside risk.",
        "following_context": "The text after discusses recovery duration.",
    }

    prompt = enrichment._build_visual_details_prompt(target, "en")

    assert "referenced chart" in prompt
    assert "Sparse MinerU chart text." in prompt
    assert "FIGURE 2.1 Drawdown and maximum drawdown." in prompt
    assert "Equity values are illustrative." in prompt
    assert "The text before introduces downside risk." in prompt
    assert "The text after discusses recovery duration." in prompt
    assert "Write the answer in this language: en." in prompt
    assert "Do not include <details>, <summary>" in prompt
    assert 'Start exactly with a "### Didactic interpretation" heading' in prompt
    assert 'Do not create a "### Visual data" section' in prompt


def test_visual_details_content_normalization_adds_didactic_section_and_strips_visual_data():
    content = enrichment._normalize_visual_details_content(
        "### Visual data\n- Axis: price\n\n### Interpretation\nA concise chart explanation."
    )

    assert content.startswith("### Didactic interpretation")
    assert "### Visual data" not in content
    assert "A concise chart explanation." in content


def test_response_wrapper_stripping_preserves_unicode_text():
    data = {
        "choices": [
            {
                "message": {
                    "content": "```markdown\n<details>\n<summary>line</summary>\n\nEquity ranges from 1×10⁴ to 3×10⁴ — useful for risk analysis.\n</details>\n```"
                }
            }
        ]
    }

    content = enrichment._extract_chat_completion_content(data)

    assert content == "Equity ranges from 1×10⁴ to 3×10⁴ — useful for risk analysis."
    assert "⁴" in content
    assert "<summary>" not in content


def test_enrichment_preserves_content_on_failure(tmp_path, monkeypatch):
    (tmp_path / "image.jpg").write_bytes(b"image")
    pdf_info = [
        {
            "page_idx": 0,
            "page_size": [100, 100],
            "para_blocks": [
                _visual_para(BlockType.IMAGE, BlockType.IMAGE_BODY, ContentType.IMAGE, "image.jpg", "old image")
            ],
            "discarded_blocks": [],
        }
    ]

    def fake_request(target, config):
        raise RuntimeError("network down")

    monkeypatch.setattr(enrichment, "_request_visual_description", fake_request)
    config = enrichment.build_details_vlm_config(
        details_image_analysis=True,
        details_vlm_url="http://localhost:11434/v1",
        details_vlm_model="qwen",
    )

    assert enrichment.enrich_visual_details(pdf_info, os.fspath(tmp_path), config) == 0
    assert pdf_info[0]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["content"] == "old image"


def test_parse_request_form_data_includes_details_parameters():
    data = build_parse_request_form_data(
        lang_list=["en"],
        backend="vlm-http-client",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        image_analysis=True,
        details_image_analysis=True,
        details_vlm_url="http://localhost:11434/v1",
        details_vlm_model="qwen",
        details_vlm_api_key="",
        details_vlm_timeout=180,
        details_vlm_max_concurrency=1,
        details_vlm_language="en",
        server_url="http://localhost:30000",
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=True,
        return_original_file=True,
    )

    assert data["details_image_analysis"] == "true"
    assert data["details_vlm_url"] == "http://localhost:11434/v1"
    assert data["details_vlm_model"] == "qwen"
    assert data["details_vlm_api_key"] == ""
    assert data["details_vlm_timeout"] == "180"
    assert data["details_vlm_max_concurrency"] == "1"
    assert data["details_vlm_language"] == "en"


def test_details_enrichment_preserves_mineru_image_chart_analysis(tmp_path, monkeypatch):
    captured = {}

    def fake_vlm_doc_analyze(*args, **kwargs):
        captured.update(kwargs)
        return {"pdf_info": []}, []

    monkeypatch.setattr(common, "vlm_doc_analyze", fake_vlm_doc_analyze)
    monkeypatch.setattr(common, "_process_output", lambda *args, **kwargs: None)

    config = enrichment.build_details_vlm_config(
        details_image_analysis=True,
        details_vlm_url="http://localhost:11434/v1",
        details_vlm_model="qwen",
    )

    common._process_vlm(
        tmp_path,
        ["doc"],
        [b"%PDF"],
        "vllm-engine",
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        MakeMode.MM_MD,
        details_vlm_config=config,
        image_analysis=True,
    )

    assert captured["image_analysis"] is True
