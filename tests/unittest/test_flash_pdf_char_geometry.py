from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mineru.model.flash.pdf import pipeline
from mineru.model.flash.pdf.char_geometry import (
    _line_loose_tier_offsets,
    apply_line_geometry_repairs,
    build_document_geometry_plan,
)
from mineru.model.flash.pdf.document import PDFPageTextGeometry
from mineru.model.flash.pdf.models import _LineItem


_PDF_FIXTURE_XOR_KEY = b"MinerU flash layout fixture"


def _read_pdf_fixture(path: Path) -> bytes:
    """读取普通 PDF，或在内存中解密以 .xor 结尾的测试样本。"""

    payload = path.read_bytes()
    if path.suffix != ".xor":
        return payload
    key_length = len(_PDF_FIXTURE_XOR_KEY)
    return bytes(value ^ _PDF_FIXTURE_XOR_KEY[index % key_length] for index, value in enumerate(payload))


def _line_fixture(
    *,
    source_index: int,
    baseline: float,
    count: int = 40,
    advance: float = 6.0,
    loose_width: float = 6.0,
    loose_top: float | None = None,
    loose_bottom: float | None = None,
    font_name: str = "ABCDEF+Fixture",
    font_size: float = 10.0,
    start_char_idx: int = 0,
    split_baseline: float | None = None,
) -> tuple[_LineItem, PDFPageTextGeometry]:
    """构造 loose/tight/origin 均可控的单行几何 fixture。"""

    chars = []
    loose_bboxes = {}
    tight_bboxes = {}
    origins = {}
    top = baseline - 8.0 if loose_top is None else loose_top
    bottom = baseline + 2.0 if loose_bottom is None else loose_bottom
    for position in range(count):
        char_idx = start_char_idx + position
        origin_y = split_baseline if split_baseline is not None and position >= count // 2 else baseline
        origin_x = 10.0 + position * advance
        loose_bbox = (
            origin_x,
            top if origin_y == baseline else origin_y - 8.0,
            origin_x + loose_width,
            bottom if origin_y == baseline else origin_y + 2.0,
        )
        tight_bbox = (origin_x + 0.5, origin_y - 7.0, origin_x + 5.0, origin_y)
        chars.append(
            {
                "char": chr(ord("A") + position % 26),
                "bbox": loose_bbox,
                "rotation": 0.0,
                "font": {
                    "name": font_name,
                    "flags": 0,
                    "size": font_size,
                    "weight": 400,
                },
                "char_idx": char_idx,
            }
        )
        loose_bboxes[char_idx] = loose_bbox
        tight_bboxes[char_idx] = tight_bbox
        origins[char_idx] = (origin_x, origin_y)
    bbox = (
        min(value[0] for value in loose_bboxes.values()),
        min(value[1] for value in loose_bboxes.values()),
        max(value[2] for value in loose_bboxes.values()),
        max(value[3] for value in loose_bboxes.values()),
    )
    line = _LineItem(
        text="".join(char["char"] for char in chars),
        bbox=bbox,
        angle=0,
        source_index=source_index,
        chars=chars,  # type: ignore[arg-type]
        effective_height=bottom - top,
    )
    return line, PDFPageTextGeometry(
        chars=chars,  # type: ignore[arg-type]
        tight_bboxes=tight_bboxes,
        origins=origins,
        loose_bboxes=loose_bboxes,
    )


def _merge_geometries(*geometries: PDFPageTextGeometry) -> PDFPageTextGeometry:
    """合并同页多行 fixture 的字符 side-map。"""

    return PDFPageTextGeometry(
        chars=[char for geometry in geometries for char in geometry.chars],
        tight_bboxes={key: value for geometry in geometries for key, value in geometry.tight_bboxes.items()},
        origins={key: value for geometry in geometries for key, value in geometry.origins.items()},
        loose_bboxes={key: value for geometry in geometries for key, value in geometry.loose_bboxes.items()},
    )


def test_two_page_repeated_loose_height_inflation_sets_canonical_em_scale() -> None:
    """验证两页内重复的 loose 高度异常即可按字号与 tight 几何校准全文。"""

    lines_by_page: list[list[_LineItem]] = []
    geometries: list[PDFPageTextGeometry] = []
    for page_index in range(2):
        page_lines = []
        page_geometries = []
        for line_index in range(3):
            baseline = 20.0 + 16.0 * line_index
            anomalous = line_index < 2
            line, geometry = _line_fixture(
                source_index=line_index,
                baseline=baseline,
                loose_top=(
                    baseline - 20.0
                    if anomalous
                    else baseline - 8.0
                ),
                loose_bottom=(
                    baseline + 4.0
                    if anomalous
                    else baseline + 2.0
                ),
                loose_width=12.0 if anomalous else 6.0,
                font_name=(
                    "ABCDEF+Anomaly"
                    if anomalous
                    else "ABCDEF+Normal"
                ),
                font_size=10.0,
                start_char_idx=(
                    page_index * 10000
                    + line_index * 100
                ),
            )
            page_lines.append(line)
            page_geometries.append(geometry)
        lines_by_page.append(page_lines)
        geometries.append(_merge_geometries(*page_geometries))

    plan = build_document_geometry_plan(
        lines_by_page,
        geometries,
        [(300.0, 200.0)] * 2,
    )
    for page_index, lines in enumerate(lines_by_page):
        apply_line_geometry_repairs(
            lines,
            page_index=page_index,
            plan=plan,
            allow_y_trim=True,
        )

    assert plan.document_style_anomaly
    assert all(lines[0].em_height == 10.0 for lines in lines_by_page)
    assert any(run["style_y_bad"] for run in plan.run_diagnostics)


def test_line_loose_tier_shrinks_repeated_largest_tier_to_second_tier() -> None:
    """验证同行重复最大 loose 档按次档的归一化 ascent/descent 回缩。"""

    offsets = _line_loose_tier_offsets(
        [(8.0, 2.0, 7.0, 10.0)] * 6
        + [(20.0, 4.0, 7.0, 10.0)] * 4,
        10.0,
    )

    assert offsets == (8.0, 2.0)


def test_line_loose_tier_preserves_legitimate_mixed_font_sizes() -> None:
    """验证按各自 em 归一化后相同的真实混合字号不会形成异常高度档。"""

    offsets = _line_loose_tier_offsets(
        [(8.0, 2.0, 7.0, 10.0)] * 4
        + [(16.0, 4.0, 14.0, 20.0)] * 4,
        10.0,
    )

    assert offsets is None


def test_line_loose_tier_ignores_single_large_outlier() -> None:
    """验证单个 loose 高度离群字符不足以触发整行档位回缩。"""

    offsets = _line_loose_tier_offsets(
        [(8.0, 2.0, 7.0, 10.0)] * 7
        + [(20.0, 4.0, 7.0, 10.0)],
        10.0,
    )

    assert offsets is None


def test_strong_x_run_repairs_advance_and_contains_tight_bbox() -> None:
    """验证强 X 异常 run 按 origin advance 收缩且仍包含 tight。"""

    line, geometry = _line_fixture(source_index=0, baseline=20.0, loose_width=12.0)
    plan = build_document_geometry_plan([[line]], [geometry], [(400.0, 100.0)])

    assert len(plan.char_repairs) == 40
    repair = plan.char_repairs[(0, 0)]
    assert repair.x_state == "abnormal"
    assert repair.layout_bbox[2] <= geometry.origins[1][0]
    assert repair.layout_bbox[0] <= repair.tight_bbox[0]
    assert repair.layout_bbox[2] >= repair.tight_bbox[2]
    assert plan.line_repairs[(0, 0)].state == "repair_x"


def test_canonical_line_metrics_propagate_without_y_bbox_rewrite() -> None:
    """验证只发生 X 修复的行仍获得 tight 字形并集和 dominant origin 基线。"""

    line, geometry = _line_fixture(
        source_index=0,
        baseline=20.0,
        loose_width=12.0,
    )
    plan = build_document_geometry_plan(
        [[line]],
        [geometry],
        [(400.0, 100.0)],
    )

    apply_line_geometry_repairs(
        [line],
        page_index=0,
        plan=plan,
        allow_y_trim=False,
    )

    assert line.baseline == 20.0
    assert line.ink_bbox == (
        10.5,
        13.0,
        249.0,
        20.0,
    )
    assert line.geometry_state == "repair_x"


def test_normal_monospace_run_keeps_identity_geometry() -> None:
    """验证 fixed cell 与 origin advance 一致时不会被判为异常。"""

    line, geometry = _line_fixture(source_index=0, baseline=20.0, loose_width=6.0)
    plan = build_document_geometry_plan([[line]], [geometry], [(400.0, 100.0)])

    assert plan.char_repairs == {}
    assert (0, 0) not in plan.line_repairs


def test_zero_rotation_ignores_untrusted_loose_side_map_shadow() -> None:
    """验证零旋转字符只使用原始 bbox，不让密集 side-map 扰动改变布局计划。"""

    line, geometry = _line_fixture(source_index=0, baseline=20.0, loose_width=6.0)
    perturbed = {
        char_idx: (bbox[0], bbox[1] - 20.0, bbox[2] + 30.0, bbox[3] + 20.0)
        for char_idx, bbox in geometry.loose_bboxes.items()
    }
    geometry.loose_bboxes.clear()
    geometry.loose_bboxes.update(perturbed)

    plan = build_document_geometry_plan([[line]], [geometry], [(400.0, 100.0)])

    assert plan.char_repairs == {}
    assert plan.line_repairs == {}


def test_rotated_char_rejects_implausible_side_map_x_expansion() -> None:
    """验证旋转字符的 loose 宽度远超原始和 tight 框时回退稳定原始几何。"""

    line, geometry = _line_fixture(source_index=0, baseline=20.0, loose_width=6.0)
    for char in line.chars:
        char["rotation"] = 0.2
    perturbed = {
        char_idx: (bbox[0], bbox[1], bbox[2] + 30.0, bbox[3])
        for char_idx, bbox in geometry.loose_bboxes.items()
    }
    geometry.loose_bboxes.clear()
    geometry.loose_bboxes.update(perturbed)

    plan = build_document_geometry_plan([[line]], [geometry], [(400.0, 100.0)])

    assert plan.char_repairs == {}
    assert plan.line_repairs == {}


def test_missing_extended_geometry_is_exact_identity() -> None:
    """验证 tight/origin 缺失时完全沿用 legacy loose 行。"""

    line, geometry = _line_fixture(source_index=0, baseline=20.0)
    empty = PDFPageTextGeometry(chars=geometry.chars, tight_bboxes={}, origins={}, loose_bboxes=geometry.loose_bboxes)
    original = line.bbox
    plan = build_document_geometry_plan([[line]], [empty], [(400.0, 100.0)])

    apply_line_geometry_repairs([line], page_index=0, plan=plan, allow_y_trim=True)

    assert line.bbox == original
    assert line.geometry_state == "healthy"


def test_repeated_neighbor_intrusion_trims_only_y() -> None:
    """验证同 run 多行 loose 侵入邻行 tight core 时仅裁剪 Y。"""

    lines = []
    geometries = []
    for index, baseline in enumerate((20.0, 32.0, 44.0, 56.0)):
        line, geometry = _line_fixture(
            source_index=index,
            baseline=baseline,
            count=12,
            loose_width=6.0,
            loose_top=baseline - 8.0,
            loose_bottom=baseline + 9.0,
            start_char_idx=index * 20,
        )
        lines.append(line)
        geometries.append(geometry)
    geometry = _merge_geometries(*geometries)
    plan = build_document_geometry_plan([lines], [geometry], [(200.0, 100.0)])

    trimmed = [repair for repair in plan.line_repairs.values() if repair.state == "trim_y"]
    assert len(trimmed) >= 3
    for repair in trimmed:
        assert repair.layout_bbox[0] == repair.source_bbox[0]
        assert repair.layout_bbox[2] == repair.source_bbox[2]
        assert repair.layout_bbox[3] - repair.layout_bbox[1] < repair.source_bbox[3] - repair.source_bbox[1]
        assert repair.ink_bbox is not None
        assert repair.layout_bbox[1] <= repair.ink_bbox[1]
        assert repair.layout_bbox[3] >= repair.ink_bbox[3]


def test_split_y_is_shadow_only() -> None:
    """验证多基线 legacy line 只记录 split 候选而不改变输出。"""

    line, geometry = _line_fixture(
        source_index=0,
        baseline=20.0,
        count=8,
        loose_width=6.0,
        split_baseline=32.0,
    )
    original = line.bbox
    plan = build_document_geometry_plan([[line]], [geometry], [(200.0, 100.0)])

    assert plan.line_repairs[(0, 0)].split_y_candidate is True
    apply_line_geometry_repairs([line], page_index=0, plan=plan, allow_y_trim=True)
    assert line.bbox == original


def test_y_trim_is_not_applied_to_formula_candidate() -> None:
    """验证公式候选行不会进入生产 Y trim。"""

    lines = []
    geometries = []
    for index, baseline in enumerate((20.0, 32.0, 44.0, 56.0)):
        line, geometry = _line_fixture(
            source_index=index,
            baseline=baseline,
            count=12,
            loose_bottom=baseline + 9.0,
            start_char_idx=index * 20,
        )
        line.formula_candidate_only = True
        lines.append(line)
        geometries.append(geometry)
    plan = build_document_geometry_plan([lines], [_merge_geometries(*geometries)], [(200.0, 100.0)])

    assert all(repair.state != "trim_y" for repair in plan.line_repairs.values())


def test_strong_bad_font_family_propagates_to_supported_sibling_run() -> None:
    """验证已确认异常字体族只向仍有 overlap 证据的 sibling 传播。"""

    strong, strong_geometry = _line_fixture(
        source_index=0,
        baseline=20.0,
        count=40,
        loose_width=12.0,
        font_name="ABCDEF+SharedFont",
        font_size=10.0,
    )
    sibling, sibling_geometry = _line_fixture(
        source_index=1,
        baseline=40.0,
        count=12,
        loose_width=7.2,
        font_name="UVWXYZ+SharedFont",
        font_size=12.0,
        start_char_idx=100,
    )
    plan = build_document_geometry_plan(
        [[strong, sibling]],
        [_merge_geometries(strong_geometry, sibling_geometry)],
        [(400.0, 100.0)],
    )

    diagnostics = {tuple(item["run_key"]): item for item in plan.run_diagnostics}
    strong_item = next(item for key, item in diagnostics.items() if key[1] == 10.0)
    sibling_item = next(item for key, item in diagnostics.items() if key[1] == 12.0)
    assert strong_item["strong_x_bad"] is True
    assert sibling_item["sibling_x_bad"] is True
    assert (0, 100) in plan.char_repairs


def test_versioned_mixed_text_fixture_keeps_normal_geometry_identity() -> None:
    """验证版本化中英混排与等宽字体样本不会误触发生产修复。"""

    project_root = Path(__file__).parents[2]
    fixture = project_root / "tests" / "unittest" / "pdfs" / "flash_layout" / "mixed_text_layout_sample.pdf.xor"
    encrypted_bytes = fixture.read_bytes()
    assert hashlib.sha256(encrypted_bytes).hexdigest() == "fc3150c68c9f88b78d10dd2c321871f63a2e08be2099528fe00cf62d8a17e3d3"
    assert not encrypted_bytes.startswith(b"%PDF-")
    pdf_bytes = _read_pdf_fixture(fixture)
    assert hashlib.sha256(pdf_bytes).hexdigest() == "87d98f6f4e42152c38b0a1f8dcfefa3e5ac163cc086a6f37169cabadf77bd107"

    diagnostics: list[dict[str, object]] = []
    with pipeline.PDFDocument(pdf_bytes) as document:
        pages = pipeline._analyze_native_document(document, geometry_diagnostics=diagnostics)  # noqa: SLF001

    geometry = diagnostics[0]
    assert not any(
        run["strong_x_bad"] or run["sibling_x_bad"]  # type: ignore[index]
        for run in geometry["run_diagnostics"]  # type: ignore[index]
    )
    assert not any(
        line["state"] != "healthy"  # type: ignore[index]
        for line in geometry["line_repairs"]  # type: ignore[index]
    )
    for page_index, section_text in (
        (4, "1 前言"),
        (5, "2 准备工作"),
        (8, "3 工具安装"),
        (20, "4 常见问题"),
        (22, "5 附录："),
    ):
        section = next(
            block
            for block in pages[page_index]
            if section_text in str(block.get("content"))
        )
        security = next(
            block
            for block in pages[page_index]
            if "文档密级：秘密" in str(block.get("content"))
        )
        assert section["type"] == "paragraph_title"
        assert security["type"] == "header"


def test_flash_layout_manifest_uses_portable_repository_paths() -> None:
    """验证版本化 layout manifest 不保存主机绝对路径且页数完整。"""

    project_root = Path(__file__).parents[2]
    payload = json.loads((project_root / "tests" / "fixtures" / "flash_layout_geometry_manifest.json").read_text())
    assert payload["schema_version"] == 1
    assert len(payload["documents"]) == 19
    assert sum(len(document["pages"]) for document in payload["documents"]) == 168
    for document in payload["documents"]:
        path = Path(document["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert hashlib.sha256((project_root / path).read_bytes()).hexdigest() == document["sha256"]
