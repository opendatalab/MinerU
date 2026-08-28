from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_html_css


def test_committed_minified_css_matches_readable_source() -> None:
    """验证仓库提交的 min 产物始终由当前可读源码确定性生成。"""
    project_root = Path(__file__).resolve().parents[2]
    resource_root = project_root / "mineru" / "resources" / "html"
    source = resource_root.joinpath("mineru.css").read_text(encoding="utf-8")
    minified = resource_root.joinpath("mineru.min.css").read_text(encoding="utf-8")

    assert minified == build_html_css.minify_css(source)


def test_visual_bodies_captions_and_footnotes_align_left() -> None:
    """验证视觉主体与说明统一贴正文左边，长说明也保持左对齐。"""
    project_root = Path(__file__).resolve().parents[2]
    css_path = project_root.joinpath("mineru", "resources", "html", "mineru.css")
    source = css_path.read_text(encoding="utf-8")

    assert "width: fit-content" not in source
    assert (
        ".mineru-document .mineru-figure > img,\n"
        ".mineru-document .mineru-visual-body > img {\n  display: block;\n  margin-inline: 0;\n}"
    ) in source
    assert ".mineru-document .mineru-flowchart {\n  margin: 1rem 0;" in source
    assert ".mineru-document .mineru-flowchart-canvas {\n  display: none;\n  min-width: 0;\n  text-align: left;" in source
    assert (
        ".mineru-document .mineru-caption {\n"
        "  color: var(--mineru-muted);\n"
        "  font-size: 0.9em;\n"
        "  margin-top: 0.5rem;\n"
        "  text-align: left;\n"
        "}"
    ) in source
    assert (
        ".mineru-document .mineru-footnote,\n"
        ".mineru-document .mineru-page-footnote {\n"
        "  color: var(--mineru-muted);\n"
        "  font-size: 0.875em;\n"
        "  margin-top: 0.4rem;\n"
        "  text-align: left;\n"
        "}"
    ) in source


def test_minify_css_preserves_strings_escapes_and_calc_spacing() -> None:
    """验证压缩过程只删除安全空白，不破坏字符串、转义和 calc 运算符。"""
    source = r"""
/* removable */
@media screen and (max-width: 40rem) {
  .demo::before {
    content: "a /* not comment */ b";
    font-family: "Courier New";
    width: calc(100% - 2rem);
  }
  .escaped\ class {
    --label: 'x\' y';
  }
}
"""

    expected = (
        '@media screen and (max-width:40rem){.demo::before{content:"a /* not comment */ b";'
        'font-family:"Courier New";width:calc(100% - 2rem)}'
        ".escaped\\ class{--label:'x\\' y'}}"
    )
    assert build_html_css.minify_css(source) == expected


def test_minify_css_preserves_descendant_combinator_before_pseudo_class() -> None:
    """验证伪类前的后代空格不会被误当作声明冒号旁的冗余空白。"""
    source = ".root :is(h1, h2) { color: red; } .root:hover { color: blue; }"

    assert build_html_css.minify_css(source) == (".root :is(h1,h2){color:red}.root:hover{color:blue}")


@pytest.mark.parametrize(
    "source",
    [
        "a { color: red;",
        'a { content: "unterminated; }',
        "a { color: red; /* unterminated",
        "a { color: red; }}",
        "a { content: \\",
    ],
)
def test_minify_css_rejects_unterminated_or_unbalanced_input(source: str) -> None:
    """验证生成器拒绝未闭合字符串、注释、转义和规则块。"""
    with pytest.raises(ValueError):
        build_html_css.minify_css(source)


def test_build_html_css_check_detects_and_repairs_stale_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 check 模式无写入地报告过期产物，普通模式再原子修复。"""
    source_path = tmp_path / "mineru.css"
    output_path = tmp_path / "mineru.min.css"
    source_path.write_text(".demo { color: red; }\n", encoding="utf-8")
    output_path.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(build_html_css, "_SOURCE_PATH", source_path)
    monkeypatch.setattr(build_html_css, "_OUTPUT_PATH", output_path)

    assert build_html_css.build_html_css(check=True) is False
    assert output_path.read_text(encoding="utf-8") == "stale"
    assert build_html_css.build_html_css(check=False) is True
    assert output_path.read_text(encoding="utf-8") == ".demo{color:red}"
    assert build_html_css.build_html_css(check=True) is True
