"""以真实原生文档验证批量输出、逐项失败和全局终止的边界。"""

from __future__ import annotations

import errno
from pathlib import Path
from shutil import copyfile
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import pytest
from typer.testing import CliRunner

from mineru.errors import InvalidRequestError
from mineru.kit.batch_parse import validate_parse_input
from mineru.kit.commands import parse
from mineru.kit.common import OUTPUT_FILE_SUFFIXES, resolve_batch_output_paths
from mineru.kit.main import app
from mineru.parser import parse as local_parse
from mineru.parser.api_client import _V1APIError


def _csv(path: Path) -> Path:
    """生成无需模型或网络即可解析的真实 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"name,value\n{path.stem},1\n", encoding="utf-8")
    return path


def _invoke(inputs: list[Path], output: Path, *args: str) -> Any:
    """通过公开 Kit CLI 执行同一组输入。"""
    return CliRunner().invoke(app, ["parse", *map(str, inputs), "-o", str(output), "--tier", "flash", *args])


@pytest.mark.parametrize("output_format", ["markdown", "middle_json", "zip"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("name", ["results.v1", "results.md"])
def test_batch_accepts_new_and_existing_dotted_directories(
    tmp_path: Path, output_format: str, existing: bool, name: str
) -> None:
    """只有一个文档的目录也固定使用目录输出，名称和格式相互独立。"""
    source = _csv(tmp_path / "docs" / "one.csv")
    output = tmp_path / name
    if existing:
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    result = _invoke([source.parent], output, "-f", output_format)
    assert result.exit_code == 0, result.output
    assert (output / f"one{OUTPUT_FILE_SUFFIXES[output_format]}").is_file()
    if existing:
        assert (output / "keep.txt").read_text() == "keep"
    assert "succeeded=1" in result.output


@pytest.mark.parametrize("name", ["result", "result.md"])
def test_directory_never_overwrites_regular_output_file(tmp_path: Path, name: str) -> None:
    """目录只剩一个输入也不能套用单文件覆盖规则。"""
    source = _csv(tmp_path / "docs" / "one.csv")
    output = tmp_path / name
    output.write_text("keep")
    result = _invoke([source.parent], output)
    assert result.exit_code == 1
    assert output.read_text() == "keep"
    assert "directory path" in " ".join(result.output.split())
    single = _invoke([source], output)
    assert single.exit_code == 0, single.output
    assert "one" in output.read_text()


def test_directory_uses_content_not_office_owner_filename(tmp_path: Path) -> None:
    """合法特殊名称保留，任何命名的非文档均由内容识别跳过。"""
    source = tmp_path / "docs"
    source.mkdir()
    fixture = Path(__file__).parents[1] / "fixtures/pr5389/table-links.docx"
    copyfile(fixture, source / "~$real.docx")
    (source / "~$owner.docx").write_bytes(b"owner lock metadata")
    (source / "normal-name.docx").write_bytes(b"owner lock metadata")
    _csv(source / "last.csv")
    output = tmp_path / "out"
    result = _invoke([source], output)
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in output.iterdir()) == ["last.md", "~$real.md"]
    assert "https://example.org/one" in (output / "~$real.md").read_text()
    assert "succeeded=2, skipped=2, failed=0" in " ".join(result.output.split())
    assert (source / "~$owner.docx").exists()


def test_corrupt_document_does_not_prevent_later_success(tmp_path: Path) -> None:
    """可识别但损坏的 PDF 记录失败，后续 CSV 仍生成真实产物。"""
    source = _csv(tmp_path / "docs" / "b.csv")
    (source.parent / "a.pdf").write_bytes(b"%PDF-1.7\ntruncated")
    output = tmp_path / "out"
    result = _invoke([source.parent], output)
    assert result.exit_code == 1, result.output
    assert (output / "b.md").exists()
    assert "file_corrupted" in result.output
    assert "succeeded=1, skipped=0, failed=1, unprocessed=0" in " ".join(result.output.split())


def test_encrypted_pdf_is_a_reported_input_failure(tmp_path: Path) -> None:
    """真实密码保护 PDF 记录为加密输入，后续原生文档继续处理。"""
    source = _csv(tmp_path / "docs" / "b.csv")
    copyfile(Path(__file__).parents[1] / "fixtures/kit/encrypted.pdf", source.parent / "a.pdf")
    result = _invoke([source.parent], tmp_path / "out")
    assert result.exit_code == 1, result.output
    assert "file_encrypted" in result.output
    assert (tmp_path / "out" / "b.md").exists()


def test_malformed_office_xml_is_reported_and_later_document_runs(tmp_path: Path) -> None:
    """真实 DOCX 容器中的损坏正文 XML 属于输入失败，不阻断后续文件。"""
    source = _csv(tmp_path / "docs" / "b.csv")
    fixture = Path(__file__).parents[1] / "fixtures/pr5389/table-links.docx"
    with ZipFile(fixture) as original, ZipFile(source.parent / "a.docx", "w", ZIP_DEFLATED) as broken:
        for item in original.infolist():
            body = b"<w:document" if item.filename == "word/document.xml" else original.read(item.filename)
            broken.writestr(item, body)
    result = _invoke([source.parent], tmp_path / "out")
    assert result.exit_code == 1, result.output
    assert "file_corrupted" in result.output
    assert (tmp_path / "out" / "b.md").exists()


def test_missing_explicit_batch_input_is_reported_and_remaining_input_runs(tmp_path: Path) -> None:
    """多文件模式隔离单项不存在错误，保留成功文件并非零退出。"""
    source = _csv(tmp_path / "good.csv")
    result = _invoke([tmp_path / "missing.csv", source], tmp_path / "out")
    assert result.exit_code == 1
    assert (tmp_path / "out" / "good.md").exists()
    assert "file_not_found" in result.output


def test_stat_permission_error_is_not_reported_as_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """路径属性查询也保留权限错误，避免 Path.is_file 吞掉异常后误报不存在。"""
    source = _csv(tmp_path / "secret.csv")
    original = Path.stat

    def denied(path: Path, **kwargs: Any) -> Any:
        """只拒绝该输入的属性读取，其他文件系统调用仍使用真实实现。"""
        if path == source:
            raise PermissionError(errno.EACCES, "denied", str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(InvalidRequestError) as raised:
        validate_parse_input(source)
    assert raised.value.code == "file_permission_denied"


@pytest.mark.parametrize(
    "code,remaining", [("file_encrypted", True), ("engine_dead", False), ("invalid_api_key", False), ("parse_failed", False)]
)
def test_remote_preserves_error_codes_and_batch_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, remaining: bool
) -> None:
    """远程只隔离明确输入错误；通用失败不能被猜测为损坏文档。"""
    first = _csv(tmp_path / "a.csv")
    last = _csv(tmp_path / "b.csv")
    calls: list[str] = []

    class Parser:
        def __init__(self, **kwargs: Any) -> None:
            """替换远程传输，保留相同的公开错误契约。"""

        def parse(self, path: Path, *, page_range: str) -> Any:
            """先返回指定错误，后续文件通过真实原生解析生成产物。"""
            calls.append(path.name)
            if path == first:
                raise _V1APIError(code, "detail", "input")
            return local_parse(path, tier="flash")

    monkeypatch.setattr(parse, "MinerUApiParser", Parser)
    result = _invoke([first, last], tmp_path / "out", "--remote-url", "https://example.invalid")
    assert result.exit_code == 1
    assert code in result.output
    assert calls == (["a.csv", "b.csv"] if remaining else ["a.csv"])
    assert (tmp_path / "out" / "b.md").exists() is remaining


def test_remote_does_not_upload_recognized_non_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """远程分流之前进行相同的内容识别，避免为非文档创建服务任务。"""
    source = tmp_path / "owner.docx"
    source.write_bytes(b"not a document")

    class Parser:
        def __init__(self, **kwargs: Any) -> None:
            """允许构造轻量客户端，但禁止请求上传。"""

        def parse(self, path: Path, *, page_range: str) -> Any:
            """已确认的非文档不应到达远程解析器。"""
            pytest.fail("must not upload")

    monkeypatch.setattr(parse, "MinerUApiParser", Parser)
    result = _invoke([source], tmp_path / "out.md", "--remote-url", "https://example.invalid")
    assert result.exit_code == 1
    assert "Unsupported file type" in result.output
    assert not (tmp_path / "out.md").exists()


@pytest.mark.parametrize("failure", [ValueError("program contract"), RuntimeError("unknown engine error")])
def test_unknown_failures_abort_instead_of_silently_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """程序错误和未知引擎故障不属于可隔离的单文件错误。"""
    sources = [_csv(tmp_path / name) for name in ["a.csv", "b.csv"]]
    calls: list[Path] = []

    def fail(path: Path, **kwargs: Any) -> Any:
        """第一项触发未知故障，后续任务不得开始。"""
        calls.append(path)
        raise failure

    monkeypatch.setattr(parse, "local_parse", fail)
    result = _invoke(sources, tmp_path / "out")
    assert result.exit_code == 1
    assert calls == sources[:1]
    assert "unprocessed=1" in result.output


def test_write_failure_stops_batch_and_preserves_prior_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """磁盘故障终止后续写入，保留已经完成的文档输出。"""
    sources = [_csv(tmp_path / name) for name in ["a.csv", "b.csv", "c.csv"]]
    save = parse.save_parse_result

    def save_or_fail(result: Any, dest: Path, format: str) -> None:
        """第二次写入模拟磁盘空间不足。"""
        if dest.stem == "b":
            raise OSError(errno.ENOSPC, "no space")
        save(result, dest, format)

    monkeypatch.setattr(parse, "save_parse_result", save_or_fail)
    result = _invoke(sources, tmp_path / "out")
    assert result.exit_code == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.md"]
    assert "succeeded=1, skipped=0, failed=1, unprocessed=1" in " ".join(result.output.split())


def test_all_skipped_is_not_reported_as_success(tmp_path: Path) -> None:
    """空结果批次即使只有跳过项也必须非零退出。"""
    source = tmp_path / "docs"
    source.mkdir()
    (source / "owner.docx").write_bytes(b"lock")
    result = _invoke([source], tmp_path / "out")
    assert result.exit_code == 1
    assert "succeeded=0, skipped=1" in result.output


def test_output_planning_rejects_collision_before_creating_outputs(tmp_path: Path) -> None:
    """目标映射冲突在执行之前拒绝，且不能写回任何源文件。"""
    first = _csv(tmp_path / "one" / "a.csv")
    second = _csv(tmp_path / "two" / "a.csv")
    result = _invoke([first, second], tmp_path / "out")
    assert result.exit_code == 1
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="same as the input"):
        resolve_batch_output_paths([first], first, "markdown", batch=False)
