from pathlib import Path

from magika import Magika


DEFAULT_LANG = "txt"
magika = Magika()

def guess_language_by_text(code):
    codebytes = code.encode(encoding="utf-8")
    lang = magika.identify_bytes(codebytes).prediction.output.label
    return lang if lang != "unknown" else DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes, file_path=None) -> str:
    # magika 识别 PDF-1.4 需 16 字节，32 字节不识别，PDF-1.6 需 32 字节，64 字节能兼容这两种格式
    # 但是 PDF-1.3 识别不了表现和 1.4/1.6不一样，而且同样是1.3，发现不一样，有的128能识别，有的不能
    # 但是发现所有的 pdf 文件头，都是以 b'%PDF-' 开头
    if file_bytes.startswith(b'%PDF-'):
        return "pdf"
    suffix = magika.identify_bytes(file_bytes).prediction.output.label
    if file_path and suffix in ["ai"] and Path(file_path).suffix.lower() in [".pdf"]:
        suffix = "pdf"
    return suffix


def guess_suffix_by_path(file_path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    suffix = magika.identify_path(file_path).prediction.output.label
    if suffix in ["ai"] and file_path.suffix.lower() in [".pdf"]:
        suffix = "pdf"
    return suffix