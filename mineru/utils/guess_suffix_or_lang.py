from magika import Magika


DEFAULT_LANG = "txt"
magika = Magika()

def guess_language_by_text(code):
    codebytes = code.encode(encoding="utf-8")
    lang = magika.identify_bytes(codebytes).prediction.output.label
    return lang if lang != "unknown" else DEFAULT_LANG


def guess_suffix_by_bytes(file_bytes) -> str:
    suffix = magika.identify_bytes(file_bytes).prediction.output.label
    return suffix


def guess_suffix_by_path(file_path) -> str:
    suffix = magika.identify_path(file_path).prediction.output.label
    return suffix