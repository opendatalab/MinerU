from magika import Magika


DEFAULT_LANG = "txt"
magika = Magika()

def guess_language(code):
    codebytes = code.encode(encoding="utf-8")
    lang = magika.identify_bytes(codebytes).prediction.output.label
    return lang if lang != "unknown" else DEFAULT_LANG