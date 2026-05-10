from mineru.direct_flow_processor import DirectFlowProcessor

DEMO_DOCX = "demo/office_docs/docx_01.docx"
DEMO_PPTX = "demo/office_docs/pptx_01.pptx"
DEMO_XLSX = "demo/office_docs/xlsx_01.xlsx"
DEMO_PDF = "/home/aidi/devtest/IMPORTANT_FORMULAE_FOR_COMPETITIVE_EXAMS.pdf"

# VLM server URL when using the http-client backend
# Example: running vLLM/SGLang locally
VLM_SERVER_URL = "http://127.0.0.1:8000"


def test_docx():
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_office(DEMO_DOCX)
    print(f"[docx] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[docx] saved: {list(saved.keys())}")


def test_pptx():
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_office(DEMO_PPTX)
    print(f"[pptx] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pptx] saved: {list(saved.keys())}")


def test_xlsx():
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_office(DEMO_XLSX)
    print(f"[xlsx] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[xlsx] saved: {list(saved.keys())}")


def test_pdf_pipeline():
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_pipeline(
        "/home/tomkey/aidi/nbc/MinerU/demo/pdfs/demo3.pdf",
        parse_method="auto",
        language="lt",
        formula_enable=True,
        table_enable=True,
        draw_layout=True,
        draw_span=True,
    )
    print(f"[pdf/pipeline] flow={result.flow} pages={len(result.pdf_info)}")
    print(f"[pdf/pipeline] saved: {list(saved.keys())}")


def test_pdf_vlm():
    """VLM flow with a local model (requires GPU + downloaded model)."""
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_vlm(
        DEMO_PDF,
        backend="auto-engine",   # Automatically detects the appropriate engine (huggingface / lmdeploy / ...)
        language="en",
        draw_layout=True,
    )
    print(f"[pdf/vlm] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pdf/vlm] saved: {list(saved.keys())}")


def test_pdf_vlm_http():
    """VLM flow via HTTP client (remote VLM server, no local GPU required).

    Requires the VLM server to be running at VLM_SERVER_URL.
    Example server startup: ./run.sh api --host 127.0.0.1 --port 8000
    """
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_vlm(
        DEMO_PDF,
        backend="http-client",
        server_url=VLM_SERVER_URL,
        language="en",
    )
    print(f"[pdf/vlm-http] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pdf/vlm-http] saved: {list(saved.keys())}")


def test_pdf_hybrid():
    """Hybrid flow = VLM + OCR refinement (requires GPU + downloaded model).

    Better than pure VLM when documents contain many mathematical formulas.
    """
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_hybrid(
        DEMO_PDF,
        backend="auto-engine",
        parse_method="auto",
        language="en",
        inline_formula_enable=True,
        draw_layout=True,
    )
    print(f"[pdf/hybrid] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pdf/hybrid] vlm_ocr_enabled={result.vlm_ocr_enabled}")
    print(f"[pdf/hybrid] saved: {list(saved.keys())}")


def test_pdf_hybrid_http():
    """Hybrid flow via HTTP client (remote VLM + local pipeline OCR).

    Requires:
    - VLM server running at VLM_SERVER_URL
    - `mineru[pipeline]` installed locally
    """
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_hybrid(
        DEMO_PDF,
        backend="http-client",
        server_url=VLM_SERVER_URL,
        parse_method="auto",
        language="en",
        inline_formula_enable=True,
    )
    print(f"[pdf/hybrid-http] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pdf/hybrid-http] vlm_ocr_enabled={result.vlm_ocr_enabled}")
    print(f"[pdf/hybrid-http] saved: {list(saved.keys())}")


if __name__ == "__main__":
    # --- Office flows ---
    # test_docx()
    # test_pptx()
    # test_xlsx()

    test_pdf_pipeline()

    # --- PDF VLM requires GPU + local model ---
    # test_pdf_vlm()

    # --- PDF VLM via HTTP client ---
    # test_pdf_vlm_http()

    # --- PDF Hybrid / local model requires GPU ---
    # test_pdf_hybrid()

    # --- PDF Hybrid via HTTP client (remote VLM + local pipeline OCR) ---
    # test_pdf_hybrid_http()