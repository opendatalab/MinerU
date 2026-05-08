from mineru.direct_flow_processor import DirectFlowProcessor

DEMO_DOCX = "demo/office_docs/docx_01.docx"
DEMO_PPTX = "demo/office_docs/pptx_01.pptx"
DEMO_XLSX = "demo/office_docs/xlsx_01.xlsx"
DEMO_PDF = "/home/aidi/devtest/IMPORTANT_FORMULAE_FOR_COMPETITIVE_EXAMS.pdf"

# URL server VLM khi dùng backend http-client (ví dụ: chạy vLLM/SGLang cục bộ)
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
        "/home/aidi/devtest/IMPORTANT_FORMULAE_FOR_COMPETITIVE_EXAMS.pdf",
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
    """VLM flow với local model (cần GPU + model đã download)."""
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_vlm(
        DEMO_PDF,
        backend="auto-engine",   # tự phát hiện engine phù hợp (huggingface / lmdeploy / ...)
        language="en",
        draw_layout=True,
    )
    print(f"[pdf/vlm] flow={result.flow} backend={result.backend} pages={len(result.pdf_info)}")
    print(f"[pdf/vlm] saved: {list(saved.keys())}")


def test_pdf_vlm_http():
    """VLM flow qua HTTP client (remote VLM server, không cần GPU cục bộ).

    Yêu cầu server VLM đang chạy tại VLM_SERVER_URL.
    Ví dụ khởi động server: ./run.sh api --host 127.0.0.1 --port 8000
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
    """Hybrid flow = VLM + OCR refinement (cần GPU + model đã download).

    Tốt hơn vlm thuần khi tài liệu có nhiều công thức toán.
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
    """Hybrid flow qua HTTP client (remote VLM + local pipeline OCR).

    Yêu cầu: server VLM tại VLM_SERVER_URL + `mineru[pipeline]` cài cục bộ.
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

    # --- PDF VLM needs GPU + local model ---
    # test_pdf_vlm()

    # --- PDF VLM to HTTP client ---
    # test_pdf_vlm_http()

    # --- PDF Hybrid / local model neads GPU ---
    # test_pdf_hybrid()

    # --- PDF Hybrid qua HTTP client (remote VLM + local pipeline OCR) ---
    # test_pdf_hybrid_http()
