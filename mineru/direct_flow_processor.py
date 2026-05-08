# Copyright (c) Opendatalab. All rights reserved.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import (
    convert_pdf_bytes_to_bytes,
    ensure_backend_dependencies,
    normalize_task_stem,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path

pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]
office_suffixes = ["docx", "pptx", "xlsx"]


@dataclass
class PreparedInput:
    """Normalized input bundle shared by all direct processing flows.

    Attributes:
        source_path: Absolute path to the original user-provided file.
        stem: Normalized task stem used to build output filenames.
        original_suffix: Guessed source suffix before any conversion.
        original_bytes: Raw bytes read from the source file.
        document_bytes: Bytes returned by MinerU's input reader.
        processed_bytes: Final PDF bytes after optional image conversion and page slicing.
        output_root: Root output directory for this run.
        output_dir: Flow-specific output directory.
        image_dir: Directory where extracted/cropped images are written.
        flow: Selected direct flow, such as ``pipeline``, ``vlm``, or ``hybrid``.
        parse_method: Parse mode forwarded to the backend when relevant.
    """
    source_path: Path
    stem: str
    original_suffix: str
    original_bytes: bytes
    document_bytes: bytes
    processed_bytes: bytes
    output_root: Path
    output_dir: Path
    image_dir: Path
    flow: str
    parse_method: str


@dataclass
class DirectFlowResult:
    """Backend result bundle returned by direct flow execution.

    This object carries both the structured MinerU outputs and the metadata
    needed to render or persist final artifacts without going through the API
    layer.
    """
    flow: str
    backend: str
    parse_method: str
    language: str
    prepared_input: PreparedInput
    middle_json: dict[str, Any]
    model_output: Any
    processed_bytes: bytes
    pdf_info: list[dict[str, Any]]
    ocr_enabled: bool | None = None
    vlm_ocr_enabled: bool | None = None


class DirectFlowProcessor:
    """Direct wrapper around MinerU backend flows without going through API.

    The public methods are intentionally grouped into four responsibilities:

    1. processing input
    2. process ocr / backend analysis
    3. post process
    4. build/save outputs
    """

    def __init__(self, output_root: str | Path = "./output") -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    # ------------------------------------------------------------------
    # Group 1: processing input
    # ------------------------------------------------------------------
    def process_input(
        self,
        input_path: str | Path,
        *,
        flow: str,
        parse_method: str = "auto",
        start_page_id: int = 0,
        end_page_id: int | None = None,
        output_root: str | Path | None = None,
    ) -> PreparedInput:
        """Read and normalize a local input file for direct backend execution.

        The method accepts a PDF or image file, uses MinerU's native readers to
        produce document bytes, optionally slices the requested page range, and
        prepares the output directories shared by later steps.
        """
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file not found: {source_path}")

        original_suffix = guess_suffix_by_path(source_path)
        if original_suffix not in pdf_suffixes + image_suffixes:
            raise ValueError(
                f"Unsupported input suffix for direct {flow} flow: {source_path.suffix}"
            )

        original_bytes = source_path.read_bytes()
        document_bytes = read_fn(source_path, file_suffix=original_suffix)
        processed_bytes = convert_pdf_bytes_to_bytes(
            document_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )

        effective_output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else self.output_root
        )
        stem = normalize_task_stem(source_path.stem)
        flow_dir_name = self._build_flow_dir_name(flow, parse_method)
        output_dir = effective_output_root / stem / flow_dir_name
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        return PreparedInput(
            source_path=source_path,
            stem=stem,
            original_suffix=original_suffix,
            original_bytes=original_bytes,
            document_bytes=document_bytes,
            processed_bytes=processed_bytes,
            output_root=effective_output_root,
            output_dir=output_dir,
            image_dir=image_dir,
            flow=flow,
            parse_method=parse_method,
        )

    # ------------------------------------------------------------------
    # Group 2: process ocr / backend analysis
    # ------------------------------------------------------------------
    def process_pipeline_ocr(
        self,
        prepared_input: PreparedInput,
        *,
        language: str = "ch",
        parse_method: str | None = None,
        formula_enable: bool = True,
        table_enable: bool = True,
    ) -> DirectFlowResult:
        """Run the classic pipeline flow directly against prepared PDF bytes.

        This path delegates to ``pipeline.doc_analyze_streaming`` and captures
        the callback payload that normally feeds the CLI/API output stage.
        """
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming

        parse_method = parse_method or prepared_input.parse_method
        image_writer = FileBasedDataWriter(str(prepared_input.image_dir))
        captured: dict[str, Any] = {}

        def on_doc_ready(
            doc_index: int,
            model_list: Any,
            middle_json: dict[str, Any],
            ocr_enable: bool,
        ) -> None:
            del doc_index
            captured["model_output"] = model_list
            captured["middle_json"] = middle_json
            captured["ocr_enabled"] = ocr_enable

        doc_analyze_streaming(
            [prepared_input.processed_bytes],
            [image_writer],
            [language],
            on_doc_ready,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )

        return self.post_process(
            DirectFlowResult(
                flow="pipeline",
                backend="pipeline",
                parse_method=parse_method,
                language=language,
                prepared_input=prepared_input,
                middle_json=captured["middle_json"],
                model_output=captured["model_output"],
                processed_bytes=prepared_input.processed_bytes,
                pdf_info=captured["middle_json"]["pdf_info"],
                ocr_enabled=captured.get("ocr_enabled"),
            )
        )

    def process_vlm_ocr(
        self,
        prepared_input: PreparedInput,
        *,
        language: str = "ch",
        backend: str = "auto-engine",
        server_url: str | None = None,
        **kwargs,
    ) -> DirectFlowResult:
        """Run the VLM flow directly and return the parsed document result.

        ``backend`` may be an explicit engine name or ``auto-engine``. Any
        extra keyword arguments are forwarded to the underlying VLM analyzer.
        """
        from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze

        resolved_backend = self._resolve_vlm_backend(backend)
        image_writer = FileBasedDataWriter(str(prepared_input.image_dir))
        middle_json, infer_result = vlm_doc_analyze(
            prepared_input.processed_bytes,
            image_writer=image_writer,
            backend=resolved_backend,
            server_url=server_url,
            **kwargs,
        )

        return self.post_process(
            DirectFlowResult(
                flow="vlm",
                backend=resolved_backend,
                parse_method="vlm",
                language=language,
                prepared_input=prepared_input,
                middle_json=middle_json,
                model_output=infer_result,
                processed_bytes=prepared_input.processed_bytes,
                pdf_info=middle_json["pdf_info"],
            )
        )

    def process_hybrid_ocr(
        self,
        prepared_input: PreparedInput,
        *,
        language: str = "ch",
        parse_method: str | None = None,
        backend: str = "auto-engine",
        inline_formula_enable: bool = True,
        server_url: str | None = None,
        **kwargs,
    ) -> DirectFlowResult:
        """Run the hybrid flow directly using VLM plus OCR/formula refinement.

        The hybrid backend first performs VLM-based extraction and may then
        invoke OCR/formula helpers depending on the parse strategy and content.
        """
        from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze

        parse_method = parse_method or prepared_input.parse_method
        resolved_backend = self._resolve_hybrid_backend(backend)
        image_writer = FileBasedDataWriter(str(prepared_input.image_dir))
        middle_json, infer_result, vlm_ocr_enabled = hybrid_doc_analyze(
            prepared_input.processed_bytes,
            image_writer=image_writer,
            backend=resolved_backend,
            parse_method=parse_method,
            language=language,
            inline_formula_enable=inline_formula_enable,
            server_url=server_url,
            **kwargs,
        )

        return self.post_process(
            DirectFlowResult(
                flow="hybrid",
                backend=resolved_backend,
                parse_method=parse_method,
                language=language,
                prepared_input=prepared_input,
                middle_json=middle_json,
                model_output=infer_result,
                processed_bytes=prepared_input.processed_bytes,
                pdf_info=middle_json["pdf_info"],
                vlm_ocr_enabled=vlm_ocr_enabled,
            )
        )

    # ------------------------------------------------------------------
    # Group 3: post process
    # ------------------------------------------------------------------
    def post_process(self, result: DirectFlowResult) -> DirectFlowResult:
        """Validate the minimum result structure expected by output builders."""
        if not isinstance(result.middle_json, dict):
            raise TypeError("middle_json must be a dict")
        if "pdf_info" not in result.middle_json:
            raise ValueError("middle_json does not contain pdf_info")
        if not isinstance(result.pdf_info, list):
            raise TypeError("pdf_info must be a list")
        return result

    # ------------------------------------------------------------------
    # Group 4: build json / md / etc and save output
    # ------------------------------------------------------------------
    def build_output_payloads(
        self,
        result: DirectFlowResult,
        *,
        make_mode: str = MakeMode.MM_MD,
        include_md: bool = True,
        include_content_list: bool = True,
        include_content_list_v2: bool = True,
        include_middle_json: bool = True,
        include_model_output: bool = True,
        include_original_pdf: bool = True,
    ) -> dict[str, str | bytes]:
        """Build in-memory output files for a completed direct flow run.

        The returned mapping uses output filenames as keys and file contents as
        values. Text artifacts are returned as ``str`` and binary artifacts as
        ``bytes``.
        """
        make_func = self._get_make_func(result.flow)
        image_dir_name = result.prepared_input.image_dir.name
        payloads: dict[str, str | bytes] = {}

        if include_md:
            payloads[f"{result.prepared_input.stem}.md"] = make_func(
                result.pdf_info,
                make_mode,
                image_dir_name,
            )

        if include_content_list:
            payloads[f"{result.prepared_input.stem}_content_list.json"] = json.dumps(
                make_func(result.pdf_info, MakeMode.CONTENT_LIST, image_dir_name),
                ensure_ascii=False,
                indent=4,
            )

        if include_content_list_v2:
            payloads[f"{result.prepared_input.stem}_content_list_v2.json"] = json.dumps(
                make_func(result.pdf_info, MakeMode.CONTENT_LIST_V2, image_dir_name),
                ensure_ascii=False,
                indent=4,
            )

        if include_middle_json:
            payloads[f"{result.prepared_input.stem}_middle.json"] = json.dumps(
                result.middle_json,
                ensure_ascii=False,
                indent=4,
            )

        if include_model_output:
            payloads[f"{result.prepared_input.stem}_model.json"] = json.dumps(
                result.model_output,
                ensure_ascii=False,
                indent=4,
            )

        if include_original_pdf:
            origin_suffix = result.prepared_input.original_suffix if result.flow == "office" else "pdf"
            payloads[f"{result.prepared_input.stem}_origin.{origin_suffix}"] = result.processed_bytes

        return payloads

    def save_output(
        self,
        result: DirectFlowResult,
        payloads: dict[str, str | bytes] | None = None,
        *,
        draw_layout: bool = False,
        draw_span: bool = False,
    ) -> dict[str, str]:
        """Persist generated artifacts to disk and return saved file paths.

        When requested, this method also writes debug PDFs containing layout or
        span bounding boxes.
        """
        writer = FileBasedDataWriter(str(result.prepared_input.output_dir))
        payloads = payloads or self.build_output_payloads(result)
        saved_paths: dict[str, str] = {}

        for filename, data in payloads.items():
            if isinstance(data, bytes):
                writer.write(filename, data)
            else:
                writer.write_string(filename, data)
            saved_paths[filename] = str(result.prepared_input.output_dir / filename)

        if draw_layout:
            layout_name = f"{result.prepared_input.stem}_layout.pdf"
            draw_layout_bbox(
                result.pdf_info,
                result.processed_bytes,
                str(result.prepared_input.output_dir),
                layout_name,
            )
            saved_paths[layout_name] = str(result.prepared_input.output_dir / layout_name)

        if draw_span and result.flow == "pipeline":
            span_name = f"{result.prepared_input.stem}_span.pdf"
            draw_span_bbox(
                result.pdf_info,
                result.processed_bytes,
                str(result.prepared_input.output_dir),
                span_name,
            )
            saved_paths[span_name] = str(result.prepared_input.output_dir / span_name)

        return saved_paths

    # ------------------------------------------------------------------
    # Convenience dispatchers
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        input_path: str | Path,
        *,
        output_root: str | Path | None = None,
        parse_method: str = "auto",
        language: str = "lt",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page_id: int = 0,
        end_page_id: int | None = None,
        save: bool = True,
        draw_layout: bool = False,
        draw_span: bool = False,
    ) -> tuple[DirectFlowResult, dict[str, str] | None]:
        """Execute the full direct ``pipeline`` workflow for one local file."""
        prepared = self.process_input(
            input_path,
            flow="pipeline",
            parse_method=parse_method,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            output_root=output_root,
        )
        result = self.process_pipeline_ocr(
            prepared,
            language=language,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )
        saved = (
            self.save_output(result, draw_layout=draw_layout, draw_span=draw_span)
            if save
            else None
        )
        return result, saved

    def run_vlm(
        self,
        input_path: str | Path,
        *,
        output_root: str | Path | None = None,
        backend: str = "auto-engine",
        server_url: str | None = None,
        language: str = "lt",
        start_page_id: int = 0,
        end_page_id: int | None = None,
        save: bool = True,
        draw_layout: bool = False,
        **kwargs,
    ) -> tuple[DirectFlowResult, dict[str, str] | None]:
        """Execute the full direct ``vlm`` workflow for one local file."""
        prepared = self.process_input(
            input_path,
            flow="vlm",
            parse_method="vlm",
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            output_root=output_root,
        )
        result = self.process_vlm_ocr(
            prepared,
            backend=backend,
            server_url=server_url,
            language=language,
            **kwargs,
        )
        saved = self.save_output(result, draw_layout=draw_layout) if save else None
        return result, saved

    def run_hybrid(
        self,
        input_path: str | Path,
        *,
        output_root: str | Path | None = None,
        parse_method: str = "auto",
        backend: str = "auto-engine",
        server_url: str | None = None,
        language: str = "lt",
        inline_formula_enable: bool = True,
        start_page_id: int = 0,
        end_page_id: int | None = None,
        save: bool = True,
        draw_layout: bool = False,
        **kwargs,
    ) -> tuple[DirectFlowResult, dict[str, str] | None]:
        """Execute the full direct ``hybrid`` workflow for one local file."""
        prepared = self.process_input(
            input_path,
            flow="hybrid",
            parse_method=parse_method,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            output_root=output_root,
        )
        result = self.process_hybrid_ocr(
            prepared,
            parse_method=parse_method,
            backend=backend,
            server_url=server_url,
            language=language,
            inline_formula_enable=inline_formula_enable,
            **kwargs,
        )
        saved = self.save_output(result, draw_layout=draw_layout) if save else None
        return result, saved

    def run_office(
        self,
        input_path: str | Path,
        *,
        output_root: str | Path | None = None,
        save: bool = True,
    ) -> tuple[DirectFlowResult, dict[str, str] | None]:
        """Execute the office document flow for one local file (docx, pptx, xlsx)."""
        from mineru.backend.office.docx_analyze import office_docx_analyze
        from mineru.backend.office.pptx_analyze import office_pptx_analyze
        from mineru.backend.office.xlsx_analyze import office_xlsx_analyze

        source_path = Path(input_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file not found: {source_path}")

        original_suffix = guess_suffix_by_path(source_path)
        if original_suffix not in office_suffixes:
            raise ValueError(f"Unsupported suffix for office flow: {source_path.suffix}")

        file_bytes = source_path.read_bytes()

        effective_output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else self.output_root
        )
        stem = normalize_task_stem(source_path.stem)
        output_dir = effective_output_root / stem / "office"
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_writer = FileBasedDataWriter(str(image_dir))

        analyze_fn = {
            "docx": office_docx_analyze,
            "pptx": office_pptx_analyze,
            "xlsx": office_xlsx_analyze,
        }[original_suffix]

        middle_json, infer_result = analyze_fn(file_bytes, image_writer=image_writer)

        prepared = PreparedInput(
            source_path=source_path,
            stem=stem,
            original_suffix=original_suffix,
            original_bytes=file_bytes,
            document_bytes=file_bytes,
            processed_bytes=file_bytes,
            output_root=effective_output_root,
            output_dir=output_dir,
            image_dir=image_dir,
            flow="office",
            parse_method="office",
        )

        result = self.post_process(
            DirectFlowResult(
                flow="office",
                backend=original_suffix,
                parse_method="office",
                language="",
                prepared_input=prepared,
                middle_json=middle_json,
                model_output=infer_result,
                processed_bytes=file_bytes,
                pdf_info=middle_json["pdf_info"],
            )
        )

        saved = self.save_output(result) if save else None
        return result, saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_flow_dir_name(flow: str, parse_method: str) -> str:
        """Return the output subdirectory name used for a given flow."""
        if flow == "pipeline":
            return parse_method
        if flow == "vlm":
            return "vlm"
        if flow == "hybrid":
            return f"hybrid_{parse_method}"
        raise ValueError(f"Unsupported flow: {flow}")

    @staticmethod
    def _get_make_func(flow: str):
        """Select the content rendering function that matches the flow type."""
        if flow == "pipeline":
            return pipeline_union_make
        if flow in {"vlm", "hybrid"}:
            return vlm_union_make
        if flow == "office":
            from mineru.backend.office.office_middle_json_mkcontent import union_make as office_union_make
            return office_union_make
        raise ValueError(f"Unsupported flow: {flow}")

    @staticmethod
    def _resolve_vlm_backend(backend: str) -> str:
        """Normalize public VLM backend aliases to analyzer backend names."""
        normalized = backend
        if normalized.startswith("vlm-"):
            normalized = normalized[4:]
        if normalized == "auto-engine":
            return get_vlm_engine(inference_engine="auto", is_async=False)
        return normalized

    @staticmethod
    def _resolve_hybrid_backend(backend: str) -> str:
        """Normalize hybrid backend aliases and verify required dependencies."""
        normalized = backend
        if normalized.startswith("hybrid-"):
            normalized = normalized[7:]
        original_backend = normalized
        if original_backend == "auto-engine":
            normalized = get_vlm_engine(inference_engine="auto", is_async=False)
        ensure_backend_dependencies(f"hybrid-{original_backend}")
        return normalized
