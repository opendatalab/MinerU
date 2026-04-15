# Copyright (c) Opendatalab. All rights reserved.
import json
from dataclasses import dataclass
from pathlib import Path

from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox


VISUALIZATION_FINISHED = "finished"
VISUALIZATION_SKIPPED = "skipped"


@dataclass(frozen=True)
class VisualizationJob:
    document_stem: str
    backend: str
    parse_method: str
    parse_dir: Path
    draw_span: bool


@dataclass(frozen=True)
class VisualizationResult:
    document_stem: str
    parse_dir: Path
    status: str
    message: str
    generated_files: tuple[str, ...] = ()


def run_visualization_job(job: VisualizationJob) -> VisualizationResult:
    middle_json_path = job.parse_dir / f"{job.document_stem}_middle.json"
    origin_pdf_path = job.parse_dir / f"{job.document_stem}_origin.pdf"

    if not middle_json_path.exists():
        return VisualizationResult(
            document_stem=job.document_stem,
            parse_dir=job.parse_dir,
            status=VISUALIZATION_SKIPPED,
            message=f"missing middle.json: {middle_json_path.name}",
        )
    if not origin_pdf_path.exists():
        return VisualizationResult(
            document_stem=job.document_stem,
            parse_dir=job.parse_dir,
            status=VISUALIZATION_SKIPPED,
            message=f"missing origin.pdf: {origin_pdf_path.name}",
        )

    try:
        payload = json.loads(middle_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return VisualizationResult(
            document_stem=job.document_stem,
            parse_dir=job.parse_dir,
            status=VISUALIZATION_SKIPPED,
            message=f"failed to read middle.json: {exc}",
        )

    pdf_info = payload.get("pdf_info")
    if not isinstance(pdf_info, list):
        return VisualizationResult(
            document_stem=job.document_stem,
            parse_dir=job.parse_dir,
            status=VISUALIZATION_SKIPPED,
            message="invalid middle.json: missing pdf_info",
        )

    try:
        pdf_bytes = origin_pdf_path.read_bytes()
        generated_files = [f"{job.document_stem}_layout.pdf"]
        draw_layout_bbox(pdf_info, pdf_bytes, str(job.parse_dir), generated_files[0])
        if job.draw_span:
            generated_files.append(f"{job.document_stem}_span.pdf")
            draw_span_bbox(pdf_info, pdf_bytes, str(job.parse_dir), generated_files[1])
    except Exception as exc:
        return VisualizationResult(
            document_stem=job.document_stem,
            parse_dir=job.parse_dir,
            status=VISUALIZATION_SKIPPED,
            message=f"visualization failed: {exc}",
        )

    return VisualizationResult(
        document_stem=job.document_stem,
        parse_dir=job.parse_dir,
        status=VISUALIZATION_FINISHED,
        message="generated visualization pdf(s)",
        generated_files=tuple(generated_files),
    )
