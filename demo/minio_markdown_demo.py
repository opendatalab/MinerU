"""Demonstrate the MinIO round-trip parsing workflow for MinerU.

The script uploads a local file to MinIO, reads it back from MinIO for parsing,
uploads all parse artifacts to MinIO, rewrites image references inside Markdown
and JSON outputs to MinIO object URLs, and finally downloads the generated
Markdown to a local directory for inspection.
"""

# Copyright (c) Opendatalab. All rights reserved.

import argparse
import os
import re
import tempfile
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from mineru.cli.common import docx_suffixes, do_parse, image_suffixes, normalize_upload_filename, pdf_suffixes
from mineru.data.data_reader_writer import S3DataReader, S3DataWriter
from mineru.utils.config_reader import get_s3_config
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes

SUPPORTED_INPUT_SUFFIXES = set(pdf_suffixes + image_suffixes + docx_suffixes)
DEFAULT_BUCKET_NAME = "mineru-bucket"
DEFAULT_INPUT_PREFIX = "input"
DEFAULT_OUTPUT_PREFIX = "output"


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Upload a local file to MinIO, read it back from MinIO for MinerU parsing, "
            "then upload and download the optimized Markdown result."
        )
    )
    parser.add_argument("input_file", help="Local file to upload and parse.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_NAME, help="MinIO bucket name.")
    parser.add_argument(
        "--minio-url",
        default=None,
        help="MinIO endpoint URL. Fallback order: CLI > MINIO_URL > mineru.json bucket config.",
    )
    parser.add_argument(
        "--minio-ak",
        default=None,
        help="MinIO access key. Fallback order: CLI > MINIO_AK > mineru.json bucket config.",
    )
    parser.add_argument(
        "--minio-sk",
        default=None,
        help="MinIO secret key. Fallback order: CLI > MINIO_SK > mineru.json bucket config.",
    )
    parser.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX, help="Source object prefix in MinIO.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Parsed artifact prefix in MinIO.")
    parser.add_argument("--language", default="ch", help="OCR language hint for MinerU.")
    parser.add_argument(
        "--backend",
        default="hybrid-auto-engine",
        help="MinerU backend, for example pipeline or hybrid-auto-engine.",
    )
    parser.add_argument(
        "--parse-method",
        default="auto",
        choices=("auto", "txt", "ocr"),
        help="MinerU parse method.",
    )
    parser.add_argument(
        "--md-mode",
        default=MakeMode.MM_MD,
        choices=(MakeMode.MM_MD, MakeMode.NLP_MD),
        help="Markdown generation mode.",
    )
    parser.add_argument(
        "--local-output-dir",
        default=str(demo_dir / "minio_output"),
        help="Local directory used to save the downloaded Markdown.",
    )
    parser.add_argument(
        "--print-md",
        action="store_true",
        help="Print the final Markdown content to stdout.",
    )
    return parser.parse_args()


def resolve_input_file(input_file: str) -> Path:
    path = Path(input_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path must be a file: {path}")

    file_suffix = guess_suffix_by_path(path)
    if file_suffix not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(
            f"Unsupported input type for Markdown generation: {path.name} ({file_suffix}). "
            f"Supported types: {supported}"
        )
    return path.resolve()


def build_task_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def resolve_minio_config(bucket_name: str, args: argparse.Namespace) -> tuple[str, str, str]:
    endpoint = args.minio_url or os.getenv("MINIO_URL")
    ak = args.minio_ak or os.getenv("MINIO_AK")
    sk = args.minio_sk or os.getenv("MINIO_SK")

    config_ak = None
    config_sk = None
    config_endpoint = None
    if not all([endpoint, ak, sk]):
        try:
            config_ak, config_sk, config_endpoint = get_s3_config(bucket_name)
        except Exception:
            config_ak, config_sk, config_endpoint = None, None, None

    endpoint = endpoint or config_endpoint
    ak = ak or config_ak
    sk = sk or config_sk

    if not all([endpoint, ak, sk]):
        raise ValueError(
            "MinIO configuration is incomplete. Provide --minio-url/--minio-ak/--minio-sk, "
            "or set MINIO_URL/MINIO_AK/MINIO_SK, or configure the bucket in mineru.json."
        )
    return ak, sk, endpoint


def normalize_http_endpoint(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint.rstrip("/")
    return f"http://{endpoint.strip('/')}"


def build_http_object_url(endpoint: str, bucket_name: str, *parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part != "."]
    object_path = "/".join(quote(part, safe="/") for part in clean_parts)
    return f"{normalize_http_endpoint(endpoint)}/{bucket_name}/{object_path}"


def build_s3_uri(bucket_name: str, prefix: str, object_key: str) -> str:
    return f"s3://{bucket_name}/{prefix.strip('/')}/{object_key.strip('/')}"


def rewrite_relative_image_paths(
    file_text: str,
    relative_key: str,
    output_root_key: str,
    endpoint: str,
    bucket_name: str,
    output_prefix: str,
) -> str:
    parent_key = Path(relative_key).parent.as_posix()
    image_base_url = build_http_object_url(
        endpoint,
        bucket_name,
        output_prefix,
        output_root_key,
        parent_key,
        "images",
    )

    # Rewrite references like:
    # ![](images/xxx.png), ![](./images/xxx.png), "img_path": "images/xxx.png", src='images/xxx.png'
    return re.sub(
        r"""(?P<prefix>["'(])(?:\./)?images/""",
        lambda match: f"{match.group('prefix')}{image_base_url}/",
        file_text,
    )


def prepare_parse_bytes(source_bytes: bytes, upload_name: str) -> bytes:
    file_suffix = guess_suffix_by_bytes(source_bytes, upload_name)
    if file_suffix in image_suffixes:
        return images_bytes_to_pdf_bytes(source_bytes)
    if file_suffix in pdf_suffixes + docx_suffixes:
        return source_bytes
    raise ValueError(
        f"Uploaded source type is not supported for Markdown generation: {upload_name} ({file_suffix})"
    )


def create_s3_client(
    *,
    bucket: str,
    prefix: str,
    ak: str,
    sk: str,
    endpoint: str,
    reader: bool,
):
    client_cls = S3DataReader if reader else S3DataWriter
    return client_cls(
        default_prefix_without_bucket=prefix,
        bucket=bucket,
        ak=ak,
        sk=sk,
        endpoint_url=endpoint,
        addressing_style="path",
    )


def upload_parse_outputs(
    output_root: Path,
    *,
    writer: S3DataWriter,
    bucket: str,
    output_prefix: str,
    output_root_key: str,
    endpoint: str,
    doc_stem: str,
) -> str:
    markdown_object_key: str | None = None

    for local_file in sorted(output_root.rglob("*")):
        if not local_file.is_file():
            continue

        relative_key = local_file.relative_to(output_root).as_posix()
        payload = local_file.read_bytes()
        if local_file.suffix.lower() in {".md", ".json"}:
            text = payload.decode("utf-8")
            text = rewrite_relative_image_paths(
                text,
                relative_key,
                output_root_key,
                endpoint,
                bucket,
                output_prefix,
            )
            payload = text.encode("utf-8")

        object_key = f"{output_root_key}/{relative_key}"
        writer.write(object_key, payload)
        print(f"uploaded: {build_s3_uri(bucket, output_prefix, object_key)}")

        if Path(relative_key).name == f"{doc_stem}.md":
            markdown_object_key = object_key

    if markdown_object_key is None:
        raise FileNotFoundError(
            f"Unable to find generated Markdown for {doc_stem} under {output_root}"
        )
    return markdown_object_key


def save_local_markdown(markdown_text: str, output_dir: str, task_id: str, doc_stem: str) -> Path:
    local_output_dir = Path(output_dir).expanduser().resolve()
    local_output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = local_output_dir / task_id / f"{doc_stem}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return markdown_path


def run_demo(args: argparse.Namespace) -> Path:
    input_path = resolve_input_file(args.input_file)
    upload_name = normalize_upload_filename(input_path.name)
    doc_stem = Path(upload_name).stem
    task_id = build_task_id()

    ak, sk, endpoint = resolve_minio_config(args.bucket, args)
    input_reader = create_s3_client(
        bucket=args.bucket,
        prefix=args.input_prefix,
        ak=ak,
        sk=sk,
        endpoint=endpoint,
        reader=True,
    )
    input_writer = create_s3_client(
        bucket=args.bucket,
        prefix=args.input_prefix,
        ak=ak,
        sk=sk,
        endpoint=endpoint,
        reader=False,
    )
    output_reader = create_s3_client(
        bucket=args.bucket,
        prefix=args.output_prefix,
        ak=ak,
        sk=sk,
        endpoint=endpoint,
        reader=True,
    )
    output_writer = create_s3_client(
        bucket=args.bucket,
        prefix=args.output_prefix,
        ak=ak,
        sk=sk,
        endpoint=endpoint,
        reader=False,
    )

    source_bytes = input_path.read_bytes()
    input_object_key = f"{task_id}/{upload_name}"
    input_writer.write(input_object_key, source_bytes)
    print(f"uploaded source: {build_s3_uri(args.bucket, args.input_prefix, input_object_key)}")
    print(
        "source http url: "
        f"{build_http_object_url(endpoint, args.bucket, args.input_prefix, input_object_key)}"
    )

    source_bytes_from_minio = input_reader.read(input_object_key)
    parse_bytes = prepare_parse_bytes(source_bytes_from_minio, upload_name)

    with tempfile.TemporaryDirectory(prefix="mineru_minio_") as tmp_dir:
        do_parse(
            output_dir=tmp_dir,
            pdf_file_names=[doc_stem],
            pdf_bytes_list=[parse_bytes],
            p_lang_list=[args.language],
            backend=args.backend,
            parse_method=args.parse_method,
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            f_make_md_mode=args.md_mode,
        )

        output_root = Path(tmp_dir) / doc_stem
        output_root_key = f"{task_id}/{doc_stem}"
        markdown_object_key = upload_parse_outputs(
            output_root,
            writer=output_writer,
            bucket=args.bucket,
            output_prefix=args.output_prefix,
            output_root_key=output_root_key,
            endpoint=endpoint,
            doc_stem=doc_stem,
        )

    markdown_text = output_reader.read(markdown_object_key).decode("utf-8")
    local_markdown_path = save_local_markdown(
        markdown_text,
        args.local_output_dir,
        task_id,
        doc_stem,
    )

    markdown_s3_uri = build_s3_uri(args.bucket, args.output_prefix, markdown_object_key)
    markdown_http_url = build_http_object_url(
        endpoint,
        args.bucket,
        args.output_prefix,
        markdown_object_key,
    )

    print(f"generated markdown: {markdown_s3_uri}")
    print(f"markdown http url: {markdown_http_url}")
    print(f"local markdown: {local_markdown_path}")

    if args.print_md:
        print("\n===== Optimized Markdown =====\n")
        print(markdown_text)
    return local_markdown_path


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    freeze_support()
    main()
