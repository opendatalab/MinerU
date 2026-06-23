# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import mimetypes
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import click
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger

from base64 import b64encode

from mineru.cli.common import (
    aio_do_parse,
    do_parse,
    image_suffixes,
    normalize_upload_filename,
    office_suffixes,
    pdf_suffixes,
    normalize_task_stem,
    read_fn,
    uniquify_task_stems,
)
from mineru.cli.api_request import ParseRequestOptions, parse_request_form
from mineru.cli.public_http_client_policy import (
    configure_public_http_client_policy,
    is_public_bind_host,
    warn_if_public_http_client_policy as _warn_if_public_http_client_policy,
)
from mineru.cli.output_paths import resolve_parse_dir
from mineru.cli.api_protocol import (
    API_PROTOCOL_VERSION,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_PROCESSING_WINDOW_SIZE,
)
from mineru.cli.vlm_preload import (
    maybe_preload_vlm_model,
    split_service_and_model_config,
)
from mineru.backend.vlm.vlm_analyze import shutdown_cached_models
from mineru.utils.cli_parser import arg_parse
from mineru.utils.check_sys_env import is_mac_environment
from mineru.utils.config_reader import (
    get_max_concurrent_requests as read_max_concurrent_requests,
    get_processing_window_size,
    read_config,
)
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.utils.pdf_image_tools import shutdown_pdf_render_executor
from mineru.version import __version__

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level)

TASK_PENDING = "pending"
TASK_PROCESSING = "processing"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_TERMINAL_STATES = {TASK_COMPLETED, TASK_FAILED}
SUPPORTED_UPLOAD_SUFFIXES = pdf_suffixes + image_suffixes + office_suffixes
RESULT_IMAGE_SUFFIXES = set(image_suffixes) | {"svg"}
DEFAULT_TASK_RETENTION_SECONDS = 24 * 60 * 60
DEFAULT_TASK_CLEANUP_INTERVAL_SECONDS = 5 * 60
DEFAULT_OUTPUT_ROOT = "./output"
FILE_PARSE_TASK_ID_HEADER = "X-MinerU-Task-Id"
FILE_PARSE_TASK_STATUS_HEADER = "X-MinerU-Task-Status"
FILE_PARSE_TASK_STATUS_URL_HEADER = "X-MinerU-Task-Status-Url"
FILE_PARSE_TASK_RESULT_URL_HEADER = "X-MinerU-Task-Result-Url"
MINERU_API_PUBLIC_BIND_EXPOSED_ENV = "MINERU_API_PUBLIC_BIND_EXPOSED"
MINERU_API_ALLOW_PUBLIC_HTTP_CLIENT_ENV = "MINERU_API_ALLOW_PUBLIC_HTTP_CLIENT"

# 并发控制器
_request_semaphore: Optional[asyncio.Semaphore] = None
_configured_max_concurrent_requests = 1


def env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def get_s3_presign_expires(default: int = 604800) -> int:
    """Presigned-URL lifetime (seconds) for S3 image uploads.

    Read from MINERU_API_S3_PRESIGN_EXPIRES; defaults to 7 days, the maximum
    for long-term access keys. Only relevant when S3 image upload is enabled.
    """
    value = os.getenv("MINERU_API_S3_PRESIGN_EXPIRES")
    if value is None:
        return default
    try:
        expires = int(value)
    except ValueError:
        logger.warning(
            f"Invalid MINERU_API_S3_PRESIGN_EXPIRES value: {value}, using default {default}"
        )
        return default
    return max(1, expires)


def is_main_multiprocessing_process() -> bool:
    try:
        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        return True


def install_stdin_shutdown_watcher(server: uvicorn.Server) -> None:
    if not env_flag_enabled("MINERU_API_SHUTDOWN_ON_STDIN_EOF"):
        return

    def _watch_stdin_for_eof() -> None:
        stdin_stream = getattr(sys.stdin, "buffer", sys.stdin)
        try:
            stdin_stream.read()
        except Exception:
            return
        server.should_exit = True

    watcher = threading.Thread(
        target=_watch_stdin_for_eof,
        name="mineru-api-stdin-shutdown",
        daemon=True,
    )
    watcher.start()


@dataclass
class StoredUpload:
    original_name: str
    stem: str
    path: str


@dataclass
class AsyncParseTask:
    task_id: str
    status: str
    backend: str
    file_names: list[str]
    created_at: str
    output_dir: str
    parse_method: str
    lang_list: list[str]
    formula_enable: bool
    table_enable: bool
    image_analysis: bool
    server_url: Optional[str]
    return_md: bool
    return_middle_json: bool
    return_model_output: bool
    return_content_list: bool
    return_images: bool
    response_format_zip: bool
    return_original_file: bool
    start_page_id: int
    end_page_id: int
    upload_names: list[str]
    uploads: list[str]
    submit_order: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    _s3_image_urls: Optional[dict[str, dict[str, str]]] = field(default=None, repr=False)

    def to_status_payload(
        self,
        request: Request,
        queued_ahead: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "status": self.status,
            "backend": self.backend,
            "file_names": self.file_names,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "status_url": str(
                request.url_for("get_async_task_status", task_id=self.task_id)
            ),
            "result_url": str(
                request.url_for("get_async_task_result", task_id=self.task_id)
            ),
        }
        if queued_ahead is not None:
            payload["queued_ahead"] = queued_ahead
        return payload


class TaskWaitAbortedError(RuntimeError):
    """Raised when a synchronous file_parse request cannot keep waiting safely."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_app_state(app)
    try:
        yield
    finally:
        await shutdown_app_state(app)


def create_app():
    # By default, the OpenAPI documentation endpoints (openapi_url, docs_url, redoc_url) are enabled.
    # To disable the FastAPI docs and schema endpoints, set the environment variable MINERU_API_ENABLE_FASTAPI_DOCS=0.
    enable_docs = env_flag_enabled("MINERU_API_ENABLE_FASTAPI_DOCS", default=True)
    app = FastAPI(
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        lifespan=lifespan,
    )

    global _request_semaphore, _configured_max_concurrent_requests

    if is_mac_environment():
        max_concurrent_requests = 1
    else:
        max_concurrent_requests = read_max_concurrent_requests(
            default=DEFAULT_MAX_CONCURRENT_REQUESTS
        )

    _configured_max_concurrent_requests = max_concurrent_requests
    app.state.max_concurrent_requests = max_concurrent_requests
    _request_semaphore = asyncio.Semaphore(max_concurrent_requests)
    if is_main_multiprocessing_process():
        logger.info(f"Request concurrency limited to {max_concurrent_requests}")

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.state.public_bind_exposed = env_flag_enabled(
        MINERU_API_PUBLIC_BIND_EXPOSED_ENV,
        default=False,
    )
    app.state.allow_public_http_client = env_flag_enabled(
        MINERU_API_ALLOW_PUBLIC_HTTP_CLIENT_ENV,
        default=False,
    )
    default_service_config, default_model_config = split_service_and_model_config(
        {
            "enable_vlm_preload": env_flag_enabled(
                "MINERU_API_ENABLE_VLM_PRELOAD",
                default=False,
            )
        }
    )
    app.state.service_config = default_service_config
    app.state.config = default_model_config
    app.state.task_manager = None
    return app


app = create_app()


async def startup_app_state(app: FastAPI) -> "AsyncTaskManager":
    task_manager = AsyncTaskManager(app)
    await task_manager.start()
    try:
        service_config = getattr(app.state, "service_config", {})
        model_config = getattr(app.state, "config", {})
        maybe_preload_vlm_model(
            bool(service_config.get("enable_vlm_preload", False)),
            model_kwargs=model_config,
        )
    except Exception:
        await task_manager.shutdown()
        app.state.task_manager = None
        raise

    app.state.task_manager = task_manager
    return task_manager


async def shutdown_app_state(app: FastAPI) -> None:
    current_task_manager = getattr(app.state, "task_manager", None)
    if current_task_manager is not None:
        await current_task_manager.shutdown()
    app.state.task_manager = None
    shutdown_runtime_resources()


def shutdown_runtime_resources() -> None:
    try:
        shutdown_cached_models()
    except Exception as exc:
        logger.warning(f"Failed to shutdown cached VLM models: {exc}")

    try:
        shutdown_pdf_render_executor()
    except Exception as exc:
        logger.warning(f"Failed to shutdown PDF render executor: {exc}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if value < minimum:
        return default
    return value


def get_max_concurrent_requests() -> int:
    return _configured_max_concurrent_requests


def get_task_retention_seconds() -> int:
    return get_int_env(
        "MINERU_API_TASK_RETENTION_SECONDS",
        DEFAULT_TASK_RETENTION_SECONDS,
        minimum=0,
    )


def get_task_cleanup_interval_seconds() -> int:
    return get_int_env(
        "MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS",
        DEFAULT_TASK_CLEANUP_INTERVAL_SECONDS,
        minimum=1,
    )


def get_output_root() -> Path:
    root = Path(os.getenv("MINERU_API_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def warn_if_public_http_client_policy(host: str, allow_public_http_client: bool) -> None:
    _warn_if_public_http_client_policy(
        service_name="API",
        host=host,
        allow_public_http_client=allow_public_http_client,
    )


def cleanup_file(file_path: str) -> None:
    """清理临时文件或目录"""
    try:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")


def build_upload_destination(upload_dir: str, filename: str) -> Path:
    destination = Path(upload_dir) / filename
    if not destination.exists():
        return destination

    base_name = Path(filename).stem
    suffix = Path(filename).suffix
    index = 2
    while True:
        candidate = Path(upload_dir) / f"{base_name}__upload_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def _get_s3_upload_config() -> Optional[dict[str, str]]:
    """Read S3/MinIO upload config from mineru.json.

    Reuses the standard config source only — ``MINERU_TOOLS_CONFIG_JSON`` (an
    explicit absolute path) or ``~/mineru.json`` via ``read_config()``. This does
    NOT walk the working directory or probe mounted paths: in a long-running
    API service those could pick up an unrelated config containing credentials.
    """
    config = read_config()
    if config is None:
        return None
    bucket_info = config.get('bucket_info')
    if not bucket_info:
        return None
    bucket_name = list(bucket_info.keys())[0]
    info = bucket_info[bucket_name]
    if len(info) < 3 or not all(info[:3]):
        return None
    access_key, secret_key, endpoint_url = info[0], info[1], info[2]
    return {
        'bucket_name': bucket_name,
        'access_key': access_key,
        'secret_key': secret_key,
        'endpoint_url': endpoint_url,
    }


def _upload_images_and_replace_md_urls(
    output_dir: str,
    pdf_file_names: list[str],
    backend: str,
    parse_method: str,
    s3_config: dict[str, str],
    presign_expires: int = 604800,
) -> dict[str, dict[str, str]]:
    """Upload images to S3/MinIO and replace local paths with URLs in markdown.

    Returns ``{pdf_name: {image_filename: url}}`` for the uploaded images so the
    caller can surface them in the API response. URLs are presigned GETs bound to
    the configured endpoint, so the bucket stays private:

    * No bucket is created — the bucket must already exist; upload is skipped
      for the task if it is not accessible.
    * No bucket policy is modified — the bucket is never made public.
    * No credentials appear in the bucket name; the presigned URL carries only
      the access key id (not the secret) as a signed query parameter.
    """
    import boto3
    from botocore.config import Config as BotocoreConfig

    endpoint_url = s3_config['endpoint_url']
    if not endpoint_url.startswith(('http://', 'https://')):
        endpoint_url = f'http://{endpoint_url}'

    bucket_name = s3_config['bucket_name']

    s3_client = boto3.client(
        service_name='s3',
        aws_access_key_id=s3_config['access_key'],
        aws_secret_access_key=s3_config['secret_key'],
        endpoint_url=endpoint_url,
        config=BotocoreConfig(
            s3={'addressing_style': 'path'},
            retries={'max_attempts': 3, 'mode': 'standard'},
        ),
    )

    # Require the bucket to already exist; never create buckets or mutate
    # bucket policy from the parse path.
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception as e:
        logger.warning(
            f"S3 bucket '{bucket_name}' is not accessible, skipping image upload: {e}"
        )
        return {}

    result_map: dict[str, dict[str, str]] = {}

    for pdf_name in pdf_file_names:
        try:
            parse_dir = get_parse_dir(output_dir, pdf_name, backend, parse_method)
        except ValueError:
            continue

        images_dir = os.path.join(parse_dir, "images")
        if not os.path.isdir(images_dir):
            continue

        # Upload each image and build filename -> URL mapping
        url_map: dict[str, str] = {}
        for img_path in sorted(Path(images_dir).iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lstrip(".").lower() not in RESULT_IMAGE_SUFFIXES:
                continue

            img_filename = img_path.name
            s3_key = f"{pdf_name}/images/{img_filename}"
            content_type = mimetypes.guess_type(str(img_path))[0] or 'image/jpeg'

            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=img_path.read_bytes(),
                    ContentType=content_type,
                )
            except Exception as e:
                logger.error(f"Failed to upload image {img_filename}: {e}")
                continue

            try:
                url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": s3_key},
                    ExpiresIn=presign_expires,
                    HttpMethod="GET",
                )
            except Exception as e:
                logger.error(f"Failed to presign image {img_filename}: {e}")
                continue

            url_map[img_filename] = url
            logger.debug(f"Uploaded image to S3: {s3_key}")

        result_map[pdf_name] = url_map

        # Replace local image paths with S3 URLs in markdown
        md_path = os.path.join(parse_dir, f"{pdf_name}.md")
        if os.path.exists(md_path) and url_map:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            for img_filename, s3_url in url_map.items():
                md_content = md_content.replace(
                    f"![](images/{img_filename})", f"![]({s3_url})"
                )

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

    return result_map


def get_images_dir_image_paths(images_dir: str) -> list[str]:
    """Return all supported image files directly under images_dir."""
    if not os.path.isdir(images_dir):
        return []

    return sorted(
        str(path)
        for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lstrip(".").lower() in RESULT_IMAGE_SUFFIXES
    )


def get_image_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        return mime_type
    return "image/jpeg"


def get_infer_result(
    file_suffix_identifier: str, pdf_name: str, parse_dir: str
) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


def normalize_lang_list(lang_list: list[str], file_count: int) -> list[str]:
    if len(lang_list) == file_count:
        return lang_list
    base_lang = lang_list[0] if lang_list else "ch"
    return [base_lang] * file_count


def get_parse_dir(output_dir: str, pdf_name: str, backend: str, parse_method: str) -> str:
    return str(
        resolve_parse_dir(
            output_dir,
            pdf_name,
            backend,
            parse_method,
            allow_office_fallback=True,
        )
    )


def is_task_terminal(status: str) -> bool:
    return status in TASK_TERMINAL_STATES


def build_result_dict(
    output_dir: str,
    pdf_file_names: list[str],
    backend: str,
    parse_method: str,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    s3_image_urls: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    result_dict: dict[str, dict[str, Any]] = {}
    for pdf_name in pdf_file_names:
        result_dict[pdf_name] = {}
        data = result_dict[pdf_name]

        try:
            parse_dir = get_parse_dir(output_dir, pdf_name, backend, parse_method)
        except ValueError:
            logger.warning(f"Unknown backend type: {backend}, skipping {pdf_name}")
            continue

        if not os.path.exists(parse_dir):
            continue

        if return_md:
            data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
        if return_middle_json:
            data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
        if return_model_output:
            data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
        if return_content_list:
            data["content_list"] = get_infer_result(
                "_content_list.json", pdf_name, parse_dir
            )
        if return_images:
            images_dir = os.path.join(parse_dir, "images")
            image_paths = get_images_dir_image_paths(images_dir)
            pdf_image_urls = (s3_image_urls or {}).get(pdf_name) or {}
            data["images"] = {
                os.path.basename(image_path): (
                    pdf_image_urls.get(os.path.basename(image_path))
                    or f"data:{get_image_mime_type(image_path)};base64,{encode_image(image_path)}"
                )
                for image_path in image_paths
            }
    return result_dict


def build_zip_arcname(
    pdf_name: str,
    parse_dir: str,
    relative_path: str,
) -> str:
    return os.path.join(pdf_name, os.path.basename(parse_dir), relative_path)


def create_result_zip(
    output_dir: str,
    pdf_file_names: list[str],
    backend: str,
    parse_method: str,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    return_original_file: bool,
) -> str:
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
    os.close(zip_fd)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_name in pdf_file_names:
            try:
                parse_dir = get_parse_dir(output_dir, pdf_name, backend, parse_method)
            except ValueError:
                logger.warning(f"Unknown backend type: {backend}, skipping {pdf_name}")
                continue

            if not os.path.exists(parse_dir):
                continue

            if return_md:
                path = os.path.join(parse_dir, f"{pdf_name}.md")
                if os.path.exists(path):
                    zf.write(
                        path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            f"{pdf_name}.md",
                        ),
                    )

            if return_middle_json:
                path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                if os.path.exists(path):
                    zf.write(
                        path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            f"{pdf_name}_middle.json",
                        ),
                    )

            if return_model_output:
                path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                if os.path.exists(path):
                    zf.write(
                        path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            f"{pdf_name}_model.json",
                        ),
                    )

            if return_content_list:
                path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                if os.path.exists(path):
                    zf.write(
                        path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            f"{pdf_name}_content_list.json",
                        ),
                    )

                path = os.path.join(parse_dir, f"{pdf_name}_content_list_v2.json")
                if os.path.exists(path):
                    zf.write(
                        path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            f"{pdf_name}_content_list_v2.json",
                        ),
                    )

            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                image_paths = get_images_dir_image_paths(images_dir)
                for image_path in image_paths:
                    zf.write(
                        image_path,
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            os.path.join("images", os.path.basename(image_path)),
                        ),
                    )

            if return_original_file:
                origin_pattern = f"{pdf_name}_origin."
                for path in sorted(Path(parse_dir).iterdir()):
                    if not path.is_file():
                        continue
                    if not path.name.startswith(origin_pattern):
                        continue
                    zf.write(
                        str(path),
                        arcname=build_zip_arcname(
                            pdf_name,
                            parse_dir,
                            path.name,
                        ),
                    )
    return zip_path


def _cleanup_generated_zip_task(task: asyncio.Task[str]) -> None:
    try:
        generated_zip_path = task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        return
    cleanup_file(generated_zip_path)


async def build_result_response(
    background_tasks: BackgroundTasks,
    status_code: int,
    output_dir: str,
    pdf_file_names: list[str],
    backend: str,
    parse_method: str,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    response_format_zip: bool,
    return_original_file: bool,
    zip_filename: str = "results.zip",
    s3_image_urls: dict[str, dict[str, str]] | None = None,
) -> Response:
    if response_format_zip:
        zip_task = asyncio.create_task(
            asyncio.to_thread(
                create_result_zip,
                output_dir=output_dir,
                pdf_file_names=pdf_file_names,
                backend=backend,
                parse_method=parse_method,
                return_md=return_md,
                return_middle_json=return_middle_json,
                return_model_output=return_model_output,
                return_content_list=return_content_list,
                return_images=return_images,
                return_original_file=return_original_file,
            )
        )
        try:
            zip_path = await asyncio.shield(zip_task)
        except asyncio.CancelledError:
            zip_task.add_done_callback(_cleanup_generated_zip_task)
            raise
        background_tasks.add_task(cleanup_file, zip_path)
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=zip_filename,
            status_code=status_code,
        )

    result_dict = await asyncio.to_thread(
        build_result_dict,
        output_dir=output_dir,
        pdf_file_names=pdf_file_names,
        backend=backend,
        parse_method=parse_method,
        return_md=return_md,
        return_middle_json=return_middle_json,
        return_model_output=return_model_output,
        return_content_list=return_content_list,
        return_images=return_images,
        s3_image_urls=s3_image_urls,
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "backend": backend,
            "version": __version__,
            "results": result_dict,
        },
    )


def build_task_submission_response(
    task: AsyncParseTask,
    request: Request,
    task_manager: "AsyncTaskManager",
) -> JSONResponse:
    payload = task_manager.build_status_payload(task, request)
    payload["message"] = "Task submitted successfully"
    return JSONResponse(status_code=202, content=payload)


async def build_sync_file_parse_response(
    background_tasks: BackgroundTasks,
    task: AsyncParseTask,
    request: Request,
) -> Response:
    task_payload = task.to_status_payload(request)
    if task.response_format_zip:
        response = await build_result_response(
            background_tasks=background_tasks,
            status_code=200,
            output_dir=task.output_dir,
            pdf_file_names=task.file_names,
            backend=task.backend,
            parse_method=task.parse_method,
            return_md=task.return_md,
            return_middle_json=task.return_middle_json,
            return_model_output=task.return_model_output,
            return_content_list=task.return_content_list,
            return_images=task.return_images,
            response_format_zip=task.response_format_zip,
            return_original_file=task.return_original_file,
            zip_filename=f"{task.task_id}.zip",
            s3_image_urls=task._s3_image_urls,
        )
        response.headers[FILE_PARSE_TASK_ID_HEADER] = task.task_id
        response.headers[FILE_PARSE_TASK_STATUS_HEADER] = task.status
        response.headers[FILE_PARSE_TASK_STATUS_URL_HEADER] = task_payload["status_url"]
        response.headers[FILE_PARSE_TASK_RESULT_URL_HEADER] = task_payload["result_url"]
        return response

    result_dict = await asyncio.to_thread(
        build_result_dict,
        output_dir=task.output_dir,
        pdf_file_names=task.file_names,
        backend=task.backend,
        parse_method=task.parse_method,
        return_md=task.return_md,
        return_middle_json=task.return_middle_json,
        return_model_output=task.return_model_output,
        return_content_list=task.return_content_list,
        return_images=task.return_images,
        s3_image_urls=task._s3_image_urls,
    )
    return JSONResponse(
        status_code=200,
        content={
            **task_payload,
            "backend": task.backend,
            "version": __version__,
            "results": result_dict,
        },
    )


async def save_upload_files(upload_dir: str, files: list[UploadFile]) -> list[StoredUpload]:
    os.makedirs(upload_dir, exist_ok=True)
    uploads: list[StoredUpload] = []

    for upload in files:
        original_name = upload.filename or f"upload-{uuid.uuid4()}"
        filename = normalize_upload_filename(original_name)
        normalized_stem = normalize_task_stem(Path(filename).stem)
        destination = build_upload_destination(upload_dir, filename)
        try:
            with open(destination, "wb") as handle:
                while True:
                    chunk = await upload.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)

            file_suffix = guess_suffix_by_path(destination)
            if file_suffix not in SUPPORTED_UPLOAD_SUFFIXES:
                cleanup_file(str(destination))
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_suffix}",
                )

            uploads.append(
                StoredUpload(
                    original_name=original_name,
                    stem=normalized_stem,
                    path=str(destination),
                )
            )
        except Exception:
            cleanup_file(str(destination))
            raise
        finally:
            await upload.close()

    normalized_stems, renamed_stems = uniquify_task_stems(
        [upload.stem for upload in uploads]
    )
    if renamed_stems:
        rename_details = ", ".join(
            f"{Path(upload.original_name).name} -> {effective_stem}"
            for upload, effective_stem in zip(uploads, normalized_stems)
            if upload.stem != effective_stem
        )
        logger.warning(
            f"Normalized duplicate upload stems within request: {rename_details}"
        )
        uploads = [
            StoredUpload(
                original_name=upload.original_name,
                stem=effective_stem,
                path=upload.path,
            )
            for upload, effective_stem in zip(uploads, normalized_stems)
        ]
    return uploads


def load_parse_inputs(uploads: list[StoredUpload]) -> tuple[list[str], list[bytes]]:
    pdf_file_names = []
    pdf_bytes_list = []

    for upload in uploads:
        try:
            pdf_bytes = read_fn(Path(upload.path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load file {upload.original_name}: {exc}") from exc
        pdf_file_names.append(upload.stem)
        pdf_bytes_list.append(pdf_bytes)
    return pdf_file_names, pdf_bytes_list


async def run_parse_job(
    output_dir: str,
    uploads: list[StoredUpload],
    request_options: ParseRequestOptions | AsyncParseTask,
    config: dict[str, Any],
) -> list[str]:
    pdf_file_names, pdf_bytes_list = await asyncio.to_thread(load_parse_inputs, uploads)
    actual_lang_list = normalize_lang_list(request_options.lang_list, len(pdf_file_names))
    response_file_names = list(pdf_file_names)

    parse_kwargs = dict(
        output_dir=output_dir,
        pdf_file_names=list(pdf_file_names),
        pdf_bytes_list=list(pdf_bytes_list),
        p_lang_list=list(actual_lang_list),
        backend=request_options.backend,
        parse_method=request_options.parse_method,
        formula_enable=request_options.formula_enable,
        table_enable=request_options.table_enable,
        image_analysis=request_options.image_analysis,
        server_url=request_options.server_url,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=request_options.return_md,
        f_dump_middle_json=request_options.return_middle_json,
        f_dump_model_output=request_options.return_model_output,
        f_dump_orig_pdf=(
            request_options.return_original_file and request_options.response_format_zip
        ),
        f_dump_content_list=request_options.return_content_list,
        start_page_id=request_options.start_page_id,
        end_page_id=request_options.end_page_id,
        **config,
    )

    if request_options.backend == "pipeline":
        await asyncio.to_thread(do_parse, **parse_kwargs)
    else:
        await aio_do_parse(**parse_kwargs)
    return response_file_names


def create_task_output_dir(task_id: str) -> str:
    output_root = get_output_root()
    task_output_dir = output_root / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    return str(task_output_dir)


async def create_async_parse_task(
    request_options: ParseRequestOptions,
) -> AsyncParseTask:
    task_id = str(uuid.uuid4())
    task_output_dir = create_task_output_dir(task_id)
    uploads_dir = os.path.join(task_output_dir, "uploads")
    task_manager = get_task_manager()

    try:
        uploads = await save_upload_files(uploads_dir, request_options.files)
        request_options.files.clear()
        file_names = [upload.stem for upload in uploads]
        task = AsyncParseTask(
            task_id=task_id,
            status=TASK_PENDING,
            backend=request_options.backend,
            file_names=file_names,
            created_at=utc_now_iso(),
            output_dir=task_output_dir,
            parse_method=request_options.parse_method,
            lang_list=request_options.lang_list,
            formula_enable=request_options.formula_enable,
            table_enable=request_options.table_enable,
            image_analysis=request_options.image_analysis,
            server_url=request_options.server_url,
            return_md=request_options.return_md,
            return_middle_json=request_options.return_middle_json,
            return_model_output=request_options.return_model_output,
            return_content_list=request_options.return_content_list,
            return_images=request_options.return_images,
            response_format_zip=request_options.response_format_zip,
            return_original_file=request_options.return_original_file,
            start_page_id=request_options.start_page_id,
            end_page_id=request_options.end_page_id,
            upload_names=[upload.original_name for upload in uploads],
            uploads=[upload.path for upload in uploads],
        )
        await task_manager.submit(task)
        return task
    except HTTPException:
        cleanup_file(task_output_dir)
        raise
    except Exception:
        cleanup_file(task_output_dir)
        raise


class AsyncTaskManager:
    def __init__(self, fastapi_app: FastAPI):
        self.app = fastapi_app
        self.tasks: dict[str, AsyncParseTask] = {}
        self.task_events: dict[str, asyncio.Event] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.dispatcher_task: Optional[asyncio.Task[Any]] = None
        self.cleanup_task: Optional[asyncio.Task[Any]] = None
        self.active_tasks: set[asyncio.Task[Any]] = set()
        self.last_worker_error: Optional[str] = None
        self.is_shutting_down = False
        self.task_retention_seconds = get_task_retention_seconds()
        self.task_cleanup_interval_seconds = get_task_cleanup_interval_seconds()
        self.manager_wakeup = asyncio.Event()
        self._next_submit_order = 1

    async def start(self) -> None:
        self.is_shutting_down = False
        self.last_worker_error = None
        self.manager_wakeup = asyncio.Event()
        if self.dispatcher_task is None or self.dispatcher_task.done():
            self.dispatcher_task = asyncio.create_task(
                self._dispatcher_loop(), name="mineru-fastapi-task-dispatcher"
            )
        if (
            self.task_retention_seconds > 0
            and (self.cleanup_task is None or self.cleanup_task.done())
        ):
            self.cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="mineru-fastapi-task-cleanup"
            )

    async def shutdown(self) -> None:
        self.is_shutting_down = True
        self._wake_waiters()
        if self.dispatcher_task is not None:
            self.dispatcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.dispatcher_task
            self.dispatcher_task = None
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.cleanup_task
            self.cleanup_task = None

        pending = list(self.active_tasks)
        for processor in pending:
            processor.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.active_tasks.clear()

    async def submit(self, task: AsyncParseTask) -> None:
        task.submit_order = self._next_submit_order
        self._next_submit_order += 1
        self.tasks[task.task_id] = task
        self.task_events[task.task_id] = asyncio.Event()
        await self.queue.put(task.task_id)

    def get(self, task_id: str) -> Optional[AsyncParseTask]:
        return self.tasks.get(task_id)

    def get_queued_ahead(self, task_id: str) -> int | None:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task.status != TASK_PENDING:
            return 0

        return sum(
            1
            for other_task in self.tasks.values()
            if (
                other_task.task_id != task_id
                and other_task.status == TASK_PENDING
                and 0 < other_task.submit_order < task.submit_order
            )
        )

    def build_status_payload(
        self,
        task: AsyncParseTask,
        request: Request,
    ) -> dict[str, Any]:
        return task.to_status_payload(
            request,
            queued_ahead=self.get_queued_ahead(task.task_id),
        )

    async def wait_for_terminal_state(self, task_id: str) -> AsyncParseTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise TaskWaitAbortedError("Task not found")
        if is_task_terminal(task.status):
            return task

        task_event = self.task_events.get(task_id)
        if task_event is None:
            raise TaskWaitAbortedError("Task wait handle is unavailable")

        event_wait_task = asyncio.create_task(task_event.wait())
        manager_wait_task = asyncio.create_task(self.manager_wakeup.wait())
        done: set[asyncio.Task[Any]] = set()
        pending: set[asyncio.Task[Any]] = set()
        try:
            done, pending = await asyncio.wait(
                {event_wait_task, manager_wait_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in pending:
                waiter.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for waiter in done:
                with suppress(asyncio.CancelledError):
                    waiter.result()

        task = self.tasks.get(task_id)
        if task is None:
            if self.is_shutting_down:
                raise TaskWaitAbortedError("Task manager is shutting down")
            raise TaskWaitAbortedError("Task was removed before completion")
        if is_task_terminal(task.status):
            return task
        if self.is_shutting_down:
            raise TaskWaitAbortedError("Task manager is shutting down")
        raise TaskWaitAbortedError(
            self.last_worker_error or "Task manager became unavailable while waiting"
        )

    def get_stats(self) -> dict[str, int]:
        stats = {
            TASK_PENDING: 0,
            TASK_PROCESSING: 0,
            TASK_COMPLETED: 0,
            TASK_FAILED: 0,
        }
        for task in self.tasks.values():
            if task.status in stats:
                stats[task.status] += 1
        return stats

    def is_healthy(self) -> bool:
        if self.dispatcher_task is None:
            return False
        if self.dispatcher_task.done() and not self.is_shutting_down:
            return False
        if self.task_retention_seconds > 0 and self.cleanup_task is None:
            return False
        if (
            self.task_retention_seconds > 0
            and self.cleanup_task is not None
            and self.cleanup_task.done()
            and not self.is_shutting_down
        ):
            return False
        return self.last_worker_error is None

    def _wake_waiters(self) -> None:
        self.manager_wakeup.set()
        for task_event in self.task_events.values():
            task_event.set()

    def _signal_task_event(self, task_id: str) -> None:
        task_event = self.task_events.get(task_id)
        if task_event is not None:
            task_event.set()

    async def _dispatcher_loop(self) -> None:
        try:
            while True:
                task_id = await self.queue.get()
                processor = asyncio.create_task(
                    self._process_task(task_id),
                    name=f"mineru-fastapi-task-{task_id}",
                )
                self.active_tasks.add(processor)
                processor.add_done_callback(self._on_processor_done)
                self.queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_worker_error = str(exc)
            self._wake_waiters()
            logger.exception("Async task dispatcher crashed")
            raise

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.task_cleanup_interval_seconds)
                self.cleanup_expired_tasks()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_worker_error = str(exc)
            logger.exception("Async task cleanup loop crashed")
            raise

    def _on_processor_done(self, processor: asyncio.Task[Any]) -> None:
        self.active_tasks.discard(processor)
        if processor.cancelled():
            return
        exception = processor.exception()
        if exception is not None:
            logger.error(f"Async task processor crashed: {exception}")
            self.last_worker_error = str(exception)

    async def _process_task(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            return

        try:
            if _request_semaphore is not None:
                async with _request_semaphore:
                    await self._run_task(task)
            else:
                await self._run_task(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            task.status = TASK_FAILED
            task.error = str(exc)
            task.completed_at = utc_now_iso()
            self._signal_task_event(task_id)
            logger.exception(f"Async task failed: {task_id}")

    async def _run_task(self, task: AsyncParseTask) -> None:
        task.status = TASK_PROCESSING
        task.started_at = utc_now_iso()
        task.error = None

        uploads = [
            StoredUpload(
                original_name=upload_name,
                stem=file_name,
                path=upload_path,
            )
            for upload_name, file_name, upload_path in zip(
                task.upload_names,
                task.file_names,
                task.uploads,
            )
        ]
        config = getattr(self.app.state, "config", {})
        await run_parse_job(
            output_dir=task.output_dir,
            uploads=uploads,
            request_options=task,
            config=config,
        )

        # Upload images to S3/MinIO only when the operator explicitly opts in
        # (MINERU_API_S3_IMAGE_UPLOAD=1) and bucket_info is configured. Off by
        # default — uploading parsed images can expose private document content,
        # so this must never happen silently.
        if env_flag_enabled("MINERU_API_S3_IMAGE_UPLOAD"):
            s3_config = _get_s3_upload_config()
            if s3_config:
                try:
                    task._s3_image_urls = await asyncio.to_thread(
                        _upload_images_and_replace_md_urls,
                        output_dir=task.output_dir,
                        pdf_file_names=task.file_names,
                        backend=task.backend,
                        parse_method=task.parse_method,
                        s3_config=s3_config,
                        presign_expires=get_s3_presign_expires(),
                    )
                except Exception as exc:
                    logger.error(f"Failed to upload images to S3/MinIO: {exc}")

        task.status = TASK_COMPLETED
        task.completed_at = utc_now_iso()
        self._signal_task_event(task.task_id)

    def cleanup_expired_tasks(self) -> int:
        if self.task_retention_seconds <= 0:
            return 0

        now = datetime.now(timezone.utc)
        expired_task_ids = [
            task_id
            for task_id, task in self.tasks.items()
            if self._is_task_expired(task, now)
        ]

        for task_id in expired_task_ids:
            task = self.tasks.pop(task_id, None)
            if task is None:
                continue
            task_event = self.task_events.pop(task_id, None)
            if task_event is not None:
                task_event.set()
            cleanup_file(task.output_dir)
            logger.info(f"Cleaned expired async task: {task_id}")
        return len(expired_task_ids)

    def _is_task_expired(self, task: AsyncParseTask, now: datetime) -> bool:
        if task.status not in (TASK_COMPLETED, TASK_FAILED):
            return False
        if not task.completed_at:
            return False
        try:
            completed_at = datetime.fromisoformat(task.completed_at)
        except ValueError:
            logger.warning(f"Invalid completed_at for task {task.task_id}: {task.completed_at}")
            return False
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        return (now - completed_at).total_seconds() >= self.task_retention_seconds


def get_task_manager() -> AsyncTaskManager:
    task_manager = getattr(app.state, "task_manager", None)
    if task_manager is None:
        raise HTTPException(status_code=503, detail="Task manager is not initialized")
    return task_manager


@app.post(
    path="/file_parse",
    status_code=200,
    summary="Synchronously parse uploaded files",
    description=(
        "Submit a parsing task to the shared async task manager, wait for it to "
        "finish, and return the final parsing result in the same response."
    ),
)
async def parse_pdf(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_options: Annotated[
        ParseRequestOptions, Depends(parse_request_form)
    ],
):
    task = await create_async_parse_task(request_options)
    request_options = None
    task_manager = get_task_manager()

    try:
        task = await task_manager.wait_for_terminal_state(task.task_id)
    except TaskWaitAbortedError as exc:
        return JSONResponse(
            status_code=503,
            content={
                **task.to_status_payload(http_request),
                "message": "Task manager became unavailable while waiting for result",
                "error": str(exc),
            },
        )

    if task.status == TASK_FAILED:
        return JSONResponse(
            status_code=409,
            content={
                **task.to_status_payload(http_request),
                "message": "Task execution failed",
            },
        )

    return await build_sync_file_parse_response(
        background_tasks=background_tasks,
        task=task,
        request=http_request,
    )


@app.post(
    path="/tasks",
    status_code=202,
    summary="Submit an asynchronous parse task",
    description=(
        "Submit files for parsing and return immediately with a task id that can be "
        "checked via the task status and result endpoints."
    ),
)
async def submit_parse_task(
    http_request: Request,
    request_options: Annotated[
        ParseRequestOptions, Depends(parse_request_form)
    ],
):
    task_manager = get_task_manager()
    task = await create_async_parse_task(request_options)
    return build_task_submission_response(task, http_request, task_manager)


@app.get(path="/tasks/{task_id}", name="get_async_task_status")
async def get_async_task_status(task_id: str, request: Request):
    task_manager = get_task_manager()
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_manager.build_status_payload(task, request)


@app.get(path="/tasks/{task_id}/result", name="get_async_task_result")
async def get_async_task_result(
    task_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    task_manager = get_task_manager()
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in (TASK_PENDING, TASK_PROCESSING):
        return JSONResponse(
            status_code=202,
            content={
                **task.to_status_payload(request),
                "message": "Task result is not ready yet",
            },
        )

    if task.status == TASK_FAILED:
        return JSONResponse(
            status_code=409,
            content={
                **task.to_status_payload(request),
                "message": "Task execution failed",
            },
        )

    return await build_result_response(
        background_tasks=background_tasks,
        status_code=200,
        output_dir=task.output_dir,
        pdf_file_names=task.file_names,
        backend=task.backend,
        parse_method=task.parse_method,
        return_md=task.return_md,
        return_middle_json=task.return_middle_json,
        return_model_output=task.return_model_output,
        return_content_list=task.return_content_list,
        return_images=task.return_images,
        response_format_zip=task.response_format_zip,
        return_original_file=task.return_original_file,
        zip_filename=f"{task.task_id}.zip",
        s3_image_urls=task._s3_image_urls,
    )


@app.get(path="/health")
async def health_check():
    task_manager = getattr(app.state, "task_manager", None)
    if task_manager is None or not task_manager.is_healthy():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": __version__,
                "error": (
                    "Task manager is not initialized"
                    if task_manager is None
                    else task_manager.last_worker_error
                    or (
                        "Task cleanup loop is not running"
                        if (
                            task_manager.task_retention_seconds > 0
                            and task_manager.cleanup_task is None
                        )
                        else "Task dispatcher is not running"
                    )
                ),
            },
        )

    stats = task_manager.get_stats()
    return {
        "status": "healthy",
        "version": __version__,
        "protocol_version": API_PROTOCOL_VERSION,
        "queued_tasks": stats[TASK_PENDING],
        "processing_tasks": stats[TASK_PROCESSING],
        "completed_tasks": stats[TASK_COMPLETED],
        "failed_tasks": stats[TASK_FAILED],
        "max_concurrent_requests": get_max_concurrent_requests(),
        "processing_window_size": get_processing_window_size(
            default=DEFAULT_PROCESSING_WINDOW_SIZE
        ),
        "task_retention_seconds": task_manager.task_retention_seconds,
        "task_cleanup_interval_seconds": task_manager.task_cleanup_interval_seconds,
    }


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
@click.option("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="Server port (default: 8000)")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
@click.option(
    "--allow-public-http-client",
    is_flag=True,
    help=(
        "Allow *-http-client backends and server_url even when binding the API to "
        "0.0.0.0 or ::."
    ),
)
@click.option(
    "--enable-vlm-preload",
    "enable_vlm_preload",
    type=bool,
    default=False,
    help="Preload the local VLM model during mineru-api startup.",
)
def main(
    ctx,
    host,
    port,
    reload,
    allow_public_http_client,
    enable_vlm_preload,
    **kwargs,
):
    del kwargs
    raw_config = arg_parse(ctx)
    raw_config["enable_vlm_preload"] = enable_vlm_preload
    service_config, model_config = split_service_and_model_config(raw_config)
    public_bind_exposed = is_public_bind_host(host)

    app.state.service_config = service_config
    app.state.config = model_config
    configure_public_http_client_policy(
        app,
        public_bind_exposed=public_bind_exposed,
        allow_public_http_client=allow_public_http_client,
    )
    os.environ["MINERU_API_ENABLE_VLM_PRELOAD"] = (
        "1" if service_config["enable_vlm_preload"] else "0"
    )
    os.environ[MINERU_API_PUBLIC_BIND_EXPOSED_ENV] = "1" if public_bind_exposed else "0"
    os.environ[MINERU_API_ALLOW_PUBLIC_HTTP_CLIENT_ENV] = (
        "1" if allow_public_http_client else "0"
    )
    warn_if_public_http_client_policy(host, allow_public_http_client)
    access_log = not env_flag_enabled("MINERU_API_DISABLE_ACCESS_LOG")

    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")

    if reload:
        uvicorn.run(
            "mineru.cli.fast_api:app",
            host=host,
            port=port,
            reload=True,
            access_log=access_log,
        )
    else:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=False,
            access_log=access_log,
        )
        server = uvicorn.Server(config)
        install_stdin_shutdown_watcher(server)
        server.run()


if __name__ == "__main__":
    main()
