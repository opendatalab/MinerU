# Copyright (c) Opendatalab. All rights reserved.
"""Pydantic response models for the MinerU FastAPI service.

These models are attached to the routes in :mod:`mineru.cli.fast_api` via
FastAPI's ``response_model`` and ``responses`` parameters so that the OpenAPI
documentation (``/openapi.json`` and ``/docs``) accurately describes the JSON
payloads returned by each endpoint.

They are intentionally documentation-oriented. The route handlers continue to
return :class:`fastapi.responses.JSONResponse` (or
:class:`fastapi.responses.FileResponse` for ZIP downloads) directly. When a
handler returns a ``Response`` object, FastAPI bypasses response-model
validation and serialization at runtime — so adding these models does not
change runtime behavior. They only describe the documented contract for API
consumers.

Every model allows extra fields
(``model_config = ConfigDict(extra='allow')``) so the schema stays
forward-compatible with additive changes to the response payloads (for
example the conditional ``queued_ahead`` key that is appended only by
``AsyncTaskManager.build_status_payload``).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


_SHARED_CONFIG = ConfigDict(extra='allow')


class TaskStatusPayload(BaseModel):
    """Shared status payload returned for parse tasks.

    Emitted by ``GET /tasks/{task_id}`` and embedded in many other responses.
    The ``queued_ahead`` key is appended by
    ``AsyncTaskManager.build_status_payload`` and is therefore declared on the
    derived models that include it, not on this base.
    """

    model_config = _SHARED_CONFIG

    task_id: str = Field(description='unique identifier for the parse task')
    status: str = Field(
        description='current task lifecycle state: pending, processing, '
        'completed, or failed'
    )
    backend: str = Field(
        description='parse backend used for the task (e.g. pipeline or vlm)'
    )
    file_names: list[str] = Field(
        description='normalized stems of the uploaded input files'
    )
    created_at: str = Field(
        description='ISO 8601 timestamp marking when the task was created'
    )
    started_at: Optional[str] = Field(
        default=None,
        description='ISO 8601 timestamp marking when processing started; '
        'absent until the task begins processing',
    )
    completed_at: Optional[str] = Field(
        default=None,
        description='ISO 8601 timestamp marking when processing finished; '
        'absent until the task reaches a terminal state',
    )
    error: Optional[str] = Field(
        default=None,
        description='error message captured when the task failed; absent '
        'otherwise',
    )
    status_url: str = Field(
        description='absolute URL to poll the task status'
    )
    result_url: str = Field(
        description='absolute URL to fetch the task result'
    )


class TaskStatusResponse(TaskStatusPayload):
    """Response for ``GET /tasks/{task_id}``.

    Extends the base status payload with the queue position.
    """

    queued_ahead: int = Field(
        description='number of pending tasks submitted before this one that '
        'are still waiting to start; 0 when the task is no longer pending'
    )


class TaskSubmissionResponse(TaskStatusPayload):
    """Response for ``POST /tasks`` (asynchronous submission)."""

    queued_ahead: int = Field(
        description='number of pending tasks submitted before this one that '
        'are still waiting to start'
    )
    message: str = Field(
        description='human-readable confirmation message'
    )


class ParseResultInner(BaseModel):
    """Per-file parsing result returned inside the ``results`` map.

    Which keys are present depends on the request's ``return_*`` flags. The
    text/JSON values are the raw file contents (or ``null`` when the
    corresponding result file is missing on disk). Image values are
    ``data:`` URLs with base64-encoded payloads.
    """

    model_config = _SHARED_CONFIG

    md_content: Optional[str] = Field(
        default=None,
        description='Markdown rendering of the parsed document',
    )
    middle_json: Optional[str] = Field(
        default=None,
        description='middle-json representation serialized as a string',
    )
    model_output: Optional[str] = Field(
        default=None,
        description='raw model output serialized as a JSON string',
    )
    content_list: Optional[str] = Field(
        default=None,
        description='content list serialized as a JSON string',
    )
    images: dict[str, str] = Field(
        default_factory=dict,
        description='map of image basename to base64 data URL for extracted '
        'images; empty when no images were emitted',
    )


class FileParseResultResponse(TaskStatusPayload):
    """JSON response for ``POST /file_parse`` on success.

    Returns the task status payload together with the MinerU version and the
    per-file parsing results. When ``response_format_zip`` is set on the
    request, the endpoint instead returns a binary ``application/zip``
    response (documented via the route's ``responses`` parameter).
    """

    version: str = Field(description='MinerU release version')
    results: dict[str, ParseResultInner] = Field(
        description='map of input file stem to its parsing result; only keys '
        'requested via the return_* flags are populated'
    )


class TaskResultResponse(BaseModel):
    """JSON response for ``GET /tasks/{task_id}/result`` on success.

    Contains only the backend, MinerU version, and the per-file parsing
    results. When ``response_format_zip`` is set on the original task, the
    endpoint instead returns a binary ``application/zip`` response
    (documented via the route's ``responses`` parameter).
    """

    model_config = _SHARED_CONFIG

    backend: str = Field(
        description='parse backend used for the task (e.g. pipeline or vlm)'
    )
    version: str = Field(description='MinerU release version')
    results: dict[str, ParseResultInner] = Field(
        description='map of input file stem to its parsing result; only keys '
        'requested via the return_* flags are populated'
    )


class TaskMessageResponse(TaskStatusPayload):
    """Status payload accompanying an informational or error message.

    Used by endpoints that return the task status together with a short
    human-readable ``message`` (e.g. result not ready yet, task execution
    failed, task manager became unavailable while waiting).
    """

    message: str = Field(
        description='human-readable status or error message'
    )


class HealthResponse(BaseModel):
    """Healthy service response for ``GET /health``."""

    model_config = _SHARED_CONFIG

    status: str = Field(description='health indicator; "healthy" on success')
    version: str = Field(description='MinerU release version')
    protocol_version: int = Field(description='API protocol version')
    queued_tasks: int = Field(
        description='number of tasks currently in the pending state'
    )
    processing_tasks: int = Field(
        description='number of tasks currently in the processing state'
    )
    completed_tasks: int = Field(
        description='number of tasks in the completed state'
    )
    failed_tasks: int = Field(
        description='number of tasks in the failed state'
    )
    max_concurrent_requests: int = Field(
        description='maximum number of parse requests the server may process '
        'concurrently'
    )
    processing_window_size: int = Field(
        description='processing window size used by the parser'
    )
    task_retention_seconds: int = Field(
        description='seconds a terminal task is retained before cleanup'
    )
    task_cleanup_interval_seconds: int = Field(
        description='interval between cleanup sweeps of expired tasks'
    )


class UnhealthyResponse(BaseModel):
    """Unhealthy service response for ``GET /health``."""

    model_config = _SHARED_CONFIG

    status: str = Field(
        description='health indicator; "unhealthy" on failure'
    )
    version: str = Field(description='MinerU release version')
    error: str = Field(
        description='description of the underlying failure (e.g. task '
        'manager is not initialized, dispatcher is not running)'
    )


class HTTPExceptionResponse(BaseModel):
    """Response body for non-success status codes raised via HTTPException.

    Matches the default FastAPI ``HTTPException`` JSON shape (``{"detail": ...}``).
    """

    model_config = _SHARED_CONFIG

    detail: str = Field(
        description='human-readable error detail'
    )
