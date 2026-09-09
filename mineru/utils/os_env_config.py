# Copyright (c) Opendatalab. All rights reserved.
import os
from typing import Optional

# Environment variables through which a deployment can pin the size of the CPU
# thread pools of the native math libraries (libgomp / MKL / OpenBLAS) used by
# the torch CPU path. If any of them is set we must not override the operator.
CPU_THREAD_LIMIT_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)

# Upper bound applied when no explicit thread limit is configured.
# libgomp keeps one worker team of ``num_threads`` threads alive per master thread
# that entered a parallel region, and the workers of a ``ThreadPoolExecutor``
# (used by ``asyncio.to_thread``) live for the whole process lifetime. Without a
# cap, every pool worker that touches a torch CPU op permanently costs one team
# sized to the whole machine, so a long-running server grows towards
# ``pool_size * os.cpu_count()`` OS threads. See issue #5472.
DEFAULT_CPU_THREAD_LIMIT = 8


def get_op_num_threads(env_name: str) -> int:
    env_value = os.getenv(env_name, None)
    return get_value_from_string(env_value, -1)


def cpu_thread_limit_already_configured() -> bool:
    """Whether the deployment already pinned the native CPU thread pools."""
    return any(os.getenv(env_name) is not None for env_name in CPU_THREAD_LIMIT_ENV_NAMES)


def resolve_cpu_thread_limit(
    default_limit: int = DEFAULT_CPU_THREAD_LIMIT,
    cpu_count: Optional[int] = None,
) -> int:
    """Resolve the number of CPU threads a single torch CPU operator may use.

    ``MINERU_CPU_NUM_THREADS`` takes precedence, then ``MINERU_INTRA_OP_NUM_THREADS``
    so a single variable can tune both the ONNX and the torch CPU paths. Invalid
    or non-positive values fall back to ``default_limit``, and the result is
    clamped to ``[1, cpu_count]`` like the ONNX session options in
    ``mineru.model.table.rec``.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1

    configured = get_op_num_threads("MINERU_CPU_NUM_THREADS")
    if configured == -1:
        configured = get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
    if configured == -1:
        configured = default_limit

    return max(1, min(configured, max(1, cpu_count)))


def apply_cpu_thread_limit(
    default_limit: int = DEFAULT_CPU_THREAD_LIMIT,
    cpu_count: Optional[int] = None,
) -> Optional[int]:
    """Cap the per-thread torch CPU pool unless the deployment configured it.

    Applied through ``torch.set_num_threads`` rather than by exporting
    ``OMP_NUM_THREADS``, so that the VLM engines keep their own
    ``OMP_NUM_THREADS=1`` default (they only set it when it is still unset).

    :return: the limit that was applied, or ``None`` when nothing was changed
        because the deployment already pinned the thread pools or torch is not
        installed.
    """
    if cpu_thread_limit_already_configured():
        return None

    try:
        import torch
    except ImportError:
        return None

    limit = resolve_cpu_thread_limit(default_limit=default_limit, cpu_count=cpu_count)
    if torch.get_num_threads() <= limit:
        return None

    torch.set_num_threads(limit)
    return limit


def get_load_images_timeout() -> int:
    env_value = os.getenv('MINERU_PDF_RENDER_TIMEOUT', None)
    return get_value_from_string(env_value, 300)


def get_load_images_threads() -> int:
    env_value = os.getenv('MINERU_PDF_RENDER_THREADS', None)
    return get_value_from_string(env_value, 3)


def get_value_from_string(env_value: str, default_value: int) -> int:
    if env_value is not None:
        try:
            num_threads = int(env_value)
            if num_threads > 0:
                return num_threads
        except ValueError:
            return default_value
    return default_value


if __name__ == '__main__':
    print(get_value_from_string('1', -1))
    print(get_value_from_string('0', -1))
    print(get_value_from_string('-1', -1))
    print(get_value_from_string('abc', -1))
    print(get_load_images_timeout())