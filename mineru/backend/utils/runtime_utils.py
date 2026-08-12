# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import time
from typing import Any


def exclude_progress_bar_idle_time(progress_bar: Any, idle_since: float | None, now: float | None = None) -> float | None:
    """Exclude non-processing idle time from a reused tqdm progress bar."""
    if progress_bar is None or idle_since is None:
        return

    if now is None:
        now = time.time()

    idle_duration = now - idle_since
    if idle_duration <= 0:
        return

    if hasattr(progress_bar, "start_t"):
        progress_bar.start_t += idle_duration
    if hasattr(progress_bar, "last_print_t"):
        progress_bar.last_print_t = now
    if hasattr(progress_bar, "last_print_n") and hasattr(progress_bar, "n"):
        progress_bar.last_print_n = progress_bar.n
