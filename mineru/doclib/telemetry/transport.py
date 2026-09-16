"""Telemetry upload transport helpers."""

from __future__ import annotations

import hmac
import time
from hashlib import sha256

from .constants import TELEMETRY_APP_KEY, TELEMETRY_APP_SECRET


def signed_headers(body: bytes, *, timestamp: int | None = None) -> dict[str, str]:
    ts = int(time.time() * 1000) if timestamp is None else timestamp
    message = TELEMETRY_APP_KEY.encode("utf-8") + str(ts).encode("utf-8") + body
    signature = hmac.new(TELEMETRY_APP_SECRET.encode("utf-8"), message, sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Track-App-Key": TELEMETRY_APP_KEY,
        "X-Track-Ts": str(ts),
        "X-Track-Sign": signature,
    }


__all__ = ["signed_headers"]
