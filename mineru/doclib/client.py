"""Product SDK — communicates with the local mineru doclib over UDS."""

from __future__ import annotations

from typing import Any

import httpx

from mineru.constants import SOCKET_PATH
from mineru.errors import MineruError, ServerNotRunningError


class MineruClient:
    """httpx-based client for the local mineru doclib."""

    def __init__(self, socket_path: str = SOCKET_PATH, timeout: int = 60) -> None:
        transport = httpx.HTTPTransport(uds=socket_path)
        self._client = httpx.Client(
            transport=transport, base_url="http://mineru", timeout=timeout
        )

    def close(self) -> None:
        self._client.close()

    # ── helpers ────────────────────────────────────────────────

    def _get(self, url_path: str, **params: Any) -> dict:
        data = self._request("GET", url_path, query_params=params or None)
        return data

    def _post(self, url_path: str, json_data: dict | None = None) -> dict:
        data = self._request("POST", url_path, json_data=json_data)
        return data

    def _delete(self, url_path: str, **params: Any) -> dict:
        data = self._request("DELETE", url_path, query_params=params or None)
        return data

    def _request(
        self,
        method: str,
        url_path: str,
        query_params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict:
        try:
            if method == "GET":
                resp = self._client.get(url_path, params=query_params or {})
            elif method == "POST":
                resp = self._client.post(url_path, json=json_data or {})
            else:
                resp = self._client.delete(url_path, params=query_params or {})
            resp.raise_for_status()
        except httpx.ConnectError:
            raise ServerNotRunningError() from None
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        if "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                raise _error_from_response(err)
        return data

    # ── server ─────────────────────────────────────────────────

    def server_status(self) -> dict:
        return self._get("/server/status")

    def shutdown(self) -> dict:
        return self._post("/shutdown")

    # ── parse ──────────────────────────────────────────────────

    def parse(
        self,
        path: str,
        *,
        tier: str | None = None,
        pages: str | None = None,
        force: bool = False,
        remote: bool = False,
        remote_url: str | None = None,
    ) -> dict:
        return self._post(
            "/parse",
            {
                "path": path,
                "tier": tier,
                "pages": pages,
                "force": force,
                "remote": remote,
                "remote_url": remote_url,
            },
        )

    def parse_status(self, sha256: str, tier: str) -> dict:
        return self._get("/parse/status", sha256=sha256, tier=tier)

    def parse_content(self, sha256: str, tier: str, output: str | None = None) -> dict:
        params = {"sha256": sha256, "tier": tier}
        if output:
            params["output"] = output
        return self._get("/parse/content", **params)

    # ── search ─────────────────────────────────────────────────

    def search(
        self, query: str, file_type: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict:
        return self._get("/search", q=query, type=file_type, limit=limit, offset=offset)

    def find(self, query: str, limit: int = 50) -> dict:
        return self._get("/find", q=query, limit=limit)

    # ── info ───────────────────────────────────────────────────

    def info(self, file_path: str) -> dict:
        return self._get("/info", path=file_path)

    # ── config ─────────────────────────────────────────────────

    def config_show(self) -> dict:
        return self._get("/config")

    def config_watch_add(
        self, path: str, *, removable: bool = False, label: str | None = None
    ) -> dict:
        return self._post(
            "/config/watch", {"path": path, "removable": removable, "label": label}
        )

    def config_watch_list(self) -> dict:
        return self._get("/config/watch")

    def config_watch_rm(self, path: str) -> dict:
        return self._delete("/config/watch", path=path)

    def config_exclude_add(self, pattern: str, priority: int = 0) -> dict:
        return self._post("/config/exclude", {"pattern": pattern, "priority": priority})

    def config_exclude_list(self) -> dict:
        return self._get("/config/exclude")

    def config_exclude_rm(self, rule_id: int) -> dict:
        return self._delete(f"/config/exclude/{rule_id}")

    def config_parsing_rules_add(
        self,
        pattern: str,
        *,
        tier: str | None = None,
        pages: str | None = None,
        remote: bool = False,
        name: str | None = None,
    ) -> dict:
        return self._post(
            "/config/parsing-rules",
            {
                "pattern": pattern,
                "tier": tier,
                "pages": pages,
                "remote": remote,
                "name": name,
            },
        )

    def config_parsing_rules_list(self) -> dict:
        return self._get("/config/parsing-rules")

    def config_parsing_rules_rm(self, rule_id: int) -> dict:
        return self._delete(f"/config/parsing-rules/{rule_id}")


def _error_from_response(error: dict) -> MineruError:
    code = error.get("code", "internal_error")
    message = error.get("message", "")
    param = error.get("param")
    return MineruError(code, message, param)
