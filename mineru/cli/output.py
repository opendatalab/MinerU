"""Rich-based output formatting for mineru CLI."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_json

Table: Any
Panel: Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def print_error(msg: str) -> None:
    if console:
        console.print("Error:", style="red", end=" ")
        console.print(msg)
    else:
        print(f"Error: {msg}", file=sys.stderr)


def print_success(msg: str) -> None:
    if console:
        console.print(msg, style="green")
    else:
        print(msg)


def print_info(msg: str) -> None:
    if console:
        console.print(msg, style="dim")
    else:
        print(msg)


def print_json(data: Any) -> None:
    if isinstance(data, BaseModel):
        print(to_json(data, indent=2).decode("utf-8"))
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


# ── parse ──────────────────────────────────────────────────────


def format_parse_result(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    status = _get(data, "status", "?")
    tip = _get(data, "tip", "")

    if status == "pending" or status == "parsing":
        print_info(f"Parse {status}... {tip}")
    elif status == "failed":
        print_error(f"Parse failed: {_get(data, 'error_code', '?')} — {_get(data, 'error_msg', '')}")
    else:
        print_success(f"Parse complete (tier={_get(data, 'tier', '?')}) {tip}")


# ── search ─────────────────────────────────────────────────────


def format_search_results(data: Any, json_mode: bool = False) -> None:
    results = _get(data, "results", [])
    total = _get(data, "total", 0)

    if json_mode:
        print_json(data)
        return

    if not results:
        print_info("No results found.")
        return

    table = Table(title=f"Search results ({total} total)")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Tier", style="green", width=10)
    table.add_column("Snippet", style="dim")
    table.add_column("Paths", style="dim")

    for r in results:
        table.add_row(
            _get(r, "title") or _get(r, "filename", "?"),
            _get(r, "tier", ""),
            (_get(r, "snippet", "") or "")[:80],
            ", ".join(_get(r, "paths", [])[:3]),
        )

    if console:
        console.print(table)
    else:
        for r in results:
            print(f"{_get(r, 'title')} [{_get(r, 'tier')}] — {', '.join(_get(r, 'paths', []))}")


# ── server status ──────────────────────────────────────────────


def format_server_status(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    if not _get(data, "running"):
        print_info("Server is not running.")
        return

    if console:
        table = Table(title="MinerU Server")
        table.add_column("Field", style="cyan")
        table.add_column("Current", style="green")
        table.add_row("PID", str(_get(data, "pid", "?")))
        table.add_row("Uptime", f"{_get(data, 'uptime_seconds', 0):.0f}s")
        table.add_row("Home", _get(data, "mineru_home", ""))
        table.add_row("Version", _get(data, "version", ""))
        table.add_row("Python", _get(data, "python_version", ""))
        table.add_row("Socket", _get(data, "socket_path", ""))
        table.add_row("Data dir", _get(data, "data_dir", ""))
        table.add_row("SQLite", _get(data, "sqlite_path", ""))
        table.add_row("SQLite size", _format_bytes(_get(data, "sqlite_size_bytes")))
        table.add_row("Log", _get(data, "log_path", ""))
        tcp_data = _get(data, "tcp")
        tcp_enabled = bool(_get(tcp_data, "enabled", False))
        tcp_host = _get(tcp_data, "host", "") or "-"
        tcp_port = _get(tcp_data, "port")
        if tcp_enabled:
            tcp_value = f"http://{tcp_host}:{tcp_port}" if tcp_port is not None else f"http://{tcp_host}:(pending)"
        else:
            tcp_value = "disabled"
        table.add_row("TCP", tcp_value)
        table.add_row("Files tracked", str(_get(data, "files_total", 0)))
        table.add_row("Docs indexed", str(_get(data, "docs_total", 0)))
        table.add_row("Active scans", str(_get(data, "active_scan_count", 0)))
        table.add_row("Last scan", _format_timestamp_ms(_get(data, "last_scan_at")))
        table.add_row("Parse queue", str(_get(data, "parse_queue_length", 0)))
        table.add_row("Ingest queue", str(_get(data, "ingest_queue_length", 0)))
        table.add_row("Watches", str(_get(data, "watch_count", 0)))
        console.print(table)

        workers = _get(data, "workers")
        if workers:
            worker_table = Table(title="Workers")
            worker_table.add_column("Component", style="cyan")
            worker_table.add_column("Running", style="green")
            worker_table.add_column("Workers", justify="right")
            worker_table.add_row("Watch", "yes" if _get(workers, "watch_running", False) else "no", "-")
            worker_table.add_row(
                "Scan",
                "yes" if _get(workers, "scan_running", False) else "no",
                str(_get(workers, "scan_workers", 0)),
            )
            worker_table.add_row(
                "Ingest",
                "yes" if _get(workers, "ingest_running", False) else "no",
                str(_get(workers, "ingest_workers", 0)),
            )
            worker_table.add_row(
                "Parse",
                "yes" if _get(workers, "parse_running", False) else "no",
                str(_get(workers, "parse_workers", 0)),
            )
            worker_table.add_row(
                "Device monitor",
                "yes" if _get(workers, "device_monitor_running", False) else "no",
                "-",
            )
            worker_table.add_row(
                "Compaction",
                "yes" if _get(workers, "compaction_running", False) else "no",
                "-",
            )
            worker_table.add_row(
                "Health check",
                "yes" if _get(workers, "health_check_running", False) else "no",
                "-",
            )
            console.print(worker_table)

        watch_stats = _get(data, "watch_stats", [])
        if watch_stats:
            watch_table = Table(title="Watch Stats")
            watch_table.add_column("Path", style="cyan", no_wrap=True)
            watch_table.add_column("Status", style="green")
            watch_table.add_column("Files", justify="right")
            watch_table.add_column("Active", justify="right")
            watch_table.add_column("Deleted", justify="right")
            watch_table.add_column("Unreachable", justify="right")
            watch_table.add_column("Pending ingest", justify="right")
            watch_table.add_column("Errors", justify="right")
            watch_table.add_column("Docs", justify="right")
            watch_table.add_column("Parses done/pending/parsing/failed", justify="right")
            for item in watch_stats:
                parse_counts = (
                    f"{_get(item, 'parse_done_count', 0)}/"
                    f"{_get(item, 'parse_pending_count', 0)}/"
                    f"{_get(item, 'parse_parsing_count', 0)}/"
                    f"{_get(item, 'parse_failed_count', 0)}"
                )
                watch_table.add_row(
                    _get(item, "path", ""),
                    _get(item, "status", ""),
                    str(_get(item, "total_files", 0)),
                    str(_get(item, "active_files", 0)),
                    str(_get(item, "deleted_files", 0)),
                    str(_get(item, "unreachable_files", 0)),
                    str(_get(item, "pending_ingest_files", 0)),
                    str(_get(item, "file_error_count", 0)),
                    str(_get(item, "doc_count", 0)),
                    parse_counts,
                )
            console.print(watch_table)

        error_summary = _get(data, "error_summary")
        error_rows = _error_summary_rows(error_summary)
        if error_rows:
            error_table = Table(title="Error Summary")
            error_table.add_column("Scope", style="cyan")
            error_table.add_column("Code", style="red")
            error_table.add_column("Count", justify="right")
            for scope, code, count in error_rows:
                error_table.add_row(scope, code, str(count))
            console.print(error_table)

        recent_scans = _get(data, "recent_scans", [])
        if recent_scans:
            scan_table = Table(title="Recent Scans")
            scan_table.add_column("ID", justify="right")
            scan_table.add_column("Kind", style="cyan")
            scan_table.add_column("Source")
            scan_table.add_column("Status", style="green")
            scan_table.add_column("Path", no_wrap=True)
            scan_table.add_column("Seen", justify="right")
            scan_table.add_column("New", justify="right")
            scan_table.add_column("Changed", justify="right")
            scan_table.add_column("Deleted", justify="right")
            scan_table.add_column("Errors", justify="right")
            scan_table.add_column("Error code", style="red")
            for item in recent_scans:
                scan_table.add_row(
                    str(_get(item, "id", "")),
                    _get(item, "kind", ""),
                    _get(item, "source", ""),
                    _get(item, "status", ""),
                    _get(item, "path", ""),
                    str(_get(item, "files_seen", 0)),
                    str(_get(item, "files_new", 0)),
                    str(_get(item, "files_changed", 0)),
                    str(_get(item, "files_deleted", 0)),
                    str(_get(item, "files_error", 0)),
                    _get(item, "error_code", "") or "",
                )
            console.print(scan_table)

        # parse-server status
        ps_data = _get(data, "parse_server")
        if ps_data:
            ps_table = Table(title="Parse Server")
            ps_table.add_column("Target", style="cyan")
            ps_table.add_column("Healthy", style="green")
            ps_table.add_column("Endpoint", style="dim")
            ps_table.add_column("Managed", style="dim")
            ps_table.add_column("Restart", justify="right")
            ps_table.add_column("Last probe", style="dim")
            ps_table.add_column("Last ok", style="dim")
            ps_table.add_column("Last fail", style="dim")
            ps_table.add_column("Tiers", style="green")
            for label, key in [("Local", "local"), ("Remote", "remote")]:
                ps = _get(ps_data, key, {})
                if _get(ps, "starting"):
                    healthy_str = "starting"
                elif _get(ps, "healthy"):
                    healthy_str = "yes"
                else:
                    healthy_str = "no"
                tiers_str = ", ".join(_get(ps, "supported_tiers", [])) or "-"
                mode = _get(ps, "mode", "")
                label_str = f"{label} ({mode})" if mode else label
                endpoint = _get(ps, "url", "") or "-"
                managed_str = "-"
                restart_str = "-"
                if key == "local":
                    mode_text = _get(ps, "mode", "")
                    if mode_text == "managed":
                        managed_tier = _get(ps, "managed_tier", "") or "-"
                        managed_pid = _get(ps, "managed_pid")
                        managed_running = "yes" if _get(ps, "managed_running", False) else "no"
                        managed_str = (
                            f"tier={managed_tier}, pid={managed_pid or '-'}, running={managed_running}"
                        )
                    elif mode_text == "self_hosted":
                        managed_str = _get(ps, "self_hosted_url", "") or "-"
                    restart_str = f"{_get(ps, 'restart_count', 0)}/{_get(ps, 'max_restart_attempts', 0)}"
                ps_table.add_row(
                    label_str,
                    healthy_str,
                    endpoint,
                    managed_str,
                    restart_str,
                    _format_age_ms(_get(ps, "last_probe_at")),
                    _format_age_ms(_get(ps, "last_success_at")),
                    _format_age_ms(_get(ps, "last_failure_at")),
                    tiers_str,
                )
            console.print(ps_table)

        # recent logs
        logs = _get(data, "recent_logs", [])
        if logs:
            log_text = "".join(logs)  # last 100 lines from server
            panel = Panel(log_text.strip() or "(empty)", title="Recent Logs", border_style="dim")
            console.print(panel)
    else:
        print(f"Server running (PID {_get(data, 'pid')}), {_get(data, 'files_total')} files")


# ── info ───────────────────────────────────────────────────────


def format_info(data: Any, json_mode: bool = False) -> None:
    if json_mode:
        print_json(data)
        return

    file_data = _get(data, "file")
    if not file_data:
        print_info("File not found in database.")
        return
    doc_data = _get(data, "doc")

    if console:
        table = Table(title=f"File Info: {_get(file_data, 'filename', '?')}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Path", _get(file_data, "path", "?"))
        table.add_row("Type", _get(file_data, "ext", "?"))
        table.add_row("Size", _format_bytes(_get(file_data, "size_bytes")))
        table.add_row("SHA-256", (_get(file_data, "sha256") or "")[:16] + "...")
        table.add_row("Page count", str(_get(doc_data, "page_count", "?") if doc_data else "?"))
        table.add_row("Title", (_get(doc_data, "title") if doc_data else None) or "—")
        table.add_row("Author", (_get(doc_data, "author") if doc_data else None) or "—")

        tiers = _get(data, "parsed_tiers", [])
        if tiers:
            tier_str = ", ".join(f"{_get(t, 'tier')}={_get(t, 'status')}" for t in tiers)
            table.add_row("Tiers", tier_str)

        console.print(table)
    else:
        print(f"{_get(file_data, 'filename')} — {(_get(doc_data, 'title') if doc_data else None) or 'No title'}")


# ── helpers ────────────────────────────────────────────────────


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"


def _format_timestamp_ms(ts: int | None) -> str:
    if ts is None:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1000))
    except Exception:
        return str(ts)


def _format_age_ms(ts: int | None) -> str:
    if ts is None:
        return "-"
    try:
        age_seconds = max(0, int(time.time() - (ts / 1000)))
    except Exception:
        return str(ts)
    if age_seconds < 60:
        return f"{age_seconds}s ago"
    age_minutes = age_seconds // 60
    if age_minutes < 60:
        return f"{age_minutes}m ago"
    age_hours = age_minutes // 60
    if age_hours < 24:
        return f"{age_hours}h ago"
    age_days = age_hours // 24
    return f"{age_days}d ago"


def _error_summary_rows(error_summary: Any) -> list[tuple[str, str, int]]:
    if not error_summary:
        return []
    rows: list[tuple[str, str, int]] = []
    for scope, key in (("file", "file_errors"), ("doc", "doc_errors"), ("parse", "parse_errors")):
        for bucket in _get(error_summary, key, []):
            rows.append((scope, _get(bucket, "code", ""), _get(bucket, "count", 0)))
    return rows
