#!/usr/bin/env python3
"""仅使用 Python 标准库的 tasks HTTP 示例：JSON 清单、逐文件上传、轮询与独立结果下载。"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

TERMINAL = frozenset({"completed", "partial", "failed", "canceled"})
STATUSES = TERMINAL | {"waiting-file", "pending", "running"}
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
BodyFactory = Callable[[], Generator[bytes, None, None]]


class ClientError(RuntimeError):
    """保存可重试属性，避免将错误 hash、鉴权或非法响应反复提交。"""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        """区分网络临时错误、HTTP 状态和确定性的协议错误。"""
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass(frozen=True)
class InputFile:
    """固定原始文件顺序，创建或恢复时绑定服务端返回的文件标识。"""

    path: Path
    size: int
    sha256sum: str | None

    def task_file(self) -> dict[str, Any]:
        """生成 JSON 清单条目；关闭 hash 时完全省略该可选字段。"""
        item: dict[str, Any] = {"file_name": self.path.name, "bytes": self.size}
        if self.sha256sum is not None:
            item["sha256sum"] = self.sha256sum
        return item


def _file_body(path: Path) -> Generator[bytes, None, None]:
    """文件上传按固定块读取，中断或重试均关闭本次文件句柄。"""
    try:
        with path.open("rb") as source:
            while chunk := source.read(CHUNK_BYTES):
                yield chunk
    except OSError as exc:
        raise ClientError(f"Cannot read input file: {path.name}: {exc}") from exc


def _bytes_body(data: bytes) -> Generator[bytes, None, None]:
    """为小型 JSON 主体提供可重新创建的发送迭代器。"""
    yield data


def _origin(url: str) -> tuple[str, str, int]:
    """严格比较协议、主机和有效端口，阻止跨源转发 API Key。"""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ClientError("Expected an http(s) URL without embedded credentials")
    return parts.scheme.lower(), parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80)


def _integer(name: str, default: int, minimum: int = 1) -> int:
    """把环境变量明确校验为有界客户端预算，错误配置立即退出。"""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ClientError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ClientError(f"{name} must be at least {minimum}")
    return value


class TaskClient:
    """公共客户端只识别 task_id、状态和服务返回的上传及下载链接。"""

    def __init__(self, files: list[InputFile]) -> None:
        """读取连接与轮询预算，不依赖 MinerU SDK 或云存储 SDK。"""
        self.base = os.environ.get("MINERU_API_URL", "").rstrip("/")
        if not self.base:
            raise ClientError("Set MINERU_API_URL to the API base URL")
        self.origin = _origin(self.base)
        if urlsplit(self.base).query or urlsplit(self.base).fragment:
            raise ClientError("MINERU_API_URL must not contain a query or fragment")
        self.key = os.environ.get("MINERU_API_KEY", "")
        self.files = files
        self.by_id: dict[str, InputFile] = {}
        self.connect_timeout = _integer("CONNECT_TIMEOUT", 10)
        self.timeout = _integer("REQUEST_TIMEOUT", 600)
        self.max_polls = _integer("MAX_POLLS", 300, minimum=0)
        try:
            self.poll_interval = float(os.environ.get("POLL_INTERVAL", "1"))
        except ValueError as exc:
            raise ClientError("POLL_INTERVAL must be a non-negative number") from exc
        if not 0 <= self.poll_interval < float("inf"):
            raise ClientError("POLL_INTERVAL must be a non-negative finite number")
        self.task_id: str | None = None
        self.saved_ids: set[str] = set()
        self.uploaded_ids: set[str] = set()
        self.upload_attempts: dict[str, int] = {}

    def resolve(self, url: str) -> str:
        """解析服务返回的绝对或相对链接，保留 OSS 签名查询串。"""
        resolved = urljoin(self.base + "/", url)
        _origin(resolved)
        return resolved

    def _headers(self, url: str, headers: Mapping[str, str] | None) -> dict[str, str]:
        """只向 API 同源地址附加 Key，同时保留明确给出的存储服务请求头。"""
        result = dict(headers or {})
        for name, value in result.items():
            if not isinstance(name, str) or not isinstance(value, str) or "\r" in name + value or "\n" in name + value:
                raise ClientError("Invalid service-provided HTTP header")
        same_origin = _origin(url) == self.origin
        if self.key and same_origin:
            if not any(name.lower() == "authorization" for name in result):
                result["Authorization"] = f"Bearer {self.key}"
        elif self.key:
            for name, value in result.items():
                if name.lower() == "authorization" and value == f"Bearer {self.key}":
                    raise ClientError("Refusing to forward the API Key to another origin")
        return result

    @contextmanager
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: BodyFactory | None = None,
        length: int | None = None,
    ) -> Iterator[http.client.HTTPResponse]:
        """流式发送与读取一次请求；创建任务不进行隐式重试或 POST 重定向。"""
        resolved = self.resolve(url)
        parts = urlsplit(resolved)
        connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parts.hostname, parts.port, timeout=self.connect_timeout)
        outgoing = self._headers(resolved, headers)
        if length is not None:
            outgoing = {
                name: value for name, value in outgoing.items() if name.lower() not in {"content-length", "transfer-encoding"}
            }
            outgoing["Content-Length"] = str(length)
        iterator = body() if body is not None else None
        try:
            try:
                connection.connect()
                if connection.sock is not None:
                    connection.sock.settimeout(self.timeout)
                target = parts.path or "/"
                if parts.query:
                    target += "?" + parts.query
                connection.request(method, target, body=iterator, headers=outgoing, encode_chunked=False)
                response = connection.getresponse()
            except (OSError, http.client.HTTPException, socket.timeout) as exc:
                raise ClientError(f"{method} connection failed: {exc}", retryable=True) from exc
            try:
                yield response
            finally:
                response.close()
        finally:
            if iterator is not None:
                iterator.close()
            connection.close()

    def _read_response(self, response: http.client.HTTPResponse, size: int) -> bytes:
        """只把响应读取异常标记为网络错误，不能把本地写盘错误错误地加入重试。"""
        try:
            return response.read(size)
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            raise ClientError(f"Response interrupted: {exc}", retryable=True) from exc

    def _http_error(self, response: http.client.HTTPResponse) -> ClientError:
        """提取小型错误消息，不在日志中输出带签名的下载地址或认证信息。"""
        content = self._read_response(response, MAX_JSON_BYTES + 1)
        message = f"HTTP {response.status}"
        try:
            value = json.loads(content)
            detail = value.get("error") if isinstance(value, dict) else None
            if isinstance(detail, dict):
                message += f": {detail.get('code', 'request_failed')}: {detail.get('message', '')}"
        except (ValueError, UnicodeError):
            pass
        return ClientError(message, retryable=response.status in RETRYABLE_HTTP, status=response.status)

    def json_request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        body: BodyFactory | None = None,
        length: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """读取有界的任务元数据，HTTP 200 的非法 JSON 也立即报告错误。"""
        if data is not None:

            def data_body() -> Generator[bytes, None, None]:
                """闭包固定本次 JSON 元数据字节，不把源文件主体放入创建请求。"""
                yield from _bytes_body(data)

            body, length = data_body, len(data)
        with self.request(method, url, body=body, length=length, headers=headers) as response:
            if response.status not in {200, 202}:
                raise self._http_error(response)
            content = self._read_response(response, MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ClientError("Task metadata response is too large")
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeError) as exc:
            raise ClientError("Task response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ClientError("Task response must be a JSON object")
        return self.validate_creation(payload) if method == "POST" else self.validate_task(payload)

    def _validate_file_map(self, payload: dict[str, Any], *, details: bool) -> dict[str, Any]:
        """按原始顺序绑定服务端 ID，此后校验同一映射，避免同名输入被错误关联。"""
        task_id = payload.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in {".", ".."}
            or (self.task_id is not None and task_id != self.task_id)
        ):
            raise ClientError("Missing or inconsistent task_id")
        if payload.get("status") not in STATUSES:
            raise ClientError("Unknown task status")
        files = payload.get("files")
        if not isinstance(files, list) or len(files) != len(self.files):
            raise ClientError("Task file count does not match local inputs")
        bindings: dict[str, InputFile] = {}
        for item, local in zip(files, self.files):
            if not isinstance(item, dict):
                raise ClientError("Invalid task file entry")
            file_id = item.get("file_id")
            if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", file_id) or file_id in bindings:
                raise ClientError("Invalid or duplicate file_id")
            if self.by_id and self.by_id.get(file_id) is not local:
                raise ClientError("Task file identity or order changed")
            if item.get("file_name") != local.path.name or (details and item.get("bytes") != local.size):
                raise ClientError("Task files differ from local inputs; use the original file list and order")
            bindings[file_id] = local
            if item.get("status") not in {"waiting-file", "pending", "running", "completed", "failed", "canceled"}:
                raise ClientError("Unknown file status")
            link = item.get("upload")
            if link is None:
                continue
            if not isinstance(link, dict):
                raise ClientError("Invalid upload descriptor")
            if not isinstance(link.get("url"), str) or not link["url"]:
                raise ClientError("Invalid upload URL")
            if not isinstance(link.get("headers"), dict) or type(link.get("expires_at")) is not int:
                raise ClientError("Invalid upload headers or expiry")
            if item["status"] != "waiting-file" or payload["status"] in TERMINAL:
                raise ClientError("Upload descriptor does not match file state")
        self.task_id = task_id
        self.by_id = bindings
        return payload

    def validate_creation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """创建回执仅确认身份和上传动作，不要求详情中的大小、格式或结果字段。"""
        return self._validate_file_map(payload, details=False)

    def validate_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """独立校验 GET 详情和产物契约，省略字段表示当前没有该值。"""
        payload = self._validate_file_map(payload, details=True)
        formats = payload.get("output_formats")
        expected = list(dict.fromkeys(value.strip() for value in os.environ.get("OUTPUT_FORMATS", "default").split(",")))
        if not isinstance(formats, list) or not formats or formats != expected:
            raise ClientError("Task output_formats differ from the original request")
        for item in payload["files"]:
            if item["status"] == "completed":
                if not isinstance(item.get("result_url"), str) or not item["result_url"]:
                    raise ClientError("Completed file has no download URL")
            elif item.get("result_url") is not None:
                raise ClientError("Unfinished file has a download URL")
        return payload

    def create(self) -> dict[str, Any]:
        """用一次 JSON 请求创建清单，文件主体只发送到服务返回的上传链接。"""
        formats = [value.strip() for value in os.environ.get("OUTPUT_FORMATS", "default").split(",")]
        if not formats or any(value not in {"default", "docx", "html", "latex"} for value in formats):
            raise ClientError("OUTPUT_FORMATS must contain default/docx/html/latex")
        body: dict[str, Any] = {
            "files": [file.task_file() for file in self.files],
            "ocr_mode": os.environ.get("OCR_MODE", "auto"),
            "page_range": os.environ.get("PAGE_RANGE", ""),
            "output_formats": list(dict.fromkeys(formats)),
        }
        if os.environ.get("MINERU_TIER"):
            body["tier"] = os.environ["MINERU_TIER"]
        return self.json_request(
            "POST",
            self.base + "/v1/tasks",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )

    def refresh(self) -> dict[str, Any]:
        """只查询已有任务，任何查询失败都不能触发创建重试。"""
        if self.task_id is None or self.task_id in {".", ".."}:
            raise ClientError("No valid task ID")
        return self.json_request("GET", self.base + "/v1/tasks/" + quote(self.task_id, safe=""))

    def refresh_after_transfer_error(self) -> dict[str, Any]:
        """传输结果不确定时先恢复状态查询，无法确认时不盲目重复发送文件。"""
        for attempt in range(3):
            try:
                return self.refresh()
            except ClientError as exc:
                if not exc.retryable or attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    def upload_missing(self, task: dict[str, Any]) -> dict[str, Any]:
        """逐文件上传与恢复，成功 PUT 或已进入后续阶段的文件不重复发送。"""
        for original in task["files"]:
            file_id = original["file_id"]
            local = self.by_id[file_id]
            link = original.get("upload")
            if original["status"] != "waiting-file":
                self.uploaded_ids.add(file_id)
            if link is None or file_id in self.uploaded_ids:
                continue
            if self.upload_attempts.get(file_id, 0) >= 3:
                raise ClientError(f"Upload failed after three attempts: {local.path.name}")
            for attempt in range(self.upload_attempts.get(file_id, 0), 3):
                self.upload_attempts[file_id] = attempt + 1

                def upload_body() -> Generator[bytes, None, None]:
                    """每次重试重新打开同一个文件，单文件从头重传。"""
                    yield from _file_body(local.path)

                try:
                    with self.request(
                        "PUT",
                        link["url"],
                        headers=link["headers"],
                        body=upload_body,
                        length=local.size,
                    ) as response:
                        if not 200 <= response.status < 300:
                            raise self._http_error(response)
                    self.uploaded_ids.add(file_id)
                    break
                except ClientError as exc:
                    if not exc.retryable and exc.status not in {403, 410}:
                        raise
                    # 请求可能已经落盘但响应丢失；先看状态，再决定是否重传。
                    current = self.refresh_after_transfer_error()
                    remote = next(item for item in current["files"] if item["file_id"] == file_id)
                    if remote["status"] != "waiting-file":
                        self.uploaded_ids.add(file_id)
                        break
                    replacement = remote.get("upload")
                    if replacement is None:
                        break
                    if exc.status == 403 and replacement["url"] == link["url"]:
                        raise
                    if attempt == 2:
                        raise ClientError(f"Upload failed after three attempts: {local.path.name}") from exc
                    link = replacement
                    time.sleep(attempt + 1)
        return self.refresh()

    def download(self, task: dict[str, Any], file: dict[str, Any], output: Path) -> None:
        """下载一个已完成文件，按所选格式校验，并安全刷新过期下载链接。"""
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
        os.close(descriptor)
        temporary = Path(name)
        url = self.resolve(file["result_url"])
        try:
            for attempt in range(3):
                try:
                    for _ in range(6):
                        with self.request("GET", url) as response:
                            if response.status in REDIRECTS:
                                location = response.getheader("Location")
                                if not location:
                                    raise ClientError("Download redirect is missing Location")
                                url = self.resolve(urljoin(url, location))
                                continue
                            if response.status != 200:
                                raise self._http_error(response)
                            with temporary.open("wb") as target:
                                while chunk := self._read_response(response, CHUNK_BYTES):
                                    try:
                                        target.write(chunk)
                                    except OSError as exc:
                                        raise ClientError(f"Cannot save result: {exc}") from exc
                            expected = response.getheader("Content-Length")
                            if expected is not None and temporary.stat().st_size != int(expected):
                                raise ClientError("Incomplete result download", retryable=True)
                        break
                    else:
                        raise ClientError("Too many download redirects")
                    formats = list(dict.fromkeys(task["output_formats"]))
                    if formats == ["html"]:
                        if "<html" not in temporary.read_text(encoding="utf-8").lower():
                            raise ClientError("Downloaded result is not an HTML document")
                    else:
                        if not zipfile.is_zipfile(temporary):
                            raise ClientError("Downloaded result is not a ZIP archive")
                        with zipfile.ZipFile(temporary) as archive:
                            if archive.testzip() is not None:
                                raise ClientError("Downloaded ZIP is corrupt")
                            if formats == ["docx"] and "word/document.xml" not in archive.namelist():
                                raise ClientError("Downloaded result is not a DOCX document")
                    temporary.replace(output)
                    return
                except ClientError as exc:
                    if attempt == 2 or (not exc.retryable and exc.status not in {401, 403, 404}):
                        raise
                    current = self.refresh_after_transfer_error()
                    remote = next(item for item in current["files"] if item["file_id"] == file["file_id"])
                    if remote["status"] != "completed":
                        raise ClientError("File no longer has a downloadable result") from exc
                    replacement = self.resolve(remote["result_url"])
                    if exc.status in {401, 403, 404} and replacement == url:
                        raise
                    url = replacement
                    time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)

    def download_completed(self, task: dict[str, Any], output: Path) -> None:
        """按 file_id 保存独立结果，同名输入和多次轮询不会相互覆盖。"""
        formats = list(dict.fromkeys(task["output_formats"]))
        extension = formats[0] if len(formats) == 1 and formats[0] in {"html", "docx"} else "zip"
        for file in task["files"]:
            file_id = file["file_id"]
            if file["status"] != "completed" or file_id in self.saved_ids:
                continue
            destination = output / file_id / (self.by_id[file_id].path.stem + "." + extension)
            self.download(task, file, destination)
            self.saved_ids.add(file_id)
            print(f"Saved {destination}")

    def run(self, output: Path) -> int:
        """上传缺失文件并逐个下载成功结果，失败与取消也保留已经完成的产物。"""
        resume = os.environ.get("TASK_ID", "")
        if resume:
            self.task_id = resume
            task = self.refresh()
        else:
            task = self.create()
        print(f"task_id={self.task_id}", file=sys.stderr)
        if not resume:
            task = self.refresh()
        self.download_completed(task, output)
        for original in list(task["files"]):
            current = next(item for item in task["files"] if item["file_id"] == original["file_id"])
            if current.get("upload") is None:
                continue
            task = self.upload_missing({**task, "files": [current]})
            self.download_completed(task, output)
        polls = 0
        while task["status"] not in TERMINAL:
            if polls >= self.max_polls:
                print(f"Polling budget exhausted; task_id={self.task_id} continues running", file=sys.stderr)
                return 124
            polls += 1
            time.sleep(self.poll_interval)
            try:
                task = self.refresh()
            except ClientError as exc:
                if not exc.retryable:
                    raise
                continue
            if any(
                item["status"] == "waiting-file" and item.get("upload") is not None and item["file_id"] not in self.uploaded_ids
                for item in task["files"]
            ):
                task = self.upload_missing(task)
            self.download_completed(task, output)
        if task["status"] in {"failed", "canceled"}:
            print(
                json.dumps({"status": task["status"], "error": task.get("error"), "files": task["files"]}, ensure_ascii=False),
                file=sys.stderr,
            )
            return 1
        return 2 if task["status"] == "partial" else 0


def main(arguments: list[str]) -> int:
    """构造真实文件清单并报告可恢复的失败，所有文件读取都采用流式处理。"""
    client: TaskClient | None = None
    try:
        if not arguments:
            raise ClientError("usage: bash scripts/http_task_example.sh <file> [file ...]")
        if os.environ.get("UPLOAD_MODE") or os.environ.get("RESULT_FILE"):
            raise ClientError("UPLOAD_MODE and RESULT_FILE were removed; use JSON preflight and RESULT_DIR")
        send_hash = os.environ.get("SEND_HASH", "1")
        if send_hash not in {"0", "1"}:
            raise ClientError("SEND_HASH must be 0 or 1")
        inputs: list[InputFile] = []
        for value in arguments:
            path = Path(value).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise ClientError(f"Input must be a non-empty file: {value}")
            hasher = hashlib.sha256() if send_hash == "1" else None
            if hasher is not None:
                for chunk in _file_body(path):
                    hasher.update(chunk)
            inputs.append(InputFile(path, path.stat().st_size, hasher.hexdigest() if hasher else None))
        client = TaskClient(inputs)
        return client.run(Path(os.environ.get("RESULT_DIR", "results")))
    except (ClientError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if client is not None and client.task_id:
            print(f"Resume with TASK_ID={client.task_id} and the original file list", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
