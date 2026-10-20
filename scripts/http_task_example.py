#!/usr/bin/env python3
"""仅使用 Python 标准库的 tasks HTTP 示例：清单、逐文件上传、轮询与 ZIP 下载。"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import secrets
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
from urllib.parse import quote, urlencode, urljoin, urlsplit

TERMINAL = frozenset({"completed", "partial", "failed", "canceled"})
STATUSES = TERMINAL | {"waiting_upload", "queued", "running"}
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
    """固定文件顺序和客户端标识，恢复任务时仍按同一份清单对应文件。"""

    path: Path
    client_file_id: str
    size: int
    sha256sum: str | None

    def manifest(self) -> dict[str, Any]:
        """生成表单清单条目；关闭 hash 时完全省略该可选字段。"""
        item: dict[str, Any] = {"client_file_id": self.client_file_id, "filename": self.path.name, "bytes": self.size}
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
    """为小型表单主体提供可重新创建的发送迭代器。"""
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
        self.by_id = {item.client_file_id: item for item in files}
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
        self.status_url: str | None = None

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
                """闭包固定本次表单字节，避免把文件内容放进 JSON 请求。"""
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
        return self.validate_task(payload)

    def validate_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """校验任务身份和文件映射，恢复请求不能误把文件上传给另一项。"""
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id or (self.task_id is not None and task_id != self.task_id):
            raise ClientError("Missing or inconsistent task_id")
        if payload.get("status") not in STATUSES:
            raise ClientError("Unknown task status")
        if not isinstance(payload.get("status_url"), str) or not payload["status_url"]:
            raise ClientError("Missing status_url")
        files = payload.get("files")
        if not isinstance(files, list) or len(files) != len(self.files):
            raise ClientError("Task file count does not match local inputs")
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise ClientError("Invalid task file entry")
            client_id = item.get("client_file_id")
            if not isinstance(client_id, str) or client_id not in self.by_id or client_id in seen:
                raise ClientError("Invalid or duplicate client_file_id")
            seen.add(client_id)
            local = self.by_id[client_id]
            if item.get("filename") != local.path.name or item.get("bytes") != local.size:
                raise ClientError("Task manifest differs from local inputs; use the original file list and order")
            if local.sha256sum is not None and item.get("sha256sum") not in {None, local.sha256sum}:
                raise ClientError("Task hash differs from the local file")
            if item.get("upload_status") not in {"pending", "ready"}:
                raise ClientError("Unknown upload_status")
        uploads = payload.get("uploads")
        if not isinstance(uploads, list):
            raise ClientError("Missing uploads list")
        seen_uploads: set[str] = set()
        pending = {item["client_file_id"] for item in files if item["upload_status"] == "pending"}
        for link in uploads:
            if not isinstance(link, dict):
                raise ClientError("Invalid upload descriptor")
            client_id = link.get("client_file_id")
            if not isinstance(client_id, str) or client_id not in self.by_id or client_id in seen_uploads:
                raise ClientError("Invalid or duplicate upload client_file_id")
            seen_uploads.add(client_id)
            if link.get("upload_method") != "PUT" or not isinstance(link.get("upload_url"), str) or not link["upload_url"]:
                raise ClientError("Invalid upload URL or method")
            if not isinstance(link.get("upload_headers"), dict):
                raise ClientError("Invalid upload headers")
        if payload["status"] == "waiting_upload":
            if seen_uploads != pending:
                raise ClientError("Upload descriptors do not match pending files")
        elif seen_uploads:
            raise ClientError("Task no longer accepts uploads but still returns upload URLs")
        if payload["status"] in {"queued", "running", "completed", "partial"} and pending:
            raise ClientError("Task started before all files became ready")
        if payload["status"] in {"completed", "partial"}:
            if not isinstance(payload.get("result_url"), str) or not payload["result_url"]:
                raise ClientError("Completed task has no ZIP download URL")
        self.task_id = task_id
        self.status_url = self.resolve(payload["status_url"])
        return payload

    def _form(self) -> list[tuple[str, str]]:
        """所有解析参数都编码到表单，包括 JSON 文本形式的 manifest。"""
        values = [
            ("manifest", json.dumps([file.manifest() for file in self.files], ensure_ascii=False)),
            ("ocr_mode", os.environ.get("OCR_MODE", "auto")),
            ("page_range", os.environ.get("PAGE_RANGE", "")),
        ]
        if os.environ.get("MINERU_TIER"):
            values.append(("tier", os.environ["MINERU_TIER"]))
        formats = [value.strip() for value in os.environ.get("OUTPUT_FORMATS", "markdown").split(",")]
        if not all(formats):
            raise ClientError("OUTPUT_FORMATS contains an empty entry")
        values.extend(("output_formats", value) for value in dict.fromkeys([*formats, "zip"]))
        return values

    def create(self, mode: str) -> dict[str, Any]:
        """创建只执行一次；单文件直传也返回异步任务，不等待服务端同步解析。"""
        values = self._form()
        if mode == "preflight":
            return self.json_request(
                "POST",
                self.base + "/v1/tasks",
                data=urlencode(values).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        boundary = "mineru-" + secrets.token_hex(16)
        prefix = bytearray()
        for name, value in values:
            prefix.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        # 逻辑文件名来自 manifest；固定 part 文件名避免引号、换行等字符破坏 MIME 头。
        prefix.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="upload.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode()
        )
        suffix = f"\r\n--{boundary}--\r\n".encode()

        def multipart_body() -> Generator[bytes, None, None]:
            """只在内存保留小型表单元数据，文件主体仍从磁盘分块发送。"""
            yield bytes(prefix)
            yield from _file_body(self.files[0].path)
            yield suffix

        return self.json_request(
            "POST",
            self.base + "/v1/tasks",
            body=multipart_body,
            length=len(prefix) + self.files[0].size + len(suffix),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def refresh(self) -> dict[str, Any]:
        """只查询已有任务，任何查询失败都不能触发创建重试。"""
        if self.status_url is None:
            raise ClientError("No task status URL")
        return self.json_request("GET", self.status_url)

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
        """逐文件上传与恢复，已经 ready 的文件永远不会重新发送。"""
        for original in task["uploads"]:
            client_id = original["client_file_id"]
            local = self.by_id[client_id]
            link = original
            for attempt in range(3):

                def upload_body() -> Generator[bytes, None, None]:
                    """每次重试重新打开同一个文件，单文件从头重传。"""
                    yield from _file_body(local.path)

                try:
                    with self.request(
                        "PUT",
                        link["upload_url"],
                        headers=link["upload_headers"],
                        body=upload_body,
                        length=local.size,
                    ) as response:
                        if not 200 <= response.status < 300:
                            raise self._http_error(response)
                    break
                except ClientError as exc:
                    if not exc.retryable and exc.status != 403:
                        raise
                    # 请求可能已经落盘但响应丢失；先看状态，再决定是否重传。
                    current = self.refresh_after_transfer_error()
                    remote = next(item for item in current["files"] if item["client_file_id"] == client_id)
                    if remote["upload_status"] == "ready":
                        break
                    if current["status"] != "waiting_upload":
                        return current
                    replacement = next((item for item in current["uploads"] if item["client_file_id"] == client_id), None)
                    if replacement is None:
                        raise ClientError("Pending file has no upload URL") from exc
                    if exc.status == 403 and replacement["upload_url"] == link["upload_url"]:
                        raise
                    if attempt == 2:
                        raise ClientError(f"Upload failed after three attempts: {local.path.name}") from exc
                    link = replacement
                    time.sleep(attempt + 1)
        return self.refresh()

    def download(self, task: dict[str, Any], output: Path) -> None:
        """只访问返回的 ZIP 链接，支持过期链接刷新和安全的跨源重定向。"""
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
        os.close(descriptor)
        temporary = Path(name)
        url = self.resolve(task["result_url"])
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
                                        raise ClientError(f"Cannot save ZIP: {exc}") from exc
                            expected = response.getheader("Content-Length")
                            if expected is not None and temporary.stat().st_size != int(expected):
                                raise ClientError("Incomplete ZIP download", retryable=True)
                        break
                    else:
                        raise ClientError("Too many download redirects")
                    if not zipfile.is_zipfile(temporary):
                        raise ClientError("Downloaded result is not a ZIP archive")
                    temporary.replace(output)
                    return
                except ClientError as exc:
                    if attempt == 2 or (not exc.retryable and exc.status not in {401, 403, 404}):
                        raise
                    current = self.refresh_after_transfer_error()
                    if current["status"] not in {"completed", "partial"}:
                        raise ClientError("Task no longer has a downloadable result") from exc
                    replacement = self.resolve(current["result_url"])
                    if exc.status in {401, 403, 404} and replacement == url:
                        raise
                    url = replacement
                    time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)

    def run(self, mode: str, output: Path) -> int:
        """一条命令等待异步任务并保存 ZIP，超时保留任务供后续恢复。"""
        resume = os.environ.get("TASK_ID", "")
        if resume:
            self.task_id = resume
            self.status_url = self.base + "/v1/tasks/" + quote(resume, safe="")
            task = self.refresh()
        else:
            task = self.create(mode)
        print(f"task_id={self.task_id}", file=sys.stderr)
        if task["status"] == "waiting_upload":
            task = self.upload_missing(task)
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
        if task["status"] in {"failed", "canceled"}:
            print(
                json.dumps({"status": task["status"], "error": task.get("error"), "files": task["files"]}, ensure_ascii=False),
                file=sys.stderr,
            )
            return 1
        self.download(task, output)
        print(f"Saved {output}")
        return 2 if task["status"] == "partial" else 0


def main(arguments: list[str]) -> int:
    """构造真实文件清单并报告可恢复的失败，所有文件读取都采用流式处理。"""
    client: TaskClient | None = None
    try:
        if not arguments:
            raise ClientError("usage: bash scripts/http_task_example.sh <file> [file ...]")
        mode = os.environ.get("UPLOAD_MODE", "preflight")
        if mode not in {"preflight", "direct"}:
            raise ClientError("UPLOAD_MODE must be preflight or direct")
        if mode == "direct" and len(arguments) != 1:
            raise ClientError("Direct multipart upload accepts one file; use preflight for multiple files")
        send_hash = os.environ.get("SEND_HASH", "1")
        if send_hash not in {"0", "1"}:
            raise ClientError("SEND_HASH must be 0 or 1")
        inputs: list[InputFile] = []
        for index, value in enumerate(arguments):
            path = Path(value).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise ClientError(f"Input must be a non-empty file: {value}")
            hasher = hashlib.sha256() if send_hash == "1" else None
            if hasher is not None:
                for chunk in _file_body(path):
                    hasher.update(chunk)
            inputs.append(InputFile(path, f"f{index + 1}", path.stat().st_size, hasher.hexdigest() if hasher else None))
        client = TaskClient(inputs)
        return client.run(mode, Path(os.environ.get("RESULT_FILE", "result.zip")))
    except (ClientError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if client is not None and client.task_id:
            print(f"Resume with TASK_ID={client.task_id} and the original file list", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
