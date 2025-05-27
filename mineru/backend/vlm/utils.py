import os
import re
from base64 import b64decode

import httpx

_timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
_file_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")
_data_uri_regex = re.compile(r"^data:[^;,]+;base64,")


def load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        response = httpx.get(uri, timeout=_timeout)
        return response.content
    if uri.startswith("file://"):
        with open(uri[len("file://") :], "rb") as file:
            return file.read()
    if uri.lower().endswith(_file_exts):
        with open(uri, "rb") as file:
            return file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)


async def aio_load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        async with httpx.AsyncClient(timeout=_timeout) as client:
            response = await client.get(uri)
            return response.content
    if uri.startswith("file://"):
        with open(uri[len("file://") :], "rb") as file:
            return file.read()
    if uri.lower().endswith(_file_exts):
        with open(uri, "rb") as file:
            return file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)
