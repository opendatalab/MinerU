# Copyright (c) Opendatalab. All rights reserved.
"""The local MinerU document library: public SDK entry points.

`DoclibClient` talks to the local MinerU server (start it with
`mineru server start`); the request/response models here are the stable
SDK surface. Less common types stay in `mineru.doclib.types`.
"""

from __future__ import annotations

from .base import DoclibInterface
from .client import DoclibClient
from .types import (
    DocContentResponse,
    DocInfo,
    FindResponse,
    ListDocsResponse,
    ParseInfo,
    ParseRequest,
    ParseResponse,
    ParseStatus,
    ScanInfo,
    ScanRequest,
    SearchResponse,
)

__all__ = [
    "DocContentResponse",
    "DocInfo",
    "DoclibClient",
    "DoclibInterface",
    "FindResponse",
    "ListDocsResponse",
    "ParseInfo",
    "ParseRequest",
    "ParseResponse",
    "ParseStatus",
    "ScanInfo",
    "ScanRequest",
    "SearchResponse",
]
