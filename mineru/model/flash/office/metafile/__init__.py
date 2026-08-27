# Copyright (c) Opendatalab. All rights reserved.
"""Office WMF/EMF 的跨平台内部渲染能力。"""

from .api import render_metafile
from .models import (
    MetafileDiagnostic,
    MetafileError,
    MetafileMalformedError,
    MetafileRenderResult,
    MetafileResourceLimitError,
    MetafileUnsupportedError,
)

__all__ = [
    "MetafileDiagnostic",
    "MetafileError",
    "MetafileMalformedError",
    "MetafileRenderResult",
    "MetafileResourceLimitError",
    "MetafileUnsupportedError",
    "render_metafile",
]
