# Copyright (c) Opendatalab. All rights reserved.

"""旧版 Office 解析使用的固定安全限制。"""

from typing import Final

MAX_ENTRY_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ASSET_TOTAL_BYTES: Final = 128 * 1024 * 1024
MAX_GRID_SLOTS: Final = 4_000_000
MAX_RECORD_DEPTH: Final = 64
MAX_RECORDS: Final = 16_000_000
MAX_PICTURE_RECORDS: Final = 100_000
MAX_USER_EDIT_CHAIN: Final = 100
