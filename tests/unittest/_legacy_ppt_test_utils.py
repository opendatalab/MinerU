"""构造不依赖 Office 的最小 PowerPoint 97–2003 测试文件。"""

from __future__ import annotations

import struct
import zlib

from _mtef_test_utils import _TINY_PNG, build_equation_object


def _ppt_record(version_instance: int, record_type: int, payload: bytes) -> bytes:
    """构造一条 PPT record。"""

    return struct.pack("<HHI", version_instance, record_type, len(payload)) + payload


def _ppt_container(instance: int, record_type: int, payload: bytes) -> bytes:
    """构造 recVer=0xF 的 PPT container。"""

    return _ppt_record((instance << 4) | 0xF, record_type, payload)


def _officeart_record(
    record_type: int,
    payload: bytes,
    *,
    version: int = 0,
    instance: int = 0,
) -> bytes:
    """构造一条可嵌入 PPT record tree 的 OfficeArt record。"""

    return _ppt_record((instance << 4) | version, record_type, payload)


def _equation_preview_bse() -> bytes:
    """构造一个内嵌 PNG 的 OfficeArt BSE。"""

    blip = _officeart_record(0xF01E, b"\x00" * 17 + _TINY_PNG)
    return _officeart_record(0xF007, b"\x00" * 36 + blip, version=2)


def _equation_shape(object_id: int, *, preview: bool) -> bytes:
    """构造带 ExObjRefAtom、坐标及可选 pib 的 PPT shape。"""

    fsp = _officeart_record(
        0xF00A,
        struct.pack("<II", object_id + 1024, 0),
        version=2,
        instance=1,
    )
    fopt = (
        _officeart_record(
            0xF00B,
            struct.pack("<HI", 0x0104, 1),
            version=3,
            instance=1,
        )
        if preview
        else b""
    )
    top = 320
    left = 480
    anchor = _officeart_record(
        0xF010,
        struct.pack("<4h", top, left, 3000, 1900),
    )
    reference = _ppt_record(0, 0x0BC1, struct.pack("<I", object_id))
    client_data = _officeart_record(0xF011, reference)
    return _officeart_record(
        0xF004,
        fsp + fopt + anchor + client_data,
        version=0xF,
    )


def build_equation_ppt(
    formulas: list[bytes],
    *,
    preview: bool = True,
    compressed: bool = True,
    declared_size: int | None = None,
) -> bytes:
    """构造每页一个 Equation Editor OLE 对象的 PPT persist fixture。"""

    if not formulas:
        raise ValueError("equation PPT fixture requires at least one formula")

    def persist_atom(persist_ref: int, slide_id: int) -> bytes:
        """构造 SlidePersistAtom。"""

        return _ppt_record(
            0,
            0x03F3,
            struct.pack("<IIIII", persist_ref, 0, 0, slide_id, 0),
        )

    slide_persist_ids = list(range(2, 2 + len(formulas)))
    storage_persist_ids = list(
        range(2 + len(formulas), 2 + len(formulas) * 2)
    )
    slide_list = _ppt_container(
        0,
        0x0FF0,
        b"".join(
            persist_atom(persist_id, 256 + index)
            for index, persist_id in enumerate(slide_persist_ids)
        ),
    )
    external_objects = []
    for index, storage_id in enumerate(storage_persist_ids, start=1):
        embed_atom = _ppt_record(0, 0x0FCD, b"\x00" * 8)
        object_atom = _ppt_record(
            1,
            0x0FC3,
            struct.pack("<6I", 1, 0, index, 6, storage_id, 0),
        )
        prog_id = _ppt_record(2 << 4, 0x0FBA, "Equation.3".encode("utf-16le"))
        external_objects.append(
            _ppt_container(0, 0x0FCC, embed_atom + object_atom + prog_id)
        )
    object_list = _ppt_container(0, 0x0409, b"".join(external_objects))
    drawing_group = (
        _officeart_record(
            0xF000,
            _officeart_record(0xF001, _equation_preview_bse(), version=0xF),
            version=0xF,
        )
        if preview
        else b""
    )
    document = _ppt_container(0, 0x03E8, slide_list + object_list + drawing_group)

    records: list[tuple[int, bytes]] = [(1, document)]
    for index, persist_id in enumerate(slide_persist_ids, start=1):
        slide_atom = _ppt_record(
            2,
            0x03EF,
            struct.pack("<I", 0) + b"\x00" * 8 + struct.pack("<IIHH", 0, 0, 0, 0),
        )
        records.append(
            (
                persist_id,
                _ppt_container(
                    0,
                    0x03EE,
                    slide_atom + _equation_shape(index, preview=preview),
                ),
            )
        )
    for persist_id, mtef in zip(storage_persist_ids, formulas, strict=True):
        storage = build_equation_object(mtef)
        if compressed:
            size = len(storage) if declared_size is None else declared_size
            payload = struct.pack("<I", size) + zlib.compress(storage)
            record = _ppt_record(1 << 4, 0x1011, payload)
        else:
            record = _ppt_record(0, 0x1011, storage)
        records.append((persist_id, record))

    records.sort(key=lambda item: item[0])
    stream = b""
    offsets: dict[int, int] = {}
    for persist_id, record in records:
        offsets[persist_id] = len(stream)
        stream += record
    count = max(offsets)
    directory = struct.pack("<I", 1 | (count << 20)) + struct.pack(
        f"<{count}I",
        *(offsets[index] for index in range(1, count + 1)),
    )
    directory_offset = len(stream)
    stream += _ppt_record(0, 0x1772, directory)
    edit_offset = len(stream)
    stream += _ppt_record(
        0,
        0x0FF5,
        struct.pack(
            "<IIIIIIHH",
            0,
            0,
            0,
            directory_offset,
            1,
            count + 1,
            0,
            0,
        ),
    )
    current_user = _ppt_record(
        0,
        0x0FF6,
        struct.pack("<III", 20, 0xE391C05F, edit_offset),
    )
    return _build_cfb(
        [("Current User", current_user), ("PowerPoint Document", stream)]
    )


def _build_cfb(streams: list[tuple[str, bytes]]) -> bytes:
    """生成所有 stream 均使用常规 FAT sector 的最小 CFB v3。"""

    sector_size = 512
    end_of_chain = 0xFFFF_FFFE
    fat_sector = 0xFFFF_FFFD
    free_sector = 0xFFFF_FFFF
    fat: list[int] = []
    sectors: list[bytes] = []

    def add_chain(data: bytes) -> int:
        """把字节追加为 FAT chain 并返回首 sector。"""

        if not data:
            return end_of_chain
        padded = data + b"\x00" * (-len(data) % sector_size)
        first = len(sectors)
        count = len(padded) // sector_size
        for index in range(count):
            sectors.append(padded[index * sector_size : (index + 1) * sector_size])
            fat.append(first + index + 1 if index + 1 < count else end_of_chain)
        return first

    def directory_entry(
        name: str,
        kind: int,
        right: int,
        child: int,
        start: int,
        size: int,
    ) -> bytes:
        """构造一个 128 字节 CFB directory entry。"""

        none = 0xFFFF_FFFF
        raw_name = name.encode("utf-16-le") + b"\x00\x00"
        return (
            raw_name
            + b"\x00" * (64 - len(raw_name))
            + struct.pack("<HBBIII", len(raw_name), kind, 1, none, right, child)
            + b"\x00" * 16
            + b"\x00" * 4
            + b"\x00" * 16
            + struct.pack("<IQ", start, size)
        )

    starts: list[tuple[int, int]] = []
    for _, data in streams:
        padded = data + b"\x00" * (4096 - len(data)) if 0 < len(data) < 4096 else data
        starts.append((add_chain(padded), len(padded)))

    none = 0xFFFF_FFFF
    entries = [
        directory_entry(
            "Root Entry",
            5,
            none,
            1 if streams else none,
            end_of_chain,
            0,
        )
    ]
    for index, ((name, _), (start, size)) in enumerate(zip(streams, starts, strict=True), start=1):
        right = index + 1 if index < len(streams) else none
        entries.append(directory_entry(name, 2, right, none, start, size))
    directory = b"".join(entries)
    directory += b"\x00" * (-len(directory) % sector_size)
    directory_start = add_chain(directory)

    fat_count = 1
    while (len(fat) + fat_count + 127) // 128 > fat_count:
        fat_count += 1
    fat_start = len(sectors)
    fat_ids = list(range(fat_start, fat_start + fat_count))
    full_fat = fat + [fat_sector] * fat_count
    full_fat += [free_sector] * (fat_count * 128 - len(full_fat))
    for index in range(fat_count):
        sectors.append(struct.pack("<128I", *full_fat[index * 128 : (index + 1) * 128]))

    difat = fat_ids + [free_sector] * (109 - len(fat_ids))
    header = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        + b"\x00" * 16
        + struct.pack(
            "<HHHHHHIIIIIIIII",
            0x003E,
            0x0003,
            0xFFFE,
            9,
            6,
            0,
            0,
            0,
            fat_count,
            directory_start,
            0,
            4096,
            end_of_chain,
            0,
            end_of_chain,
        )
        + struct.pack("<I", 0)
        + struct.pack("<109I", *difat)
    )
    return header + b"".join(sectors)


def build_sparse_notes_ppt() -> bytes:
    """构造仅第二页含 speaker notes 的两页 PPT。"""

    def slide(text: str) -> bytes:
        """构造含一段直接文本的 slide container。"""

        slide_atom = _ppt_record(2, 0x03EF, struct.pack("<I", 0) + b"\x00" * 8 + struct.pack("<IIHH", 0, 0, 0, 0))
        text_header = _ppt_record(0, 0x0F9F, struct.pack("<I", 1))
        text_bytes = _ppt_record(0, 0x0FA8, text.encode("ascii"))
        return _ppt_container(0, 0x03EE, slide_atom + text_header + text_bytes)

    def notes(slide_ref: int, text: str) -> bytes:
        """构造绑定到指定 slide id 的 notes container。"""

        notes_atom = _ppt_record(1, 0x03F1, struct.pack("<IHH", slide_ref, 0, 0))
        text_header = _ppt_record(0, 0x0F9F, struct.pack("<I", 2))
        text_bytes = _ppt_record(0, 0x0FA8, text.encode("ascii"))
        return _ppt_container(0, 0x03F0, notes_atom + text_header + text_bytes)

    def persist_atom(persist_ref: int, slide_id: int) -> bytes:
        """构造 SlidePersistAtom。"""

        return _ppt_record(0, 0x03F3, struct.pack("<IIIII", persist_ref, 0, 0, slide_id, 0))

    slide_list = _ppt_container(0, 0x0FF0, persist_atom(2, 256) + persist_atom(3, 257))
    notes_list = _ppt_container(2, 0x0FF0, persist_atom(4, 1301))
    document = _ppt_container(0, 0x03E8, slide_list + notes_list)
    records = [
        (1, document),
        (2, slide("First slide text\r")),
        (3, slide("Second slide text\r")),
        (4, notes(257, "Notes for the second slide\r")),
    ]
    stream = b""
    offsets: dict[int, int] = {}
    for persist_id, record in records:
        offsets[persist_id] = len(stream)
        stream += record
    directory = struct.pack("<I", 1 | (4 << 20)) + struct.pack("<4I", *(offsets[index] for index in range(1, 5)))
    directory_offset = len(stream)
    stream += _ppt_record(0, 0x1772, directory)
    edit_offset = len(stream)
    stream += _ppt_record(0, 0x0FF5, struct.pack("<IIIIIIHH", 0, 0, 0, directory_offset, 1, 5, 0, 0))
    current_user = _ppt_record(0, 0x0FF6, struct.pack("<III", 20, 0xE391C05F, edit_offset))
    return _build_cfb([("Current User", current_user), ("PowerPoint Document", stream)])


def build_multimaster_ppt() -> bytes:
    """构造两页分别继承不同母版 bullet/粗斜体默认值的 PPT。"""

    def master_style(bullet_on: bool, character_mask: int, character_style: int) -> bytes:
        """构造一个层级的 TextMasterStyleAtom。"""

        paragraph = struct.pack("<IH", 0x0001, 0x0001 if bullet_on else 0x0000)
        character = struct.pack("<IH", character_mask, character_style)
        return _ppt_record(1 << 4, 0x0FA3, struct.pack("<H", 1) + paragraph + character)

    def slide(master_id: int, text: str) -> bytes:
        """构造引用指定 master id 的 slide。"""

        slide_atom = _ppt_record(
            2,
            0x03EF,
            struct.pack("<I", 0) + b"\x00" * 8 + struct.pack("<IIHH", master_id, 0, 0, 0),
        )
        text_header = _ppt_record(0, 0x0F9F, struct.pack("<I", 1))
        text_bytes = _ppt_record(0, 0x0FA8, text.encode("ascii"))
        return _ppt_container(0, 0x03EE, slide_atom + text_header + text_bytes)

    def persist_atom(persist_ref: int, stable_id: int) -> bytes:
        """构造 slide/master 共用的 persist atom。"""

        return _ppt_record(0, 0x03F3, struct.pack("<IIIII", persist_ref, 0, 0, stable_id, 0))

    slide_list = _ppt_container(0, 0x0FF0, persist_atom(4, 256) + persist_atom(5, 257))
    master_list = _ppt_container(1, 0x0FF0, persist_atom(2, 1001) + persist_atom(3, 1002))
    document = _ppt_container(0, 0x03E8, slide_list + master_list)
    master_a = _ppt_container(0, 0x03F8, master_style(True, 0x0001, 0x0001))
    master_b = _ppt_container(0, 0x03F8, master_style(False, 0x0002, 0x0002))
    records = [
        (1, document),
        (2, master_a),
        (3, master_b),
        (4, slide(1001, "Alpha master body text\r")),
        (5, slide(1002, "Beta master body text\r")),
    ]
    stream = b""
    offsets: dict[int, int] = {}
    for persist_id, record in records:
        offsets[persist_id] = len(stream)
        stream += record
    directory = struct.pack("<I", 1 | (5 << 20)) + struct.pack(
        "<5I", *(offsets[index] for index in range(1, 6))
    )
    directory_offset = len(stream)
    stream += _ppt_record(0, 0x1772, directory)
    edit_offset = len(stream)
    stream += _ppt_record(
        0,
        0x0FF5,
        struct.pack("<IIIIIIHH", 0, 0, 0, directory_offset, 1, 6, 0, 0),
    )
    current_user = _ppt_record(0, 0x0FF6, struct.pack("<III", 20, 0xE391C05F, edit_offset))
    return _build_cfb([("Current User", current_user), ("PowerPoint Document", stream)])


def build_deep_nested_ppt() -> bytes:
    """构造超过固定 record depth 的攻击形状。"""

    nested = _ppt_record(0, 0x0FA8, b"deep")
    for _ in range(200):
        nested = _ppt_container(0, 0x03EE, nested)
    return _build_cfb([("Current User", b""), ("PowerPoint Document", nested)])
