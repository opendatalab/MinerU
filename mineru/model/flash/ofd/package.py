# Copyright (c) Opendatalab. All rights reserved.
"""在固定预算内读取 OFD ZIP/XML 包。"""

from __future__ import annotations

import posixpath
import re
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from .constants import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_DOCUMENT_COUNT,
    MAX_ENTRY_BYTES,
    MAX_ENTRY_COUNT,
    MAX_TOTAL_BYTES,
    MAX_XML_DEPTH,
    MAX_XML_NODES,
    OFD_KNOWN_VERSIONS,
    OFD_NAMESPACES,
)
from .errors import OfdEncryptedError, OfdParseError, OfdResourceLimitError
from .models import OfdDocumentRef


def local_name(tag: object) -> str:
    """返回 XML 标签不含命名空间的本地名。"""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def namespace_name(tag: object) -> str:
    """返回 Clark notation 标签中的命名空间。"""
    if not isinstance(tag, str) or not tag.startswith("{"):
        return ""
    return tag[1:].split("}", 1)[0]


def first_child(element: etree._Element | None, name: str) -> etree._Element | None:
    """返回指定本地名的首个直接子元素。"""
    if element is None:
        return None
    return next((child for child in element if local_name(child.tag) == name), None)


def first_descendant(element: etree._Element | None, name: str) -> etree._Element | None:
    """返回指定本地名的首个后代元素。"""
    if element is None:
        return None
    return next((child for child in element.iter() if local_name(child.tag) == name), None)


def element_text(element: etree._Element | None) -> str:
    """返回元素折叠首尾空白后的完整文本。"""
    return "" if element is None else "".join(element.itertext()).strip()


def parse_int(value: object) -> int | None:
    """把非负整数字段安全解析为 Python int。"""
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _xml_parser() -> etree.XMLParser:
    """为每个 OFD XML part 创建禁用实体、DTD 和网络的解析器。"""
    return etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
        remove_blank_text=False,
        huge_tree=False,
    )


class OfdPackage:
    """负责 OFD 包身份、成员访问和受限 XML 解析。"""

    def __init__(self, file_bytes: bytes) -> None:
        """打开内存包并在读取正文前校验中央目录。"""
        if len(file_bytes) > MAX_TOTAL_BYTES:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
        try:
            self._zip = ZipFile(BytesIO(file_bytes))
        except (BadZipFile, OSError, ValueError) as exc:
            raise OfdParseError(f"Malformed OFD package: {exc}") from exc
        try:
            self._infos = self._validate_members(self._zip.infolist())
        except Exception:
            self._zip.close()
            raise
        self._cache: dict[str, bytes] = {}
        self._total_read = 0
        self._asset_parts: set[str] = set()
        self._asset_bytes = 0
        self._root: etree._Element | None = None

    @staticmethod
    def _is_safe_member_name(name: str) -> bool:
        """判断 ZIP 成员是否为包内安全 POSIX 路径。"""
        if not name or "\x00" in name or "\\" in name or name.startswith("/"):
            return False
        parts = PurePosixPath(name).parts
        return bool(parts) and all(part not in {"", ".", ".."} for part in parts)

    @classmethod
    def _validate_members(cls, infos: list[ZipInfo]) -> dict[str, ZipInfo]:
        """校验成员数量、路径、加密、压缩方式和声明体积。"""
        if len(infos) > MAX_ENTRY_COUNT:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_entry_count={MAX_ENTRY_COUNT}")
        members: dict[str, ZipInfo] = {}
        total_size = 0
        for info in infos:
            name = info.filename
            if not cls._is_safe_member_name(name):
                raise OfdParseError(f"Malformed OFD package: unsafe member path {name!r}")
            if name in members:
                raise OfdParseError(f"Malformed OFD package: duplicate member {name!r}")
            if info.flag_bits & 0x1:
                raise OfdEncryptedError(f"Encrypted OFD ZIP member is not supported: {name!r}")
            if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                raise OfdParseError(f"Malformed OFD package: unsupported ZIP compression for {name!r}")
            if info.file_size > MAX_ENTRY_BYTES:
                raise OfdResourceLimitError(
                    f"OFD resource limit exceeded: member {name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
                )
            total_size += info.file_size
            if total_size > MAX_TOTAL_BYTES:
                raise OfdResourceLimitError(f"OFD resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
            members[name] = info
        if "OFD.xml" not in members:
            raise OfdParseError("Malformed OFD package: missing required root 'OFD.xml'")
        return members

    def has_part(self, part_name: str) -> bool:
        """返回包内是否存在指定规范成员。"""
        return part_name in self._infos

    def read_part(self, part_name: str, *, required: bool = False, asset: bool = False) -> bytes | None:
        """在成员和累计预算内读取一个包内 part。"""
        info = self._infos.get(part_name)
        if info is None:
            if required:
                raise OfdParseError(f"Malformed OFD package: missing required part {part_name!r}")
            return None
        if part_name in self._cache:
            data = self._cache[part_name]
            if asset:
                self._charge_asset(part_name, len(data))
            return data
        try:
            with self._zip.open(info) as source:
                data = source.read(MAX_ENTRY_BYTES + 1)
        except (BadZipFile, OSError, RuntimeError, ValueError) as exc:
            if required:
                raise OfdParseError(f"Malformed OFD package: cannot read {part_name!r}: {exc}") from exc
            return None
        if len(data) > MAX_ENTRY_BYTES:
            raise OfdResourceLimitError(
                f"OFD resource limit exceeded: member {part_name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        self._total_read += len(data)
        if self._total_read > MAX_TOTAL_BYTES:
            raise OfdResourceLimitError(
                f"OFD resource limit exceeded while reading {part_name!r}: max_total_bytes={MAX_TOTAL_BYTES}"
            )
        if asset:
            self._charge_asset(part_name, len(data))
        self._cache[part_name] = data
        return data

    def _charge_asset(self, part_name: str, byte_count: int) -> None:
        """按唯一成员累计保留的资源字节。"""
        if part_name in self._asset_parts:
            return
        self._asset_parts.add(part_name)
        self._asset_bytes += byte_count
        if self._asset_bytes > MAX_ASSET_TOTAL_BYTES:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}")

    def xml_part(self, part_name: str, *, required: bool = False) -> etree._Element | None:
        """禁用 DTD/实体后解析 XML，并校验节点数量与深度。"""
        data = self.read_part(part_name, required=required)
        if data is None:
            return None
        try:
            root = etree.fromstring(data, parser=_xml_parser())
        except (etree.XMLSyntaxError, ValueError) as exc:
            if required:
                raise OfdParseError(f"Malformed OFD package: invalid XML part {part_name!r}: {exc}") from exc
            return None
        if root.getroottree().docinfo.doctype:
            raise OfdParseError(f"Malformed OFD package: DTD is not allowed in {part_name!r}")
        self._validate_xml_shape(root, part_name)
        return root

    @staticmethod
    def _validate_xml_shape(root: etree._Element, part_name: str) -> None:
        """迭代校验 XML 节点数和最大深度。"""
        node_count = 0
        stack: list[tuple[etree._Element, int]] = [(root, 1)]
        while stack:
            element, depth = stack.pop()
            node_count += 1
            if node_count > MAX_XML_NODES:
                raise OfdResourceLimitError(f"OFD resource limit exceeded: {part_name!r} exceeds max_xml_nodes={MAX_XML_NODES}")
            if depth > MAX_XML_DEPTH:
                raise OfdResourceLimitError(f"OFD resource limit exceeded: {part_name!r} exceeds max_xml_depth={MAX_XML_DEPTH}")
            stack.extend((child, depth + 1) for child in element if isinstance(child.tag, str))

    def root(self) -> etree._Element:
        """读取并验证 OFD.xml 根节点、命名空间、版本和文档类型。"""
        if self._root is not None:
            return self._root
        root = self.xml_part("OFD.xml", required=True)
        assert root is not None
        namespace = namespace_name(root.tag)
        if local_name(root.tag) != "OFD" or namespace not in OFD_NAMESPACES:
            raise OfdParseError(f"Malformed OFD package: unsupported root namespace {namespace!r}")
        version = (root.get("Version") or "").strip()
        if not re.fullmatch(r"1(?:\.\d+)?", version):
            raise OfdParseError(f"Unsupported OFD version: {version or '<missing>'}")
        if version not in OFD_KNOWN_VERSIONS:
            logger.warning(f"OFD_COMPAT_VERSION: parsing unrecognized 1.x version {version!r}")
        doc_type = (root.get("DocType") or "OFD").strip().upper()
        if doc_type not in {"OFD", "OFD-A"}:
            raise OfdParseError(f"Unsupported OFD DocType: {doc_type!r}")
        if not any(local_name(child.tag) == "DocBody" for child in root):
            raise OfdParseError("Malformed OFD package: OFD.xml has no DocBody")
        self._root = root
        return root

    def resolve_reference(self, base_part: str, location: str | None) -> str | None:
        """解析大小写敏感的 ST_Loc，并拒绝包外与网络位置。"""
        raw = unquote((location or "").strip()).replace("\\", "/")
        if not raw:
            return None
        try:
            parsed = urlsplit(raw)
        except ValueError:
            return None
        if parsed.scheme or parsed.netloc or parsed.query:
            return None
        raw_path = parsed.path
        if raw_path.startswith("/"):
            resolved = posixpath.normpath(raw_path).lstrip("/")
        else:
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base_part), raw_path))
        if resolved in {"", ".", ".."} or resolved.startswith("../") or resolved.startswith("/"):
            return None
        return resolved.removeprefix("./")

    def document_refs(self) -> list[OfdDocumentRef]:
        """按 DocBody 声明顺序返回全部文档入口。"""
        refs: list[OfdDocumentRef] = []
        for body in self.root():
            if local_name(body.tag) != "DocBody":
                continue
            if len(refs) >= MAX_DOCUMENT_COUNT:
                raise OfdResourceLimitError(f"OFD resource limit exceeded: max_document_count={MAX_DOCUMENT_COUNT}")
            doc_root = first_child(body, "DocRoot")
            document_part = self.resolve_reference("OFD.xml", element_text(doc_root))
            if document_part is None:
                raise OfdParseError("Malformed OFD package: DocBody has invalid DocRoot")
            signatures = first_child(body, "Signatures")
            signatures_part = self.resolve_reference("OFD.xml", element_text(signatures)) if signatures is not None else None
            metadata: dict[str, str] = {}
            doc_info = first_child(body, "DocInfo")
            if doc_info is not None:
                for key in ("Title", "Author", "Subject", "Keywords", "DocUsage", "Creator", "CreatorVersion"):
                    value = element_text(first_child(doc_info, key))
                    if value:
                        metadata[key] = value
            refs.append(OfdDocumentRef(document_part=document_part, signatures_part=signatures_part, metadata=metadata))
        return refs

    def close(self) -> None:
        """关闭底层 ZipFile。"""
        self._zip.close()

    def __enter__(self) -> OfdPackage:
        """返回当前包以支持 with 生命周期。"""
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """退出 with 块时关闭包。"""
        self.close()


def detect_ofd(file_bytes: bytes) -> bool:
    """按受限 OFD 包身份识别内存字节。"""
    try:
        with OfdPackage(file_bytes) as package:
            package.root()
            return True
    except (BadZipFile, OfdParseError, OSError, ValueError):
        return False


def detect_ofd_path(file_path: str | Path) -> bool:
    """直接从 ZIP 路径读取有限 OFD.xml 并验证包身份。"""
    try:
        path = Path(file_path)
        if path.stat().st_size > MAX_TOTAL_BYTES:
            return False
        with ZipFile(path) as package:
            info = package.getinfo("OFD.xml")
            if info.file_size > MAX_ENTRY_BYTES or info.flag_bits & 0x1 or info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                return False
            with package.open(info) as source:
                data = source.read(MAX_ENTRY_BYTES + 1)
            if len(data) > MAX_ENTRY_BYTES:
                return False
            root = etree.fromstring(data, parser=_xml_parser())
            namespace = namespace_name(root.tag)
            version = (root.get("Version") or "").strip()
            doc_type = (root.get("DocType") or "OFD").strip().upper()
            return (
                not root.getroottree().docinfo.doctype
                and local_name(root.tag) == "OFD"
                and namespace in OFD_NAMESPACES
                and re.fullmatch(r"1(?:\.\d+)?", version) is not None
                and doc_type in {"OFD", "OFD-A"}
                and any(local_name(child.tag) == "DocBody" for child in root)
            )
    except (BadZipFile, KeyError, OSError, RuntimeError, ValueError, etree.XMLSyntaxError):
        return False


__all__ = [
    "OfdPackage",
    "detect_ofd",
    "detect_ofd_path",
    "element_text",
    "first_child",
    "first_descendant",
    "local_name",
    "namespace_name",
    "parse_int",
]
