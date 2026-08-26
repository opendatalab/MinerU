# Copyright (c) Opendatalab. All rights reserved.
"""在固定资源预算内读取 EPUB OCF 容器、OPF manifest 与 spine。"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from .constants import (
    EPUB_MIME,
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
    MAX_ENTRY_COUNT,
    MAX_TOTAL_BYTES,
    MAX_XML_DEPTH,
    MAX_XML_NODES,
    SVG_MEDIA_TYPE,
    XHTML_MEDIA_TYPES,
)
from .errors import EpubEncryptedError, EpubParseError, EpubResourceLimitError


@dataclass(frozen=True, slots=True)
class EpubTarget:
    """保存解析后的 OCF 成员路径和可选 fragment。"""

    path: str
    fragment: str | None = None


@dataclass(frozen=True, slots=True)
class EpubManifestItem:
    """保存 OPF manifest 中一个资源的规范信息。"""

    item_id: str
    path: str
    media_type: str
    properties: frozenset[str]
    fallback: str | None


@dataclass(frozen=True, slots=True)
class EpubSpineItem:
    """保存默认阅读顺序中一个已解析的逻辑内容项。"""

    index: int
    idref: str
    path: str | None
    media_type: str | None
    linear: bool
    properties: frozenset[str]


@dataclass(frozen=True, slots=True)
class EpubMetadata:
    """保存 OPF 中供 doclib 使用的基础出版物元数据。"""

    title: str | None
    author: str | None
    subject: str | None
    keywords: str | None
    layout: str


def _xml_parser() -> etree.XMLParser:
    """为每个 EPUB XML part 创建禁用实体、DTD 和网络的 parser。"""
    return etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
        remove_blank_text=False,
        huge_tree=False,
    )


def _local_name(element: etree._Element) -> str:
    """返回 XML 元素不含命名空间的本地名。"""
    return etree.QName(element).localname


def _element_text(element: etree._Element | None) -> str | None:
    """返回元素折叠首尾空白后的完整文本。"""
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


class EpubPackage:
    """负责 EPUB 包身份、资源读取、OPF 和 spine 的受限解析。"""

    def __init__(self, file_bytes: bytes) -> None:
        """打开内存 EPUB，并在读取正文前校验中央目录和必需结构。"""
        if len(file_bytes) > MAX_TOTAL_BYTES:
            raise EpubResourceLimitError(f"EPUB resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
        try:
            self._zip = ZipFile(BytesIO(file_bytes))
        except (BadZipFile, OSError, ValueError) as exc:
            raise EpubParseError(f"Malformed EPUB package: {exc}") from exc
        try:
            self._infos = self._validate_members(self._zip.infolist())
            self._cache: dict[str, bytes] = {}
            self._total_read = 0
            self._asset_parts: set[str] = set()
            self._asset_bytes = 0
            self._encrypted_parts = self._read_encrypted_parts()
            self._validate_mimetype()
            self.opf_path = self._read_default_rootfile()
            opf_root = self.xml_part(self.opf_path, required=True)
            if opf_root is None:
                raise EpubParseError("Malformed EPUB package: OPF root could not be parsed")
            self.opf_root: etree._Element = opf_root
            self.manifest = self._read_manifest()
            self.spine = self._read_spine()
            self.navigation_path = self._read_navigation_path()
            self.ncx_path = self._read_ncx_path()
            self.metadata = self._read_metadata()
        except Exception:
            self._zip.close()
            raise

    @staticmethod
    def _validate_members(infos: list[ZipInfo]) -> dict[str, ZipInfo]:
        """校验成员数量、体积、路径、重名和 ZIP 级加密。"""
        if len(infos) > MAX_ENTRY_COUNT:
            raise EpubResourceLimitError(f"EPUB resource limit exceeded: max_entry_count={MAX_ENTRY_COUNT}")
        total_size = 0
        members: dict[str, ZipInfo] = {}
        for info in infos:
            name = info.filename
            if not EpubPackage._is_safe_member_name(name):
                raise EpubParseError(f"Malformed EPUB package: unsafe member path {name!r}")
            if name in members:
                raise EpubParseError(f"Malformed EPUB package: duplicate member {name!r}")
            if info.flag_bits & 0x1:
                raise EpubEncryptedError(f"Encrypted EPUB ZIP member is not supported: {name!r}")
            if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                raise EpubParseError(f"Malformed EPUB package: unsupported ZIP compression for {name!r}")
            if info.file_size > MAX_ENTRY_BYTES:
                raise EpubResourceLimitError(
                    f"EPUB resource limit exceeded: member {name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
                )
            total_size += info.file_size
            if total_size > MAX_TOTAL_BYTES:
                raise EpubResourceLimitError(f"EPUB resource limit exceeded: max_total_bytes={MAX_TOTAL_BYTES}")
            members[name] = info
        return members

    @staticmethod
    def _is_safe_member_name(name: str) -> bool:
        """判断 ZIP 成员是否为无绝对路径、反斜杠和上跳段的 POSIX 路径。"""
        if not name or "\x00" in name or "\\" in name or name.startswith("/"):
            return False
        parts = PurePosixPath(name).parts
        return bool(parts) and all(part not in {"", ".", ".."} for part in parts)

    def _validate_mimetype(self) -> None:
        """验证存在时的 EPUB mimetype，并对顺序或压缩不规范记录兼容告警。"""
        info = self._infos.get("mimetype")
        if info is None:
            logger.warning("EPUB package has no mimetype member; using container.xml compatibility detection")
            return
        data = self.read_part("mimetype", required=True)
        assert data is not None
        try:
            value = data.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise EpubParseError("Malformed EPUB package: mimetype is not ASCII") from exc
        if value != EPUB_MIME:
            raise EpubParseError(f"Malformed EPUB package: invalid mimetype {value!r}")
        first_name = self._zip.infolist()[0].filename if self._zip.infolist() else ""
        if first_name != "mimetype" or info.compress_type != ZIP_STORED:
            logger.warning("EPUB mimetype is not the first uncompressed member; continuing in compatibility mode")

    def read_part(self, part_name: str, *, required: bool = False, asset: bool = False) -> bytes | None:
        """读取一个已校验成员，并按唯一资源累计图片载荷。"""
        if part_name in self._encrypted_parts:
            raise EpubEncryptedError(f"Encrypted EPUB resource is not supported: {part_name!r}")
        info = self._infos.get(part_name)
        if info is None:
            if required:
                raise EpubParseError(f"Malformed EPUB package: missing required part {part_name!r}")
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
                raise EpubParseError(f"Malformed EPUB package: cannot read {part_name!r}: {exc}") from exc
            return None
        if len(data) > MAX_ENTRY_BYTES:
            raise EpubResourceLimitError(
                f"EPUB resource limit exceeded: member {part_name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        self._charge_total(part_name, len(data))
        if asset:
            self._charge_asset(part_name, len(data))
        self._cache[part_name] = data
        return data

    def _charge_total(self, part_name: str, byte_count: int) -> None:
        """按首次实际读取字节累计全包解压预算。"""
        self._total_read += byte_count
        if self._total_read > MAX_TOTAL_BYTES:
            raise EpubResourceLimitError(
                f"EPUB resource limit exceeded while reading {part_name!r}: max_total_bytes={MAX_TOTAL_BYTES}"
            )

    def _charge_asset(self, part_name: str, byte_count: int) -> None:
        """按唯一包成员累计保留图片字节，重复引用不重复计费。"""
        if part_name in self._asset_parts:
            return
        self._asset_parts.add(part_name)
        self._asset_bytes += byte_count
        if self._asset_bytes > MAX_ASSET_TOTAL_BYTES:
            raise EpubResourceLimitError(f"EPUB resource limit exceeded: max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}")

    def xml_part(
        self,
        part_name: str,
        *,
        required: bool = False,
        allow_external_doctype: bool = False,
    ) -> etree._Element | None:
        """安全解析 XML/XHTML part，并校验节点数和最大深度。"""
        data = self.read_part(part_name, required=required)
        if data is None:
            return None
        try:
            root = etree.fromstring(data, parser=_xml_parser())
        except (etree.XMLSyntaxError, ValueError) as exc:
            if required:
                raise EpubParseError(f"Malformed EPUB package: invalid XML part {part_name!r}: {exc}") from exc
            return None
        docinfo = root.getroottree().docinfo
        if docinfo.doctype:
            internal_dtd = docinfo.internalDTD
            has_entities = internal_dtd is not None and bool(internal_dtd.entities())
            if not allow_external_doctype or has_entities:
                raise EpubParseError(f"Malformed EPUB package: DTD declarations are not allowed in {part_name!r}")
        self._validate_xml_shape(root, part_name)
        return root

    @staticmethod
    def _validate_xml_shape(root: etree._Element, part_name: str) -> None:
        """迭代统计 XML 节点与深度，避免超大或深层 DOM 继续传播。"""
        node_count = 0
        stack: list[tuple[etree._Element, int]] = [(root, 1)]
        while stack:
            element, depth = stack.pop()
            node_count += 1
            if node_count > MAX_XML_NODES:
                raise EpubResourceLimitError(
                    f"EPUB resource limit exceeded: {part_name!r} exceeds max_xml_nodes={MAX_XML_NODES}"
                )
            if depth > MAX_XML_DEPTH:
                raise EpubResourceLimitError(
                    f"EPUB resource limit exceeded: {part_name!r} exceeds max_xml_depth={MAX_XML_DEPTH}"
                )
            stack.extend((child, depth + 1) for child in element if isinstance(child.tag, str))

    def resolve_reference(self, href: str, *, base_part: str) -> EpubTarget | None:
        """按 OCF URI 规则解析相对引用，拒绝外部地址和编码后的结构字符。"""
        raw_href = (href or "").strip()
        if not raw_href:
            return None
        try:
            split = urlsplit(raw_href)
        except ValueError:
            return None
        if split.scheme or split.netloc:
            return None
        raw_path = split.path
        base_segments = [] if raw_path.startswith("/") else [part for part in posixpath.dirname(base_part).split("/") if part]
        segments = list(base_segments)
        for raw_segment in raw_path.split("/"):
            if raw_segment in {"", "."}:
                continue
            if raw_segment == "..":
                if not segments:
                    return None
                segments.pop()
                continue
            decoded = unquote(raw_segment)
            if decoded in {"", ".", ".."} or "/" in decoded or "\\" in decoded or "\x00" in decoded:
                return None
            segments.append(decoded)
        path = "/".join(segments) if raw_path else base_part
        if not path or path not in self._infos:
            return None
        fragment = unquote(split.fragment) if split.fragment else None
        return EpubTarget(path=path, fragment=fragment)

    def _read_encrypted_parts(self) -> frozenset[str]:
        """读取 encryption.xml 中的 CipherReference URI，损坏时按加密文件失败。"""
        if "META-INF/encryption.xml" not in self._infos:
            return frozenset()
        data = self._read_unchecked_part("META-INF/encryption.xml")
        try:
            root = etree.fromstring(data, parser=_xml_parser())
        except (etree.XMLSyntaxError, ValueError) as exc:
            raise EpubEncryptedError(f"Malformed EPUB encryption.xml: {exc}") from exc
        if root.getroottree().docinfo.doctype:
            raise EpubEncryptedError("Malformed EPUB encryption.xml: DTD is not allowed")
        encrypted: set[str] = set()
        for element in root.iter():
            if not isinstance(element.tag, str) or _local_name(element) != "CipherReference":
                continue
            uri = element.get("URI")
            if not uri:
                continue
            # OCF 规定 META-INF 控制文件中的 URI 以容器根为基准。
            target = self._resolve_reference_against_members(uri, base_part="")
            if target:
                encrypted.add(target.path)
        return frozenset(encrypted)

    def _read_unchecked_part(self, part_name: str) -> bytes:
        """在加密成员集合尚未建立时读取已验证的小型控制 part。"""
        info = self._infos.get(part_name)
        if info is None:
            raise EpubParseError(f"Malformed EPUB package: missing required part {part_name!r}")
        try:
            with self._zip.open(info) as source:
                data = source.read(MAX_ENTRY_BYTES + 1)
        except (BadZipFile, OSError, RuntimeError, ValueError) as exc:
            raise EpubParseError(f"Malformed EPUB package: cannot read {part_name!r}: {exc}") from exc
        if len(data) > MAX_ENTRY_BYTES:
            raise EpubResourceLimitError(
                f"EPUB resource limit exceeded: member {part_name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        self._charge_total(part_name, len(data))
        return data

    def _resolve_reference_against_members(self, href: str, *, base_part: str) -> EpubTarget | None:
        """在初始化阶段仅依赖中央目录解析安全包内引用。"""
        raw_href = (href or "").strip()
        if not raw_href:
            return None
        split = urlsplit(raw_href)
        if split.scheme or split.netloc:
            return None
        raw_path = split.path
        segments = [] if raw_path.startswith("/") else [part for part in posixpath.dirname(base_part).split("/") if part]
        for raw_segment in raw_path.split("/"):
            if raw_segment in {"", "."}:
                continue
            if raw_segment == "..":
                if not segments:
                    return None
                segments.pop()
                continue
            decoded = unquote(raw_segment)
            if decoded in {"", ".", ".."} or "/" in decoded or "\\" in decoded or "\x00" in decoded:
                return None
            segments.append(decoded)
        path = "/".join(segments) if raw_path else base_part
        return EpubTarget(path, unquote(split.fragment) if split.fragment else None) if path in self._infos else None

    def _read_default_rootfile(self) -> str:
        """从 container.xml 读取第一个默认 rendition 的 OPF 路径。"""
        container = self.xml_part("META-INF/container.xml", required=True)
        assert container is not None
        for element in container.iter():
            if not isinstance(element.tag, str) or _local_name(element) != "rootfile":
                continue
            full_path = element.get("full-path")
            if not full_path:
                continue
            target = self._resolve_reference_against_members(full_path, base_part="")
            if target is not None:
                return target.path
        raise EpubParseError("Malformed EPUB package: container.xml has no usable rootfile")

    def _read_manifest(self) -> dict[str, EpubManifestItem]:
        """读取 OPF manifest，并把 href 规范化为实际包成员路径。"""
        manifest: dict[str, EpubManifestItem] = {}
        for element in self.opf_root.iter():
            if not isinstance(element.tag, str) or _local_name(element) != "item":
                continue
            item_id = (element.get("id") or "").strip()
            href = (element.get("href") or "").strip()
            if not item_id or not href or item_id in manifest:
                continue
            fallback = (element.get("fallback") or "").strip() or None
            target = self.resolve_reference(href, base_part=self.opf_path)
            if target is None and fallback is None:
                continue
            manifest[item_id] = EpubManifestItem(
                item_id=item_id,
                path=target.path if target else "",
                media_type=(element.get("media-type") or "").strip().casefold(),
                properties=frozenset((element.get("properties") or "").split()),
                fallback=fallback,
            )
        return manifest

    def _supported_manifest_item(self, item_id: str) -> EpubManifestItem | None:
        """沿 manifest fallback chain 找到首个支持的 XHTML 或 SVG 内容项。"""
        visited: set[str] = set()
        current_id: str | None = item_id
        while current_id and current_id not in visited:
            visited.add(current_id)
            item = self.manifest.get(current_id)
            if item is None:
                return None
            if item.media_type in XHTML_MEDIA_TYPES or item.media_type == SVG_MEDIA_TYPE:
                return item
            current_id = item.fallback
        return None

    def _read_spine(self) -> list[EpubSpineItem]:
        """按 OPF itemref 顺序建立稳定逻辑页，并保留 non-linear 内容。"""
        spine_element = next(
            (element for element in self.opf_root.iter() if isinstance(element.tag, str) and _local_name(element) == "spine"),
            None,
        )
        if spine_element is None:
            raise EpubParseError("Malformed EPUB package: OPF has no spine")
        result: list[EpubSpineItem] = []
        for element in spine_element:
            if not isinstance(element.tag, str) or _local_name(element) != "itemref":
                continue
            idref = (element.get("idref") or "").strip()
            item = self._supported_manifest_item(idref)
            result.append(
                EpubSpineItem(
                    index=len(result),
                    idref=idref,
                    path=item.path if item else None,
                    media_type=item.media_type if item else None,
                    linear=(element.get("linear") or "yes").casefold() != "no",
                    properties=frozenset((element.get("properties") or "").split()),
                )
            )
        if not result:
            raise EpubParseError("Malformed EPUB package: OPF spine has no itemref")
        return result

    def _read_navigation_path(self) -> str | None:
        """返回 manifest 中首个 EPUB3 navigation document 路径。"""
        for item in self.manifest.values():
            if "nav" in item.properties and item.media_type in XHTML_MEDIA_TYPES and item.path:
                return item.path
        return None

    def _read_ncx_path(self) -> str | None:
        """按 OPF spine 的 toc ID 返回 EPUB2 NCX 资源路径。"""
        spine_element = next(
            (element for element in self.opf_root.iter() if isinstance(element.tag, str) and _local_name(element) == "spine"),
            None,
        )
        if spine_element is None:
            return None
        toc_id = (spine_element.get("toc") or "").strip()
        item = self.manifest.get(toc_id)
        return item.path if item is not None and item.path else None

    def _read_metadata(self) -> EpubMetadata:
        """提取 OPF 的首个标题、作者、主题、关键词和布局模式。"""
        title = author = subject = None
        keywords: list[str] = []
        layout = "reflowable"
        for element in self.opf_root.iter():
            if not isinstance(element.tag, str):
                continue
            name = _local_name(element)
            value = _element_text(element)
            if name == "title" and title is None:
                title = value
            elif name == "creator" and author is None:
                author = value
            elif name == "subject" and value:
                subject = subject or value
                keywords.append(value)
            elif name == "meta":
                property_name = (element.get("property") or element.get("name") or "").casefold()
                content = value or (element.get("content") or "").strip() or None
                if property_name in {"rendition:layout", "fixed-layout"} and content:
                    layout = "pre-paginated" if content in {"pre-paginated", "true"} else content
                if property_name in {"keywords", "keyword"} and content:
                    keywords.append(content)
        return EpubMetadata(title, author, subject, ", ".join(dict.fromkeys(keywords)) or None, layout)

    def content_type_for(self, part_name: str) -> str | None:
        """返回 manifest 为指定成员声明的媒体类型。"""
        for item in self.manifest.values():
            if item.path == part_name:
                return item.media_type or None
        return None

    def close(self) -> None:
        """关闭底层 ZipFile。"""
        self._zip.close()


def _detect_epub_zip(package: ZipFile) -> bool:
    """在已打开 ZIP 中按 mimetype 或有效 container rootfile 识别 EPUB。"""
    try:
        mime_info = package.getinfo("mimetype")
        if mime_info.file_size <= len(EPUB_MIME):
            with package.open(mime_info) as source:
                if source.read(len(EPUB_MIME) + 1) == EPUB_MIME.encode("ascii"):
                    return True
    except (KeyError, BadZipFile, OSError, RuntimeError, ValueError):
        pass
    infos = EpubPackage._validate_members(package.infolist())
    container_info = infos.get("META-INF/container.xml")
    if container_info is None:
        return False
    with package.open(container_info) as source:
        data = source.read(MAX_ENTRY_BYTES + 1)
    if len(data) > MAX_ENTRY_BYTES:
        return False
    root = etree.fromstring(data, parser=_xml_parser())
    if root.getroottree().docinfo.doctype:
        return False
    for element in root.iter():
        if not isinstance(element.tag, str) or _local_name(element) != "rootfile":
            continue
        full_path = (element.get("full-path") or "").lstrip("/")
        if full_path in infos:
            return True
    return False


def detect_epub(file_bytes: bytes) -> bool:
    """从内存字节按 EPUB mimetype 或有效 container rootfile 识别 OCF 包。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as package:
            return _detect_epub_zip(package)
    except (BadZipFile, EpubEncryptedError, EpubParseError, EpubResourceLimitError, OSError, ValueError, etree.XMLSyntaxError):
        return False


def detect_epub_path(file_path: str | Path) -> bool:
    """从文件路径打开 ZIP 并验证 EPUB 强内容身份。"""
    try:
        with ZipFile(file_path) as package:
            return _detect_epub_zip(package)
    except (BadZipFile, EpubEncryptedError, EpubParseError, EpubResourceLimitError, OSError, ValueError, etree.XMLSyntaxError):
        return False


__all__ = [
    "EpubManifestItem",
    "EpubMetadata",
    "EpubPackage",
    "EpubSpineItem",
    "EpubTarget",
    "detect_epub",
    "detect_epub_path",
]
