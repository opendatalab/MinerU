
import base64
import re
from collections import defaultdict

from loguru import logger

from mineru.backend.office.office_magic_model import MagicModel
from mineru.utils.enum_class import BlockType
from mineru.utils.hash_utils import str_sha256
from mineru.version import __version__


def _save_base64_image(b64_data_uri: str, image_writer, page_index: int):
    """将 data-URI 格式的 base64 图片解码并通过 image_writer 保存到本地。

    Args:
        b64_data_uri: 形如 ``data:image/{fmt};base64,{data}`` 的字符串。
        image_writer:  DataWriter 实例，用于将字节写入本地存储。
        page_index:    当前页索引，仅用于日志信息。

    Returns:
        保存成功时返回相对路径字符串（如 ``"abc123.png"``），否则返回 ``None``。
    """
    m = re.match(r'data:image/(\w+);base64,(.+)', b64_data_uri, re.DOTALL)
    if not m:
        logger.warning(f"Unrecognized image_base64 format in page {page_index}, skipping.")
        return None
    fmt = m.group(1)
    ext = "jpg" if fmt == "jpeg" else fmt
    try:
        img_bytes = base64.b64decode(m.group(2))
    except Exception as e:
        logger.warning(f"Failed to decode image_base64 on page {page_index}: {e}")
        return None
    img_path = f"{str_sha256(b64_data_uri)}.{ext}"
    image_writer.write(img_path, img_bytes)
    return img_path


def blocks_to_page_info(page_blocks, image_writer, page_index) -> dict:
    """将blocks转换为页面信息"""

    magic_model = MagicModel(page_blocks)
    image_blocks = magic_model.get_image_blocks()

    # Write embedded images to local storage via image_writer
    if image_writer:
        for img_block in image_blocks:
            for sub_block in img_block.get("blocks", []):
                if sub_block.get("type") != "image_body":
                    continue
                for line in sub_block.get("lines", []):
                    for span in line.get("spans", []):
                        img_b64 = span.get("image_base64", "")
                        if not img_b64:
                            continue
                        img_path = _save_base64_image(img_b64, image_writer, page_index)
                        if img_path:
                            span["image_path"] = img_path
                            del span["image_base64"]

    table_blocks = magic_model.get_table_blocks()

    # Replace inline base64 images inside table HTML with local paths
    if image_writer:
        for tbl_block in table_blocks:
            for sub_block in tbl_block.get("blocks", []):
                if sub_block.get("type") != "table_body":
                    continue
                for line in sub_block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("type") != "table":
                            continue
                        html = span.get("html", "")
                        if not html or "base64," not in html:
                            continue

                        def _replace_src(m_src, _writer=image_writer, _idx=page_index):
                            img_path = _save_base64_image(m_src.group(1), _writer, _idx)
                            if img_path:
                                return f'src="{img_path}"'
                            return m_src.group(0)  # keep original on failure

                        span["html"] = re.sub(
                            r'src="(data:image/[^"]+)"',
                            _replace_src,
                            html,
                        )

    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    list_blocks = magic_model.get_list_blocks()
    index_blocks = magic_model.get_index_blocks()
    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    page_blocks = []
    page_blocks.extend([
        *image_blocks,
        *table_blocks,
        *title_blocks,
        *text_blocks,
        *interline_equation_blocks,
        *list_blocks,
        *index_blocks,
    ])
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {"para_blocks": page_blocks, "discarded_blocks": discarded_blocks, "page_idx": page_index}
    return page_info


def _extract_section_parts_from_content(content: str, level: int):
    """Try to extract a leading section number (e.g. '1.2.1') from title content.

    Returns a list of ints [n1, n2, ..., nL] when the number of parts equals
    `level`, otherwise None.  Handles formats like:
        '1心肌特异性...'       (no separator)
        '1.2.1建立...'         (Chinese text immediately after number)
        '2.2.1 ALKBH5 ...'    (space separator)
    """
    match = re.match(r'^(\d+(?:\.\d+)*)', content.strip())
    if match:
        parts = [int(p) for p in match.group(1).split('.')]
        if len(parts) == level:
            return parts
    return None


_FULLWIDTH_TO_HALFWIDTH = str.maketrans(
    '：；，。（）！？【】｛｝\u201c\u201d\u2018\u2019／＼＆＠＃＄％＾＊＋＝｜～＜＞',
    ':;,.()!?[]{}""\'\'/\\@&#$%^*+=|~<>',
)


def _normalize_for_match(text: str) -> str:
    """Aggressively normalize text for TOC-to-body matching.

    * fullwidth punctuation → halfwidth
    * unicode minus / en-dash / em-dash → ASCII hyphen
    * strip ALL whitespace
    * lowercase
    """
    text = text.translate(_FULLWIDTH_TO_HALFWIDTH)
    text = text.replace('\u2212', '-')  # − MINUS SIGN
    text = text.replace('\u2013', '-')  # – EN DASH
    text = text.replace('\u2014', '-')  # — EM DASH
    text = re.sub(r'\s+', '', text)
    return text.lower()


def _strip_section_number(text: str) -> str:
    """Strip a leading section number like '1.', '1.2', '1.2.3.' from *raw* text."""
    return re.sub(r'^\d+(\.\d+)*\.?\s*', '', text.strip())


def _strip_fig_tab_number(text: str) -> str:
    """Strip a leading figure/table number like '图1.3.1' or '表2.1' from *raw* text.

    These prefixes start with the Chinese character 图 or 表 followed by a
    dotted number.  Stripping them allows matching even when the TOC and body
    use different numbering (common in informally formatted documents).
    """
    return re.sub(r'^[图表]\s*\d+(?:[.\-]\d+)*\s*', '', text.strip())


def _get_block_text(block: dict) -> str:
    """Concatenate all span content from a block's lines."""
    parts = []
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            c = span.get('content', '')
            if c:
                parts.append(c)
    return ''.join(parts).strip()


def _collect_toc_spans(index_block: dict, result: list) -> None:
    """Depth-first collect all leaf text spans from an index block tree."""
    for child in index_block.get('blocks', []):
        if child.get('type') == BlockType.INDEX:
            _collect_toc_spans(child, result)
        elif child.get('type') == BlockType.TEXT:
            for line in child.get('lines', []):
                for span in line.get('spans', []):
                    if span.get('content', '').strip():
                        result.append(span)


def _link_index_spans_to_body_blocks(middle_json: dict) -> None:
    """Annotate each leaf span inside index blocks with target_page_idx / target_block_idx.

    Uses sequential order-based matching so that duplicate titles (e.g.
    multiple "本章小结") are correctly assigned to successive body blocks
    in document order.  Matches against title blocks only.
    """
    pdf_info = middle_json.get('pdf_info', [])

    # ------------------------------------------------------------------
    # 1. Collect body targets in document order
    #    Each entry: (page_idx, block_idx, raw_text)
    #    Only title blocks are matched.
    # ------------------------------------------------------------------
    body_targets: list[tuple[int, int, str]] = []
    for page_info in pdf_info:
        page_idx = page_info.get('page_idx', 0)
        for block in page_info.get('para_blocks', []):
            block_idx = block.get('index')
            if block_idx is None:
                continue
            btype = block.get('type')
            if btype == BlockType.TITLE:
                text = _get_block_text(block)
                if text:
                    body_targets.append((page_idx, block_idx, text))

    # ------------------------------------------------------------------
    # 2. Build lookup: normalized_text -> ordered list of indices into body_targets
    # ------------------------------------------------------------------
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_pid, _bidx, raw) in enumerate(body_targets):
        seen_keys: set[str] = set()
        for key in (
            _normalize_for_match(raw),
            _normalize_for_match(_strip_section_number(raw)),
            _normalize_for_match(_strip_fig_tab_number(raw)),
        ):
            if key and key not in seen_keys:
                norm_to_indices[key].append(i)
                seen_keys.add(key)

    # ------------------------------------------------------------------
    # 3. Collect TOC spans in document order
    # ------------------------------------------------------------------
    toc_spans: list[dict] = []
    for page_info in pdf_info:
        for block in page_info.get('para_blocks', []):
            if block.get('type') == BlockType.INDEX:
                _collect_toc_spans(block, toc_spans)

    # ------------------------------------------------------------------
    # 4. Pass 1 — exact-normalized sequential matching.
    #    A counter per key ensures the n-th TOC entry for "X" maps to the
    #    n-th body block with the same normalized text (handles duplicates
    #    like multiple "本章小结" chapters).
    # ------------------------------------------------------------------
    used_counts: dict[str, int] = defaultdict(int)
    consumed_targets: set[int] = set()

    for span in toc_spans:
        content = span.get('content', '')
        if '\t' in content:
            toc_text = content.split('\t', 1)[0].strip()
        else:
            toc_text = content.strip()

        if not toc_text:
            continue

        for normalized in (
            _normalize_for_match(toc_text),
            _normalize_for_match(_strip_section_number(toc_text)),
            _normalize_for_match(_strip_fig_tab_number(toc_text)),
        ):
            if not normalized or normalized not in norm_to_indices:
                continue
            indices = norm_to_indices[normalized]
            count = used_counts[normalized]
            # Skip indices already consumed by another normalization key
            while count < len(indices) and indices[count] in consumed_targets:
                count += 1
            if count < len(indices):
                target_idx = indices[count]
                pid, bidx, _ = body_targets[target_idx]
                span['target_anchor'] = [pid, bidx]
                consumed_targets.add(target_idx)
                used_counts[normalized] = count + 1
                break

    # ------------------------------------------------------------------
    # 5. Pass 2 — prefix / coverage-based fuzzy matching for unresolved spans.
    #    One of the two texts must be a prefix of the other (handles truncated
    #    captions), AND the coverage ratio min/max >= 0.5 (prevents matching
    #    a short TOC entry against a long body paragraph).
    # ------------------------------------------------------------------
    _MIN_COVERAGE = 0.5
    remaining = sorted(set(range(len(body_targets))) - consumed_targets)

    for span in toc_spans:
        if 'target_anchor' in span:
            continue
        content = span.get('content', '')
        toc_text = content.split('\t', 1)[0].strip() if '\t' in content else content.strip()
        if not toc_text:
            continue

        for toc_norm in (
            _normalize_for_match(toc_text),
            _normalize_for_match(_strip_section_number(toc_text)),
            _normalize_for_match(_strip_fig_tab_number(toc_text)),
        ):
            if not toc_norm or len(toc_norm) < 5:
                continue
            for j, body_idx in enumerate(remaining):
                body_norm = _normalize_for_match(body_targets[body_idx][2])
                if not body_norm:
                    continue
                len_t, len_b = len(toc_norm), len(body_norm)
                coverage = min(len_t, len_b) / max(len_t, len_b)
                if coverage < _MIN_COVERAGE:
                    continue
                if body_norm.startswith(toc_norm) or toc_norm.startswith(body_norm):
                    pid, bidx, _ = body_targets[body_idx]
                    span['target_anchor'] = [pid, bidx]
                    remaining.pop(j)
                    break
            if 'target_anchor' in span:
                break


def result_to_middle_json(model_output_blocks_list, image_writer):
    middle_json = {"pdf_info": [], "_backend":"office", "_version_name": __version__}
    for index, page_blocks in enumerate(model_output_blocks_list):
        page_info = blocks_to_page_info(page_blocks, image_writer, index)
        middle_json["pdf_info"].append(page_info)

    section_counters: dict[int, int] = defaultdict(int)
    for page_info in middle_json["pdf_info"]:
        for block in page_info.get("para_blocks", []):
            if block.get("type") != BlockType.TITLE:
                continue
            level = block.get("level", 1)
            if block.get("is_numbered_style", False):
                # Ensure all ancestor levels start at 1 (never 0)
                for ancestor in range(1, level):
                    if section_counters[ancestor] == 0:
                        section_counters[ancestor] = 1
                # Increment current level counter and reset all deeper levels
                section_counters[level] += 1
                for deeper in list(section_counters.keys()):
                    if deeper > level:
                        section_counters[deeper] = 0
                # Build section number string, e.g. "1.2.1."
                section_number = ".".join(
                    str(section_counters[l]) for l in range(1, level + 1)
                ) + "."
                block["section_number"] = section_number
            else:
                # Some documents embed the section number directly in the content
                # (is_numbered_style=False).  Parse it and sync the counters so
                # that subsequent numbered blocks continue from the right base.
                lines = block.get("lines", [])
                content = ""
                if lines and lines[0].get("spans"):
                    content = lines[0]["spans"][0].get("content", "")
                parts = _extract_section_parts_from_content(content, level)
                if parts:
                    for k, v in enumerate(parts, start=1):
                        section_counters[k] = v
                    # Reset all deeper levels
                    for deeper in list(section_counters.keys()):
                        if deeper > level:
                            section_counters[deeper] = 0

    _link_index_spans_to_body_blocks(middle_json)
    return middle_json