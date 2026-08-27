# Copyright (c) Opendatalab. All rights reserved.
"""基于静态 DOM 指标执行保守的 HTML 正文自动选择。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from lxml import etree  # type: ignore[reportMissingImports]

from .._shared.markup import MarkupStylesheet, TextStyle
from .._shared.markup.projector import local_name


_CANDIDATE_TAGS = frozenset({"article", "div", "main", "section"})
_SEMANTIC_TAGS = frozenset({"figure", "img", "math", "pre", "svg", "table"})
_PARAGRAPH_TAGS = frozenset({"dd", "dt", "li", "p", "pre"})
_BOILERPLATE_TAGS = frozenset({"footer", "form", "nav"})
_POSITIVE_TOKENS = frozenset({"article", "body", "content", "entry", "main", "post", "story", "text"})
_NEGATIVE_TOKENS = frozenset(
    {
        "advert",
        "banner",
        "comment",
        "cookie",
        "footer",
        "menu",
        "nav",
        "newsletter",
        "related",
        "share",
        "sidebar",
        "social",
        "widget",
    }
)
_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_MIN_TEXT_CHARS = 200
_MIN_SEMANTIC_OBJECTS = 2
_MIN_RETAINED_RATIO = 1 / 7
_MIN_SCORE_MARGIN = 1.25


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """保存一个 DOM 子树的可见正文与噪声统计。"""

    text_chars: int = 0
    link_chars: int = 0
    paragraph_chars: int = 0
    heading_count: int = 0
    semantic_count: int = 0
    boilerplate_count: int = 0

    def __add__(self, other: CandidateMetrics) -> CandidateMetrics:
        """合并两个子树指标。"""
        return CandidateMetrics(
            text_chars=self.text_chars + other.text_chars,
            link_chars=self.link_chars + other.link_chars,
            paragraph_chars=self.paragraph_chars + other.paragraph_chars,
            heading_count=self.heading_count + other.heading_count,
            semantic_count=self.semantic_count + other.semantic_count,
            boilerplate_count=self.boilerplate_count + other.boilerplate_count,
        )


@dataclass(frozen=True, slots=True)
class ContentSelection:
    """保存最终内容根、正文命中状态和可诊断的保留率。"""

    root: etree._Element
    mode_used: str
    confidence: float
    retained_text_ratio: float
    reason: str


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    """绑定候选元素、指标、分数和显式语义标记。"""

    element: etree._Element
    metrics: CandidateMetrics
    score: float
    explicit: bool


def select_auto_content(body: etree._Element, stylesheet: MarkupStylesheet) -> ContentSelection:
    """高置信选择正文候选，任一保守门槛失败时回退完整 body。"""
    metrics_by_element: dict[etree._Element, CandidateMetrics] = {}
    body_metrics = _collect_metrics(body, stylesheet, metrics_by_element, TextStyle(), False, False)
    candidates: list[_ScoredCandidate] = []
    for element in body.iter():
        if not isinstance(element.tag, str) or element is body:
            continue
        name = local_name(element)
        metrics = metrics_by_element.get(element, CandidateMetrics())
        explicit = _is_explicit_candidate(element)
        if name not in _CANDIDATE_TAGS and not explicit:
            continue
        if not explicit and metrics.text_chars < _MIN_TEXT_CHARS and metrics.semantic_count < _MIN_SEMANTIC_OBJECTS:
            continue
        candidates.append(_ScoredCandidate(element, metrics, _candidate_score(element, metrics), explicit))
    candidates.sort(key=lambda item: (-item.score, _document_order(item.element)))

    for candidate in candidates:
        if candidate.score <= 0:
            continue
        if _is_repeated_candidate_item(candidate.element, metrics_by_element):
            continue
        second = next(
            (
                item
                for item in candidates
                if item is not candidate and item.score > 0 and not _is_containment_equivalent(candidate, item)
            ),
            None,
        )
        if not candidate.explicit and second is not None and candidate.score < second.score * _MIN_SCORE_MARGIN:
            continue
        selected = _copy_candidate_with_ancestors(candidate.element, body)
        selected_metrics: dict[etree._Element, CandidateMetrics] = {}
        final_metrics = _collect_metrics(selected, stylesheet, selected_metrics, TextStyle(), False, False)
        retained_ratio = final_metrics.text_chars / max(1, body_metrics.text_chars)
        if final_metrics.text_chars < _MIN_TEXT_CHARS and final_metrics.semantic_count < _MIN_SEMANTIC_OBJECTS:
            continue
        if body_metrics.text_chars >= _MIN_TEXT_CHARS and retained_ratio < _MIN_RETAINED_RATIO:
            continue
        if candidate.metrics.semantic_count and final_metrics.semantic_count == 0:
            continue
        confidence = candidate.score / max(candidate.score + (second.score if second is not None else 0), 1)
        return ContentSelection(selected, "main", confidence, retained_ratio, "high_confidence_candidate")

    return ContentSelection(deepcopy(body), "document", 0.0, 1.0, "body_fallback")


def _copy_candidate_with_ancestors(candidate: etree._Element, body: etree._Element) -> etree._Element:
    """复制正文候选及其到 body 的空祖先链，保留继承样式但不带入周边正文。"""
    selected = deepcopy(candidate)
    selected.tail = None
    _soft_prune(selected)
    root = selected
    for ancestor in candidate.iterancestors():
        if not isinstance(ancestor.tag, str):
            continue
        wrapper = etree.Element(ancestor.tag, nsmap=ancestor.nsmap)
        for name, value in ancestor.attrib.items():
            wrapper.set(name, value)
        wrapper.append(root)
        root = wrapper
        if ancestor is body:
            return root
    return root


def _collect_metrics(
    element: etree._Element,
    stylesheet: MarkupStylesheet,
    output: dict[etree._Element, CandidateMetrics],
    inherited: TextStyle,
    inherited_visibility_hidden: bool,
    inside_link: bool,
) -> CandidateMetrics:
    """单次深度优先遍历计算所有元素的可见指标，避免候选间重复扫描。"""
    resolved = stylesheet.resolve(element, inherited, inherited_visibility_hidden)
    if resolved.subtree_hidden:
        output[element] = CandidateMetrics()
        return output[element]
    name = local_name(element)
    current_inside_link = inside_link or name == "a"
    own_text = 0 if resolved.visibility_hidden else len(_normalized_text(element.text))
    metrics = CandidateMetrics(
        text_chars=own_text,
        link_chars=own_text if current_inside_link else 0,
        paragraph_chars=own_text if name in _PARAGRAPH_TAGS else 0,
        heading_count=1 if name in {"h1", "h2", "h3", "h4", "h5", "h6"} else 0,
        semantic_count=1 if name in _SEMANTIC_TAGS else 0,
        boilerplate_count=1 if name in _BOILERPLATE_TAGS or bool(_tokens(element) & _NEGATIVE_TOKENS) else 0,
    )
    for child in element:
        if isinstance(child.tag, str):
            metrics += _collect_metrics(
                child,
                stylesheet,
                output,
                resolved.text,
                resolved.visibility_hidden,
                current_inside_link,
            )
        if not resolved.visibility_hidden:
            tail_chars = len(_normalized_text(child.tail))
            metrics += CandidateMetrics(
                text_chars=tail_chars,
                link_chars=tail_chars if current_inside_link else 0,
                paragraph_chars=tail_chars if name in _PARAGRAPH_TAGS else 0,
            )
    output[element] = metrics
    return metrics


def _candidate_score(element: etree._Element, metrics: CandidateMetrics) -> float:
    """按正文、结构对象、链接与模板噪声计算确定性候选分数。"""
    tokens = _tokens(element)
    token_bonus = 200 if tokens & _POSITIVE_TOKENS else 0
    token_penalty = 240 if tokens & _NEGATIVE_TOKENS else 0
    link_density = metrics.link_chars / max(1, metrics.text_chars)
    repeated_penalty = _repeated_short_sibling_penalty(element)
    return (
        metrics.text_chars
        + metrics.paragraph_chars
        + metrics.heading_count * 40
        + metrics.semantic_count * 120
        + token_bonus
        - metrics.link_chars * (1.0 + link_density)
        - metrics.boilerplate_count * 120
        - repeated_penalty * 80
        - token_penalty
    )


def _is_explicit_candidate(element: etree._Element) -> bool:
    """识别标准语义 main/article/role/itemprop 正文候选。"""
    name = local_name(element)
    roles = frozenset((element.get("role") or "").casefold().split())
    itemprop = frozenset((element.get("itemprop") or "").casefold().split())
    return name in {"main", "article"} or "main" in roles or "articlebody" in itemprop


def _tokens(element: etree._Element) -> frozenset[str]:
    """把 class/id 拆成完整小写 token，避免任意 substring 误判。"""
    value = f"{element.get('id') or ''} {element.get('class') or ''}".casefold()
    return frozenset(token for token in _TOKEN_RE.split(value) if token)


def _repeated_short_sibling_penalty(element: etree._Element) -> int:
    """统计候选内部重复短同级节点簇，降低菜单和文章列表分数。"""
    penalty = 0
    for parent in element.iter():
        if not isinstance(parent.tag, str):
            continue
        groups: dict[tuple[str, tuple[str, ...]], int] = {}
        for child in parent:
            if not isinstance(child.tag, str):
                continue
            text = _normalized_text(" ".join(child.itertext()))
            if not text or len(text) > 160:
                continue
            signature = (local_name(child), tuple(sorted(_tokens(child))))
            groups[signature] = groups.get(signature, 0) + 1
        penalty += sum(count - 2 for count in groups.values() if count > 2)
    return penalty


def _is_repeated_candidate_item(
    element: etree._Element,
    metrics_by_element: dict[etree._Element, CandidateMetrics],
) -> bool:
    """拒绝论坛、文档页或聚合页中只代表单个重复同级条目的候选。"""
    parent = element.getparent()
    if parent is None:
        return False
    element_name = local_name(element)
    repeated = [
        child
        for child in parent
        if isinstance(child.tag, str)
        and (
            local_name(child) == element_name
            or bool(_tokens(child) & _tokens(element) & {"article", "content", "entry", "post", "section"})
        )
        and metrics_by_element.get(child, CandidateMetrics()).text_chars >= 100
    ]
    return len(repeated) >= 2 and element in repeated


def _is_containment_equivalent(first: _ScoredCandidate, second: _ScoredCandidate) -> bool:
    """忽略高度重叠的祖先/后代候选，避免嵌套 main/article 互相压低置信度。"""
    contains = first.element in second.element.iterancestors() or second.element in first.element.iterancestors()
    if not contains:
        return False
    smaller = min(first.metrics.text_chars, second.metrics.text_chars)
    larger = max(first.metrics.text_chars, second.metrics.text_chars, 1)
    return smaller / larger >= 0.8


def _soft_prune(root: etree._Element) -> None:
    """在候选副本中删除确定的导航/表单和高噪声 token 子树。"""
    for element in list(root.iterdescendants()):
        if not isinstance(element.tag, str):
            continue
        name = local_name(element)
        tokens = _tokens(element)
        valuable = any(
            isinstance(child.tag, str) and local_name(child) in _SEMANTIC_TAGS for child in element.iterdescendants()
        )
        text = _normalized_text(" ".join(element.itertext()))
        links = " ".join(
            " ".join(link.itertext())
            for link in element.iterdescendants()
            if isinstance(link.tag, str) and local_name(link) == "a"
        )
        link_density = len(_normalized_text(links)) / max(1, len(text))
        should_remove = name in _BOILERPLATE_TAGS or (
            bool(tokens & _NEGATIVE_TOKENS) and not valuable and (len(text) < 200 or link_density > 0.5)
        )
        if should_remove:
            _drop_tree_preserve_tail(element)


def _drop_tree_preserve_tail(element: etree._Element) -> None:
    """删除噪声子树，同时把 tail 归还到相邻文本位置。"""
    parent = element.getparent()
    if parent is None:
        return
    tail = element.tail or ""
    previous = element.getprevious()
    if tail:
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


def _normalized_text(value: str | None) -> str:
    """折叠文本空白，供字符计数和候选比较使用。"""
    return re.sub(r"\s+", " ", value or "").strip()


def _document_order(element: etree._Element) -> tuple[int, ...]:
    """返回元素从根到自身的逐层索引，作为稳定排序键。"""
    path: list[int] = []
    current: etree._Element | None = element
    while current is not None and current.getparent() is not None:
        parent = current.getparent()
        path.append(parent.index(current))
        current = parent
    return tuple(reversed(path))


__all__ = ["CandidateMetrics", "ContentSelection", "select_auto_content"]
