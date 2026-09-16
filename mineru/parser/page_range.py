# Copyright (c) Opendatalab. All rights reserved.
"""共享页范围算法，并在产品边界映射 MinerU 错误。"""

from __future__ import annotations
from typing import Iterable
from docvortex.document import page_range as _ranges
from docvortex.errors import InvalidRequestError as _DocumentInvalidRequest
from ..errors import InvalidRequestError

PAGE_RANGE_DESCRIPTION = _ranges.PAGE_RANGE_DESCRIPTION


def get_end_page_id(end_page_id: int | None, pdf_page_num: int) -> int:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.get_end_page_id(end_page_id, pdf_page_num)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def normalize_page_range_input(raw: str | None) -> str:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.normalize_page_range_input(raw)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def parse_page_range(raw: str, page_count: int) -> list[int]:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.parse_page_range(raw, page_count)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def expand_page_range(raw: str | None, page_count: int) -> str:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.expand_page_range(raw, page_count)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def normalize_result_page_range(raw: str) -> str:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.normalize_result_page_range(raw)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def parse_page_range_set(raw: str) -> set[int]:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.parse_page_range_set(raw)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def count_pages_in_range(raw: str) -> int:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.count_pages_in_range(raw)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


def format_page_range(page_numbers: Iterable[int]) -> str:
    """复用页范围逻辑并保留宿主错误合同。"""
    try:
        return _ranges.format_page_range(page_numbers)
    except _DocumentInvalidRequest as error:
        raise InvalidRequestError(error.code, error.message, error.param) from error


__all__ = [
    "PAGE_RANGE_DESCRIPTION",
    "count_pages_in_range",
    "expand_page_range",
    "format_page_range",
    "get_end_page_id",
    "normalize_page_range_input",
    "normalize_result_page_range",
    "parse_page_range",
    "parse_page_range_set",
]
