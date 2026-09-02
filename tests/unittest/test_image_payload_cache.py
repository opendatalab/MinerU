from __future__ import annotations

import pytest

from mineru.utils.image_payload import ImagePayloadCache, validate_image_sidecar_path


@pytest.mark.parametrize(
    ("image_path", "expected"),
    [
        ("chart.png", "chart.png"),
        ("images/chart.png", "images/chart.png"),
        ("images/./chart.png", "images/chart.png"),
    ],
)
def test_image_payload_cache_registers_safe_explicit_paths(image_path: str, expected: str) -> None:
    """验证显式图片路径进入缓存时会被统一校验并规范化。"""
    cache = ImagePayloadCache()

    registered = cache.register_bytes(b"image-bytes", "png", image_path=image_path)

    assert registered == expected
    assert cache.images() == {expected: b"image-bytes"}


def test_image_payload_cache_validates_constructor_and_update_paths() -> None:
    """验证构造函数和 update 不再绕过 sidecar 路径校验。"""
    cache = ImagePayloadCache({"images/./constructor.png": b"constructor"})

    cache.update({"updated.png": b"updated"})

    assert cache.images() == {
        "images/constructor.png": b"constructor",
        "updated.png": b"updated",
    }


def test_image_payload_cache_validates_path_key_and_generated_hash_paths() -> None:
    """验证 path_key 和哈希路径也会经过统一校验。"""
    cache = ImagePayloadCache()

    keyed_path = cache.register_bytes(b"keyed", "jpeg", path_key="images/page_0")
    hashed_path = cache.register_bytes(b"hashed", "png")

    assert keyed_path == "images/page_0.jpg"
    assert hashed_path.endswith(".png")
    assert "/" not in hashed_path
    assert cache.images()[keyed_path] == b"keyed"
    assert cache.images()[hashed_path] == b"hashed"


@pytest.mark.parametrize(
    "image_path",
    [
        "../escape.png",
        "/tmp/escape.png",
        "C:\\escape.png",
        "\\escape.png",
        "data:image/png;base64,AAAA",
        "http://example.test/escape.png",
        "//example.test/escape.png",
        "images/control\nname.png",
        "images/control\x00name.png",
    ],
)
def test_image_payload_cache_rejects_unsafe_explicit_paths(image_path: str) -> None:
    """验证不安全显式路径不能进入图片缓存。"""
    cache = ImagePayloadCache()

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        cache.register_bytes(b"image-bytes", "png", image_path=image_path)


@pytest.mark.parametrize(
    "image_path",
    [
        "../constructor.png",
        "http://example.test/constructor.png",
    ],
)
def test_image_payload_cache_rejects_unsafe_constructor_and_update_paths(image_path: str) -> None:
    """验证构造函数和 update 对不安全路径采用 fail-fast。"""
    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        ImagePayloadCache({image_path: b"constructor"})

    cache = ImagePayloadCache()
    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        cache.update({image_path: b"updated"})


@pytest.mark.parametrize("path_key", ["../escape", "http://example.test/escape"])
def test_image_payload_cache_rejects_unsafe_path_keys(path_key: str) -> None:
    """验证由 path_key 派生出的不安全路径也会被拒绝。"""
    cache = ImagePayloadCache()

    with pytest.raises(ValueError, match="Unsafe image sidecar path"):
        cache.register_bytes(b"image-bytes", "png", path_key=path_key)


@pytest.mark.parametrize(
    ("image_path", "expected"),
    [
        ("chart.png", "chart.png"),
        ("images/./chart.png", "images/chart.png"),
    ],
)
def test_validate_image_sidecar_path_returns_normalized_posix_path(image_path: str, expected: str) -> None:
    """验证底层路径校验器返回规范化 POSIX 相对路径。"""
    assert validate_image_sidecar_path(image_path) == expected
