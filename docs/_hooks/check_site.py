"""检查真实构建产物的发布边界、旧链接和下载内容，避免归档重新进入搜索。"""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class PageLinks(HTMLParser):
    """读取页面的锚点和链接，不执行页面脚本。"""

    def __init__(self) -> None:
        """为每个 HTML 页面独立保存目标 ID、链接及跳转标记。"""
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.redirect = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """收集真实输出的 href 和 id，供本地及站内绝对链接检查。"""
        values = dict(attrs)
        if value := values.get("id"):
            self.ids.add(value)
        if tag == "a" and (value := values.get("href")):
            self.links.append(value)
        if values.get("data-mineru-redirect") == "true":
            self.redirect = True


def check_site(site_dir: Path) -> tuple[int, int]:
    """验证搜索、归档排除、内部跳转和历史附件的字节身份。"""
    docs_dir = Path(__file__).resolve().parents[1]
    site_dir = site_dir.resolve()
    config_path = docs_dir.parent / "mkdocs.yml"
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    site_url = config["site_url"].rstrip("/") + "/"
    redirects = json.loads((docs_dir / "_redirects.json").read_text(encoding="utf-8"))
    for directory in ("archive", "proposals", "_hooks"):
        if (site_dir / directory).exists() or (site_dir / "zh" / directory).exists():
            raise ValueError(f"Excluded directory was published: {directory}")
    pages: dict[Path, PageLinks] = {}
    for path in site_dir.rglob("*.html"):
        page = PageLinks()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path] = page
    index = json.loads((site_dir / "search/search_index.json").read_text(encoding="utf-8"))
    search_locations = {entry["location"].split("#", 1)[0] for entry in index["docs"]}
    for location in search_locations:
        path = site_dir / unquote(location)
        if not path.suffix:
            path /= "index.html"
        if path not in pages or pages[path].redirect:
            raise ValueError(f"Search includes a missing or redirected page: {location}")
    for source, destination in redirects.items():
        for prefix in ("", "zh/"):
            if not source.endswith(".md"):
                if (site_dir / prefix / source).read_bytes() != (docs_dir / destination).read_bytes():
                    raise ValueError(f"Historical download changed: {prefix}{source}")
                continue
            output = site_dir / prefix / source.removesuffix(".md") / "index.html"
            if output not in pages or not pages[output].redirect:
                raise ValueError(f"Missing old-address redirect: {prefix}{source}")
    errors: list[str] = []
    for path, page in pages.items():
        current_url = urljoin(site_url, path.relative_to(site_dir).as_posix())
        for link in page.links:
            resolved = urlsplit(urljoin(current_url, link))
            full_url = resolved._replace(fragment="", query="").geturl()
            if not full_url.startswith(site_url):
                continue
            target = site_dir / unquote(full_url.removeprefix(site_url))
            if not target.suffix or full_url.endswith("/"):
                target /= "index.html"
            if not target.is_file():
                errors.append(f"Broken published link: {path.relative_to(site_dir)} -> {link}")
                continue
            if resolved.fragment and target in pages and not pages[target].redirect:
                if unquote(resolved.fragment) not in pages[target].ids:
                    errors.append(f"Broken published anchor: {path.relative_to(site_dir)} -> {link}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(search_locations), sum(page.redirect for page in pages.values())


def main() -> None:
    """接受已构建目录并输出检查结果，任一错误使文档 CI 失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", type=Path)
    args = parser.parse_args()
    pages, redirects = check_site(args.site_dir)
    print(f"Verified {pages} searchable pages and {redirects} redirects; historical downloads match their sources.")


if __name__ == "__main__":
    main()
