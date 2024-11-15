import re


def minify_html(html):
    # 移除多余的空白字符
    html = re.sub(r'\s+', ' ', html)
    # 移除行尾的空白字符
    html = re.sub(r'\s*>\s*', '>', html)
    # 移除标签前的空白字符
    html = re.sub(r'\s*<\s*', '<', html)
    return html.strip()