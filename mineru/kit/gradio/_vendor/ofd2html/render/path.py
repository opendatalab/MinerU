"""OFD ``AbbreviatedData`` -> SVG ``d`` attribute.

OFD path operators (GB/T 33190-2016 sec 9):
    S x y                   start (move-to)
    M x y                   move-to
    L x y                   line-to
    Q x1 y1 x2 y2           quadratic bezier
    B x1 y1 x2 y2 x3 y3     cubic bezier
    A rx ry rot la sw x y   elliptic arc (same args as SVG)
    C                       close path

We map them to the corresponding SVG path commands.
"""

from __future__ import annotations

# Number of numeric arguments per OFD operator.
_OP_ARITY = {
    "S": 2,
    "M": 2,
    "L": 2,
    "Q": 4,
    "B": 6,
    "A": 7,
    "C": 0,
}

# Mapping to SVG operators.
_OP_TO_SVG = {
    "S": "M",
    "M": "M",
    "L": "L",
    "Q": "Q",
    "B": "C",
    "A": "A",
    "C": "Z",
}


# 将 OFD 缩写路径指令转换为 SVG 路径数据。
def abbr_data_to_svg_d(raw: str) -> str:
    """Convert an OFD ``AbbreviatedData`` string into an SVG ``d`` value."""
    if not raw:
        return ""
    tokens = raw.replace(",", " ").split()
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        op = tokens[i]
        if op not in _OP_ARITY:
            # Skip unknown tokens defensively rather than aborting the whole page.
            i += 1
            continue
        arity = _OP_ARITY[op]
        args = tokens[i + 1 : i + 1 + arity]
        if len(args) < arity:
            break
        svg_op = _OP_TO_SVG[op]
        if arity == 0:
            out.append(svg_op)
        else:
            out.append(svg_op + " " + " ".join(args))
        i += 1 + arity
    return " ".join(out)
