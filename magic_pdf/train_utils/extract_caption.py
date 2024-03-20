from magic_pdf.libs.boxbase import _is_in


def extract_caption_bbox(outer: list, inner: list) -> list:
    """
    ret: list of {
                    "bbox": [1,2,3,4],
                    "caption": [5,6,7,8] # may existed
                }

    """
    found_count = 0  # for debug
    print(outer, inner)

    def is_float_equal(a, b):
        if 0.01 > abs(a - b):  # non strict float equal compare
            return True
        return False

    outer_h = {i: outer[i] for i in range(len(outer))}
    ret = []
    for v in inner:
        ix0, iy0, ix1, iy1 = v
        found_idx = None
        d = {"bbox": v[:4]}
        for k in outer_h:
            ox0, oy0, ox1, oy1 = outer_h[k]
            equal_float_flags = [
                is_float_equal(ix0, ox0),
                is_float_equal(iy0, oy0),
                is_float_equal(ix1, ox1),
                is_float_equal(iy1, oy1),
            ]
            if _is_in(v, outer_h[k]) and not all(equal_float_flags):
                found_idx = k
                break
        if found_idx is not None:
            found_count += 1
            captions: list[list] = []
            ox0, oy0, ox1, oy1 = outer_h[found_idx]
            captions = [
                [ox0, oy0, ix0, oy1],
                [ox0, oy0, ox1, iy0],
                [ox0, iy1, ox1, oy1],
                [ix1, oy0, ox1, oy1],
            ]
            captions = sorted(
                captions,
                key=lambda rect: abs(rect[0] - rect[2]) * abs(rect[1] - rect[3]),
            )  # 面积最大的框就是caption
            d["caption"] = captions[-1]
            outer_h.pop(
                found_idx
            )  # 同一个 outer box 只能用于确定一个 inner box 的 caption 位置。

        ret.append(d)

    print("found_count: ", found_count)
    return ret
