from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_in


def _remove_overlap_between_bbox_for_span(spans):
    res = []

    keeps = [True] * len(spans)
    for i in range(len(spans)):
        for j in range(len(spans)):
            if i == j:
                continue
            if _is_in(spans[i]["bbox"], spans[j]["bbox"]):
                keeps[i] = False

    for idx, v in enumerate(spans):
        if not keeps[idx]:
            continue

        for i in range(len(res)):
            if _is_in(v["bbox"], res[i]["bbox"]):
                continue
            if _is_in_or_part_overlap(res[i]["bbox"], v["bbox"]):
                ix0, iy0, ix1, iy1 = res[i]["bbox"]
                x0, y0, x1, y1 = v["bbox"]

                diff_x = min(x1, ix1) - max(x0, ix0)
                diff_y = min(y1, iy1) - max(y0, iy0)

                if diff_y > diff_x:
                    if x1 >= ix1:
                        mid = (x0 + ix1) // 2
                        ix1 = min(mid - 0.25, ix1)
                        x0 = max(mid + 0.25, x0)
                    else:
                        mid = (ix0 + x1) // 2
                        ix0 = max(mid + 0.25, ix0)
                        x1 = min(mid - 0.25, x1)
                else:
                    if y1 >= iy1:
                        mid = (y0 + iy1) // 2
                        y0 = max(mid + 0.25, y0)
                        iy1 = min(iy1, mid-0.25)
                    else:
                        mid = (iy0 + y1) // 2
                        y1 = min(y1, mid-0.25)
                        iy0 = max(mid + 0.25, iy0)
                res[i]["bbox"] = [ix0, iy0, ix1, iy1]
                v["bbox"] = [x0, y0, x1, y1]

        res.append(v)
    return res


def _remove_overlap_between_bbox_for_block(all_bboxes):
    res = []

    keeps = [True] * len(all_bboxes)
    for i in range(len(all_bboxes)):
        for j in range(len(all_bboxes)):
            if i == j:
                continue
            if _is_in(all_bboxes[i][:4], all_bboxes[j][:4]):
                keeps[i] = False

    for idx, v in enumerate(all_bboxes):
        if not keeps[idx]:
            continue

        for i in range(len(res)):
            if _is_in(v[:4], res[i][:4]):
                continue
            if _is_in_or_part_overlap(res[i][:4], v[:4]):
                ix0, iy0, ix1, iy1 = res[i][:4]
                x0, y0, x1, y1 = v[:4]

                diff_x = min(x1, ix1) - max(x0, ix0)
                diff_y = min(y1, iy1) - max(y0, iy0)

                if diff_y > diff_x:
                    if x1 >= ix1:
                        mid = (x0 + ix1) // 2
                        ix1 = min(mid - 0.25, ix1)
                        x0 = max(mid + 0.25, x0)
                    else:
                        mid = (ix0 + x1) // 2
                        ix0 = max(mid + 0.25, ix0)
                        x1 = min(mid - 0.25, x1)
                else:
                    if y1 >= iy1:
                        mid = (y0 + iy1) // 2
                        y0 = max(mid + 0.25, y0)
                        iy1 = min(iy1, mid-0.25)
                    else:
                        mid = (iy0 + y1) // 2
                        y1 = min(y1, mid-0.25)
                        iy0 = max(mid + 0.25, iy0)
                res[i][:4] = [ix0, iy0, ix1, iy1]
                v[:4] = [x0, y0, x1, y1]

        res.append(v)
    return res


def remove_overlap_between_bbox_for_span(spans):
    return _remove_overlap_between_bbox_for_span(spans)


def remove_overlap_between_bbox_for_block(all_bboxes):
    return _remove_overlap_between_bbox_for_block(all_bboxes)
