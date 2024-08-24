from magic_pdf.libs.boxbase import _is_in_or_part_overlap, _is_in, _is_part_overlap
from magic_pdf.libs.drop_reason import DropReason

def _remove_overlap_between_bbox(bbox1, bbox2):
   if _is_part_overlap(bbox1, bbox2):
        ix0, iy0, ix1, iy1 = bbox1
        x0, y0, x1, y1 = bbox2

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

        if ix1 > ix0 and iy1 > iy0 and y1 > y0 and x1 > x0:
            bbox1 = [ix0, iy0, ix1, iy1]
            bbox2 = [x0, y0, x1, y1]
            return bbox1, bbox2, None
        else:
            return bbox1, bbox2, DropReason.NEGATIVE_BBOX_AREA
   else:
       return bbox1, bbox2, None


def _remove_overlap_between_bboxes(arr):
    drop_reasons = []
    N = len(arr)
    keeps = [True] * N
    res = [None] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if _is_in(arr[i]["bbox"], arr[j]["bbox"]):
                keeps[i] = False

    for idx, v in enumerate(arr):
        if not keeps[idx]:
            continue
        for i in range(N):
            if res[i] is None:
                continue
        
            bbox1, bbox2, drop_reason = _remove_overlap_between_bbox(v["bbox"], res[i]["bbox"])
            if drop_reason is None:
                v["bbox"] = bbox1
                res[i]["bbox"] = bbox2
            else:
                if v["score"] > res[i]["score"]:
                    keeps[i] = False
                    res[i] = None
                else:
                    keeps[idx] = False
                drop_reasons.append(drop_reasons)
        if keeps[idx]:
            res[idx] = v
    return res, drop_reasons


def remove_overlap_between_bbox_for_span(spans):
    arr = [{"bbox": span["bbox"], "score": span.get("score", 0.1)} for span in spans ]
    res, drop_reasons = _remove_overlap_between_bboxes(arr)
    ret = []
    for i in range(len(res)):
        if res[i] is None:
            continue
        spans[i]["bbox"] = res[i]["bbox"]
        ret.append(spans[i])
    return ret, drop_reasons


def remove_overlap_between_bbox_for_block(all_bboxes):
    arr = [{"bbox": bbox[:4], "score": bbox[-1]} for bbox in all_bboxes ]
    res, drop_reasons = _remove_overlap_between_bboxes(arr)
    ret = []
    for i in range(len(res)):
        if res[i] is None:
            continue
        all_bboxes[i][:4] = res[i]["bbox"]
        ret.append(all_bboxes[i])
    return ret, drop_reasons

