"""
包含两个MagicModel类中重复使用的方法和逻辑
"""
from typing import List, Dict, Any, Callable
from mineru.utils.boxbase import bbox_distance, bbox_center_distance, is_in


def reduct_overlap(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去除重叠的bbox，保留不被其他bbox包含的bbox

    Args:
        bboxes: 包含bbox信息的字典列表

    Returns:
        去重后的bbox列表
    """
    N = len(bboxes)
    keep = [True] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                keep[i] = False
    return [bboxes[i] for i in range(N) if keep[i]]


def tie_up_category_by_distance_v3(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None
):
    """
    通用的类别关联方法，用于将主体对象与客体对象进行关联

    参数:
        get_subjects_func: 函数，提取主体对象
        get_objects_func: 函数，提取客体对象
        extract_subject_func: 函数，自定义提取主体属性（默认使用bbox和其他属性）
        extract_object_func: 函数，自定义提取客体属性（默认使用bbox和其他属性）

    返回:
        关联后的对象列表
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # 如果没有提供自定义提取函数，使用默认函数
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    ret = []
    N, M = len(subjects), len(objects)
    subjects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)
    objects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)

    OBJ_IDX_OFFSET = 10000
    SUB_BIT_KIND, OBJ_BIT_KIND = 0, 1

    all_boxes_with_idx = [(i, SUB_BIT_KIND, sub["bbox"][0], sub["bbox"][1]) for i, sub in enumerate(subjects)] + [
        (i + OBJ_IDX_OFFSET, OBJ_BIT_KIND, obj["bbox"][0], obj["bbox"][1]) for i, obj in enumerate(objects)
    ]
    seen_idx = set()
    seen_sub_idx = set()

    while N > len(seen_sub_idx):
        candidates = []
        for idx, kind, x0, y0 in all_boxes_with_idx:
            if idx in seen_idx:
                continue
            candidates.append((idx, kind, x0, y0))

        if len(candidates) == 0:
            break
        left_x = min([v[2] for v in candidates])
        top_y = min([v[3] for v in candidates])

        candidates.sort(key=lambda x: (x[2] - left_x) ** 2 + (x[3] - top_y) ** 2)

        fst_idx, fst_kind, left_x, top_y = candidates[0]
        fst_bbox = subjects[fst_idx]['bbox'] if fst_kind == SUB_BIT_KIND else objects[fst_idx - OBJ_IDX_OFFSET]['bbox']
        candidates.sort(
            key=lambda x: bbox_distance(fst_bbox, subjects[x[0]]['bbox']) if x[1] == SUB_BIT_KIND else bbox_distance(
                fst_bbox, objects[x[0] - OBJ_IDX_OFFSET]['bbox']))
        nxt = None

        for i in range(1, len(candidates)):
            if candidates[i][1] ^ fst_kind == 1:
                nxt = candidates[i]
                break
        if nxt is None:
            break

        if fst_kind == SUB_BIT_KIND:
            sub_idx, obj_idx = fst_idx, nxt[0] - OBJ_IDX_OFFSET
        else:
            sub_idx, obj_idx = nxt[0], fst_idx - OBJ_IDX_OFFSET

        pair_dis = bbox_distance(subjects[sub_idx]["bbox"], objects[obj_idx]["bbox"])
        nearest_dis = float("inf")
        for i in range(N):
            # 取消原先算法中 1对1 匹配的偏置
            # if i in seen_idx or i == sub_idx:continue
            nearest_dis = min(nearest_dis, bbox_distance(subjects[i]["bbox"], objects[obj_idx]["bbox"]))

        if pair_dis >= 3 * nearest_dis:
            seen_idx.add(sub_idx)
            continue

        seen_idx.add(sub_idx)
        seen_idx.add(obj_idx + OBJ_IDX_OFFSET)
        seen_sub_idx.add(sub_idx)

        ret.append(
            {
                "sub_bbox": extract_subject_func(subjects[sub_idx]),
                "obj_bboxes": [extract_object_func(objects[obj_idx])],
                "sub_idx": sub_idx,
            }
        )

    for i in range(len(objects)):
        j = i + OBJ_IDX_OFFSET
        if j in seen_idx:
            continue
        seen_idx.add(j)
        nearest_dis, nearest_sub_idx = float("inf"), -1
        for k in range(len(subjects)):
            dis = bbox_distance(objects[i]["bbox"], subjects[k]["bbox"])
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_sub_idx = k

        for k in range(len(subjects)):
            if k != nearest_sub_idx:
                continue
            if k in seen_sub_idx:
                for kk in range(len(ret)):
                    if ret[kk]["sub_idx"] == k:
                        ret[kk]["obj_bboxes"].append(extract_object_func(objects[i]))
                        break
            else:
                ret.append(
                    {
                        "sub_bbox": extract_subject_func(subjects[k]),
                        "obj_bboxes": [extract_object_func(objects[i])],
                        "sub_idx": k,
                    }
                )
            seen_sub_idx.add(k)
            seen_idx.add(k)

    for i in range(len(subjects)):
        if i in seen_sub_idx:
            continue
        ret.append(
            {
                "sub_bbox": extract_subject_func(subjects[i]),
                "obj_bboxes": [],
                "sub_idx": i,
            }
        )

    return ret


def tie_up_category_by_index(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None
):
    """
    基于index的类别关联方法，用于将主体对象与客体对象进行关联
    客体优先匹配给index最接近的主体，index差值相同时使用bbox中心点距离作为tiebreaker

    参数:
        get_subjects_func: 函数，提取主体对象
        get_objects_func: 函数，提取客体对象
        extract_subject_func: 函数，自定义提取主体属性（默认使用bbox和其他属性）
        extract_object_func: 函数，自定义提取客体属性（默认使用bbox和其他属性）

    返回:
        关联后的对象列表，按主体index升序排列
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # 如果没有提供自定义提取函数，使用默认函数
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    # 初始化结果字典，key为主体索引，value为关联信息
    result_dict = {}

    # 初始化所有主体
    for i, subject in enumerate(subjects):
        result_dict[i] = {
            "sub_bbox": extract_subject_func(subject),
            "obj_bboxes": [],
            "sub_idx": i,
        }

    # 为每个客体找到最匹配的主体
    for obj in objects:
        if len(subjects) == 0:
            # 如果没有主体，跳过客体
            continue

        obj_index = obj["index"]
        min_index_diff = float("inf")
        best_subject_indices = []

        # 找出index差值最小的所有主体
        for i, subject in enumerate(subjects):
            sub_index = subject["index"]
            index_diff = abs(obj_index - sub_index)

            if index_diff < min_index_diff:
                min_index_diff = index_diff
                best_subject_indices = [i]
            elif index_diff == min_index_diff:
                best_subject_indices.append(i)

        # 如果有多个主体的index差值相同，使用中心点距离作为tiebreaker
        if len(best_subject_indices) > 1:
            min_center_dist = float("inf")
            best_subject_idx = best_subject_indices[0]

            for idx in best_subject_indices:
                center_dist = bbox_center_distance(obj["bbox"], subjects[idx]["bbox"])
                if center_dist < min_center_dist:
                    min_center_dist = center_dist
                    best_subject_idx = idx
        else:
            best_subject_idx = best_subject_indices[0]

        # 将客体添加到最佳主体的obj_bboxes中
        result_dict[best_subject_idx]["obj_bboxes"].append(extract_object_func(obj))

    # 转换为列表并按主体index排序
    ret = list(result_dict.values())
    ret.sort(key=lambda x: x["sub_idx"])

    return ret
