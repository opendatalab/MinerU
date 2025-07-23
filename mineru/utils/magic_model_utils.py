"""
布局处理的公共工具类
包含两个MagicModel类中重复使用的方法和逻辑
"""
from typing import List, Dict, Any, Union
from mineru.utils.boxbase import bbox_relative_pos, calculate_iou, bbox_distance, is_in


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


def bbox_distance_with_relative_check(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算两个bbox之间的距离，考虑相对位置约束

    Args:
        bbox1: 第一个bbox [x1, y1, x2, y2]
        bbox2: 第二个bbox [x1, y1, x2, y2]

    Returns:
        距离值，如果不满足条件返回无穷大
    """
    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
    flags = [left, right, bottom, top]
    count = sum([1 if v else 0 for v in flags])
    if count > 1:
        return float('inf')
    if left or right:
        l1 = bbox1[3] - bbox1[1]
        l2 = bbox2[3] - bbox2[1]
    else:
        l1 = bbox1[2] - bbox1[0]
        l2 = bbox2[2] - bbox2[0]

    if l2 > l1 and (l2 - l1) / l1 > 0.3:
        return float('inf')

    return bbox_distance(bbox1, bbox2)


def tie_up_category_by_distance_v3(
    data_source: Union[List[Dict], Dict],
    subject_category_filter,
    object_category_filter,
    extract_bbox_func=None,
    extract_score_func=None,
    create_item_func=None
) -> List[Dict[str, Any]]:
    """
    基于距离关联不同类型的区块/元素

    Args:
        data_source: 数据源，可以是列表或包含layout_dets的字典
        subject_category_filter: 主体类别过滤函数或值
        object_category_filter: 对象类别过滤函数或值
        extract_bbox_func: 提取bbox的函数，默认使用'bbox'键
        extract_score_func: 提取score的函数，默认使用'score'键
        create_item_func: 创建返回项的函数

    Returns:
        关联结果列表
    """
    # 默认函数
    if extract_bbox_func is None:
        extract_bbox_func = lambda x: x['bbox']
    if extract_score_func is None:
        extract_score_func = lambda x: x['score']
    if create_item_func is None:
        create_item_func = lambda x: {'bbox': extract_bbox_func(x), 'score': extract_score_func(x)}

    # 处理数据源
    if isinstance(data_source, dict) and 'layout_dets' in data_source:
        items = data_source['layout_dets']
    else:
        items = data_source

    # 过滤主体和对象
    if callable(subject_category_filter):
        subjects = list(filter(subject_category_filter, items))
    else:
        subjects = list(filter(lambda x: x.get('category_id') == subject_category_filter or x.get('type') == subject_category_filter, items))

    if callable(object_category_filter):
        objects = list(filter(object_category_filter, items))
    else:
        objects = list(filter(lambda x: x.get('category_id') == object_category_filter or x.get('type') == object_category_filter, items))

    # 转换为标准格式并去重
    subjects = reduct_overlap([create_item_func(x) for x in subjects])
    objects = reduct_overlap([create_item_func(x) for x in objects])

    ret = []
    N, M = len(subjects), len(objects)
    subjects.sort(key=lambda x: extract_bbox_func(x)[0] ** 2 + extract_bbox_func(x)[1] ** 2)
    objects.sort(key=lambda x: extract_bbox_func(x)[0] ** 2 + extract_bbox_func(x)[1] ** 2)

    OBJ_IDX_OFFSET = 10000
    SUB_BIT_KIND, OBJ_BIT_KIND = 0, 1

    all_boxes_with_idx = [(i, SUB_BIT_KIND, extract_bbox_func(sub)[0], extract_bbox_func(sub)[1]) for i, sub in enumerate(subjects)] + \
                        [(i + OBJ_IDX_OFFSET, OBJ_BIT_KIND, extract_bbox_func(obj)[0], extract_bbox_func(obj)[1]) for i, obj in enumerate(objects)]
    seen_idx = set()
    seen_sub_idx = set()

    seen_sub_idx_len = len(seen_sub_idx)
    while N > seen_sub_idx_len:
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
        fst_bbox = extract_bbox_func(subjects[fst_idx]) if fst_kind == SUB_BIT_KIND else extract_bbox_func(objects[fst_idx - OBJ_IDX_OFFSET])
        candidates.sort(
            key=lambda x: bbox_distance(fst_bbox, extract_bbox_func(subjects[x[0]])) if x[1] == SUB_BIT_KIND else bbox_distance(
                fst_bbox, extract_bbox_func(objects[x[0] - OBJ_IDX_OFFSET])))
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

        pair_dis = bbox_distance(extract_bbox_func(subjects[sub_idx]), extract_bbox_func(objects[obj_idx]))
        nearest_dis = float('inf')
        for i in range(N):
            # 取消原先算法中 1对1 匹配的偏置
            # if i in seen_idx or i == sub_idx:continue
            nearest_dis = min(nearest_dis, bbox_distance(extract_bbox_func(subjects[i]), extract_bbox_func(objects[obj_idx])))

        if pair_dis >= 3 * nearest_dis:
            seen_idx.add(sub_idx)
            continue

        seen_idx.add(sub_idx)
        seen_idx.add(obj_idx + OBJ_IDX_OFFSET)
        seen_sub_idx.add(sub_idx)

        ret.append({
            'sub_bbox': subjects[sub_idx],
            'obj_bboxes': [objects[obj_idx]],
            'sub_idx': sub_idx,
        })

    # 处理剩余的对象
    for i in range(len(objects)):
        j = i + OBJ_IDX_OFFSET
        if j in seen_idx:
            continue
        seen_idx.add(j)
        nearest_dis, nearest_sub_idx = float('inf'), -1
        for k in range(len(subjects)):
            dis = bbox_distance(extract_bbox_func(objects[i]), extract_bbox_func(subjects[k]))
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_sub_idx = k

        for k in range(len(subjects)):
            if k != nearest_sub_idx:
                continue
            if k in seen_sub_idx:
                for kk in range(len(ret)):
                    if ret[kk]['sub_idx'] == k:
                        ret[kk]['obj_bboxes'].append(objects[i])
                        break
            else:
                ret.append({
                    'sub_bbox': subjects[k],
                    'obj_bboxes': [objects[i]],
                    'sub_idx': k,
                })
            seen_sub_idx.add(k)
            seen_idx.add(k)

    # 处理剩余的主体
    for i in range(len(subjects)):
        if i in seen_sub_idx:
            continue
        ret.append({
            'sub_bbox': subjects[i],
            'obj_bboxes': [],
            'sub_idx': i,
        })

    return ret


def remove_high_iou_low_confidence(layout_dets: List[Dict], iou_threshold: float = 0.9):
    """
    删除高IOU且置信度较低的检测结果

    Args:
        layout_dets: 布局检测结果列表
        iou_threshold: IOU阈值
    """
    need_remove_list = []

    for i in range(len(layout_dets)):
        for j in range(i + 1, len(layout_dets)):
            layout_det1 = layout_dets[i]
            layout_det2 = layout_dets[j]

            if calculate_iou(layout_det1['bbox'], layout_det2['bbox']) > iou_threshold:
                layout_det_need_remove = layout_det1 if layout_det1['score'] < layout_det2['score'] else layout_det2
                if layout_det_need_remove not in need_remove_list:
                    need_remove_list.append(layout_det_need_remove)

    for need_remove in need_remove_list:
        if need_remove in layout_dets:
            layout_dets.remove(need_remove)


def remove_low_confidence(layout_dets: List[Dict], confidence_threshold: float = 0.05):
    """
    删除置信度特别低的检测结果

    Args:
        layout_dets: 布局检测结果列表
        confidence_threshold: 置信度阈值
    """
    need_remove_list = []
    for layout_det in layout_dets:
        if layout_det['score'] <= confidence_threshold:
            need_remove_list.append(layout_det)

    for need_remove in need_remove_list:
        if need_remove in layout_dets:
            layout_dets.remove(need_remove)

