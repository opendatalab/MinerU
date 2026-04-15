# Copyright (c) Opendatalab. All rights reserved.
"""
包含两个MagicModel类中重复使用的方法和逻辑
"""
from typing import List, Dict, Any, Callable

from loguru import logger
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


def tie_up_category_by_index(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None,
        object_block_type: str = "object",
        include_bbox: bool = True,
):
    """
    基于index的类别关联方法，用于将主体对象与客体对象进行关联
    客体优先匹配给index最接近的主体，匹配优先级为：
    1. index差值（最高优先级）
    2. bbox边缘距离（相邻边距离）
    3. bbox中心点距离（最低优先级，作为最终tiebreaker）

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

    # 提取所有客体的index集合，用于计算有效index差值
    object_indices = set(obj["index"] for obj in objects)

    def calc_effective_index_diff(obj_index: int, sub_index: int) -> int:
        """
        计算有效的index差值
        有效差值 = 绝对差值 - 区间内其他客体的数量
        即：如果obj_index和sub_index之间的差值是由其他客体造成的，则应该扣除这部分差值
        """
        if obj_index == sub_index:
            return 0

        start, end = min(obj_index, sub_index), max(obj_index, sub_index)
        abs_diff = end - start

        # 计算区间(start, end)内有多少个其他客体的index
        other_objects_count = 0
        for idx in range(start + 1, end):
            if idx in object_indices:
                other_objects_count += 1

        return abs_diff - other_objects_count

    # 为每个客体找到最匹配的主体
    for obj in objects:
        if len(subjects) == 0:
            # 如果没有主体，跳过客体
            continue

        obj_index = obj["index"]
        min_index_diff = float("inf")
        best_subject_indices = []

        # 找出有效index差值最小的所有主体
        for i, subject in enumerate(subjects):
            sub_index = subject["index"]
            index_diff = calc_effective_index_diff(obj_index, sub_index)

            if index_diff < min_index_diff:
                min_index_diff = index_diff
                best_subject_indices = [i]
            elif index_diff == min_index_diff:
                best_subject_indices.append(i)

        if len(best_subject_indices) == 1:
            best_subject_idx = best_subject_indices[0]
        # 如果有多个主体的index差值相同（最多两个），根据边缘距离进行筛选
        elif len(best_subject_indices) == 2:
            # 只有在包含bbox信息时才进行边缘距离的计算和比较，否则直接匹配第一个主体
            if include_bbox:
                # 计算所有候选主体的边缘距离
                edge_distances = [(idx, bbox_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                edge_dist_diff = abs(edge_distances[0][1] - edge_distances[1][1])

                for idx, edge_dist in edge_distances:
                    logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Edge distance: {edge_dist}")

                if edge_dist_diff > 2:
                    # 边缘距离差值大于2，匹配边缘距离更小的主体
                    best_subject_idx = min(edge_distances, key=lambda x: x[1])[0]
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff > 2, matching to subject with min edge distance, index: {subjects[best_subject_idx]['index']}")
                elif object_block_type == "table_caption":
                    # 边缘距离差值<=2且为table_caption，匹配index更大的主体
                    best_subject_idx = max(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and table_caption, matching to later subject with index: {subjects[best_subject_idx]['index']}")
                elif object_block_type.endswith("footnote"):
                    # 边缘距离差值<=2且为footnote，匹配index更小的主体
                    best_subject_idx = min(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and footnote, matching to earlier subject with index: {subjects[best_subject_idx]['index']}")
                else:
                    # 边缘距离差值<=2 且不适用特殊匹配规则，使用中心点距离匹配
                    center_distances = [(idx, bbox_center_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                    for idx, center_dist in center_distances:
                        logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Center distance: {center_dist}")
                    best_subject_idx = min(center_distances, key=lambda x: x[1])[0]
            else:
                best_subject_idx = best_subject_indices[0]
        else:
            raise ValueError("More than two subjects have the same minimal index difference, which is unexpected.")

        # 将客体添加到最佳主体的obj_bboxes中
        result_dict[best_subject_idx]["obj_bboxes"].append(extract_object_func(obj))

    # 转换为列表并按主体index排序
    ret = list(result_dict.values())
    ret.sort(key=lambda x: x["sub_idx"])

    return ret
