import json
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import os
from sklearn.metrics import classification_report
from sklearn import metrics
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from io import TextIOWrapper
import zipfile



def process_equations_and_blocks(json_data, is_standard):
    """
    处理JSON数据，提取公式、文本块、图片块和表格块的边界框和文本信息。
    
    参数:
    - json_data: 列表，包含标准文档或测试文档的JSON数据。
    - is_standard: 布尔值，指示处理的数据是否为标准文档。
    
    返回:
    - 字典，包含处理后的数据。
    """
    equations_bboxs = {"inline": [], "interline": []}
    equations_texts = {"inline": [], "interline": []}
    dropped_bboxs = {"text": [], "image": [], "table": []}
    dropped_tags = {"text": []}
    para_texts = []
    para_nums = []

    for i in json_data:
        mid_json = pd.DataFrame(i).iloc[:,:-1] if is_standard else pd.DataFrame(i)
        page_data = {
            "equations_bboxs_list": {"inline": [], "interline": []},
            "equations_texts_list": {"inline": [], "interline": []},
            "dropped_bboxs_list": {"text": [], "image": [], "table": []},
            "dropped_tags_list": {"text": []},
            "para_texts_list": [],
            "para_nums_list": []
        }

        for eq_type in ["inline", "interline"]:
            for equations in mid_json.loc[f"{eq_type}_equations", :]:
                bboxs = [eq['bbox'] for eq in equations]
                texts = [eq.get('latex_text' if is_standard else 'content', '') for eq in equations]
                page_data["equations_bboxs_list"][eq_type].append(bboxs)
                page_data["equations_texts_list"][eq_type].append(texts)
        
        equations_bboxs["inline"].append(page_data["equations_bboxs_list"]["inline"])
        equations_bboxs["interline"].append(page_data["equations_bboxs_list"]["interline"])
        equations_texts["inline"].append(page_data["equations_texts_list"]["inline"])
        equations_texts["interline"].append(page_data["equations_texts_list"]["interline"])


        # 提取丢弃的文本块信息
        for dropped_text_blocks in mid_json.loc['droped_text_block',:]:
            bboxs, tags = [], []
            for block in dropped_text_blocks:
                bboxs.append(block['bbox'])
                tags.append(block.get('tag', 'None'))
            
            page_data["dropped_bboxs_list"]["text"].append(bboxs)
            page_data["dropped_tags_list"]["text"].append(tags)
        
        dropped_bboxs["text"].append(page_data["dropped_bboxs_list"]["text"])
        dropped_tags["text"].append(page_data["dropped_tags_list"]["text"])


      
        # 同时处理删除的图片块和表格块
        for block_type in ['image', 'table']:
            # page_blocks_list = []
            for blocks in mid_json.loc[f'droped_{block_type}_block', :]:
                # 如果是标准数据，直接添加整个块的列表
                if is_standard:
                    page_data["dropped_bboxs_list"][block_type].append(blocks)
                # 如果是测试数据，检查列表是否非空，并提取每个块的边界框
                else:
                    page_blocks = [block['bbox'] for block in blocks] if blocks else []
                    page_data["dropped_bboxs_list"][block_type].append(page_blocks)
            
        # 将当前页面的块边界框列表添加到结果字典中
        dropped_bboxs['image'].append(page_data["dropped_bboxs_list"]['image'])
        dropped_bboxs['table'].append(page_data["dropped_bboxs_list"]['table'])
        
        
        # 处理段落
        for para_blocks in mid_json.loc['para_blocks', :]:
            page_data["para_nums_list"].append(len(para_blocks))  # 计算段落数

            for para_block in para_blocks:
                if is_standard:
                    # 标准数据直接提取文本
                    page_data["para_texts_list"].append(para_block['text'])
                else:
                    # 测试数据可能需要检查'content'是否存在
                    if 'spans' in para_block[0] and para_block[0]['spans'][0]['type'] == 'text':
                        page_data["para_texts_list"].append(para_block[0]['spans'][0].get('content', ''))
            
            
        
        para_texts.append(page_data["para_texts_list"])
        para_nums.append(page_data["para_nums_list"])

    return {
        "equations_bboxs": equations_bboxs,
        "equations_texts": equations_texts,
        "dropped_bboxs": dropped_bboxs,
        "dropped_tags": dropped_tags,
        "para_texts": para_texts,
        "para_nums": para_nums
    }







def bbox_match_indicator_general(test_bboxs_list, standard_bboxs_list):
    """
    计算边界框匹配指标，支持掉落的表格、图像和文本块。
    此版本的函数专注于计算基于边界框的匹配指标，而不涉及标签匹配逻辑。
    
    参数:
    - test_bboxs: 测试集的边界框列表，按页面组织。
    - standard_bboxs: 标准集的边界框列表，按页面组织。

    返回:
    - 一个字典，包含准确度、精确度、召回率和F1分数。
    """
        # 如果两个列表都完全为空，返回0值指标
    if all(len(page) == 0 for page in test_bboxs_list) and all(len(page) == 0 for page in standard_bboxs_list):
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    

    matched_bbox = []
    matched_standard_bbox = []

    for test_page, standard_page in zip(test_bboxs_list, standard_bboxs_list):
        test_page_bbox, standard_page_bbox = [], []
        for standard_bbox in standard_page:
            if len(standard_bbox) != 4:
                continue
            matched = False
            for test_bbox in test_page:
                if len(test_bbox) == 4 and bbox_offset(standard_bbox, test_bbox):
                    matched = True
                    break
            test_page_bbox.append(int(matched))
            standard_page_bbox.append(1)

        # 后处理以处理多删情况，保持原逻辑不变
        diff_num = len(test_page) + test_page_bbox.count(0) - len(standard_page)
        if diff_num > 0:
            test_page_bbox.extend([1] * diff_num)
            standard_page_bbox.extend([0] * diff_num)

        matched_bbox.extend(test_page_bbox)
        matched_standard_bbox.extend(standard_page_bbox)

    block_report = {
        'accuracy': metrics.accuracy_score(matched_standard_bbox, matched_bbox),
        'precision': metrics.precision_score(matched_standard_bbox, matched_bbox, zero_division=0),
        'recall': metrics.recall_score(matched_standard_bbox, matched_bbox, zero_division=0),
        'f1_score': metrics.f1_score(matched_standard_bbox, matched_bbox, zero_division=0)
    }

    return block_report






def bbox_offset(b_t, b_s):
    """
    判断两个边界框（bounding box）之间的重叠程度是否符合给定的标准。
    
    参数:
    - b_t: 测试文档中的边界框（bbox），格式为(x1, y1, x2, y2)，
           其中(x1, y1)是左上角的坐标，(x2, y2)是右下角的坐标。
    - b_s: 标准文档中的边界框（bbox），格式同上。
    
    返回:
    - True: 如果两个边界框的重叠面积与两个边界框合计面积的差的比例超过0.95，
            表明它们足够接近。
    - False: 否则，表示两个边界框不足够接近。
    
    注意:
    - 函数首先计算两个bbox的交集区域，如果这个区域的面积相对于两个bbox的面积差非常大，
      则认为这两个bbox足够接近。
    - 如果交集区域的计算结果导致无效区域（比如宽度或高度为负值），或者分母为0（即两个bbox完全不重叠），
      则函数会返回False。
    """

    # 分别提取两个bbox的坐标
    x1_t, y1_t, x2_t, y2_t = b_t
    x1_s, y1_s, x2_s, y2_s = b_s
  
    # 计算两个bbox交集区域的坐标
    x1 = max(x1_t, x1_s)
    x2 = min(x2_t, x2_s)
    y1 = max(y1_t, y1_s)
    y2 = min(y2_t, y2_s)
    
    # 如果计算出的交集区域有效，则计算其面积
    if x2 > x1 and y2 > y1:
        area_overlap = (x2 - x1) * (y2 - y1)
    else:
        # 交集区域无效，视为无重叠
        area_overlap = 0

    # 计算两个bbox的总面积，减去重叠部分避免重复计算
    area_t = (x2_t - x1_t) * (y2_t - y1_t) + (x2_s - x1_s) * (y2_s - y1_s) - area_overlap

    # 判断重叠面积是否符合标准
    
    if area_t-area_overlap==0 or area_overlap/area_t>0.95:
        return True
    else:
        return False
    

def Levenshtein_Distance(str1, str2):
    """
    计算并返回两个字符串之间的Levenshtein编辑距离。
    
    参数:
    - str1: 字符串，第一个比较字符串。
    - str2: 字符串，第二个比较字符串。
    
    返回:
    - int: str1和str2之间的Levenshtein距离。
    
    方法:
    - 使用动态规划构建一个矩阵(matrix)，其中matrix[i][j]表示str1的前i个字符和str2的前j个字符之间的Levenshtein距离。
    - 矩阵的初始值设定为边界情况，即一个字符串与空字符串之间的距离。
    - 遍历矩阵填充每个格子的值，根据字符是否相等选择插入、删除或替换操作的最小代价。
    """
    # 初始化矩阵，大小为(len(str1)+1) x (len(str2)+1)，边界情况下的距离为i和j
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    # 遍历str1和str2的每个字符，更新矩阵中的值
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            # 如果当前字符相等，替换代价为0；否则为1
            d = 0 if (str1[i - 1] == str2[j - 1]) else 1
            # 更新当前位置的值为从str1[i]转换到str2[j]的最小操作数
            matrix[i][j] = min(matrix[i - 1][j] + 1,  # 删除操作
                               matrix[i][j - 1] + 1,  # 插入操作
                               matrix[i - 1][j - 1] + d)  # 替换操作
    # 返回右下角的值，即str1和str2之间的Levenshtein距离
    return matrix[len(str1)][len(str2)]


def equations_indicator(test_equations_bboxs, standard_equations_bboxs, test_equations, standard_equations):
    """
    根据边界框匹配的方程计算编辑距离和BLEU分数。
    
    参数:
    - test_equations_bboxs: 测试方程的边界框列表。
    - standard_equations_bboxs: 标准方程的边界框列表。
    - test_equations: 测试方程的列表。
    - standard_equations: 标准方程的列表。
    
    返回:
    - 一个元组，包含匹配方程的平均Levenshtein编辑距离和BLEU分数。
    """
    
    # 初始化匹配方程列表
    test_match_equations = []
    standard_match_equations = []

    # 匹配方程基于边界框重叠
    for index, (test_bbox, standard_bbox) in enumerate(zip(test_equations_bboxs, standard_equations_bboxs)):
        if not (test_bbox and standard_bbox):  # 跳过任一空列表
            continue
        for i, sb in enumerate(standard_bbox):
            for j, tb in enumerate(test_bbox):
                if bbox_offset(sb, tb):
                    standard_match_equations.append(standard_equations[index][i])
                    test_match_equations.append(test_equations[index][j])
                    break  # 找到第一个匹配后即跳出循环

    # 使用Levenshtein距离和BLEU分数计算编辑距离
    dis = [Levenshtein_Distance(a, b) for a, b in zip(test_match_equations, standard_match_equations) if a and b]
    # 应用平滑函数计算BLEU分数
    sm_func = SmoothingFunction().method1
    bleu = [sentence_bleu([a.split()], b.split(), smoothing_function=sm_func) for a, b in zip(test_match_equations, standard_match_equations) if a and b]

    # 计算平均编辑距离和BLEU分数，处理空列表情况
    equations_edit = np.mean(dis) if dis else float('0.0')
    equations_bleu = np.mean(bleu) if bleu else float('0.0')

    return equations_edit, equations_bleu



def bbox_match_indicator_general(test_bboxs_list, standard_bboxs_list):
    """
    计算边界框匹配指标，支持掉落的表格、图像和文本块。
    此版本的函数专注于计算基于边界框的匹配指标，而不涉及标签匹配逻辑。
    
    参数:
    - test_bboxs: 测试集的边界框列表，按页面组织。
    - standard_bboxs: 标准集的边界框列表，按页面组织。

    返回:
    - 一个字典，包含准确度、精确度、召回率和F1分数。
    """
        # 如果两个列表都完全为空，返回0值指标
    if all(len(page) == 0 for page in test_bboxs_list) and all(len(page) == 0 for page in standard_bboxs_list):
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    

    matched_bbox = []
    matched_standard_bbox = []

    for test_page, standard_page in zip(test_bboxs_list, standard_bboxs_list):
        test_page_bbox, standard_page_bbox = [], []
        for standard_bbox in standard_page:
            if len(standard_bbox) != 4:
                continue
            matched = False
            for test_bbox in test_page:
                if len(test_bbox) == 4 and bbox_offset(standard_bbox, test_bbox):
                    matched = True
                    break
            test_page_bbox.append(int(matched))
            standard_page_bbox.append(1)

        # 后处理以处理多删情况，保持原逻辑不变
        diff_num = len(test_page) + test_page_bbox.count(0) - len(standard_page)
        if diff_num > 0:
            test_page_bbox.extend([1] * diff_num)
            standard_page_bbox.extend([0] * diff_num)

        matched_bbox.extend(test_page_bbox)
        matched_standard_bbox.extend(standard_page_bbox)

    block_report = {
        'accuracy': metrics.accuracy_score(matched_standard_bbox, matched_bbox),
        'precision': metrics.precision_score(matched_standard_bbox, matched_bbox, zero_division=0),
        'recall': metrics.recall_score(matched_standard_bbox, matched_bbox, zero_division=0),
        'f1_score': metrics.f1_score(matched_standard_bbox, matched_bbox, zero_division=0)
    }

    return block_report


def bbox_match_indicator_dropped_text_block(test_dropped_text_bboxs, standard_dropped_text_bboxs, standard_dropped_text_tag, test_dropped_text_tag):
    """
    计算丢弃文本块的边界框匹配相关指标，包括准确率、精确率、召回率和F1分数，
    同时也计算文本块标签的匹配指标。

    参数:
    - test_dropped_text_bboxs: 测试集的丢弃文本块边界框列表
    - standard_dropped_text_bboxs: 标准集的丢弃文本块边界框列表
    - standard_dropped_text_tag: 标准集的丢弃文本块标签列表
    - test_dropped_text_tag: 测试集的丢弃文本块标签列表

    返回:
    - 一个包含边界框匹配指标和文本块标签匹配指标的元组
    """
    test_text_bbox, standard_text_bbox = [], []
    test_tag, standard_tag = [], []

    for index, (test_page, standard_page) in enumerate(zip(test_dropped_text_bboxs, standard_dropped_text_bboxs)):
        # 初始化每个页面的结果列表
        test_page_tag, standard_page_tag = [], []
        test_page_bbox, standard_page_bbox = [], []

        for i, standard_bbox in enumerate(standard_page):
            matched = False
            for j, test_bbox in enumerate(test_page):
                if bbox_offset(standard_bbox, test_bbox):
                    # 匹配成功，记录标签和边界框匹配结果
                    matched = True
                    test_page_tag.append(test_dropped_text_tag[index][j])
                    test_page_bbox.append(1)
                    break

            if not matched:
                # 未匹配，记录'None'和边界框未匹配结果
                test_page_tag.append('None')
                test_page_bbox.append(0)

            # 标准边界框和标签总是被视为匹配的
            standard_page_tag.append(standard_dropped_text_tag[index][i])
            standard_page_bbox.append(1)

        # 处理可能的多删情况
        handle_multi_deletion(test_page, test_page_tag, test_page_bbox, standard_page_tag, standard_page_bbox)

        # 合并当前页面的结果到整体结果中
        test_tag.extend(test_page_tag)
        standard_tag.extend(standard_page_tag)
        test_text_bbox.extend(test_page_bbox)
        standard_text_bbox.extend(standard_page_bbox)

    # 计算和返回匹配指标
    text_block_report = {
        'accuracy': metrics.accuracy_score(standard_text_bbox, test_text_bbox),
        'precision': metrics.precision_score(standard_text_bbox, test_text_bbox, zero_division=0),
        'recall': metrics.recall_score(standard_text_bbox, test_text_bbox, zero_division=0),
        'f1_score': metrics.f1_score(standard_text_bbox, test_text_bbox, zero_division=0)
    }

    # 计算和返回标签匹配指标
    text_block_tag_report = classification_report(y_true=standard_tag, y_pred=test_tag, labels=list(set(standard_tag) - {'None'}), output_dict=True, zero_division=0)
    del text_block_tag_report["macro avg"]
    del text_block_tag_report["weighted avg"]
    
    return text_block_report, text_block_tag_report

def handle_multi_deletion(test_page, test_page_tag, test_page_bbox, standard_page_tag, standard_page_bbox):
    """
    处理多删情况，即测试页面的边界框或标签数量多于标准页面。
    """
    excess_count = len(test_page) + test_page_bbox.count(0) - len(standard_page_tag)
    if excess_count > 0:
        # 对于多出的项，将它们视为正确匹配的边界框，但标签视为'None'
        test_page_bbox.extend([1] * excess_count)
        standard_page_bbox.extend([0] * excess_count)
        test_page_tag.extend(['None'] * excess_count)
        standard_page_tag.extend(['None'] * excess_count)



def check_json_files_in_zip_exist(zip_file_path, standard_json_path_in_zip, test_json_path_in_zip):
    """
    检查ZIP文件中是否存在指定的JSON文件
    """
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # 获取ZIP文件中所有文件的列表
        all_files_in_zip = z.namelist()
        # 检查标准文件和测试文件是否都在ZIP文件中
        if standard_json_path_in_zip not in all_files_in_zip or test_json_path_in_zip not in all_files_in_zip:
            raise FileNotFoundError("One or both of the required JSON files are missing from the ZIP archive.")


def read_json_files_from_streams(standard_file_stream, test_file_stream):
    """
    从文件流中读取JSON文件内容
    """
    pdf_json_standard = [json.loads(line) for line in standard_file_stream]
    pdf_json_test = [json.loads(line) for line in test_file_stream]

    json_standard_origin = pd.DataFrame(pdf_json_standard)
    json_test_origin = pd.DataFrame(pdf_json_test)

    return json_standard_origin, json_test_origin

def read_json_files_from_zip(zip_file_path, standard_json_path_in_zip, test_json_path_in_zip):
    """
    从ZIP文件中读取两个JSON文件并返回它们的DataFrame
    """
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(standard_json_path_in_zip) as standard_file_stream, \
             z.open(test_json_path_in_zip) as test_file_stream:

            standard_file_text_stream = TextIOWrapper(standard_file_stream, encoding='utf-8')
            test_file_text_stream = TextIOWrapper(test_file_stream, encoding='utf-8')

            json_standard_origin, json_test_origin = read_json_files_from_streams(
                standard_file_text_stream, test_file_text_stream
            )
    
    return json_standard_origin, json_test_origin


def merge_json_data(json_test_df, json_standard_df):
    """
    基于ID合并测试和标准数据集，并返回合并后的数据及存在性检查结果。

    参数:
    - json_test_df: 测试数据的DataFrame。
    - json_standard_df: 标准数据的DataFrame。

    返回:
    - inner_merge: 内部合并的DataFrame，包含匹配的数据行。
    - standard_exist: 标准数据存在性的Series。
    - test_exist: 测试数据存在性的Series。
    """
    test_data = json_test_df[['id', 'mid_json']].drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    standard_data = json_standard_df[['id', 'mid_json', 'pass_label']].drop_duplicates(subset='id', keep='first').reset_index(drop=True)

    outer_merge = pd.merge(test_data, standard_data, on='id', how='outer')
    outer_merge.columns = ['id', 'test_mid_json', 'standard_mid_json', 'pass_label']

    standard_exist = outer_merge.standard_mid_json.notnull()
    test_exist = outer_merge.test_mid_json.notnull()

    inner_merge = pd.merge(test_data, standard_data, on='id', how='inner')
    inner_merge.columns = ['id', 'test_mid_json', 'standard_mid_json', 'pass_label']

    return inner_merge, standard_exist, test_exist


def consolidate_data(test_data, standard_data, key_path):
    """
    Consolidates data from test and standard datasets based on the provided key path.
    
    :param test_data: Dictionary containing the test dataset.
    :param standard_data: Dictionary containing the standard dataset.
    :param key_path: List of keys leading to the desired data within the dictionaries.
    :return: List containing all items from both test and standard data at the specified key path.
    """
    # Initialize an empty list to hold the consolidated data
    overall_data_standard = []
    overall_data_test = []
    
    # Helper function to recursively navigate through the dictionaries based on the key path
    def extract_data(source_data, keys):
        for key in keys[:-1]:
            source_data = source_data.get(key, {})
        return source_data.get(keys[-1], [])
    
    for data in extract_data(standard_data, key_path):
    # 假设每个 single_table_tags 已经是一个列表，直接将它的元素添加到总列表中
        overall_data_standard.extend(data)
    
    for data in extract_data(test_data, key_path):
         overall_data_test.extend(data)
    # Extract and extend the overall data list with items from both test and standard datasets

    
    return overall_data_standard, overall_data_test

def overall_calculate_metrics(inner_merge, json_test, json_standard,standard_exist, test_exist):

    process_data_standard = process_equations_and_blocks(json_standard, is_standard=True)
    process_data_test = process_equations_and_blocks(json_test, is_standard=False)


    overall_report = {}
    overall_report['accuracy']=metrics.accuracy_score(standard_exist,test_exist)
    overall_report['precision']=metrics.precision_score(standard_exist,test_exist)
    overall_report['recall']=metrics.recall_score(standard_exist,test_exist)
    overall_report['f1_score']=metrics.f1_score(standard_exist,test_exist)
    overall_report

    test_para_text = np.asarray(process_data_test['para_texts'], dtype=object)[inner_merge['pass_label'] == 'yes']
    standard_para_text = np.asarray(process_data_standard['para_texts'], dtype=object)[inner_merge['pass_label'] == 'yes']
    ids_yes = inner_merge['id'][inner_merge['pass_label'] == 'yes'].tolist()

    pdf_dis = {}
    pdf_bleu = {}

    # 对pass_label为'yes'的数据计算编辑距离和BLEU得分
    for idx,(a, b, id) in enumerate(zip(test_para_text, standard_para_text, ids_yes)):
        a1 = ''.join(a)
        b1 = ''.join(b)
        pdf_dis[id] = Levenshtein_Distance(a, b)
        pdf_bleu[id] = sentence_bleu([a1], b1)

    overall_report['pdf间的平均编辑距离'] = np.mean(list(pdf_dis.values()))
    overall_report['pdf间的平均bleu'] = np.mean(list(pdf_bleu.values()))

    # Consolidate equations bboxs inline
    overall_equations_bboxs_inline_standard,overall_equations_bboxs_inline_test = consolidate_data(process_data_test, process_data_standard, ["equations_bboxs", "inline"])

    # # Consolidate equations texts inline
    overall_equations_texts_inline_standard,overall_equations_texts_inline_test = consolidate_data(process_data_test, process_data_standard, ["equations_texts", "inline"])

    # Consolidate equations bboxs interline
    overall_equations_bboxs_interline_standard,overall_equations_bboxs_interline_test = consolidate_data(process_data_test, process_data_standard, ["equations_bboxs", "interline"])

    # Consolidate equations texts interline
    overall_equations_texts_interline_standard,overall_equations_texts_interline_test = consolidate_data(process_data_test, process_data_standard, ["equations_texts", "interline"])

    overall_dropped_bboxs_text_standard,overall_dropped_bboxs_text_test = consolidate_data(process_data_test, process_data_standard, ["dropped_bboxs","text"])

    overall_dropped_tags_text_standard,overall_dropped_tags_text_test = consolidate_data(process_data_test, process_data_standard, ["dropped_tags","text"])

    overall_dropped_bboxs_image_standard,overall_dropped_bboxs_image_test = consolidate_data(process_data_test, process_data_standard, ["dropped_bboxs","image"])


    overall_dropped_bboxs_table_standard,overall_dropped_bboxs_table_test=consolidate_data(process_data_test, process_data_standard,["dropped_bboxs","table"])


    para_nums_test = process_data_test['para_nums']
    para_nums_standard=process_data_standard['para_nums']
    overall_para_nums_standard = [item for sublist in para_nums_standard for item in (sublist if isinstance(sublist, list) else [sublist])]
    overall_para_nums_test = [item for sublist in para_nums_test for item in (sublist if isinstance(sublist, list) else [sublist])]


    test_para_num=np.array(overall_para_nums_test)
    standard_para_num=np.array(overall_para_nums_standard)
    acc_para=np.mean(test_para_num==standard_para_num)


    overall_report['分段准确率'] = acc_para

    # 行内公式准确率和编辑距离、bleu
    overall_report['行内公式准确率'] = bbox_match_indicator_general(
        overall_equations_bboxs_inline_test,
        overall_equations_bboxs_inline_standard)

    overall_report['行内公式编辑距离'], overall_report['行内公式bleu'] = equations_indicator(
        overall_equations_bboxs_inline_test,
        overall_equations_bboxs_inline_standard,
        overall_equations_texts_inline_test,
        overall_equations_texts_inline_standard)

    # 行间公式准确率和编辑距离、bleu
    overall_report['行间公式准确率'] = bbox_match_indicator_general(
        overall_equations_bboxs_interline_test,
        overall_equations_bboxs_interline_standard)

    overall_report['行间公式编辑距离'], overall_report['行间公式bleu'] = equations_indicator(
        overall_equations_bboxs_interline_test,
        overall_equations_bboxs_interline_standard,
        overall_equations_texts_interline_test,
        overall_equations_texts_interline_standard)

    # 丢弃文本准确率，丢弃文本标签准确率
    overall_report['丢弃文本准确率'], overall_report['丢弃文本标签准确率'] = bbox_match_indicator_dropped_text_block(
        overall_dropped_bboxs_text_test,
        overall_dropped_bboxs_text_standard,
        overall_dropped_tags_text_standard,
        overall_dropped_tags_text_test)

    # 丢弃图片准确率
    overall_report['丢弃图片准确率'] = bbox_match_indicator_general(
        overall_dropped_bboxs_image_test,
        overall_dropped_bboxs_image_standard)

    # 丢弃表格准确率
    overall_report['丢弃表格准确率'] = bbox_match_indicator_general(
        overall_dropped_bboxs_table_test,
        overall_dropped_bboxs_table_standard)

    return overall_report



def calculate_metrics(inner_merge, json_test, json_standard, json_standard_origin):
    """
    计算指标
    """
    # 创建ID到file_id的映射
    id_to_file_id_map = pd.Series(json_standard_origin.file_id.values, index=json_standard_origin.id).to_dict()

    # 处理标准数据和测试数据
    process_data_standard = process_equations_and_blocks(json_standard, is_standard=True)
    process_data_test = process_equations_and_blocks(json_test, is_standard=False)

    # 从inner_merge中筛选出pass_label为'yes'的数据
    test_para_text = np.asarray(process_data_test['para_texts'], dtype=object)[inner_merge['pass_label'] == 'yes']
    standard_para_text = np.asarray(process_data_standard['para_texts'], dtype=object)[inner_merge['pass_label'] == 'yes']
    ids_yes = inner_merge['id'][inner_merge['pass_label'] == 'yes'].tolist()

    pdf_dis = {}
    pdf_bleu = {}

    # 对pass_label为'yes'的数据计算编辑距离和BLEU得分
    for idx, (a, b, id) in enumerate(zip(test_para_text, standard_para_text, ids_yes)):
        a1 = ''.join(a)
        b1 = ''.join(b)
        pdf_dis[id] = Levenshtein_Distance(a, b)
        pdf_bleu[id] = sentence_bleu([a1], b1)

        
    result_dict = {}
    acc_para=[]

    # 对所有数据计算其他指标
    for index, id_value in enumerate(inner_merge['id'].tolist()):
        result = {}
        
        # 增加file_id到结果中
        file_id = id_to_file_id_map.get(id_value, "Unknown")
        result['file_id'] = file_id
        

        
        # 根据id判断是否需要计算pdf_dis和pdf_bleu
        if id_value in ids_yes:
            result['pdf_dis'] = pdf_dis[id_value]
            result['pdf_bleu'] = pdf_bleu[id_value]
        
        

        # 计算分段准确率
        single_test_para_num = np.array(process_data_test['para_nums'][index])
        single_standard_para_num = np.array(process_data_standard['para_nums'][index])
        acc_para.append(np.mean(single_test_para_num == single_standard_para_num))
        
        result['分段准确率'] = acc_para[index]
    
        # 行内公式准确率和编辑距离、bleu
        result['行内公式准确率'] = bbox_match_indicator_general(
            process_data_test["equations_bboxs"]["inline"][index],
            process_data_standard["equations_bboxs"]["inline"][index])
        
        result['行内公式编辑距离'], result['行内公式bleu'] = equations_indicator(
            process_data_test["equations_bboxs"]["inline"][index],
            process_data_standard["equations_bboxs"]["inline"][index],
            process_data_test["equations_texts"]["inline"][index],
            process_data_standard["equations_texts"]["inline"][index])

        # 行间公式准确率和编辑距离、bleu
        result['行间公式准确率'] = bbox_match_indicator_general(
            process_data_test["equations_bboxs"]["interline"][index],
            process_data_standard["equations_bboxs"]["interline"][index])
        
        result['行间公式编辑距离'], result['行间公式bleu'] = equations_indicator(
            process_data_test["equations_bboxs"]["interline"][index],
            process_data_standard["equations_bboxs"]["interline"][index],
            process_data_test["equations_texts"]["interline"][index],
            process_data_standard["equations_texts"]["interline"][index])

        # 丢弃文本准确率，丢弃文本标签准确率
        result['丢弃文本准确率'], result['丢弃文本标签准确率'] = bbox_match_indicator_dropped_text_block(
            process_data_test["dropped_bboxs"]["text"][index],
            process_data_standard["dropped_bboxs"]["text"][index],
            process_data_standard["dropped_tags"]["text"][index],
            process_data_test["dropped_tags"]["text"][index])

        # 丢弃图片准确率
        result['丢弃图片准确率'] = bbox_match_indicator_general(
            process_data_test["dropped_bboxs"]["image"][index],
            process_data_standard["dropped_bboxs"]["image"][index])

        # 丢弃表格准确率
        result['丢弃表格准确率'] = bbox_match_indicator_general(
            process_data_test["dropped_bboxs"]["table"][index],
            process_data_standard["dropped_bboxs"]["table"][index])


        # 将结果存入result_dict
        result_dict[id_value] = result

    return result_dict



def save_results(result_dict,overall_report_dict,badcase_path,overall_path,):
    """
    将结果字典保存为JSON文件至指定路径。

    参数:
    - result_dict: 包含计算结果的字典。
    - overall_path: 结果文件的保存路径，包括文件名。
    """
    # 打开指定的文件以写入
    with open(badcase_path, 'w', encoding='utf-8') as f:
        # 将结果字典转换为JSON格式并写入文件
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"计算结果已经保存到文件：{badcase_path}")

    with open(overall_path, 'w', encoding='utf-8') as f:
    # 将结果字典转换为JSON格式并写入文件
        json.dump(overall_report_dict, f, ensure_ascii=False, indent=4)

    print(f"计算结果已经保存到文件：{overall_path}")

def upload_to_s3(file_path, bucket_name, s3_file_name,AWS_ACCESS_KEY,AWS_SECRET_KEY,END_POINT_URL):
    """
    上传文件到Amazon S3
    """
    s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY,endpoint_url=END_POINT_URL)
    try:
        # 上传文件到S3
        s3.upload_file(file_path, bucket_name, s3_file_name)
        print(f"文件 {s3_file_name} 成功上传到S3存储桶 {bucket_name} 中的路径 {file_path}")
    except FileNotFoundError:
        print(f"文件 {s3_file_name} 未找到，请检查文件路径是否正确。")
    except NoCredentialsError:
        print("无法找到AWS凭证，请确认您的AWS访问密钥和密钥ID是否正确。")
    except ClientError as e:
        print(f"上传文件时发生错误：{e}")

def generate_filename(badcase_path,overall_path):
    """
    生成带有当前时间戳的输出文件名。

    参数:
    - base_path: 基础路径和文件名前缀。

    返回:
    - 带有当前时间戳的完整输出文件名。
    """
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 构建并返回完整的输出文件名
    return f"{badcase_path}_{current_time}.json",f"{overall_path}_{current_time}.json"



def compare_edit_distance(json_file, overall_report):
    with open(json_file, 'r',encoding='utf-8') as f:
        json_data = json.load(f)
    
    json_edit_distance = json_data['pdf间的平均编辑距离']
    
    if overall_report['pdf间的平均编辑距离'] > json_edit_distance:
        return 0
    else:
        return 1



def main(standard_file, test_file, zip_file, badcase_path, overall_path,base_data_path,s3_bucket_name=None, s3_file_name=None, AWS_ACCESS_KEY=None, AWS_SECRET_KEY=None, END_POINT_URL=None):
    """
    主函数，执行整个评估流程。
    
    参数:
    - standard_file: 标准文件的路径。
    - test_file: 测试文件的路径。
    - zip_file: 压缩包的路径的路径。
    - badcase_path: badcase文件的基础路径和文件名前缀。
    - overall_path: overall文件的基础路径和文件名前缀。
    - s3_bucket_name: S3桶名称（可选）。
    - s3_file_name: S3上的文件名（可选）。
    - AWS_ACCESS_KEY, AWS_SECRET_KEY, END_POINT_URL: AWS访问凭证和端点URL（可选）。
    """
    # 检查文件是否存在
    check_json_files_in_zip_exist(zip_file, standard_file, test_file)

    # 读取JSON文件内容
    json_standard_origin, json_test_origin = read_json_files_from_zip(zip_file, standard_file, test_file)

    # 合并JSON数据
    inner_merge, standard_exist, test_exist = merge_json_data(json_test_origin, json_standard_origin)

    #计算总体指标
    overall_report_dict=overall_calculate_metrics(inner_merge, inner_merge['test_mid_json'], inner_merge['standard_mid_json'],standard_exist, test_exist)
    # 计算指标
    result_dict = calculate_metrics(inner_merge, inner_merge['test_mid_json'], inner_merge['standard_mid_json'], json_standard_origin)

    # 生成带时间戳的输出文件名
    badcase_file,overall_file = generate_filename(badcase_path,overall_path)

    # 保存结果到JSON文件
    save_results(result_dict, overall_report_dict,badcase_file,overall_file)

    result=compare_edit_distance(base_data_path, overall_report_dict)
    print(result)
    assert result == 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="主函数，执行整个评估流程。")
    parser.add_argument('standard_file', type=str, help='标准文件的路径。')
    parser.add_argument('test_file', type=str, help='测试文件的路径。')
    parser.add_argument('zip_file', type=str, help='压缩包的路径。')
    parser.add_argument('badcase_path', type=str, help='badcase文件的基础路径和文件名前缀。')
    parser.add_argument('overall_path', type=str, help='overall文件的基础路径和文件名前缀。')
    parser.add_argument('base_data_path', type=str, help='基准文件的基础路径和文件名前缀。')
    parser.add_argument('--s3_bucket_name', type=str, help='S3桶名称。', default=None)
    parser.add_argument('--s3_file_name', type=str, help='S3上的文件名。', default=None)
    parser.add_argument('--AWS_ACCESS_KEY', type=str, help='AWS访问密钥。', default=None)
    parser.add_argument('--AWS_SECRET_KEY', type=str, help='AWS秘密密钥。', default=None)
    parser.add_argument('--END_POINT_URL', type=str, help='AWS端点URL。', default=None)

    args = parser.parse_args()

    main(args.standard_file, args.test_file, args.zip_file, args.badcase_path,args.overall_path,args.base_data_path,args.s3_bucket_name, args.s3_file_name, args.AWS_ACCESS_KEY, args.AWS_SECRET_KEY, args.END_POINT_URL)

