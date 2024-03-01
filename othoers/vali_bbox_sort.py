import numpy as np
import tqdm
import json
from validation import cal_edit_distance, format_gt_bbox
from pdf_tools.layout.layout_sort import sort_with_layout

with open('/mnt/petrelfs/share_data/ouyanglinke/OCR/OCR_validation_dataset_final_rotated_formulafix_highdpi_scihub.json', 'r') as f:
    samples = json.load(f)


# labels = []
# det_res = []
edit_distance_dict = []
edit_distance_list = []
for i, sample in tqdm.tqdm(enumerate(samples)):
    pdf_name = sample['pdf_name']
    s3_pdf_path = sample['s3_path']
    page_num = sample['page']
    page_width = sample['annotations']['width']
    page_height = sample['annotations']['height']

    # pre = main(s3_pdf_path, pdf_bin_file_profile, join_path(pdf_model_dir, pdf_name), pdf_model_profile, save_path, page_num)
    # pre_dict_list = []
    # for item in pre:
    #     pre_sample = {
    #         'box': [item[0],item[1],item[2],item[3]],
    #         'type': item[7],
    #         'score': 1
    #     }
    #     pre_dict_list.append(pre_sample)

    # det_res.append(pre_dict_list)

    # match_change_dict = {   # 待确认
    #     "figure": "image",
    #     "svg_figure": "image",
    #     "inline_fomula": "equations_inline",
    #     "fomula": "equation_interline",
    #     "figure_caption": "text",
    #     "table_caption": "text",
    #     "fomula_caption": "text"
    # }
    
    gt_annos = sample['annotations']
    # matched_label = label_match(gt_annos, match_change_dict)
    # labels.append(matched_label)

    # 判断排序函数的精度
    # 目前不考虑caption与图表相同序号的问题
    ignore_category = ['abandon', 'figure_caption', 'table_caption', 'formula_caption', 'inline_fomula'] 
    gt_bboxes = format_gt_bbox(gt_annos, ignore_category)
    sorted_bboxes, _ = sort_with_layout(gt_bboxes, page_width, page_height)
    if sorted_bboxes:
        edit_distance = cal_edit_distance(sorted_bboxes)
        edit_distance_list.append(edit_distance)
        edit_distance_dict.append({
            "sample_id": i,
            "s3_path": s3_pdf_path,
            "page_num": page_num,
            "page_s2_path": sample['page_path'],
            "edit_distance": edit_distance
        })


# label_classes = ["image", "text", "table", "equation_interline"]
# detect_matrix = detect_val(labels, det_res, label_classes)
# print('detect_matrix', detect_matrix)
edit_distance_mean = np.mean(edit_distance_list)
print('edit_distance_mean', edit_distance_mean)

edit_distance_dict_sorted = sorted(edit_distance_dict, key=lambda x: x['edit_distance'], reverse=True)
# print(edit_distance_dict_sorted)

result = {
    "edit_distance_mean": edit_distance_mean,
    "edit_distance_dict_sorted": edit_distance_dict_sorted
}

with open('vali_bbox_sort_result.json', 'w') as f:
    json.dump(result, f)