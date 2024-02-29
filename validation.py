import numpy as np
from mmeval import COCODetection
import distance


def reformat_gt_and_pred(labels, det_res, label_classes):
    preds = []
    gts = []
    
    for idx, (ann, pred) in enumerate(zip(labels, det_res)):
        # with open(label_path, "r") as f:
        #     ann = json.load(f)
        gt_bboxes = []
        gt_labels = []
        for item in ann['step_1']['result']:
            if item['attribute'] in label_classes:
                gt_bboxes.append([item['x'], item['y'], item['x']+item['width'], item['y']+item['height']])
                gt_labels.append(label_classes.index(item['attribute']))
        
        gts.append({
            'img_id': idx,
            'width': ann['width'],
            'height': ann['height'],
            'bboxes': np.array(gt_bboxes),
            'labels': np.array(gt_labels),
            'ignore_flags': [False]*len(gt_labels),
        })
        
        bboxes = []
        labels = []
        scores = []
        for item in pred:
            bboxes.append(item['box'])
            labels.append(label_classes.index(item['type']))
            scores.append(item['score'])
        preds.append({
            'img_id': idx,
            'bboxes': np.array(bboxes),
            'scores': np.array(scores),
            'labels': np.array(labels),
        })
    
    return gts, preds


def detect_val(labels, det_res, label_classes):
    # label_classes = ['inline_formula', "formula"]
    meta={'CLASSES':tuple(label_classes)}
    coco_det_metric = COCODetection(dataset_meta=meta, metric=['bbox'])

    gts, preds = reformat_gt_and_pred(labels, det_res, label_classes)
    
    res = coco_det_metric(predictions=preds, groundtruths=gts)

    return res


def label_match(annotations, match_change_dict):
    for item in annotations['step_1']['result']:
        if item['attribute'] in match_change_dict.keys():
            item['attribute'] = match_change_dict[item['attribute']]
    return annotations


def format_gt_bbox(annotations, ignore_category):
    gt_bboxes = []
    for item in annotations['step_1']['result']:
        if item['textAttribute'] and item['attribute'] not in ignore_category:
            x0 = item['x']
            y0 = item['y']
            x1 = item['x'] + item['width']
            y1 = item['y'] + item['height']
            order = item['textAttribute']
            category = item['attribute']
            gt_bboxes.append([x0, y0, x1, y1, order, None, None, category])
    return gt_bboxes


def cal_edit_distance(sorted_bboxes):  
    # order_list = [int(bbox[4]) for bbox in sorted_bboxes]
    # print(sorted_bboxes[0][0][12])
    order_list = [int(bbox[12]) for bbox in sorted_bboxes]
    sorted_order = sorted(order_list, key=int)
    distance_cal = distance.levenshtein(order_list, sorted_order)
    if len(order_list) > 0:
        return distance_cal / len(order_list)
    else:
        return 0
    