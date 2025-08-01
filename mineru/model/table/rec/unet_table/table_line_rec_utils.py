# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import math

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage import measure


def bbox_decode(heat, wh, reg=None, K=100):
    """bbox组成：[V1, V2, V3, V4]
    V1~V4: bbox的4个坐标点
    """
    batch = heat.shape[0]
    heat, keep = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.reshape(batch, K, 8)
    clses = clses.reshape(batch, K, 1).astype(np.float32)
    scores = scores.reshape(batch, K, 1)

    bboxes = np.concatenate(
        [
            xs - wh[..., 0:1],
            ys - wh[..., 1:2],
            xs - wh[..., 2:3],
            ys - wh[..., 3:4],
            xs - wh[..., 4:5],
            ys - wh[..., 5:6],
            xs - wh[..., 6:7],
            ys - wh[..., 7:8],
        ],
        axis=2,
    )
    detections = np.concatenate([bboxes, scores, clses], axis=2)
    return detections, inds


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = max_pool(heat, kernel_size=kernel, stride=1, padding=pad)
    keep = hmax == heat
    return heat * keep, keep


def max_pool(img, kernel_size, stride, padding):
    h, w = img.shape[2:]
    img = np.pad(
        img,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        "constant",
        constant_values=0,
    )

    res_h = ((h + 2 - kernel_size) // stride) + 1
    res_w = ((w + 2 - kernel_size) // stride) + 1
    res = np.zeros((img.shape[0], img.shape[1], res_h, res_w))
    for i in range(res_h):
        for j in range(res_w):
            temp = img[
                :,
                :,
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            res[:, :, i, j] = temp.max()
    return res


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = find_topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds / width
    topk_xs = np.float32(np.int32(topk_inds % width))

    topk_score, topk_ind = find_topk(topk_scores.reshape(batch, -1), K)
    topk_clses = np.int32(topk_ind / K)
    topk_inds = _gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)

    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)

        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis
        )
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis
        )
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def _gather_feat(feat, ind):
    dim = feat.shape[2]
    ind = np.broadcast_to(ind[:, :, None], (ind.shape[0], ind.shape[1], dim))
    feat = _gather_np(feat, 1, ind)
    return feat


def _gather_np(data, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    data_xsection_shape = data.shape[:dim] + data.shape[dim + 1 :]
    if idx_xsection_shape != data_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and data should be the same size"
        )

    if index.dtype != np.int64:
        raise TypeError("The values of index must be integers")

    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def _tranpose_and_gather_feat(feat, ind):
    feat = np.ascontiguousarray(np.transpose(feat, [0, 2, 3, 1]))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat


def gbox_decode(mk, st_reg, reg=None, K=400):
    """gbox的组成：[V1, P1, P2, P3, P4]
    P1~P4: 四个框的中心点
    V1: 四个框的交点
    """
    batch = mk.shape[0]
    mk, keep = _nms(mk)
    scores, inds, clses, ys, xs = _topk(mk, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    scores = scores.reshape(batch, K, 1)
    clses = clses.reshape(batch, K, 1).astype(np.float32)
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)

    bboxes = np.concatenate(
        [
            xs - st_Reg[..., 0:1],
            ys - st_Reg[..., 1:2],
            xs - st_Reg[..., 2:3],
            ys - st_Reg[..., 3:4],
            xs - st_Reg[..., 4:5],
            ys - st_Reg[..., 5:6],
            xs - st_Reg[..., 6:7],
            ys - st_Reg[..., 7:8],
        ],
        axis=2,
    )
    return np.concatenate([xs, ys, bboxes, scores, clses], axis=2), keep


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def bbox_post_process(bbox, c, s, h, w):
    for i in range(bbox.shape[0]):
        bbox[i, :, 0:2] = transform_preds(bbox[i, :, 0:2], c[i], s[i], (w, h))
        bbox[i, :, 2:4] = transform_preds(bbox[i, :, 2:4], c[i], s[i], (w, h))
        bbox[i, :, 4:6] = transform_preds(bbox[i, :, 4:6], c[i], s[i], (w, h))
        bbox[i, :, 6:8] = transform_preds(bbox[i, :, 6:8], c[i], s[i], (w, h))
    return bbox


def gbox_post_process(gbox, c, s, h, w):
    for i in range(gbox.shape[0]):
        gbox[i, :, 0:2] = transform_preds(gbox[i, :, 0:2], c[i], s[i], (w, h))
        gbox[i, :, 2:4] = transform_preds(gbox[i, :, 2:4], c[i], s[i], (w, h))
        gbox[i, :, 4:6] = transform_preds(gbox[i, :, 4:6], c[i], s[i], (w, h))
        gbox[i, :, 6:8] = transform_preds(gbox[i, :, 6:8], c[i], s[i], (w, h))
        gbox[i, :, 8:10] = transform_preds(gbox[i, :, 8:10], c[i], s[i], (w, h))
    return gbox


def nms(dets, thresh):
    if len(dets) < 2:
        return dets

    index_keep, keep = [], []
    for i in range(len(dets)):
        box = dets[i]
        if box[-1] < thresh:
            break

        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4

        for j in range(len(dets)):
            if i == j or dets[j][-1] < thresh:
                break

            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if all(x > 0 for x in (a, b, c, d)) or all(x < 0 for x in (a, b, c, d)):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break

        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)

    keep = [dets[index_keep[i]] for i in range(len(index_keep))]
    return np.array(keep)


def group_bbox_by_gbox(
    bboxes, gboxes, score_thred=0.3, v2c_dist_thred=2, c2v_dist_thred=0.5
):
    def point_in_box(box, point):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x3, y3, x4, y4 = box[4], box[5], box[6], box[7]
        ctx, cty = point[0], point[1]
        a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
        b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
        c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
        d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
        if all(x > 0 for x in (a, b, c, d)) or all(x < 0 for x in (a, b, c, d)):
            return True
        return False

    def get_distance(pt1, pt2):
        return math.sqrt(
            (pt1[0] - pt2[0]) * (pt1[0] - pt2[0])
            + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])
        )

    dets = copy.deepcopy(bboxes)
    sign = np.zeros((len(dets), 4))

    for gbox in gboxes:
        if gbox[10] < score_thred:
            break

        vertex = [gbox[0], gbox[1]]
        for i in range(4):
            center = [gbox[2 * i + 2], gbox[2 * i + 3]]
            if get_distance(vertex, center) < v2c_dist_thred:
                continue

            for k, bbox in enumerate(dets):
                if bbox[8] < score_thred:
                    break

                if sum(sign[k]) == 4:
                    continue

                w = (abs(bbox[6] - bbox[0]) + abs(bbox[4] - bbox[2])) / 2
                h = (abs(bbox[3] - bbox[1]) + abs(bbox[5] - bbox[7])) / 2
                m = max(w, h)
                if point_in_box(bbox, center):
                    min_dist, min_id = 1e4, -1
                    for j in range(4):
                        dist = get_distance(vertex, [bbox[2 * j], bbox[2 * j + 1]])
                        if dist < min_dist:
                            min_dist = dist
                            min_id = j

                    if (
                        min_id > -1
                        and min_dist < c2v_dist_thred * m
                        and sign[k][min_id] == 0
                    ):
                        bboxes[k][2 * min_id] = vertex[0]
                        bboxes[k][2 * min_id + 1] = vertex[1]
                        sign[k][min_id] = 1
    return bboxes


def get_table_line(binimg, axis=0, lineW=10):
    ##获取表格线
    ##axis=0 横线
    ##axis=1 竖线
    labels = measure.label(binimg > 0, connectivity=2)  # 8连通区域标记
    regions = measure.regionprops(labels)
    if axis == 1:
        lineboxes = [
            min_area_rect(line.coords)
            for line in regions
            if line.bbox[2] - line.bbox[0] > lineW
        ]
    else:
        lineboxes = [
            min_area_rect(line.coords)
            for line in regions
            if line.bbox[3] - line.bbox[1] > lineW
        ]
    return lineboxes


def min_area_rect(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = image_location_sort_box(box)

    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = calculate_center_rotate_angle(box)
    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2

    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2
    # degree,w,h,cx,cy = solve(box)
    # x1,y1,x2,y2,x3,y3,x4,y4 = box
    # return {'degree':degree,'w':w,'h':h,'cx':cx,'cy':cy}
    return [xmin, ymin, xmax, ymax]


def image_location_sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def calculate_center_rotate_angle(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标,能一定程度缓解图片的内部倾斜，但是还是依赖模型稳妥
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    ) / 2
    h = (
        np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
    ) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    ---------------------
    本项目中是为了排序后得到[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    作者：Tong_T
    来源：CSDN
    原文：https://blog.csdn.net/Tong_T/article/details/81907132
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def adjust_lines(lines, alph=50, angle=50):
    lines_n = len(lines)
    new_lines = []
    for i in range(lines_n):
        x1, y1, x2, y2 = lines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(lines_n):
            if i != j:
                x3, y3, x4, y4 = lines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (
                    x1 < cx2 < x2 or y1 < cy2 < y2
                ):  # 判断两个横线在y方向的投影重不重合
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    k = abs((y3 - y1) / (x3 - x1 + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if r < alph and a < angle:
                        new_lines.append((x1, y1, x3, y3))

                    r = sqrt((x1, y1), (x4, y4))
                    k = abs((y4 - y1) / (x4 - x1 + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if r < alph and a < angle:
                        new_lines.append((x1, y1, x4, y4))

                    r = sqrt((x2, y2), (x3, y3))
                    k = abs((y3 - y2) / (x3 - x2 + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if r < alph and a < angle:
                        new_lines.append((x2, y2, x3, y3))
                    r = sqrt((x2, y2), (x4, y4))
                    k = abs((y4 - y2) / (x4 - x2 + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if r < alph and a < angle:
                        new_lines.append((x2, y2, x4, y4))
    return new_lines


def final_adjust_lines(rowboxes, colboxes):
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], alpha=20, angle=30)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], alpha=20, angle=30)
    return rowboxes, colboxes


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
    boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]

    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        cv2.line(
            tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA
        )

    return tmp


def line_to_line(points1, points2, alpha=10, angle=30):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    xy = np.array([(x1, y1), (x2, y2)], dtype="float32")
    A1, B1, C1 = fit_line(xy)
    oxy = np.array([(ox1, oy1), (ox2, oy2)], dtype="float32")
    A2, B2, C2 = fit_line(oxy)
    flag1 = point_line_cor(np.array([x1, y1], dtype="float32"), A2, B2, C2)
    flag2 = point_line_cor(np.array([x2, y2], dtype="float32"), A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):  # 横线或者竖线在竖线或者横线的同一侧
        if (A1 * B2 - A2 * B1) != 0:
            x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
            y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
            # x, y = round(x, 2), round(y, 2)
            p = (x, y)  # 横线与竖线的交点
            r0 = sqrt(p, (x1, y1))
            r1 = sqrt(p, (x2, y2))

            if min(r0, r1) < alpha:  # 若交点与线起点或者终点的距离小于alpha，则延长线到交点
                if r0 < r1:
                    k = abs((y2 - p[1]) / (x2 - p[0] + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if a < angle or abs(90 - a) < angle:
                        points1 = np.array([p[0], p[1], x2, y2], dtype="float32")
                else:
                    k = abs((y1 - p[1]) / (x1 - p[0] + 1e-10))
                    a = math.atan(k) * 180 / math.pi
                    if a < angle or abs(90 - a) < angle:
                        points1 = np.array([x1, y1, p[0], p[1]], dtype="float32")
    return points1


def min_area_rect_box(
    regions, flag=True, W=0, H=0, filtersmall=False, adjust_box=False
):
    """
    多边形外接矩形
    """
    boxes = []
    for region in regions:
        if region.bbox_area > H * W * 3 / 4:  # 过滤大的单元格
            continue
        rect = cv2.minAreaRect(region.coords[:, ::-1])

        box = cv2.boxPoints(rect)
        box = box.reshape((8,)).tolist()
        box = image_location_sort_box(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        angle, w, h, cx, cy = calculate_center_rotate_angle(box)
        # if adjustBox:
        #     x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w + 5, h + 5, angle=0, degree=None)
        #     x1, x4 = max(x1, 0), max(x4, 0)
        #     y1, y2 = max(y1, 0), max(y2, 0)

        # if w > 32 and h > 32 and flag:
        #     if abs(angle / np.pi * 180) < 20:
        #         if filtersmall and (w < 10 or h < 10):
        #             continue
        #         boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        # else:
        if w * h < 0.5 * W * H:
            if filtersmall and (
                w < 15 or h < 15
            ):  # or w / h > 30 or h / w > 30): # 过滤小的单元格
                continue
            boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return boxes


def point_line_cor(p, A, B, C):
    ##判断点与线之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r


def fit_line(p):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p[0]
    x2, y2 = p[1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C
