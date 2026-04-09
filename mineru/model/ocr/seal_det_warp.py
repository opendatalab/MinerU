# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import cv2
import numpy as np
from loguru import logger
from numpy import arctan, cos, sin, sqrt


def Homography(
    image,
    img_points,
    world_width,
    world_height,
    interpolation=None,
    ratio_width=1.0,
    ratio_height=1.0,
):
    if interpolation is None:
        interpolation = cv2.INTER_CUBIC

    _points = np.array(img_points).reshape(-1, 2).astype(np.float32)

    expand_x = int(0.5 * world_width * (ratio_width - 1))
    expand_y = int(0.5 * world_height * (ratio_height - 1))

    pt_lefttop = [expand_x, expand_y]
    pt_righttop = [expand_x + world_width, expand_y]
    pt_leftbottom = [expand_x + world_width, expand_y + world_height]
    pt_rightbottom = [expand_x, expand_y + world_height]

    pts_std = np.float32([pt_lefttop, pt_righttop, pt_leftbottom, pt_rightbottom])

    img_crop_width = int(world_width * ratio_width)
    img_crop_height = int(world_height * ratio_height)

    M = cv2.getPerspectiveTransform(_points, pts_std)

    dst_img = cv2.warpPerspective(
        image,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_CONSTANT,
        flags=interpolation,
    )

    return dst_img


class PlanB:
    def __call__(
        self,
        image,
        points,
        curveTextRectifier,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
        loss_thresh=5.0,
        square=False,
    ):
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        h, w = image.shape[:2]
        _points = np.array(points).reshape(-1, 2).astype(np.float32)
        x_min = int(np.min(_points[:, 0]))
        y_min = int(np.min(_points[:, 1]))
        x_max = int(np.max(_points[:, 0]))
        y_max = int(np.max(_points[:, 1]))
        dx = x_max - x_min
        dy = y_max - y_min
        max_d = max(dx, dy)
        mean_pt = np.mean(_points, 0)

        expand_x = (ratio_width - 1.0) * 0.5 * max_d
        expand_y = (ratio_height - 1.0) * 0.5 * max_d

        if square:
            x_min = np.clip(int(mean_pt[0] - max_d - expand_x), 0, w - 1)
            y_min = np.clip(int(mean_pt[1] - max_d - expand_y), 0, h - 1)
            x_max = np.clip(int(mean_pt[0] + max_d + expand_x), 0, w - 1)
            y_max = np.clip(int(mean_pt[1] + max_d + expand_y), 0, h - 1)
        else:
            x_min = np.clip(int(x_min - expand_x), 0, w - 1)
            y_min = np.clip(int(y_min - expand_y), 0, h - 1)
            x_max = np.clip(int(x_max + expand_x), 0, w - 1)
            y_max = np.clip(int(y_max + expand_y), 0, h - 1)

        new_image = image[y_min:y_max, x_min:x_max, :].copy()
        new_points = _points.copy()
        new_points[:, 0] -= x_min
        new_points[:, 1] -= y_min

        dst_img, loss = curveTextRectifier(
            new_image,
            new_points,
            interpolation,
            ratio_width,
            ratio_height,
            mode="calibration",
        )

        return dst_img, loss


class CurveTextRectifier:
    def __init__(self):
        self.get_virtual_camera_parameter()

    def get_virtual_camera_parameter(self):
        vcam_thz = 0
        vcam_thx1 = 180
        vcam_thy = 180
        vcam_thx2 = 0

        vcam_x = 0
        vcam_y = 0
        vcam_z = 100

        radian = np.pi / 180

        angle_z = radian * vcam_thz
        angle_x1 = radian * vcam_thx1
        angle_y = radian * vcam_thy
        angle_x2 = radian * vcam_thx2

        optic_x = vcam_x
        optic_y = vcam_y
        optic_z = vcam_z

        fu = 100
        fv = 100

        matT = np.zeros((4, 4))
        matT[0, 0] = cos(angle_z) * cos(angle_y) - sin(angle_z) * sin(angle_x1) * sin(
            angle_y
        )
        matT[0, 1] = cos(angle_z) * sin(angle_y) * sin(angle_x2) - sin(angle_z) * (
            cos(angle_x1) * cos(angle_x2) - sin(angle_x1) * cos(angle_y) * sin(angle_x2)
        )
        matT[0, 2] = cos(angle_z) * sin(angle_y) * cos(angle_x2) + sin(angle_z) * (
            cos(angle_x1) * sin(angle_x2) + sin(angle_x1) * cos(angle_y) * cos(angle_x2)
        )
        matT[0, 3] = optic_x
        matT[1, 0] = sin(angle_z) * cos(angle_y) + cos(angle_z) * sin(angle_x1) * sin(
            angle_y
        )
        matT[1, 1] = sin(angle_z) * sin(angle_y) * sin(angle_x2) + cos(angle_z) * (
            cos(angle_x1) * cos(angle_x2) - sin(angle_x1) * cos(angle_y) * sin(angle_x2)
        )
        matT[1, 2] = sin(angle_z) * sin(angle_y) * cos(angle_x2) - cos(angle_z) * (
            cos(angle_x1) * sin(angle_x2) + sin(angle_x1) * cos(angle_y) * cos(angle_x2)
        )
        matT[1, 3] = optic_y
        matT[2, 0] = -cos(angle_x1) * sin(angle_y)
        matT[2, 1] = cos(angle_x1) * cos(angle_y) * sin(angle_x2) + sin(angle_x1) * cos(
            angle_x2
        )
        matT[2, 2] = cos(angle_x1) * cos(angle_y) * cos(angle_x2) - sin(angle_x1) * sin(
            angle_x2
        )
        matT[2, 3] = optic_z
        matT[3, 0] = 0
        matT[3, 1] = 0
        matT[3, 2] = 0
        matT[3, 3] = 1

        matS = np.zeros((4, 4))
        matS[2, 3] = 0.5
        matS[3, 2] = 0.5

        self.ifu = 1 / fu
        self.ifv = 1 / fv

        self.matT = matT
        self.matS = matS
        self.K = np.dot(matT.T, matS)
        self.K = np.dot(self.K, matT)

    def vertical_text_process(self, points, org_size):
        org_w, org_h = org_size
        _points = np.array(points).reshape(-1).tolist()
        _points = np.array(_points[2:] + _points[:2]).reshape(-1, 2)

        adjusted_points = np.zeros(_points.shape, dtype=np.float32)
        adjusted_points[:, 0] = _points[:, 1]
        adjusted_points[:, 1] = org_h - _points[:, 0] - 1

        _image_coord, _world_coord, _new_image_size = self.horizontal_text_process(
            adjusted_points
        )

        image_coord = _points.reshape(1, -1, 2)
        world_coord = np.zeros(_world_coord.shape, dtype=np.float32)
        world_coord[:, :, 0] = 0 - _world_coord[:, :, 1]
        world_coord[:, :, 1] = _world_coord[:, :, 0]
        world_coord[:, :, 2] = _world_coord[:, :, 2]
        new_image_size = (_new_image_size[1], _new_image_size[0])

        return image_coord, world_coord, new_image_size

    def horizontal_text_process(self, points):
        poly = np.array(points).reshape(-1)

        dx_list = []
        dy_list = []
        for i in range(1, len(poly) // 2):
            xdx = poly[i * 2] - poly[(i - 1) * 2]
            xdy = poly[i * 2 + 1] - poly[(i - 1) * 2 + 1]
            d = sqrt(xdx**2 + xdy**2)
            dx_list.append(d)

        for i in range(0, len(poly) // 4):
            ydx = poly[i * 2] - poly[len(poly) - 1 - (i * 2 + 1)]
            ydy = poly[i * 2 + 1] - poly[len(poly) - 1 - (i * 2)]
            d = sqrt(ydx**2 + ydy**2)
            dy_list.append(d)

        dx_list = [
            (dx_list[i] + dx_list[len(dx_list) - 1 - i]) / 2
            for i in range(len(dx_list) // 2)
        ]

        height = np.around(np.mean(dy_list))

        rect_coord = [0, 0]
        for i in range(0, len(poly) // 4 - 1):
            x = rect_coord[-2]
            x += dx_list[i]
            y = 0
            rect_coord.append(x)
            rect_coord.append(y)

        rect_coord_half = copy.deepcopy(rect_coord)
        for i in range(0, len(poly) // 4):
            x = rect_coord_half[len(rect_coord_half) - 2 * i - 2]
            y = height
            rect_coord.append(x)
            rect_coord.append(y)

        np_rect_coord = np.array(rect_coord).reshape(-1, 2)
        x_min = np.min(np_rect_coord[:, 0])
        y_min = np.min(np_rect_coord[:, 1])
        x_max = np.max(np_rect_coord[:, 0])
        y_max = np.max(np_rect_coord[:, 1])
        new_image_size = (int(x_max - x_min + 0.5), int(y_max - y_min + 0.5))
        x_mean = (x_max - x_min) / 2
        y_mean = (y_max - y_min) / 2
        np_rect_coord[:, 0] -= x_mean
        np_rect_coord[:, 1] -= y_mean
        rect_coord = np_rect_coord.reshape(-1).tolist()

        rect_coord = np.array(rect_coord).reshape(-1, 2)
        world_coord = np.ones((len(rect_coord), 3)) * 0

        world_coord[:, :2] = rect_coord

        image_coord = np.array(poly).reshape(1, -1, 2)
        world_coord = world_coord.reshape(1, -1, 3)

        return image_coord, world_coord, new_image_size

    def horizontal_text_estimate(self, points):
        pts = np.array(points).reshape(-1, 2)
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        x = x_max - x_min
        y = y_max - y_min
        is_horizontal_text = True
        if y / x > 1.5:
            is_horizontal_text = False
        return is_horizontal_text

    def virtual_camera_to_world(self, size):
        ifu, ifv = self.ifu, self.ifv
        K, matT = self.K, self.matT

        ppu = size[0] / 2 + 1e-6
        ppv = size[1] / 2 + 1e-6

        P = np.zeros((size[1], size[0], 3))

        lu = np.array([i for i in range(size[0])])
        lv = np.array([i for i in range(size[1])])
        u, v = np.meshgrid(lu, lv)

        yp = (v - ppv) * ifv
        xp = (u - ppu) * ifu
        angle_a = arctan(sqrt(xp * xp + yp * yp))
        angle_b = arctan(yp / xp)

        D0 = sin(angle_a) * cos(angle_b)
        D1 = sin(angle_a) * sin(angle_b)
        D2 = cos(angle_a)

        D0[xp <= 0] = -D0[xp <= 0]
        D1[xp <= 0] = -D1[xp <= 0]

        ratio_a = (
            K[0, 0] * D0 * D0
            + K[1, 1] * D1 * D1
            + K[2, 2] * D2 * D2
            + (K[0, 1] + K[1, 0]) * D0 * D1
            + (K[0, 2] + K[2, 0]) * D0 * D2
            + (K[1, 2] + K[2, 1]) * D1 * D2
        )
        ratio_b = (
            (K[0, 3] + K[3, 0]) * D0
            + (K[1, 3] + K[3, 1]) * D1
            + (K[2, 3] + K[3, 2]) * D2
        )
        ratio_c = K[3, 3] * np.ones(ratio_b.shape)

        delta = ratio_b * ratio_b - 4 * ratio_a * ratio_c
        t = np.zeros(delta.shape)
        t[ratio_a == 0] = -ratio_c[ratio_a == 0] / ratio_b[ratio_a == 0]
        t[ratio_a != 0] = (-ratio_b[ratio_a != 0] + sqrt(delta[ratio_a != 0])) / (
            2 * ratio_a[ratio_a != 0]
        )
        t[delta < 0] = 0

        P[:, :, 0] = matT[0, 3] + t * (
            matT[0, 0] * D0 + matT[0, 1] * D1 + matT[0, 2] * D2
        )
        P[:, :, 1] = matT[1, 3] + t * (
            matT[1, 0] * D0 + matT[1, 1] * D1 + matT[1, 2] * D2
        )
        P[:, :, 2] = matT[2, 3] + t * (
            matT[2, 0] * D0 + matT[2, 1] * D1 + matT[2, 2] * D2
        )

        return P

    def world_to_image(self, image_size, world, intrinsic, distCoeffs, rotation, tvec):
        r11 = rotation[0, 0]
        r12 = rotation[0, 1]
        r13 = rotation[0, 2]
        r21 = rotation[1, 0]
        r22 = rotation[1, 1]
        r23 = rotation[1, 2]
        r31 = rotation[2, 0]
        r32 = rotation[2, 1]
        r33 = rotation[2, 2]

        t1 = tvec[0]
        t2 = tvec[1]
        t3 = tvec[2]

        k1 = distCoeffs[0]
        k2 = distCoeffs[1]
        p1 = distCoeffs[2]
        p2 = distCoeffs[3]
        k3 = distCoeffs[4]
        k4 = distCoeffs[5]
        k5 = distCoeffs[6]
        k6 = distCoeffs[7]

        if len(distCoeffs) > 8:
            s1 = distCoeffs[8]
            s2 = distCoeffs[9]
            s3 = distCoeffs[10]
            s4 = distCoeffs[11]
        else:
            s1 = s2 = s3 = s4 = 0

        if len(distCoeffs) > 12:
            tx = distCoeffs[12]
            ty = distCoeffs[13]
        else:
            tx = ty = 0

        fu = intrinsic[0, 0]
        fv = intrinsic[1, 1]
        ppu = intrinsic[0, 2]
        ppv = intrinsic[1, 2]

        cos_tx = cos(tx)
        cos_ty = cos(ty)
        sin_tx = sin(tx)
        sin_ty = sin(ty)

        tao11 = cos_ty * cos_tx * cos_ty + sin_ty * cos_tx * sin_ty
        tao12 = cos_ty * cos_tx * sin_ty * sin_tx - sin_ty * cos_tx * cos_ty * sin_tx
        tao13 = -cos_ty * cos_tx * sin_ty * cos_tx + sin_ty * cos_tx * cos_ty * cos_tx
        tao21 = -sin_tx * sin_ty
        tao22 = cos_ty * cos_tx * cos_tx + sin_tx * cos_ty * sin_tx
        tao23 = cos_ty * cos_tx * sin_tx - sin_tx * cos_ty * cos_tx

        P = np.zeros((image_size[1], image_size[0], 2))

        c3 = r31 * world[:, :, 0] + r32 * world[:, :, 1] + r33 * world[:, :, 2] + t3
        c1 = r11 * world[:, :, 0] + r12 * world[:, :, 1] + r13 * world[:, :, 2] + t1
        c2 = r21 * world[:, :, 0] + r22 * world[:, :, 1] + r23 * world[:, :, 2] + t2

        x1 = c1 / c3
        y1 = c2 / c3
        x12 = x1 * x1
        y12 = y1 * y1
        x1y1 = 2 * x1 * y1
        r2 = x12 + y12
        r4 = r2 * r2
        r6 = r2 * r4

        radial_distortion = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
            1 + k4 * r2 + k5 * r4 + k6 * r6
        )
        x2 = (
            x1 * radial_distortion + p1 * x1y1 + p2 * (r2 + 2 * x12) + s1 * r2 + s2 * r4
        )
        y2 = (
            y1 * radial_distortion + p2 * x1y1 + p1 * (r2 + 2 * y12) + s3 * r2 + s4 * r4
        )

        x3 = tao11 * x2 + tao12 * y2 + tao13
        y3 = tao21 * x2 + tao22 * y2 + tao23

        P[:, :, 0] = fu * x3 + ppu
        P[:, :, 1] = fv * y3 + ppv
        P[c3 <= 0] = 0

        return P

    def spatial_transform(
        self, image_data, new_image_size, mtx, dist, rvecs, tvecs, interpolation
    ):
        rotation, _ = cv2.Rodrigues(rvecs)
        world_map = self.virtual_camera_to_world(new_image_size)
        image_map = self.world_to_image(
            new_image_size, world_map, mtx, dist, rotation, tvecs
        )
        image_map = image_map.astype(np.float32)
        dst = cv2.remap(
            image_data, image_map[:, :, 0], image_map[:, :, 1], interpolation
        )
        return dst

    def calibrate(self, org_size, image_coord, world_coord):
        flag = cv2.CALIB_RATIONAL_MODEL
        flag2 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_TILTED_MODEL
        flag3 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        flag4 = (
            cv2.CALIB_RATIONAL_MODEL
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_ASPECT_RATIO
        )
        flag5 = (
            cv2.CALIB_RATIONAL_MODEL
            | cv2.CALIB_TILTED_MODEL
            | cv2.CALIB_ZERO_TANGENT_DIST
        )
        flag6 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_ASPECT_RATIO
        flag_list = [flag2, flag3, flag4, flag5, flag6]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            world_coord.astype(np.float32),
            image_coord.astype(np.float32),
            org_size,
            None,
            None,
            flags=flag,
        )
        if ret > 2:
            min_ret = ret
            for flag in flag_list:
                _ret, _mtx, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
                    world_coord.astype(np.float32),
                    image_coord.astype(np.float32),
                    org_size,
                    None,
                    None,
                    flags=flag,
                )
                if _ret < min_ret:
                    min_ret = _ret
                    ret, mtx, dist, rvecs, tvecs = _ret, _mtx, _dist, _rvecs, _tvecs

        return ret, mtx, dist, rvecs, tvecs

    def dc_homo(
        self,
        img,
        img_points,
        obj_points,
        is_horizontal_text,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
    ):
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR

        _img_points = img_points.reshape(-1, 2)
        _obj_points = obj_points.reshape(-1, 3)

        homo_img_list = []
        width_list = []
        height_list = []
        for i in range(len(_img_points) // 2 - 1):
            new_img_points = np.zeros((4, 2)).astype(np.float32)
            new_obj_points = np.zeros((4, 2)).astype(np.float32)

            new_img_points[0:2, :] = _img_points[i : (i + 2), :2]
            new_img_points[2:4, :] = _img_points[::-1, :][i : (i + 2), :2][::-1, :]

            new_obj_points[0:2, :] = _obj_points[i : (i + 2), :2]
            new_obj_points[2:4, :] = _obj_points[::-1, :][i : (i + 2), :2][::-1, :]

            if is_horizontal_text:
                world_width = np.abs(new_obj_points[1, 0] - new_obj_points[0, 0])
                world_height = np.abs(new_obj_points[3, 1] - new_obj_points[0, 1])
            else:
                world_width = np.abs(new_obj_points[1, 1] - new_obj_points[0, 1])
                world_height = np.abs(new_obj_points[3, 0] - new_obj_points[0, 0])

            homo_img = Homography(
                img,
                new_img_points,
                world_width,
                world_height,
                interpolation=interpolation,
                ratio_width=ratio_width,
                ratio_height=ratio_height,
            )

            homo_img_list.append(homo_img)
            _h, _w = homo_img.shape[:2]
            width_list.append(_w)
            height_list.append(_h)

        rectified_image = np.zeros((np.max(height_list), sum(width_list), 3)).astype(
            np.uint8
        )

        st = 0
        for homo_img, w, h in zip(homo_img_list, width_list, height_list):
            rectified_image[:h, st : st + w, :] = homo_img
            st += w

        if not is_horizontal_text:
            rectified_image = np.rot90(rectified_image, 3)

        return rectified_image

    def __call__(
        self,
        image_data,
        points,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
        mode="calibration",
    ):
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR

        org_h, org_w = image_data.shape[:2]
        org_size = (org_w, org_h)
        self.image = image_data

        is_horizontal_text = self.horizontal_text_estimate(points)
        if is_horizontal_text:
            image_coord, world_coord, new_image_size = self.horizontal_text_process(
                points
            )
        else:
            image_coord, world_coord, new_image_size = self.vertical_text_process(
                points, org_size
            )

        if mode.lower() == "calibration":
            ret, mtx, dist, rvecs, tvecs = self.calibrate(
                org_size, image_coord, world_coord
            )

            st_size = (
                int(new_image_size[0] * ratio_width),
                int(new_image_size[1] * ratio_height),
            )
            dst = self.spatial_transform(
                image_data, st_size, mtx, dist[0], rvecs[0], tvecs[0], interpolation
            )
        elif mode.lower() == "homography":
            ret = 0.01
            dst = self.dc_homo(
                image_data,
                image_coord,
                world_coord,
                is_horizontal_text,
                interpolation=interpolation,
                ratio_width=1.0,
                ratio_height=1.0,
            )
        else:
            raise ValueError(
                'mode must be ["calibration", "homography"], but got {}'.format(mode)
            )

        return dst, ret


class AutoRectifier:
    def __init__(self):
        self.npoints = 10
        self.curveTextRectifier = CurveTextRectifier()

    @staticmethod
    def get_rotate_crop_image(
        img, points, interpolation=None, ratio_width=1.0, ratio_height=1.0
    ):
        if interpolation is None:
            interpolation = cv2.INTER_CUBIC
        h, w = img.shape[:2]
        _points = np.array(points).reshape(-1, 2).astype(np.float32)

        if len(_points) != 4:
            x_min = int(np.min(_points[:, 0]))
            y_min = int(np.min(_points[:, 1]))
            x_max = int(np.max(_points[:, 0]))
            y_max = int(np.max(_points[:, 1]))
            dx = x_max - x_min
            dy = y_max - y_min
            expand_x = int(0.5 * dx * (ratio_width - 1))
            expand_y = int(0.5 * dy * (ratio_height - 1))
            x_min = np.clip(int(x_min - expand_x), 0, w - 1)
            y_min = np.clip(int(y_min - expand_y), 0, h - 1)
            x_max = np.clip(int(x_max + expand_x), 0, w - 1)
            y_max = np.clip(int(y_max + expand_y), 0, h - 1)

            dst_img = img[y_min:y_max, x_min:x_max, :].copy()
        else:
            img_crop_width = int(
                max(
                    np.linalg.norm(_points[0] - _points[1]),
                    np.linalg.norm(_points[2] - _points[3]),
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(_points[0] - _points[3]),
                    np.linalg.norm(_points[1] - _points[2]),
                )
            )

            dst_img = Homography(
                img,
                _points,
                img_crop_width,
                img_crop_height,
                interpolation,
                ratio_width,
                ratio_height,
            )

        return dst_img

    def visualize(self, image_data, points_list):
        visualization = image_data.copy()

        for box in points_list:
            box = np.array(box).reshape(-1, 2).astype(np.int32)
            cv2.drawContours(
                visualization, [np.array(box).reshape((-1, 1, 2))], -1, (0, 0, 255), 2
            )
            for i, p in enumerate(box):
                if i != 0:
                    cv2.circle(
                        visualization,
                        tuple(p),
                        radius=1,
                        color=(255, 0, 0),
                        thickness=2,
                    )
                else:
                    cv2.circle(
                        visualization,
                        tuple(p),
                        radius=1,
                        color=(255, 255, 0),
                        thickness=2,
                    )
        return visualization

    def __call__(
        self,
        image_data,
        points,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
        loss_thresh=5.0,
        mode="calibration",
    ):
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        _points = np.array(points).reshape(-1, 2)
        if len(_points) >= self.npoints and len(_points) % 2 == 0:
            try:
                curveTextRectifier = CurveTextRectifier()

                dst_img, loss = curveTextRectifier(
                    image_data, points, interpolation, ratio_width, ratio_height, mode
                )
                if loss >= 2:
                    img_list, loss_list = [dst_img], [loss]
                    _dst_img, _loss = PlanB()(
                        image_data,
                        points,
                        curveTextRectifier,
                        interpolation,
                        ratio_width,
                        ratio_height,
                        loss_thresh=loss_thresh,
                        square=True,
                    )
                    img_list += [_dst_img]
                    loss_list += [_loss]

                    _dst_img, _loss = PlanB()(
                        image_data,
                        points,
                        curveTextRectifier,
                        interpolation,
                        ratio_width,
                        ratio_height,
                        loss_thresh=loss_thresh,
                        square=False,
                    )
                    img_list += [_dst_img]
                    loss_list += [_loss]

                    min_loss = min(loss_list)
                    dst_img = img_list[loss_list.index(min_loss)]

                    if min_loss >= loss_thresh:
                        logger.warning(
                            "calibration loss: {} is too large for spatial transformer. It is failed. Using get_rotate_crop_image".format(
                                loss
                            )
                        )
                        dst_img = self.get_rotate_crop_image(
                            image_data, points, interpolation, ratio_width, ratio_height
                        )
            except Exception as e:
                logger.warning(f"Exception caught: {e}")
                dst_img = self.get_rotate_crop_image(
                    image_data, points, interpolation, ratio_width, ratio_height
                )
        else:
            dst_img = self.get_rotate_crop_image(
                image_data, _points, interpolation, ratio_width, ratio_height
            )

        return dst_img

    def run(
        self,
        image_data,
        points_list,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
        loss_thresh=5.0,
        mode="calibration",
    ):
        if image_data is None:
            raise ValueError
        if not isinstance(points_list, list):
            raise ValueError
        for points in points_list:
            if not isinstance(points, list):
                raise ValueError
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR

        if ratio_width < 1.0 or ratio_height < 1.0:
            raise ValueError(
                "ratio_width and ratio_height cannot be smaller than 1, but got {}",
                (ratio_width, ratio_height),
            )

        if mode.lower() != "calibration" and mode.lower() != "homography":
            raise ValueError(
                'mode must be ["calibration", "homography"], but got {}'.format(mode)
            )

        if mode.lower() == "homography" and ratio_width != 1.0 and ratio_height != 1.0:
            raise ValueError(
                "ratio_width and ratio_height must be 1.0 when mode is homography, but got mode:{}, ratio:({},{})".format(
                    mode, ratio_width, ratio_height
                )
            )

        res = []
        for points in points_list:
            rectified_img = self(
                image_data,
                points,
                interpolation,
                ratio_width,
                ratio_height,
                loss_thresh=loss_thresh,
                mode=mode,
            )
            res.append(rectified_img)

        visualized_image = self.visualize(image_data, points_list)

        return res, visualized_image
