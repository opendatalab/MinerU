# Copyright (c) Opendatalab. All rights reserved.

from PIL import Image
from collections import defaultdict
import inspect
from typing import List, Dict
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


# 旋转候选门控回到旧规则，先尽量召回疑似旋转表，再由 OCR rec 评分决定最终角度。
ROTATED_TEXT_ASPECT_RATIO_THRESHOLD = 0.8
ROTATED_TEXT_RATIO_THRESHOLD = 0.28
ROTATED_TEXT_MIN_BOXES = 3
# OCR rec 角度评分参数，控制抽样成本和 0 度优先的保守阈值。
ORIENTATION_SCORE_MAX_SAMPLE_BOXES = 18
ORIENTATION_SCORE_MIN_VALID_RESULTS = 5
ORIENTATION_ZERO_SCORE_PRIORITY_THRESHOLD = 0.9
ORIENTATION_SCORE_TIE_THRESHOLD = 0.08
ORIENTATION_SCORE_LABELS = ("0", "90", "270")


class MineruTableOrientationClsModel:
    def __init__(self, ocr_engine):
        self.ocr_engine = ocr_engine

    def predict(self, input_img):
        np_img = self._to_numpy_image(input_img)

        # 单张预测作为 batch_predict 的特例，保证门控、det 和 OCR 评分逻辑完全一致。
        return self.batch_predict([{"table_img": np_img}], det_batch_size=1)[0]

    @staticmethod
    def _to_numpy_image(input_img) -> np.ndarray:
        """统一将 Pillow/ndarray 输入转为 numpy 图像，保持外部入参校验一致。"""
        if isinstance(input_img, Image.Image):
            return np.asarray(input_img)
        if isinstance(input_img, np.ndarray):
            return input_img
        raise ValueError("Input must be a pillow object or a numpy array.")

    @classmethod
    def _to_bgr_table_image(cls, table_info: Dict) -> np.ndarray:
        """从表格信息中读取 table_img，并转换为 OCR detector 使用的 BGR 图像。"""
        table_img = cls._to_numpy_image(table_info["table_img"])
        return cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ceil_to_stride(value: int, stride: int) -> int:
        """将尺寸向上对齐到 stride 倍数，已经整除时保持原尺寸。"""
        if stride <= 0:
            raise ValueError("stride must be positive")
        if value <= 0:
            return 0
        return ((value + stride - 1) // stride) * stride

    @staticmethod
    def _box_width_height(box_ocr_res) -> tuple[float, float]:
        """从 OCR 四点框中提取宽高，统一处理 list/ndarray 两种输入。"""
        points = np.asarray(box_ocr_res, dtype=np.float32)
        p1 = points[0]
        p3 = points[2]
        return float(p3[0] - p1[0]), float(p3[1] - p1[1])

    @staticmethod
    def _count_rotated_text_boxes(det_boxes) -> int:
        """统计符合旧规则的高窄 OCR 框数量，作为疑似旋转表候选证据。"""
        vertical_count = 0
        for box_ocr_res in det_boxes:
            width, height = MineruTableOrientationClsModel._box_width_height(box_ocr_res)
            aspect_ratio = width / height if height > 0 else 1.0

            # 旧规则允许更宽的高窄框进入候选，最终是否旋转交给 OCR rec 评分。
            if aspect_ratio < ROTATED_TEXT_ASPECT_RATIO_THRESHOLD:
                vertical_count += 1
        return vertical_count

    @classmethod
    def _is_rotation_candidate_by_det_boxes(cls, det_boxes) -> bool:
        """用旧竖框规则判断是否进入 OCR 多角度评分。"""
        if det_boxes is None or len(det_boxes) == 0:
            return False

        vertical_count = cls._count_rotated_text_boxes(det_boxes)
        return (
            vertical_count >= len(det_boxes) * ROTATED_TEXT_RATIO_THRESHOLD
            and vertical_count >= ROTATED_TEXT_MIN_BOXES
        )

    @staticmethod
    def _rotate_image_by_label(img: np.ndarray, label: str) -> np.ndarray:
        """按候选角度旋转图像，0 度返回副本以避免后续误改原图。"""
        if label == "270":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if label == "90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img.copy()

    @staticmethod
    def _sample_det_boxes(det_boxes) -> list:
        """对 OCR det 框做均匀抽样，限制 rec 打分成本并覆盖整张表。"""
        if det_boxes is None or len(det_boxes) == 0:
            return []
        if len(det_boxes) <= ORIENTATION_SCORE_MAX_SAMPLE_BOXES:
            return list(det_boxes)

        indexes = np.linspace(
            0,
            len(det_boxes) - 1,
            ORIENTATION_SCORE_MAX_SAMPLE_BOXES,
        )
        sampled_indexes = sorted({int(round(index)) for index in indexes})
        return [det_boxes[index] for index in sampled_indexes]

    @staticmethod
    def _crop_image_without_text_rotation(img: np.ndarray, points) -> np.ndarray | None:
        """为方向评分按外接矩形切图，不做透视修正或文本方向自动转正。"""
        points = np.asarray(points, dtype=np.float32)
        if len(points) != 4:
            return None

        img_height, img_width = img.shape[:2]
        xmin = max(0, int(np.floor(np.min(points[:, 0]))))
        xmax = min(img_width, int(np.ceil(np.max(points[:, 0]))))
        ymin = max(0, int(np.floor(np.min(points[:, 1]))))
        ymax = min(img_height, int(np.ceil(np.max(points[:, 1]))))
        if xmax <= xmin or ymax <= ymin:
            return None
        return img[ymin:ymax, xmin:xmax].copy()

    def _build_orientation_score_task_from_det_boxes(
        self,
        label: str,
        img_bgr: np.ndarray,
        det_boxes,
    ) -> Dict:
        """根据已有 OCR det 框构造评分任务，复用 0 度门控结果并统一裁图逻辑。"""
        sampled_boxes = self._sample_det_boxes(det_boxes)

        img_crop_list = []
        for box in sampled_boxes:
            crop_img = self._crop_image_without_text_rotation(img_bgr, box)
            if crop_img is not None and crop_img.size > 0:
                img_crop_list.append(crop_img)

        return {
            "label": label,
            "crops": img_crop_list,
            "crop_count": len(img_crop_list),
            "crop_start": 0,
            "crop_end": len(img_crop_list),
        }

    def _build_orientation_score_task(self, label: str, img_bgr: np.ndarray) -> Dict:
        """为单个角度构造评分任务，只做 det、抽样和切图，不执行 rec。"""
        det_ocr_res = self.ocr_engine.ocr(img_bgr, rec=False)
        det_res = det_ocr_res[0] if det_ocr_res else None
        return self._build_orientation_score_task_from_det_boxes(
            label,
            img_bgr,
            det_res,
        )

    def _build_orientation_score_tasks(self, img_bgr: np.ndarray) -> List[Dict]:
        """为一张表构造 0/90/270 三个角度的评分任务。"""
        tasks = []
        for label in ORIENTATION_SCORE_LABELS:
            rotated_img = self._rotate_image_by_label(img_bgr, label)
            tasks.append(self._build_orientation_score_task(label, rotated_img))
        return tasks

    @staticmethod
    def _score_rec_results(rec_res) -> tuple[float, int, int]:
        """根据 OCR rec 结果计算平均置信度、有效文本数和字符数。"""
        valid_scores = []
        char_count = 0
        for rec_item in rec_res or []:
            if not rec_item or len(rec_item) < 2:
                continue
            text, score = rec_item
            text = str(text)
            if not text.strip():
                continue
            valid_scores.append(float(score))
            char_count += len(text)

        if len(valid_scores) < ORIENTATION_SCORE_MIN_VALID_RESULTS:
            return 0.0, len(valid_scores), char_count

        return float(np.mean(valid_scores)), len(valid_scores), char_count

    def _score_orientation_tasks_with_rec(self, tasks: List[Dict], rec_res) -> Dict[str, tuple[float, int, int]]:
        """按任务记录的 crop slice 回填 rec 结果，得到每个角度的评分。"""
        score_by_label = {}
        rec_res = rec_res or []
        for task in tasks:
            if "score" in task:
                score_by_label[task["label"]] = task["score"]
                continue
            crop_start = task.get("crop_start", 0)
            crop_end = task.get("crop_end", crop_start + task.get("crop_count", 0))
            task_rec_res = rec_res[crop_start:crop_end]
            score_by_label[task["label"]] = self._score_rec_results(task_rec_res)
        return score_by_label

    @staticmethod
    def _debug_log_orientation_rec_scores(
        table_index: int,
        score_by_label: Dict[str, tuple[float, int, int]],
    ) -> None:
        """输出单张表格各旋转候选的 OCR-rec 分数，便于排查误旋转。"""
        score_parts = []
        for label in ORIENTATION_SCORE_LABELS:
            score, valid_count, char_count = score_by_label.get(label, (0.0, 0, 0))
            score_parts.append(
                f"{label} score={float(score):.4f} "
                f"valid_count={valid_count} char_count={char_count}"
            )
        logger.debug(
            f"Table orientation rec scores table_index={table_index}: "
            f"{'; '.join(score_parts)}"
        )

    def _score_rotation_candidate_by_ocr(self, img_bgr: np.ndarray) -> tuple[float, int, int]:
        """对单个候选角度执行 OCR det+抽样 rec，返回平均置信度、有效文本数和字符数。"""
        task = self._build_orientation_score_task("", img_bgr)
        if task["crop_count"] == 0:
            return 0.0, 0, 0

        rec_ocr_res = self.ocr_engine.ocr(task["crops"], det=False, rec=True)
        rec_res = rec_ocr_res[0] if rec_ocr_res else None
        return self._score_rec_results(rec_res)

    @staticmethod
    def _select_rotation_label_by_scores(score_by_label: Dict[str, tuple[float, int, int]]) -> str:
        """按 OCR 评分选择最终角度，分差较小时优先保持 0 度。"""
        if not score_by_label:
            return "0"

        zero_score = score_by_label.get("0", (0.0, 0, 0))[0]
        if zero_score >= ORIENTATION_ZERO_SCORE_PRIORITY_THRESHOLD:
            return "0"

        best_label = max(
            ORIENTATION_SCORE_LABELS,
            key=lambda label: score_by_label.get(label, (0.0, 0, 0)),
        )
        best_score = score_by_label.get(best_label, (0.0, 0, 0))[0]
        if (
            best_label != "0"
            and best_score - zero_score < ORIENTATION_SCORE_TIE_THRESHOLD
        ):
            return "0"
        return best_label

    def _select_rotation_by_ocr_score(self, img_bgr: np.ndarray) -> str:
        """比较 0/90/270 三个角度的 OCR rec 分数，分差很小时优先保持 0 度。"""
        score_by_label = {}
        for label in ORIENTATION_SCORE_LABELS:
            rotated_img = self._rotate_image_by_label(img_bgr, label)
            score_by_label[label] = self._score_rotation_candidate_by_ocr(rotated_img)

        self._debug_log_orientation_rec_scores(-1, score_by_label)
        return self._select_rotation_label_by_scores(score_by_label)

    @staticmethod
    def _set_progress_description(progress_bar, desc: str):
        """切换复用进度条的阶段描述，并兼容测试替身和 tqdm 对象。"""
        if progress_bar is None:
            return
        if hasattr(progress_bar, "set_description"):
            progress_bar.set_description(desc)
        else:
            progress_bar.desc = desc

    @staticmethod
    def _extend_progress_total(progress_bar, count: int):
        """按新增工作量动态扩展总进度，保证最终 total 覆盖 det/score/rec。"""
        if progress_bar is None or count <= 0:
            return
        current_total = progress_bar.total if progress_bar.total is not None else progress_bar.n
        progress_bar.total = current_total + count
        if hasattr(progress_bar, "refresh"):
            progress_bar.refresh()

    @classmethod
    def _collect_portrait_image_groups(
        cls,
        imgs: List[Dict],
        resolution_group_stride: int,
    ) -> Dict[tuple[int, int], list[Dict]]:
        """兼容旧私有入口，实际收集逻辑已不再按表格宽高比过滤。"""
        return cls._collect_orientation_image_groups(
            imgs,
            resolution_group_stride,
        )

    @classmethod
    def _collect_orientation_image_groups(
        cls,
        imgs: List[Dict],
        resolution_group_stride: int,
    ) -> Dict[tuple[int, int], list[Dict]]:
        """按归一化分辨率收集所有有效表格图，旋转判断交给 OCR det/rec 评分。"""
        resolution_groups = defaultdict(list)
        for index, img in enumerate(imgs):
            bgr_img = cls._to_bgr_table_image(img)
            img_height, img_width = bgr_img.shape[:2]
            if img_height <= 0 or img_width <= 0:
                continue

            group_key = (
                cls._ceil_to_stride(img_height, resolution_group_stride),
                cls._ceil_to_stride(img_width, resolution_group_stride),
            )
            resolution_groups[group_key].append(
                {
                    "index": index,
                    "table_img_bgr": bgr_img,
                }
            )
        return resolution_groups

    @classmethod
    def _collect_orientation_images(cls, imgs: List[Dict]) -> list[Dict]:
        """扁平收集有效表格图，首轮 det 的分桶和 batch 交给 OCR detector 内部处理。"""
        orientation_imgs = []
        for index, img in enumerate(imgs):
            bgr_img = cls._to_bgr_table_image(img)
            img_height, img_width = bgr_img.shape[:2]
            if img_height <= 0 or img_width <= 0:
                continue

            orientation_imgs.append(
                {
                    "index": index,
                    "table_img_bgr": bgr_img,
                }
            )
        return orientation_imgs

    @classmethod
    def _pad_group_images(
        cls,
        group_imgs: list[Dict],
        resolution_group_stride: int,
    ) -> list[np.ndarray]:
        """将同组表格 padding 到统一尺寸，便于 OCR detector 批处理。"""
        max_h = max(img["table_img_bgr"].shape[0] for img in group_imgs)
        max_w = max(img["table_img_bgr"].shape[1] for img in group_imgs)
        target_h = cls._ceil_to_stride(max_h, resolution_group_stride)
        target_w = cls._ceil_to_stride(max_w, resolution_group_stride)

        batch_images = []
        for img in group_imgs:
            bgr_img = img["table_img_bgr"]
            h, w = bgr_img.shape[:2]
            padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            padded_img[:h, :w] = bgr_img
            batch_images.append(padded_img)
        return batch_images

    def _batch_detect_text_boxes(
        self,
        img_list: list[np.ndarray],
        det_batch_size: int,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-det Predict",
        progress_bar=None,
    ):
        """统一调用 OCR detector batch_predict，并兼容不支持进度参数的测试替身。"""
        if not img_list:
            return []

        max_batch_size = max(1, min(len(img_list), int(det_batch_size)))
        batch_predict = self.ocr_engine.text_detector.batch_predict

        progress_kwargs = {}
        try:
            signature = inspect.signature(batch_predict)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        if "tqdm_enable" in params:
            progress_kwargs["tqdm_enable"] = tqdm_enable
        if "tqdm_desc" in params:
            progress_kwargs["tqdm_desc"] = tqdm_desc
        if "tqdm_progress_bar" in params:
            progress_kwargs["tqdm_progress_bar"] = progress_bar

        batch_results = batch_predict(img_list, max_batch_size, **progress_kwargs)

        if progress_bar is not None and "tqdm_progress_bar" not in progress_kwargs:
            progress_bar.update(len(img_list))

        return batch_results

    def _detect_rotation_candidates(
        self,
        orientation_imgs: list[Dict],
        det_batch_size: int,
        tqdm_enable: bool = False,
        tqdm_desc: str = "Table orientation",
        progress_bar=None,
    ) -> list[Dict]:
        """对表格批量做 OCR det，并筛选需要进入多角度评分的候选。"""
        rotated_imgs = []
        batch_images = [img_info["table_img_bgr"] for img_info in orientation_imgs]
        batch_results = self._batch_detect_text_boxes(
            batch_images,
            det_batch_size,
            tqdm_enable=tqdm_enable and progress_bar is None,
            tqdm_desc=f"{tqdm_desc} det",
            progress_bar=progress_bar,
        )

        for img_info, (dt_boxes, _elapse) in zip(orientation_imgs, batch_results):
            if not self._is_rotation_candidate_by_det_boxes(dt_boxes):
                continue
            candidate_info = dict(img_info)
            candidate_info["gate_det_boxes"] = dt_boxes
            rotated_imgs.append(candidate_info)
        return rotated_imgs

    @staticmethod
    def _add_score_task_crops(task: Dict, all_crop_imgs: list[np.ndarray]) -> None:
        """将有效评分任务加入 OCR-rec 输入，不足阈值的任务直接记为 0 分。"""
        if task["crop_count"] < ORIENTATION_SCORE_MIN_VALID_RESULTS:
            task["score"] = (0.0, 0, 0)
            task["crop_start"] = len(all_crop_imgs)
            task["crop_end"] = len(all_crop_imgs)
            return

        crop_start = len(all_crop_imgs)
        all_crop_imgs.extend(task["crops"])
        crop_end = len(all_crop_imgs)
        task["crop_start"] = crop_start
        task["crop_end"] = crop_end

    def _build_score_tasks_for_candidates(
        self,
        rotated_imgs: list[Dict],
        det_batch_size: int,
        progress_bar=None,
    ) -> tuple[list[tuple[Dict, list[Dict]]], list[np.ndarray]]:
        """为所有旋转候选构造三角度评分任务，并汇总成一次 OCR rec 输入。"""
        img_score_tasks = []
        all_crop_imgs = []
        score_det_images = []
        score_det_tasks = []

        for img_info in rotated_imgs:
            table_img_bgr = img_info["table_img_bgr"]
            tasks = [
                self._build_orientation_score_task_from_det_boxes(
                    "0",
                    table_img_bgr,
                    img_info.get("gate_det_boxes"),
                )
            ]
            for label in ("90", "270"):
                rotated_img = self._rotate_image_by_label(table_img_bgr, label)
                task = {
                    "label": label,
                    "rotated_img_bgr": rotated_img,
                    "crops": [],
                    "crop_count": 0,
                    "crop_start": 0,
                    "crop_end": 0,
                }
                tasks.append(task)
                score_det_images.append(rotated_img)
                score_det_tasks.append(task)
            img_score_tasks.append((img_info, tasks))

        score_det_results = self._batch_detect_text_boxes(
            score_det_images,
            det_batch_size,
        )
        for task, (dt_boxes, _elapse) in zip(score_det_tasks, score_det_results):
            score_task = self._build_orientation_score_task_from_det_boxes(
                task["label"],
                task["rotated_img_bgr"],
                dt_boxes,
            )
            task.update(score_task)
            task.pop("rotated_img_bgr", None)

        for _img_info, tasks in img_score_tasks:
            for task in tasks:
                self._add_score_task_crops(task, all_crop_imgs)
            if progress_bar is not None:
                progress_bar.update(1)
        return img_score_tasks, all_crop_imgs

    def _recognize_orientation_crops(
        self,
        all_crop_imgs: list[np.ndarray],
        tqdm_enable: bool = False,
        tqdm_desc: str = "Table orientation",
        tqdm_progress_bar=None,
    ) -> list:
        """对所有候选角度 crop 合并执行 OCR rec，返回可按 slice 回填的结果。"""
        if not all_crop_imgs:
            return []

        rec_ocr_res = self.ocr_engine.ocr(
            all_crop_imgs,
            det=False,
            rec=True,
            tqdm_enable=tqdm_enable and tqdm_progress_bar is None,
            tqdm_desc=f"{tqdm_desc} rec",
            tqdm_progress_bar=tqdm_progress_bar,
        )
        return rec_ocr_res[0] if rec_ocr_res else []

    def _score_rotation_candidates(
        self,
        rotated_imgs: list[Dict],
        det_batch_size: int,
        tqdm_enable: bool = False,
        tqdm_desc: str = "Table orientation",
        progress_bar=None,
    ) -> Dict[int, str]:
        """批量评分旋转候选，并返回原始表格下标到最终角度标签的映射。"""
        if not rotated_imgs:
            return {}

        label_by_index = {}
        img_score_tasks, all_crop_imgs = self._build_score_tasks_for_candidates(
            rotated_imgs,
            det_batch_size,
            progress_bar=progress_bar,
        )
        self._extend_progress_total(progress_bar, len(all_crop_imgs))
        self._set_progress_description(progress_bar, f"{tqdm_desc} rec")
        if progress_bar is not None and hasattr(progress_bar, "refresh"):
            progress_bar.refresh()
        rec_res = self._recognize_orientation_crops(
            all_crop_imgs,
            tqdm_enable=tqdm_enable,
            tqdm_desc=tqdm_desc,
            tqdm_progress_bar=progress_bar,
        )

        for img_info, tasks in img_score_tasks:
            score_by_label = self._score_orientation_tasks_with_rec(tasks, rec_res)
            self._debug_log_orientation_rec_scores(img_info["index"], score_by_label)
            label_by_index[img_info["index"]] = self._select_rotation_label_by_scores(
                score_by_label
            )
        return label_by_index

    def batch_predict(
        self,
        imgs: List[Dict],
        det_batch_size: int,
        tqdm_enable: bool = False,
        tqdm_desc: str = "Table orientation",
    ) -> List[str]:
        """
        批量预测传入表格图片的旋转角度，只返回角度，不修改输入图片。
        """
        rotate_labels = ["0"] * len(imgs)
        orientation_imgs = self._collect_orientation_images(imgs)
        total_images = len(orientation_imgs)
        progress_bar = None
        if tqdm_enable:
            progress_bar = tqdm(
                total=total_images,
                desc=f"{tqdm_desc} det",
                leave=True,
            )
        try:
            rotated_imgs = self._detect_rotation_candidates(
                orientation_imgs,
                det_batch_size,
                tqdm_enable=tqdm_enable,
                tqdm_desc=tqdm_desc,
                progress_bar=progress_bar,
            )
            self._extend_progress_total(progress_bar, len(rotated_imgs))
            self._set_progress_description(progress_bar, f"{tqdm_desc} score")
            if progress_bar is not None and hasattr(progress_bar, "refresh"):
                progress_bar.refresh()
            label_by_index = self._score_rotation_candidates(
                rotated_imgs,
                det_batch_size,
                tqdm_enable=tqdm_enable,
                tqdm_desc=tqdm_desc,
                progress_bar=progress_bar,
            )
            for index, label in label_by_index.items():
                rotate_labels[index] = label
        finally:
            self._set_progress_description(progress_bar, tqdm_desc)
            if progress_bar is not None and hasattr(progress_bar, "refresh"):
                progress_bar.refresh()
            if progress_bar is not None:
                progress_bar.close()

        return rotate_labels
