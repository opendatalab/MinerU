import base64
import html
import os

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...model.table.rec.unet_table.utils_table_recover import (
    gather_ocr_list_by_row as gather_table_items_by_row,
    plot_html_table,
    sorted_ocr_boxes as sorted_table_boxes,
)
from ...utils.config_reader import (
    get_formula_enable,
    get_ocr_det_mask_inline_formula_enable,
    get_table_enable,
)
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_rotate_crop_image
from ...utils.pdf_image_tools import get_crop_np_img

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16
TABLE_ORI_CLS_BATCH_SIZE = 16
TABLE_Wired_Wireless_CLS_BATCH_SIZE = 16
TABLE_RICH_IMAGE_LABELS = {"image", "chart", "seal"}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TABLE_OCR_DET_DEBUG_DIR = os.path.join(PROJECT_ROOT, "output_images", "table_ocr_det_debug")


class BatchAnalyze:
    def __init__(
        self,
        model_manager,
        batch_ratio: int,
        formula_enable,
        table_enable,
        enable_ocr_det_batch: bool = True,
        mask_inline_formula_for_ocr_det: bool = True,
    ):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.model_manager = model_manager
        self.enable_ocr_det_batch = enable_ocr_det_batch
        self.mask_inline_formula_for_ocr_det = (
            get_ocr_det_mask_inline_formula_enable(mask_inline_formula_for_ocr_det)
        )

    @staticmethod
    def _normalize_bbox(box) -> list[int] | None:
        return normalize_to_int_bbox(box)

    @staticmethod
    def _bbox_center(bbox) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _point_in_bbox(point, bbox, tolerance: float = 1.0) -> bool:
        return (
            bbox[0] - tolerance <= point[0] <= bbox[2] + tolerance
            and bbox[1] - tolerance <= point[1] <= bbox[3] + tolerance
        )

    @staticmethod
    def _rotate_local_bbox(local_bbox: list[float], table_width: float, table_height: float, rotate_label: str) -> list[int]:
        if rotate_label not in {"90", "270"}:
            return normalize_to_int_bbox(local_bbox)

        corners = np.asarray(
            [
                [local_bbox[0], local_bbox[1]],
                [local_bbox[2], local_bbox[1]],
                [local_bbox[2], local_bbox[3]],
                [local_bbox[0], local_bbox[3]],
            ],
            dtype=np.float32,
        )
        if rotate_label == "270":
            rotated = np.asarray(
                [[table_height - y, x] for x, y in corners],
                dtype=np.float32,
            )
        else:
            rotated = np.asarray(
                [[y, table_width - x] for x, y in corners],
                dtype=np.float32,
            )
        return normalize_to_int_bbox(rotated)

    def _page_bbox_to_table_bbox(self, item_bbox: list[float], table_bbox: list[float], rotate_label: str) -> list[float]:
        table_width = float(table_bbox[2] - table_bbox[0])
        table_height = float(table_bbox[3] - table_bbox[1])
        local_bbox = [
            float(item_bbox[0] - table_bbox[0]),
            float(item_bbox[1] - table_bbox[1]),
            float(item_bbox[2] - table_bbox[0]),
            float(item_bbox[3] - table_bbox[1]),
        ]
        return self._rotate_local_bbox(local_bbox, table_width, table_height, rotate_label)

    @staticmethod
    def _crop_image_to_data_uri(page_np_img: np.ndarray, bbox: list[float]) -> str:
        h, w = page_np_img.shape[:2]
        int_bbox = normalize_to_int_bbox(bbox, image_size=(h, w))
        if int_bbox is None:
            return ""
        x0, y0, x1, y1 = int_bbox

        crop = page_np_img[y0:y1, x0:x1]
        if crop.size == 0:
            return ""

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return ""
        img_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f'<img src="data:image/jpg;base64,{img_base64}">'

    @staticmethod
    def _apply_mask_boxes_to_image(
        bgr_image: np.ndarray,
        mask_boxes: list[dict] | None,
    ) -> np.ndarray:
        if not mask_boxes:
            return bgr_image

        masked_image = bgr_image.copy()
        image_h, image_w = masked_image.shape[:2]
        for mask_box in mask_boxes:
            bbox = mask_box.get("bbox")
            if bbox is None:
                continue

            int_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
            if int_bbox is None:
                continue

            x0, y0, x1, y1 = int_bbox
            masked_image[y0:y1, x0:x1] = 255

        return masked_image

    def _get_masked_det_image(
        self,
        bgr_image: np.ndarray,
        mask_boxes: list[dict] | None,
    ) -> np.ndarray:
        if not self.mask_inline_formula_for_ocr_det:
            return bgr_image
        return self._apply_mask_boxes_to_image(bgr_image, mask_boxes)

    @staticmethod
    def _dump_table_det_debug_images(
        table_id: int,
        original_bgr_image: np.ndarray,
        masked_bgr_image: np.ndarray,
        dt_boxes_final: list,
    ) -> None:
        os.makedirs(TABLE_OCR_DET_DEBUG_DIR, exist_ok=True)

        original_path = os.path.join(
            TABLE_OCR_DET_DEBUG_DIR,
            f"table_{table_id:04d}_orig.png",
        )
        masked_path = os.path.join(
            TABLE_OCR_DET_DEBUG_DIR,
            f"table_{table_id:04d}_masked.png",
        )
        det_boxes_path = os.path.join(
            TABLE_OCR_DET_DEBUG_DIR,
            f"table_{table_id:04d}_det_boxes.png",
        )

        cv2.imwrite(original_path, original_bgr_image)
        cv2.imwrite(masked_path, masked_bgr_image)

        det_boxes_image = original_bgr_image.copy()
        for dt_box in dt_boxes_final or []:
            points = np.asarray(dt_box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(det_boxes_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(det_boxes_path, det_boxes_image)

    def _collect_table_rich_items(self, table_res_dict: dict) -> list[dict]:
        rich_items = []
        table_bbox = table_res_dict["table_res"]["bbox"]
        rotate_label = table_res_dict.get("rotate_label", "0")

        for layout_item in table_res_dict["page_layout_res"]:
            label = layout_item.get("label")
            if label not in {"display_formula", "inline_formula", *TABLE_RICH_IMAGE_LABELS}:
                continue

            item_bbox = layout_item.get("bbox")
            if item_bbox is None:
                continue

            center = self._bbox_center(item_bbox)
            if not self._point_in_bbox(center, table_bbox, tolerance=0.5):
                continue

            local_bbox = self._page_bbox_to_table_bbox(item_bbox, table_bbox, rotate_label)
            if label in {"display_formula", "inline_formula"}:
                latex = layout_item.get("latex", "").strip()
                if not latex:
                    continue
                rich_items.append(
                    {
                        "type": "formula",
                        "bbox": local_bbox,
                        "content": f"<eq>{latex}</eq>",
                    }
                )
                continue

            image_tag = self._crop_image_to_data_uri(table_res_dict["page_np_img"], item_bbox)
            if not image_tag:
                continue
            rich_items.append(
                {
                    "type": "image",
                    "bbox": local_bbox,
                    "content": image_tag,
                }
            )

        return rich_items

    def _should_remove_table_internal_layout_item(
        self,
        layout_item: dict,
        table_bbox: list[float],
    ) -> bool:
        label = layout_item.get("label")
        if label not in {"display_formula", "inline_formula", *TABLE_RICH_IMAGE_LABELS}:
            return False

        item_bbox = layout_item.get("bbox")
        if item_bbox is None:
            return False

        center = self._bbox_center(item_bbox)
        return self._point_in_bbox(center, table_bbox, tolerance=0.5)

    def _remove_table_internal_layout_items(self, table_res_list_all_page: list[dict]) -> None:
        for table_res_dict in table_res_list_all_page:
            table_bbox = table_res_dict["table_res"].get("bbox")
            if table_bbox is None:
                continue

            page_layout_res = table_res_dict.get("page_layout_res")
            if not page_layout_res:
                continue

            page_layout_res[:] = [
                layout_item
                for layout_item in page_layout_res
                if not self._should_remove_table_internal_layout_item(
                    layout_item,
                    table_bbox,
                )
            ]

    @staticmethod
    def _match_cell_index(item_bbox: list[float], cell_bboxes: list[list[float]]) -> int | None:
        if not cell_bboxes:
            return None

        center = BatchAnalyze._bbox_center(item_bbox)
        for cell_index, cell_bbox in enumerate(cell_bboxes):
            if BatchAnalyze._point_in_bbox(center, cell_bbox, tolerance=1.5):
                return cell_index

        distances = []
        for cell_index, cell_bbox in enumerate(cell_bboxes):
            cell_center = BatchAnalyze._bbox_center(cell_bbox)
            distances.append(
                (
                    (center[0] - cell_center[0]) ** 2 + (center[1] - cell_center[1]) ** 2,
                    cell_index,
                )
            )
        if not distances:
            return None
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    def _merge_cell_items(self, cell_items: list[dict]) -> str:
        if not cell_items:
            return ""

        item_boxes = [item["bbox"] for item in cell_items]
        _, sorted_idx = sorted_table_boxes(item_boxes, threhold=0.3)
        ordered_items = [
            [[*cell_items[index]["bbox"]], cell_items[index]["content"]]
            for index in sorted_idx
        ]
        merged_rows = gather_table_items_by_row(ordered_items, threhold=0.3)
        return "<br>".join(row[1] for row in merged_rows if row[1])

    def _rebuild_table_html(self, cell_bboxes, logic_points, content_items: list[dict]) -> str | None:
        if cell_bboxes is None or logic_points is None:
            return None

        normalized_cells = []
        for cell_bbox in cell_bboxes:
            normalized_bbox = self._normalize_bbox(cell_bbox)
            if normalized_bbox is None:
                return None
            normalized_cells.append(normalized_bbox)

        if len(normalized_cells) == 0:
            return None

        cell_items_map = defaultdict(list)
        for item in content_items:
            item_bbox = item.get("bbox")
            if item_bbox is None:
                continue
            cell_index = self._match_cell_index(item_bbox, normalized_cells)
            if cell_index is None:
                continue
            cell_items_map[cell_index].append(item)

        cell_box_map = {}
        for cell_index, items in cell_items_map.items():
            cell_html = self._merge_cell_items(items)
            if cell_html:
                cell_box_map[cell_index] = [cell_html]

        if not cell_box_map:
            return None

        normalized_logic_points = [
            logic_point.tolist() if hasattr(logic_point, "tolist") else list(logic_point)
            for logic_point in logic_points
        ]
        return plot_html_table(normalized_logic_points, cell_box_map)

    def _get_selected_table_structure(self, table_res_dict: dict) -> tuple[object, object]:
        if table_res_dict.get("selected_table_model") == "wired":
            return (
                table_res_dict.get("wired_cell_bboxes"),
                table_res_dict.get("wired_logic_points"),
            )
        return (
            table_res_dict.get("wireless_cell_bboxes"),
            table_res_dict.get("wireless_logic_points"),
        )

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )
        atom_model_manager = AtomModelSingleton()

        pil_images = [image for image, _, _ in images_with_extra_info]

        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info]

        # pp-doclayout_v2
        images_layout_res += self.model.layout_model.batch_predict(
            pil_images, LAYOUT_BASE_BATCH_SIZE
        )

        if self.formula_enable:
            images_mfd_res = []
            for layout_res in images_layout_res:
                page_formula_res = []
                for res in layout_res:
                    if res.get("label") in ["display_formula", "inline_formula"]:
                        res.setdefault("latex", "")
                        page_formula_res.append(res)
                images_mfd_res.append(page_formula_res)

            # 公式识别
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                np_images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(np_images)):
                mfr_count += len(images_formula_list[image_index])
                for formula_res, formula_with_latex in zip(
                    images_mfd_res[image_index], images_formula_list[image_index]
                ):
                    formula_res["latex"] = formula_with_latex.get("latex", "")
        else:
            for layout_res in images_layout_res:
                # 移除所有的"inline_formula"
                layout_res[:] = [res for res in layout_res if res.get("label") != "inline_formula"]

        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(np_images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    bbox = normalize_to_int_bbox(
                        [float(v) / float(scale) for v in table_res["bbox"]]
                    )
                    if bbox is None:
                        return np_img[0:0, 0:0]
                    return get_crop_np_img(bbox, np_img, scale=scale)

                wireless_table_img = get_crop_table_img(scale = 1)
                wired_table_img = get_crop_table_img(scale = 10/3)

                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':wireless_table_img,
                                                'wired_table_img':wired_table_img,
                                                'page_layout_res':layout_res,
                                                'page_np_img':np_img,
                                                'rotate_label':'0',
                                              })

        # 表格识别 table recognition
        if self.table_enable:
            os.makedirs(TABLE_OCR_DET_DEBUG_DIR, exist_ok=True)

            # 图片旋转批量处理
            img_orientation_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.ImgOrientationCls,
            )
            try:
                if self.enable_ocr_det_batch:
                    img_orientation_cls_model.batch_predict(table_res_list_all_page,
                                                            det_batch_size=self.batch_ratio * OCR_DET_BASE_BATCH_SIZE,
                                                            batch_size=TABLE_ORI_CLS_BATCH_SIZE)
                else:
                    for table_res in table_res_list_all_page:
                        rotate_label = img_orientation_cls_model.predict(table_res['table_img'])
                        img_orientation_cls_model.img_rotate(table_res, rotate_label)
            except Exception as e:
                logger.warning(
                    f"Image orientation classification failed: {e}, using original image"
                )

            # 表格分类
            table_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.TableCls,
            )
            try:
                table_cls_model.batch_predict(table_res_list_all_page,
                                              batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE)
            except Exception as e:
                logger.warning(
                    f"Table classification failed: {e}, using default model"
                )

            for table_res_dict in table_res_list_all_page:
                table_res_dict["table_rich_items"] = self._collect_table_rich_items(
                    table_res_dict
                )
                table_res_dict["table_det_mask_boxes"] = [
                    {"bbox": item["bbox"]}
                    for item in table_res_dict["table_rich_items"]
                ]
                table_res_dict["table_formula_mask_boxes"] = [
                    {"bbox": item["bbox"]}
                    for item in table_res_dict["table_rich_items"]
                    if item.get("type") == "formula"
                ]

            # OCR det 过程，顺序执行
            rec_img_lang_group = defaultdict(list)
            det_ocr_engine = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                enable_merge_det_boxes=False,
            )
            for index, table_res_dict in enumerate(
                    tqdm(table_res_list_all_page, desc="Table-ocr det")
            ):
                bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                det_image = self._apply_mask_boxes_to_image(
                    bgr_image,
                    table_res_dict.get("table_det_mask_boxes") or [],
                )
                ocr_result = det_ocr_engine.ocr(det_image, rec=False)[0]
                dt_boxes_sorted = sorted_boxes(np.asarray(ocr_result)) if ocr_result else []
                dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
                table_formula_mask_boxes = (
                    table_res_dict.get("table_formula_mask_boxes") or []
                )
                dt_boxes_final = (
                    update_det_boxes(dt_boxes_merged, table_formula_mask_boxes)
                    if dt_boxes_merged and table_formula_mask_boxes
                    else dt_boxes_merged
                )

                # debug 输出检测结果和调试信息
                # self._dump_table_det_debug_images(index, bgr_image, det_image, dt_boxes_final)

                # 构造需要 OCR 识别的图片字典，包括cropped_img, dt_box, table_id，并按照语言进行分组
                for dt_box in dt_boxes_final:
                    rec_img_lang_group[table_res_dict["lang"]].append(
                        {
                            "cropped_img": get_rotate_crop_image(
                                bgr_image, np.asarray(dt_box, dtype=np.float32)
                            ),
                            "dt_box": np.asarray(dt_box, dtype=np.float32),
                            "table_id": index,
                        }
                    )

            # OCR rec，按照语言分批处理
            for _lang, rec_img_list in rec_img_lang_group.items():
                if not rec_img_list:
                    continue
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}")[0]
                # 按照 table_id 将识别结果进行回填
                for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                    if table_res_list_all_page[img_dict["table_id"]].get("ocr_result"):
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        )
                    else:
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        ]

            clean_vram(self.model.device, vram_threshold=8)

            # 先对所有表格使用无线表格模型，然后对分类为有线的表格使用有线表格模型
            wireless_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
            )
            wireless_table_model.batch_predict(table_res_list_all_page)
            for table_res_dict in table_res_list_all_page:
                table_res_dict["selected_table_model"] = "wireless"

            # 单独拿出有线表格进行预测
            wired_table_res_list = []
            for table_res_dict in table_res_list_all_page:
                # logger.debug(f"Table classification result: {table_res_dict["table_res"]["cls_label"]} with confidence {table_res_dict["table_res"]["cls_score"]}")
                if (
                    (table_res_dict["table_res"]["cls_label"] == AtomicModel.WirelessTable and table_res_dict["table_res"]["cls_score"] < 0.9)
                    or table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable
                ):
                    wired_table_res_list.append(table_res_dict)
                del table_res_dict["table_res"]["cls_label"]
                del table_res_dict["table_res"]["cls_score"]
            if wired_table_res_list:
                for table_res_dict in tqdm(
                        wired_table_res_list, desc="Table-wired Predict"
                ):
                    wired_table_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.WiredTable,
                        lang=table_res_dict["lang"],
                    )
                    wired_result = wired_table_model.predict(
                        table_res_dict["wired_table_img"],
                        table_res_dict.get("ocr_result") or [],
                        table_res_dict["table_res"].get("html", None),
                        return_metadata=True,
                    )
                    table_res_dict["table_res"]["html"] = wired_result.get("html", "")
                    table_res_dict["selected_table_model"] = wired_result.get(
                        "selected_model", "wireless"
                    )
                    table_res_dict["wired_cell_bboxes"] = wired_result.get(
                        "wired_cell_bboxes"
                    )
                    table_res_dict["wired_logic_points"] = wired_result.get(
                        "wired_logic_points"
                    )

            for table_res_dict in table_res_list_all_page:
                if not table_res_dict.get("table_rich_items"):
                    continue

                selected_cell_bboxes, selected_logic_points = (
                    self._get_selected_table_structure(table_res_dict)
                )
                content_items = []
                for ocr_res in table_res_dict.get("ocr_result") or []:
                    ocr_bbox = self._normalize_bbox(ocr_res[0])
                    if ocr_bbox is None or not ocr_res[1]:
                        continue
                    content_items.append(
                        {
                            "type": "text",
                            "bbox": ocr_bbox,
                            "content": ocr_res[1],
                        }
                    )
                content_items.extend(table_res_dict["table_rich_items"])

                rebuilt_html = self._rebuild_table_html(
                    selected_cell_bboxes,
                    selected_logic_points,
                    content_items,
                )
                if rebuilt_html:
                    table_res_dict["table_res"]["html"] = rebuilt_html

            # 表格格式清理
            for table_res_dict in table_res_list_all_page:
                html_code = table_res_dict["table_res"].get("html", "") or ""

                # 检查html_code是否包含'<table>'和'</table>'
                if "<table>" in html_code and "</table>" in html_code:
                    # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                    start_index = html_code.find("<table>")
                    end_index = html_code.rfind("</table>") + len("</table>")
                    table_res_dict["table_res"]["html"] = html_code[start_index:end_index]

            self._remove_table_internal_layout_items(table_res_list_all_page)

        # OCR det
        if self.enable_ocr_det_batch:
            # 批处理模式 - 按语言和分辨率分组
            # 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page:
                _lang = ocr_res_list_dict['lang']

                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )

                    # BGR转换
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    det_image = self._get_masked_det_image(
                        bgr_image,
                        adjusted_mfdetrec_res,
                    )

                    all_cropped_images_info.append((
                        bgr_image,
                        det_image,
                        useful_list,
                        ocr_res_list_dict,
                        adjusted_mfdetrec_res,
                        _lang,
                    ))

            # 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                RESOLUTION_GROUP_STRIDE = 64

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[1]
                    h, w = cropped_img.shape[:2]
                    # 直接计算目标尺寸并用作分组键
                    target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (target_h, target_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                    # 对所有图像进行padding到统一尺寸
                    batch_images = []
                    for crop_info in group_crops:
                        img = crop_info[1]
                        h, w = img.shape[:2]
                        # 创建目标尺寸的白色背景
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        padded_img[:h, :w] = img
                        batch_images.append(padded_img)

                    # 批处理检测
                    det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                    batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

                    # 处理批处理结果
                    for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                        (
                            bgr_image,
                            _det_image,
                            useful_list,
                            ocr_res_list_dict,
                            adjusted_mfdetrec_res,
                            _lang,
                        ) = crop_info

                        if dt_boxes is not None and len(dt_boxes) > 0:
                            # 处理检测框
                            dt_boxes_sorted = sorted_boxes(dt_boxes)
                            dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                            # 根据公式位置更新检测框
                            dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                              if dt_boxes_merged and adjusted_mfdetrec_res
                                              else dt_boxes_merged)

                            if dt_boxes_final:
                                ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res,
                                    useful_list,
                                    ocr_res_list_dict['ocr_enable'],
                                    bgr_image,
                                    _lang,
                                )
                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    # OCR-det
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    det_image = self._get_masked_det_image(
                        bgr_image,
                        adjusted_mfdetrec_res,
                    )
                    ocr_res = ocr_model.ocr(
                        det_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res,
                            useful_list,
                            ocr_res_list_dict['ocr_enable'],
                            bgr_image,
                            _lang,
                        )

                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # OCR rec
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if not layout_res_item.get("_need_ocr_rec"):
                    continue
                if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                    lang = layout_res_item['lang']

                    # Initialize lists for this language if not exist
                    if lang not in need_ocr_lists_by_lang:
                        need_ocr_lists_by_lang[lang] = []
                        img_crop_lists_by_lang[lang] = []

                    # Add to the appropriate language-specific lists
                    need_ocr_lists_by_lang[lang].append((layout_res, layout_res_item))
                    img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                    # Remove temporary fields after collecting
                    layout_res_item.pop('np_img', None)
                    layout_res_item.pop('lang', None)
                    layout_res_item.pop('_need_ocr_rec', None)

        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    items_to_remove = []
                    # Process OCR results for this language
                    for index, (page_layout_res, layout_res_item) in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        should_remove = False
                        if ocr_score < OcrConfidence.min_confidence:
                            should_remove = True
                        else:
                            layout_res_bbox = layout_res_item['bbox']
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if (
                                    ocr_text in [
                                        '（204号', '（20', '（2', '（2号', '（20号', '号', '（204',
                                        '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                        ':)', '√:)', '√i:)', '−i:)', '−:', 'i:)',
                                    ]
                                    and ocr_score < 0.8
                                    and layout_res_width < layout_res_height
                            ):
                                should_remove = True

                        if should_remove:
                            items_to_remove.append((page_layout_res, layout_res_item))

                    for page_layout_res, layout_res_item in items_to_remove:
                        if layout_res_item in page_layout_res:
                            page_layout_res.remove(layout_res_item)

                    total_processed += len(img_crop_list)

        seal_ocr_model = None
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="Seal-ocr Predict"):
            layout_res = ocr_res_list_dict['layout_res']
            np_img = ocr_res_list_dict['np_img']
            image_h, image_w = np_img.shape[:2]

            for layout_res_item in layout_res:
                if layout_res_item.get("label") != "seal":
                    continue

                layout_res_item["text"] = ""
                seal_bbox = normalize_to_int_bbox(
                    layout_res_item.get("bbox"),
                    image_size=(image_h, image_w),
                )
                if seal_bbox is None:
                    continue

                x0, y0, x1, y1 = seal_bbox
                seal_crop_rgb = np_img[y0:y1, x0:x1]
                if seal_crop_rgb.size == 0:
                    continue

                if seal_ocr_model is None:
                    seal_ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        lang="seal",
                    )

                seal_crop_bgr = cv2.cvtColor(seal_crop_rgb, cv2.COLOR_RGB2BGR)
                seal_ocr_res = seal_ocr_model.ocr(seal_crop_bgr, det=True, rec=True)[0]
                if not seal_ocr_res:
                    continue

                seal_texts = []
                for seal_item in seal_ocr_res:
                    if not seal_item or len(seal_item) != 2:
                        continue
                    rec_result = seal_item[1]
                    if not rec_result or len(rec_result) < 1:
                        continue
                    rec_text = rec_result[0]
                    if rec_text:
                        seal_texts.append(rec_text)

                layout_res_item["text"] = seal_texts

        return images_layout_res
