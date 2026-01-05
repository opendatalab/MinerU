# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from mineru.utils.os_env_config import get_op_num_threads
from .table_structure_utils import (
    OrtInferSession,
    TableLabelDecode,
    TablePreprocess,
    BatchTablePreprocess,
)


class TableStructurer:
    def __init__(self, config: Dict[str, Any]):
        self.preprocess_op = TablePreprocess()
        self.batch_preprocess_op = BatchTablePreprocess()

        config["intra_op_num_threads"] = get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
        config["inter_op_num_threads"] = get_op_num_threads("MINERU_INTER_OP_NUM_THREADS")

        self.session = OrtInferSession(config)

        self.character = self.session.get_metadata()
        self.postprocess_op = TableLabelDecode(self.character)

    def process(self, img):
        starttime = time.time()
        data = {"image": img}
        data = self.preprocess_op(data)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        outputs = self.session([img])

        preds = {"loc_preds": outputs[0], "structure_probs": outputs[1]}

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        bbox_list = post_result["bbox_batch_list"][0]

        structure_str_list = post_result["structure_batch_list"][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )
        elapse = time.time() - starttime
        return structure_str_list, bbox_list, elapse

    def batch_process(
        self, img_list: List[np.ndarray]
    ) -> List[Tuple[List[str], np.ndarray, float]]:
        """批量处理图像列表
        Args:
            img_list: 图像列表
        Returns:
            结果列表，每个元素包含 (table_struct_str, cell_bboxes, elapse)
        """
        starttime = time.perf_counter()

        batch_data = self.batch_preprocess_op(img_list)
        preprocessed_images = batch_data[0]
        shape_lists = batch_data[1]

        preprocessed_images = np.array(preprocessed_images)
        bbox_preds, struct_probs = self.session([preprocessed_images])

        batch_size = preprocessed_images.shape[0]
        results = []
        for bbox_pred, struct_prob, shape_list in zip(
            bbox_preds, struct_probs, shape_lists
        ):
            preds = {
                "loc_preds": np.expand_dims(bbox_pred, axis=0),
                "structure_probs": np.expand_dims(struct_prob, axis=0),
            }
            shape_list = np.expand_dims(shape_list, axis=0)
            post_result = self.postprocess_op(preds, [shape_list])
            bbox_list = post_result["bbox_batch_list"][0]
            structure_str_list = post_result["structure_batch_list"][0]
            structure_str_list = structure_str_list[0]
            structure_str_list = (
                ["<html>", "<body>", "<table>"]
                + structure_str_list
                + ["</table>", "</body>", "</html>"]
            )
            results.append((structure_str_list, bbox_list, 0))

        total_elapse = time.perf_counter() - starttime
        for i in range(len(results)):
            results[i] = (results[i][0], results[i][1], total_elapse / batch_size)

        return results
