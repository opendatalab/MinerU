import re
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision import transforms

from .unitable_modules import Encoder, GPTFastDecoder

IMG_SIZE = 448
EOS_TOKEN = "<eos>"
BBOX_TOKENS = [f"bbox-{i}" for i in range(IMG_SIZE + 1)]

HTML_BBOX_HTML_TOKENS = [
    "<td></td>",
    "<td>[",
    "]</td>",
    "<td",
    ">[",
    "></td>",
    "<tr>",
    "</tr>",
    "<tbody>",
    "</tbody>",
    "<thead>",
    "</thead>",
    ' rowspan="2"',
    ' rowspan="3"',
    ' rowspan="4"',
    ' rowspan="5"',
    ' rowspan="6"',
    ' rowspan="7"',
    ' rowspan="8"',
    ' rowspan="9"',
    ' rowspan="10"',
    ' rowspan="11"',
    ' rowspan="12"',
    ' rowspan="13"',
    ' rowspan="14"',
    ' rowspan="15"',
    ' rowspan="16"',
    ' rowspan="17"',
    ' rowspan="18"',
    ' rowspan="19"',
    ' colspan="2"',
    ' colspan="3"',
    ' colspan="4"',
    ' colspan="5"',
    ' colspan="6"',
    ' colspan="7"',
    ' colspan="8"',
    ' colspan="9"',
    ' colspan="10"',
    ' colspan="11"',
    ' colspan="12"',
    ' colspan="13"',
    ' colspan="14"',
    ' colspan="15"',
    ' colspan="16"',
    ' colspan="17"',
    ' colspan="18"',
    ' colspan="19"',
    ' colspan="25"',
]

VALID_HTML_BBOX_TOKENS = [EOS_TOKEN] + HTML_BBOX_HTML_TOKENS + BBOX_TOKENS
TASK_TOKENS = [
    "[table]",
    "[html]",
    "[cell]",
    "[bbox]",
    "[cell+bbox]",
    "[html+bbox]",
]


class TableStructureUnitable:
    def __init__(self, config):
        # encoder_path: str, decoder_path: str, vocab_path: str, device: str
        vocab_path = config["model_path"]["vocab"]
        encoder_path = config["model_path"]["encoder"]
        decoder_path = config["model_path"]["decoder"]
        device = config.get("device", "cuda:0") if config["use_cuda"] else "cpu"

        self.vocab = Tokenizer.from_file(vocab_path)
        self.token_white_list = [
            self.vocab.token_to_id(i) for i in VALID_HTML_BBOX_TOKENS
        ]
        self.bbox_token_ids = set(self.vocab.token_to_id(i) for i in BBOX_TOKENS)
        self.bbox_close_html_token = self.vocab.token_to_id("]</td>")
        self.prefix_token_id = self.vocab.token_to_id("[html+bbox]")
        self.eos_id = self.vocab.token_to_id(EOS_TOKEN)
        self.max_seq_len = 1024
        self.device = device
        self.img_size = IMG_SIZE

        # init encoder
        encoder_state_dict = torch.load(encoder_path, map_location=device)
        self.encoder = Encoder()
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.eval().to(device)

        # init decoder
        decoder_state_dict = torch.load(decoder_path, map_location=device)
        self.decoder = GPTFastDecoder()
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.eval().to(device)

        # define img transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.86597056, 0.88463002, 0.87491087],
                    std=[0.20686628, 0.18201602, 0.18485524],
                ),
            ]
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray):
        start_time = time.time()
        ori_h, ori_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.decoder.setup_caches(
            max_batch_size=1,
            max_seq_length=self.max_seq_len,
            dtype=image.dtype,
            device=self.device,
        )
        context = (
            torch.tensor([self.prefix_token_id], dtype=torch.int32)
            .repeat(1, 1)
            .to(self.device)
        )
        eos_id_tensor = torch.tensor(self.eos_id, dtype=torch.int32).to(self.device)
        memory = self.encoder(image)
        context = self.loop_decode(context, eos_id_tensor, memory)
        bboxes, html_tokens = self.decode_tokens(context)
        bboxes = bboxes.astype(np.float32)

        # rescale boxes
        scale_h = ori_h / self.img_size
        scale_w = ori_w / self.img_size
        bboxes[:, 0::2] *= scale_w  # 缩放 x 坐标
        bboxes[:, 1::2] *= scale_h  # 缩放 y 坐标
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, ori_w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, ori_h - 1)
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + html_tokens
            + ["</table>", "</body>", "</html>"]
        )
        return structure_str_list, bboxes, time.time() - start_time

    def decode_tokens(self, context):
        pred_html = context[0]
        pred_html = pred_html.detach().cpu().numpy()
        pred_html = self.vocab.decode(pred_html, skip_special_tokens=False)
        seq = pred_html.split("<eos>")[0]
        token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
        for i in token_black_list:
            seq = seq.replace(i, "")

        tr_pattern = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
        td_pattern = re.compile(r"<td(.*?)>(.*?)</td>", re.DOTALL)
        bbox_pattern = re.compile(r"\[ bbox-(\d+) bbox-(\d+) bbox-(\d+) bbox-(\d+) \]")

        decoded_list = []
        bbox_coords = []

        # 查找所有的 <tr> 标签
        for tr_match in tr_pattern.finditer(pred_html):
            tr_content = tr_match.group(1)
            decoded_list.append("<tr>")

            # 查找所有的 <td> 标签
            for td_match in td_pattern.finditer(tr_content):
                td_attrs = td_match.group(1).strip()
                td_content = td_match.group(2).strip()
                if td_attrs:
                    decoded_list.append("<td")
                    # 可能同时存在行列合并，需要都添加
                    attrs_list = td_attrs.split()
                    for attr in attrs_list:
                        decoded_list.append(" " + attr)
                    decoded_list.append(">")
                    decoded_list.append("</td>")
                else:
                    decoded_list.append("<td></td>")

                # 查找 bbox 坐标
                bbox_match = bbox_pattern.search(td_content)
                if bbox_match:
                    xmin, ymin, xmax, ymax = map(int, bbox_match.groups())
                    # 将坐标转换为从左上角开始顺时针到左下角的点的坐标
                    coords = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                    bbox_coords.append(coords)
                else:
                    # 填充占位的bbox，保证后续流程统一
                    bbox_coords.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            decoded_list.append("</tr>")

        bbox_coords_array = np.array(bbox_coords)
        return bbox_coords_array, decoded_list

    def loop_decode(self, context, eos_id_tensor, memory):
        box_token_count = 0
        for _ in range(self.max_seq_len):
            eos_flag = (context == eos_id_tensor).any(dim=1)
            if torch.all(eos_flag):
                break

            next_tokens = self.decoder(memory, context)
            if next_tokens[0] in self.bbox_token_ids:
                box_token_count += 1
                if box_token_count > 4:
                    next_tokens = torch.tensor(
                        [self.bbox_close_html_token], dtype=torch.int32
                    )
                    box_token_count = 0
            context = torch.cat([context, next_tokens], dim=1)
        return context
