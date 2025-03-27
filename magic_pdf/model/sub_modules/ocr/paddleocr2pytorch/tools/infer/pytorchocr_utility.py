import os, sys
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse

def init_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default='cpu')
    # parser.add_argument("--ir_optim", type=str2bool, default=True)
    # parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    # parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_box_type", type=str, default='box')
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)
    parser.add_argument("--det_fce_box_type", type=str, default='poly')

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_path", type=str)
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)

    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--limited_max_width", type=int, default=1280)
    parser.add_argument("--limited_min_width", type=int, default=16)

    parser.add_argument(
        "--vis_font_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'doc/fonts/simfang.ttf'))
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'pytorchocr/utils/ppocr_keys_v1.txt'))

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_path", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_path", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'pytorchocr/utils/ic15_dict.txt'))
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # SR parmas
    parser.add_argument("--sr_model_path", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    # params .yaml
    parser.add_argument("--det_yaml_path", type=str, default=None)
    parser.add_argument("--rec_yaml_path", type=str, default=None)
    parser.add_argument("--cls_yaml_path", type=str, default=None)
    parser.add_argument("--e2e_yaml_path", type=str, default=None)
    parser.add_argument("--sr_yaml_path", type=str, default=None)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)

    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

def get_default_config(args):
    return vars(args)


def read_network_config_from_yaml(yaml_path, char_num=None):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    if res['Architecture']['Head']['name'] == 'MultiHead' and char_num is not None:
        res['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'SARLabelDecode': char_num + 2,
            'NRTRLabelDecode': char_num + 3
        }
    return res['Architecture']

def AnalysisConfig(weights_path, yaml_path=None, char_num=None):
    if not os.path.exists(os.path.abspath(weights_path)):
        raise FileNotFoundError('{} is not found.'.format(weights_path))

    if yaml_path is not None:
        return read_network_config_from_yaml(yaml_path, char_num=char_num)

    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()

    # supported_weights = ['ch_ptocr_server_v2.0_det_infer.pth',
    #                      'ch_ptocr_server_v2.0_rec_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_det_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_rec_infer.pth',
    #                      'ch_ptocr_mobile_v2.0_cls_infer.pth',
    #                    ]
    # assert weights_name in supported_weights, \
    #     "supported weights are {} but input weights is {}".format(supported_weights, weights_name)

    if weights_name == 'ch_ptocr_server_v2.0_det_infer.pth':
        network_config = {'model_type':'det',
                          'algorithm':'DB',
                          'Transform':None,
                          'Backbone':{'name':'ResNet_vd', 'layers':18, 'disable_se':True},
                          'Neck':{'name':'DBFPN', 'out_channels':256},
                          'Head':{'name':'DBHead', 'k':50}}

    elif weights_name == 'ch_ptocr_server_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'name':'ResNet', 'layers':34},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':256, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name in ['ch_ptocr_mobile_v2.0_det_infer.pth']:
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name =='ch_ptocr_mobile_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_cls_infer.pth':
        network_config = {'model_type':'cls',
                          'algorithm':'CLS',
                          'Transform':None,
                          'Backbone':{'name':'MobileNetV3', 'model_name':'small', 'scale':0.35},
                          'Neck':None,
                          'Head':{'name':'ClsHead', 'class_dim':2}}

    elif weights_name == 'ch_ptocr_v2_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV1Enhance', 'scale': 0.5},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 64, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'mid_channels': 96, 'fc_decay': 2e-05}}

    elif weights_name == 'ch_ptocr_v2_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_v3_rec_infer.pth':
        network_config = {'model_type':'rec',
           'algorithm':'CRNN',
           'Transform':None,
           'Backbone':{'name':'MobileNetV1Enhance',
                       'scale':0.5,
                       'last_conv_stride': [1, 2],
                       'last_pool_type': 'avg'},
           'Neck':{'name':'SequenceEncoder',
                   'dims': 64,
                   'depth': 2,
                   'hidden_dims': 120,
                   'use_guide': True,
                   'encoder_type':'svtr'},
           'Head':{'name':'CTCHead', 'fc_decay': 2e-05}
           }

    elif weights_name == 'ch_ptocr_v3_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_mv3_db_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large'},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_r50_vd_db_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_vd', 'layers': 50},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_mv3_east_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'EAST',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large'},
                          'Neck': {'name': 'EASTFPN', 'model_name': 'small'},
                          'Head': {'name': 'EASTHead', 'model_name': 'small'}}

    elif weights_name == 'det_r50_vd_east_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'EAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_vd', 'layers': 50},
                          'Neck': {'name': 'EASTFPN', 'model_name': 'large'},
                          'Head': {'name': 'EASTHead', 'model_name': 'large'}}

    elif weights_name == 'det_r50_vd_sast_icdar15_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'SAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_SAST', 'layers': 50},
                          'Neck': {'name': 'SASTFPN', 'with_cab': True},
                          'Head': {'name': 'SASTHead'}}

    elif weights_name == 'det_r50_vd_sast_totaltext_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'SAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_SAST', 'layers': 50},
                          'Neck': {'name': 'SASTFPN', 'with_cab': True},
                          'Head': {'name': 'SASTHead'}}

    elif weights_name == 'en_server_pgneta_infer.pth':
        network_config = {'model_type': 'e2e',
                          'algorithm': 'PGNet',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet', 'layers': 50},
                          'Neck': {'name': 'PGFPN'},
                          'Head': {'name': 'PGHead'}}

    elif weights_name == 'en_ptocr_mobile_v2.0_table_det_infer.pth':
        network_config = {'model_type': 'det','algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': False},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'en_ptocr_mobile_v2.0_table_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'large', 'name': 'MobileNetV3', },
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 96, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    elif 'om_' in weights_name and '_rec_' in weights_name:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'om'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    else:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}
        # raise NotImplementedError

    return network_config


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image