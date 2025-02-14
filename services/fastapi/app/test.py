'''
Author: FutureMeng be_loving@163.com
Date: 2024-11-13 19:05:01
LastEditors: FutureMeng be_loving@163.com
LastEditTime: 2024-11-13 19:06:17
FilePath: \lzmineru\api\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import urllib.request
import os
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

current_script_dir = os.path.dirname(os.path.abspath(__file__))
local_image_dir = os.path.join(current_script_dir, 'images')
image_dir = str(os.path.basename(local_image_dir))
imageUrl = 'https://one-jiulu.oss-cn-beijing.aliyuncs.com/9250ba5ccbf34249b054d063d32ec8f8.pdf?OSSAccessKeyId=LTAI5tABhdnCgSeVaptuWLfx&Expires=1732100601&Signature=XZqGPO%2BJ76bEJ0ou8GZQUO7vhjs%3D'

pdf_bytes = urllib.request.urlopen(imageUrl).read()
image_writer = DiskReaderWriter(local_image_dir)
jso_useful_key = {"_pdf_type": "", "model_list": []}
pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
pipe.pipe_classify()
pipe.pipe_analyze()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
print(md_content)