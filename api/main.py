'''
Author: FutureMeng be_loving@163.com
Date: 2024-11-13 19:44:33
LastEditors: dt_4541218930 abcstorms@163.com
LastEditTime: 2024-11-13 22:36:37
FilePath: \MinerU\scripts\fastapitest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from fastapi import FastAPI
import urllib.request
import os
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf_parse_util import pdf_parse

app = FastAPI()

current_script_dir = os.path.dirname(os.path.abspath(__file__))
local_image_dir = os.path.join(current_script_dir, 'images')
image_dir = str(os.path.basename(local_image_dir))

@app.get("/hello")
async def hello():
    return 'Hello, World'

@app.post("/parse_pdf")
async def parse_pdf(imageUrl: str, parse_method: str = 'auto'):
    pdf_bytes = urllib.request.urlopen(imageUrl).read()
    content_list, md_content = pdf_parse(pdf_bytes, parse_method)
    return {"content_list": content_list, "md_content": md_content}