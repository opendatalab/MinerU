'''
Author: FutureMeng be_loving@163.com
Date: 2024-11-13 19:44:33
LastEditors: FutureMeng be_loving@163.com
LastEditTime: 2024-11-14 15:47:27
FilePath: \MinerU\scripts\fastapitest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from fastapi import FastAPI
import urllib.request
from . import magic_pdf_parse_util

app = FastAPI()

@app.post("/parse_pdf")
async def parse_pdf(imageUrl: str, parse_method: str = 'auto'):
    pdf_bytes = urllib.request.urlopen(imageUrl).read()
    content_list, md_content = magic_pdf_parse_util.pdf_parse(pdf_bytes, parse_method)
    return {"content_list": content_list, "md_content": md_content}