'''
Author: dt_4541218930 abcstorms@163.com
Date: 2024-11-14 17:04:42
LastEditors: dt_4541218930 abcstorms@163.com
LastEditTime: 2024-11-19 19:39:15
FilePath: \lzmineru\services\fastapi\app\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from fastapi import FastAPI
import urllib.request
import hashlib
import queue
import threading
from . import magic_pdf_parse_util
from . import redis_util

message_queue = queue.Queue(20)

app = FastAPI()

def calc_md5(byteContent: bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(byteContent)
    return hash_md5.hexdigest()

def commit_parse_task(md5_value, byteContent: bytes, parse_method):
    message_queue.put({"md5": md5_value, "byteContent": byteContent, "parse_method": parse_method})

def queue_consumer(q):
    while True:
        item = q.get()
        if (item):
            magic_pdf_parse_util.pdf_parse(item['md5'], item['byteContent'], item['parse_method'])

consumer_thread = threading.Thread(target=queue_consumer, args=(message_queue,))
consumer_thread.start()

@app.post("/parse_pdf")
async def parse_pdf(imageUrl: str = None, md5: str = None, parse_method: str = 'auto'):
    if (imageUrl is None and md5 is None):
        return {"state": "failed", "error": "imageUrl or md5 is required" }
    if (md5):
        file_info = redis_util.get_file_info(md5)
        if (file_info):
            return file_info
    if (imageUrl is None):
        return {"state": "failed", "error": "imageUrl is required when md5 is not valid"}

    try:
        pdf_bytes = urllib.request.urlopen(imageUrl).read()
    except Exception:
        return {"state": "failed", "error": "imageUrl is not valid"}
    if (pdf_bytes is None):
        return {"state": "failed", "error": "imageUrl is not valid"}
    md5_value = calc_md5(pdf_bytes)
    file_info = redis_util.get_file_info(md5_value)
    if file_info:
        return file_info
    try:
        commit_parse_task(md5_value, pdf_bytes, parse_method)
    except Exception:
        redis_util.set_parse_deny(md5_value)
        return redis_util.get_file_info(md5_value)
    redis_util.set_parse_init(md5_value)
    return redis_util.get_file_info(md5_value)