'''
Author: dt_4541218930 abcstorms@163.com
Date: 2024-11-14 17:04:42
LastEditors: FutureMeng futuremeng@gmail.com
LastEditTime: 2025-02-14 09:07:31
FilePath: /MinerU/services/fastapi/app/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from fastapi import FastAPI
import urllib.request
import urllib.parse
import hashlib
import queue
import threading
from . import magic_pdf_parse_util
from . import redis_util
from loguru import logger

message_queue = queue.Queue(20)

app = FastAPI()

def calc_md5(byteContent: bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(byteContent)
    return hash_md5.hexdigest()

def commit_parse_task(md5_value, parse_method):
    message_queue.put({"md5": md5_value, "parse_method": parse_method})

def queue_consumer(q):
    while True:
        item = q.get()
        if (item):
            file_path='/gateway/tmp/'+item['md5']+'.pdf'
            if os.access(file_path, os.R_OK):
                byteContent = open(file_path,'rb').read() 
                logger.info('start to parse '+item['md5'])
                magic_pdf_parse_util.pdf_parse(item['md5'], byteContent, item['parse_method'])
            else:
                logger.error(file_path+' can not read')

consumer_thread = threading.Thread(target=queue_consumer, args=(message_queue,))
consumer_thread.start()


list = redis_util.get_init_list()
logger.info(list)
for item in list:
    redis_util.set_parse_init(item['md5'])
    commit_parse_task(item['md5'], 'auto')
    


@app.post("/parse_pdf")
async def parse_pdf(encodeUrl: str = None, md5: str = None, parse_method: str = 'auto'):
    if (encodeUrl is None and md5 is None):
        return {"state": "failed", "error": "encodeUrl or md5 is required" }
    if (md5):
        file_info = redis_util.get_file_info(md5)
        if (file_info):
            return file_info
    if (encodeUrl is None):
        return {"state": "failed", "error": "encodeUrl is required when md5 is not valid"}

    try:
        decodeUrl = urllib.parse.unquote(encodeUrl)
        pdf_bytes = urllib.request.urlopen(decodeUrl).read()
    except Exception:
        return {"state": "failed", "error": "encodeUrl is not valid"}
    if (pdf_bytes is None):
        return {"state": "failed", "error": "download faild"}
    
    md5_value = calc_md5(pdf_bytes)
    
    file_path='/gateway/tmp/'+md5_value+'.pdf'
    
    if not os.path.isfile(file_path):
        tmp_file = open(file_path,'wb')
        tmp_file.write(pdf_bytes)
        tmp_file.close()
    
    
    file_info = redis_util.get_file_info(md5_value)
    if file_info:
        file_info['queue'] = message_queue.qsize()
        return file_info
    try:
        commit_parse_task(md5_value, parse_method)
    except Exception:
        redis_util.set_parse_deny(md5_value)
        return redis_util.get_file_info(md5_value)
    
    redis_util.set_parse_init(md5_value)
    file_info = redis_util.get_file_info(md5_value)
    file_info['queue'] = message_queue.qsize()
    return file_info
