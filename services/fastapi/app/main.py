import hashlib
import os
import queue
import threading
from urllib import parse, request

from fastapi import FastAPI
from loguru import logger

from . import magic_pdf_parse_util
from . import redis_util

message_queue = queue.Queue(20)

app = FastAPI()

def calc_md5(byteContent: bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(byteContent)
    return hash_md5.hexdigest()

def commit_parse_task(md5_value, parse_method, cbUrl, cbkey):
    message_queue.put({"md5": md5_value, "parse_method": parse_method, "cbUrl": cbUrl, "cbkey": cbkey})

def queue_consumer(q):
    while True:
        item = q.get()
        if (item):
            file_path='/gateway/tmp/'+item['md5']+'.pdf'
            if os.access(file_path, os.R_OK):
                byteContent = open(file_path,'rb').read() 
                logger.info('start to parse '+item['md5'])
                magic_pdf_parse_util.pdf_parse(item['md5'], byteContent, item['parse_method'], item['cbUrl'], item['cbkey'])
            else:
                logger.error(file_path+' can not read')
            q.task_done()

consumer_thread = threading.Thread(target=queue_consumer, args=(message_queue,))
consumer_thread.start()


initial_pdf_list = redis_util.get_init_list()
logger.info(initial_pdf_list)
for item in initial_pdf_list:
    redis_util.set_parse_init(item['md5'])
    commit_parse_task(item['md5'], 'auto')
    
MAX_FILE_SIZE = 1024 * 1024 * 50

@app.post("/parse_pdf")
async def parse_pdf(encodeUrl: str = None, md5: str = None, cbUrl: str = None, cbkey: str = None, parse_method: str = 'auto'):
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
        with urllib.request.urlopen(decodeUrl) as response:
            if response.getheader('Content-Length') and int(response.getheader('Content-Length')) > MAX_FILE_SIZE:
                return {"state": "failed", "error": "File size exceeds the limit."}
            pdf_bytes = bytearray()
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                pdf_bytes.extend(chunk)
                if len(pdf_bytes) > MAX_FILE_SIZE:
                    return {"state": "failed", "error": "File size exceeds the limit."}
    except Exception:
        logger.error(f"Error downloading PDF: {e}")
        return {"state": "failed", "error": "encodeUrl is not valid."}

    md5_value = calc_md5(pdf_bytes)
    
    file_path=f'/gateway/tmp/{md5_value+}.pdf'
    
    if not os.path.isfile(file_path):
        tmp_file = open(file_path,'wb')
        tmp_file.write(pdf_bytes)
        tmp_file.close()
    
    
    file_info = redis_util.get_file_info(md5_value)
    if file_info:
        file_info['queue'] = message_queue.qsize()
        return file_info
    try:
        commit_parse_task(md5_value, parse_method, cbUrl, cbkey)
    except Exception:
        redis_util.set_parse_deny(md5_value)
        return redis_util.get_file_info(md5_value)
    
    redis_util.set_parse_init(md5_value)
    file_info = redis_util.get_file_info(md5_value)
    file_info['queue'] = message_queue.qsize()
    return file_info
