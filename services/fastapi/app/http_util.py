import requests
import asyncio
from loguru import logger
def post_result_callback(url, cbkey, md5_value, content_list, md_content):
    if (url is None):
        return
    asyncio.run(commit_callback_task(url, cbkey, md5_value, content_list, md_content))

async def commit_callback_task(url, cbkey, md5_value, content_list, md_content):
    asyncio.create_task(post_callback(url, cbkey, md5_value, content_list, md_content))

async def post_callback(url, cbkey, md5_value, content_list, md_content):
    try:
        json_result = {"cbkey": cbkey, "md5": md5_value, "content_list": content_list, "md_content": md_content}
        requests.post(url, data=json_result)
        logger.info("post_callback success, url:{}", url)
    except Exception as e:
        logger.exception(e)