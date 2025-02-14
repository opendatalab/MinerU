import requests

def post_result_callback(url, cbkey, md5_value, content_list = "", md_content = ""):
    if (url is None):
        return
    json_result = {"cbkey": cbkey, "md5": md5_value, "content_list": content_list, "md_content": md_content}
    requests.post(url, data=json_result)
