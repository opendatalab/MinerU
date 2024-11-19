import redis
import json
import enum

ParseState = enum.Enum('ParseState', ('deny', 'init', 'start', 'done', 'failed'))

redis_conn = redis.Redis(host='host.docker.internal', port=6380, db=0)

def get_file_info(md5_value):
    json_str = redis_conn.get(md5_value)
    if json_str:
        return json.loads(json_str)

def del_file_info(md5_value):
    redis_conn.delete(md5_value)

def set_file_info_expire(md5_value, expire_seconds):
    redis_conn.expire(md5_value, expire_seconds)

def set_file_info(md5_value, state: ParseState, content_list = "", md_content = ""):
    json_str = json.dumps({"state": state.name, "md5": md5_value, "content_list": content_list, "md_content": md_content})
    redis_conn.set(md5_value, json_str)

def set_parse_deny(md5_value):
    set_file_info(md5_value, ParseState.deny)
    set_file_info_expire(md5_value, 5)

def set_parse_failed(md5_value):
    set_file_info(md5_value, ParseState.failed)
    set_file_info_expire(md5_value, 10)

def set_parse_init(md5_value):
    set_file_info(md5_value, ParseState.init)
    set_file_info_expire(md5_value, 60 * 60)

def set_parse_parsing(md5_value):
    set_file_info(md5_value, ParseState.start)
    set_file_info_expire(md5_value, 60 * 30)

def set_parse_parsed(md5_value, content_list, md_content):
    set_file_info(md5_value, ParseState.done, content_list, md_content)
    set_file_info_expire(md5_value, 60 * 60 * 24)