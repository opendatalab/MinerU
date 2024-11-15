import redis
import json
import enum

ParseState = enum.Enum('ParseState', ('deny', 'init', 'parsing', 'parsed', 'failed'))

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
    json_str = json.dumps({"state": state.name, "content_list": content_list, "md_content": md_content})
    redis_conn.set(md5_value, json_str)
    redis_conn.expire(md5_value, 60 * 60 * 24)