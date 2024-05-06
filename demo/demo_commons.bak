import json

from magic_pdf.libs.config_reader import get_s3_config_dict
from magic_pdf.libs.commons import join_path, read_file, json_dump_path


local_json_path = "Z:/format.json"
local_jsonl_path = "Z:/format.jsonl"

def get_json_from_local_or_s3(book_name=None):
    if book_name is None:
        with open(local_json_path, "r", encoding="utf-8") as json_file:
            json_line = json_file.read()
            json_object = json.loads(json_line)
    else:
        # error_log_path & json_dump_path
        # 可配置从上述两个地址获取源json
        json_path = join_path(json_dump_path, book_name + ".json")
        s3_config = get_s3_config_dict(json_path)
        file_content = read_file(json_path, s3_config)
        json_str = file_content.decode("utf-8")
        # logger.info(json_str)
        json_object = json.loads(json_str)
    return json_object


def write_json_to_local(jso, book_name=None):
    if book_name is None:
        with open(local_json_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(jso, ensure_ascii=False))
    else:
        pass