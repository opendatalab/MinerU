import datetime
import json
import os, re, configparser
import subprocess
import time

import boto3
from loguru import logger
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

import fitz # 1.23.9中已经切换到rebase
# import fitz_old as fitz  # 使用1.23.9之前的pymupdf库


def get_delta_time(input_time):
    return round(time.time() - input_time, 2)


def join_path(*args):
    return '/'.join(str(s).rstrip('/') for s in args)


#配置全局的errlog_path，方便demo同步引用
error_log_path = "s3://llm-pdf-text/err_logs/"
# json_dump_path = "s3://pdf_books_temp/json_dump/" # 这条路径仅用于临时本地测试,不能提交到main
json_dump_path = "s3://llm-pdf-text/json_dump/"

# s3_image_save_path = "s3://mllm-raw-media/pdf2md_img/" # 基础库不应该有这些存在的路径，应该在业务代码中定义


def get_top_percent_list(num_list, percent):
    """
    获取列表中前百分之多少的元素
    :param num_list:
    :param percent:
    :return:
    """
    if len(num_list) == 0:
        top_percent_list = []
    else:
        # 对imgs_len_list排序
        sorted_imgs_len_list = sorted(num_list, reverse=True)
        # 计算 percent 的索引
        top_percent_index = int(len(sorted_imgs_len_list) * percent)
        # 取前80%的元素
        top_percent_list = sorted_imgs_len_list[:top_percent_index]
    return top_percent_list


def formatted_time(time_stamp):
    dt_object = datetime.datetime.fromtimestamp(time_stamp)
    output_time = dt_object.strftime("%Y-%m-%d-%H:%M:%S")
    return output_time


def mymax(alist: list):
    if len(alist) == 0:
        return 0  # 空是0， 0*0也是0大小q
    else:
        return max(alist)

def parse_aws_param(profile):
    if isinstance(profile, str):
        # 解析配置文件
        config_file = join_path(os.path.expanduser("~"), ".aws", "config")
        credentials_file = join_path(os.path.expanduser("~"), ".aws", "credentials")
        config = configparser.ConfigParser()
        config.read(credentials_file)
        config.read(config_file)
        # 获取 AWS 账户相关信息
        ak = config.get(profile, "aws_access_key_id")
        sk = config.get(profile, "aws_secret_access_key")
        if profile == "default":
            s3_str = config.get(f"{profile}", "s3")
        else:
            s3_str = config.get(f"profile {profile}", "s3")
        end_match = re.search("endpoint_url[\s]*=[\s]*([^\s\n]+)[\s\n]*$", s3_str, re.MULTILINE)
        if end_match:
            endpoint = end_match.group(1)
        else:
            raise ValueError(f"aws 配置文件中没有找到 endpoint_url")
        style_match = re.search("addressing_style[\s]*=[\s]*([^\s\n]+)[\s\n]*$", s3_str, re.MULTILINE)
        if style_match:
            addressing_style = style_match.group(1)
        else:
            addressing_style = "path"
    elif isinstance(profile, dict):
        ak = profile["ak"]
        sk = profile["sk"]
        endpoint = profile["endpoint"]
        addressing_style = "auto"

    return ak, sk, endpoint, addressing_style


def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key


def read_file(pdf_path: str, s3_profile):
    if pdf_path.startswith("s3://"):
        ak, sk, end_point, addressing_style = parse_aws_param(s3_profile)
        cli = boto3.client(service_name="s3", aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=end_point,
                           config=Config(s3={'addressing_style': addressing_style}, retries={'max_attempts': 10, 'mode': 'standard'}))
        bucket_name, bucket_key = parse_bucket_key(pdf_path)
        res = cli.get_object(Bucket=bucket_name, Key=bucket_key)
        file_content = res["Body"].read()
        return file_content
    else:
        with open(pdf_path, "rb") as f:
            return f.read()


def get_docx_model_output(pdf_model_output, page_id):

    model_output_json = pdf_model_output[page_id]

    return model_output_json


def list_dir(dir_path:str, s3_profile:str):
    """
    列出dir_path下的所有文件
    """
    ret = []
    
    if dir_path.startswith("s3"):
        ak, sk, end_point, addressing_style = parse_aws_param(s3_profile)
        s3info = re.findall(r"s3:\/\/([^\/]+)\/(.*)", dir_path)
        bucket, path = s3info[0][0], s3info[0][1]
        try:
            cli = boto3.client(service_name="s3", aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=end_point,
                                            config=Config(s3={'addressing_style': addressing_style}))
            def list_obj_scluster():
                marker = None
                while True:
                    list_kwargs = dict(MaxKeys=1000, Bucket=bucket, Prefix=path)
                    if marker:
                        list_kwargs['Marker'] = marker
                    response = cli.list_objects(**list_kwargs)
                    contents = response.get("Contents", [])
                    yield from contents
                    if not response.get("IsTruncated") or len(contents)==0:
                        break
                    marker = contents[-1]['Key']


            for info in list_obj_scluster():
                file_path = info['Key']
                #size = info['Size']

                if path!="":
                    afile = file_path[len(path):]
                    if afile.endswith(".json"):
                        ret.append(f"s3://{bucket}/{file_path}")
                        
            return ret

        except Exception as e:
            logger.exception(e)
            exit(-1)
    else: #本地的目录，那么扫描本地目录并返会这个目录里的所有jsonl文件
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".json"):
                    ret.append(join_path(root, file))
        ret.sort()
        return ret

def get_img_s3_client(save_path:str, image_s3_config:str):
    """
    """
    if save_path.startswith("s3://"):  # 放这里是为了最少创建一个s3 client
        ak, sk, end_point, addressing_style = parse_aws_param(image_s3_config)
        img_s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=end_point,
            config=Config(s3={"addressing_style": addressing_style}, retries={'max_attempts': 5, 'mode': 'standard'}),
        )
    else:
        img_s3_client = None
        
    return img_s3_client

if __name__=="__main__":
    s3_path = "s3://llm-pdf-text/layout_det/scihub/scimag07865000-07865999/10.1007/s10729-011-9175-6.pdf/"
    s3_profile = "langchao"
    ret = list_dir(s3_path, s3_profile)
    print(ret)
    