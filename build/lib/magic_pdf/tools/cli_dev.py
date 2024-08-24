import os
import json as json_parse
import click
from pathlib import Path
from magic_pdf.libs.path_utils import (
    parse_s3path,
    parse_s3_range_params,
    remove_non_official_s3_args,
)
from magic_pdf.libs.config_reader import (
    get_s3_config,
)
from magic_pdf.rw.S3ReaderWriter import S3ReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
import magic_pdf.model as model_config
from magic_pdf.tools.common import parse_pdf_methods, do_parse
from magic_pdf.libs.version import __version__


def read_s3_path(s3path):
    bucket, key = parse_s3path(s3path)

    s3_ak, s3_sk, s3_endpoint = get_s3_config(bucket)
    s3_rw = S3ReaderWriter(
        s3_ak, s3_sk, s3_endpoint, "auto", remove_non_official_s3_args(s3path)
    )
    may_range_params = parse_s3_range_params(s3path)
    if may_range_params is None or 2 != len(may_range_params):
        byte_start, byte_end = 0, None
    else:
        byte_start, byte_end = int(may_range_params[0]), int(may_range_params[1])
        byte_end += byte_start - 1
    return s3_rw.read_jsonl(
        remove_non_official_s3_args(s3path),
        byte_start,
        byte_end,
        AbsReaderWriter.MODE_BIN,
    )


@click.group()
@click.version_option(__version__, "--version", "-v", help="显示版本信息")
def cli():
    pass


@cli.command()
@click.option(
    "-j",
    "--jsonl",
    "jsonl",
    type=str,
    help="输入 jsonl 路径，本地或者 s3 上的文件",
    required=True,
)
@click.option(
    "-m",
    "--method",
    "method",
    type=parse_pdf_methods,
    help="指定解析方法。txt: 文本型 pdf 解析方法， ocr: 光学识别解析 pdf, auto: 程序智能选择解析方法",
    default="auto",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=str,
    help="输出到本地目录",
    default="",
)
def jsonl(jsonl, method, output_dir):
    print("haha")
    model_config.__use_inside_model__ = False
    full_jsonl_path = os.path.realpath(jsonl)
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(full_jsonl_path), "output")

    if jsonl.startswith("s3://"):
        jso = json_parse.loads(read_s3_path(jsonl).decode("utf-8"))
    else:
        with open(jsonl) as f:
            jso = json_parse.loads(f.readline())
    s3_file_path = jso.get("file_location")
    if s3_file_path is None:
        s3_file_path = jso.get("path")
    pdf_file_name = Path(s3_file_path).stem
    pdf_data = read_s3_path(s3_file_path)


    print(pdf_file_name, jso, method)
    do_parse(
        output_dir,
        pdf_file_name,
        pdf_data,
        jso["doc_layout_result"],
        method,
        f_dump_content_list=True,
    )


@cli.command()
@click.option(
    "-p",
    "--pdf",
    "pdf",
    type=click.Path(exists=True),
    required=True,
    help="本地 PDF 文件",
)
@click.option(
    "-j",
    "--json",
    "json_data",
    type=click.Path(exists=True),
    required=True,
    help="本地模型推理出的 json 数据",
)
@click.option(
    "-o", "--output-dir", "output_dir", type=str, help="本地输出目录", default=""
)
@click.option(
    "-m",
    "--method",
    "method",
    type=parse_pdf_methods,
    help="指定解析方法。txt: 文本型 pdf 解析方法， ocr: 光学识别解析 pdf, auto: 程序智能选择解析方法",
    default="auto",
)
def pdf(pdf, json_data, output_dir, method):
    model_config.__use_inside_model__ = False
    full_pdf_path = os.path.realpath(pdf)
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(full_pdf_path), "output")

    def read_fn(path):
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    model_json_list = json_parse.loads(read_fn(json_data).decode("utf-8"))

    file_name = str(Path(full_pdf_path).stem)
    pdf_data = read_fn(full_pdf_path)
    do_parse(
        output_dir,
        file_name,
        pdf_data,
        model_json_list,
        method,
        f_dump_content_list=True,
    )


if __name__ == "__main__":
    cli()
