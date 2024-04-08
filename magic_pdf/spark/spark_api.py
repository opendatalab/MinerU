
"""
用户输入：
    model数组，每个元素代表一个页面
    pdf在s3的路径
    截图保存的s3位置

然后：
    1）根据s3路径，调用spark集群的api,拿到ak,sk,endpoint，构造出s3PDFReader
    2）根据用户输入的s3地址，调用spark集群的api,拿到ak,sk,endpoint，构造出s3ImageWriter

其余部分至于构造s3cli, 获取ak,sk都在code-clean里写代码完成。不要反向依赖！！！

"""


from magic_pdf.io import AbsReaderWriter


def parse_txt_pdf(pdf_bytes:bytes, pdf_models:list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0, *args, **kwargs):
    """
    解析文本类pdf
    """
    pass


def parse_ocr_pdf(pdf_bytes:bytes,  pdf_models:list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0, *args, **kwargs):
    """
    解析ocr类pdf
    """
    pass


def parse_union_pdf(pdf_bytes:bytes,  pdf_models:list, imageWriter: AbsReaderWriter, is_debug=False, start_page=0,  *args, **kwargs):
    """
    ocr和文本混合的pdf，全部解析出来
    """
    pass