"""
这里实现2个click命令：
第一个：
 接收一个完整的s3路径，例如：s3://llm-pdf-text/pdf_ebook_and_paper/pre-clean-mm-markdown/v014/part-660420b490be-000008.jsonl?bytes=0,81350
    1）根据~/magic-pdf.json里的ak,sk等，构造s3cliReader读取到这个jsonl的对应行，返回json对象。
    2）根据Json对象里的pdf的s3路径获取到他的ak,sk,endpoint，构造出s3cliReader用来读取pdf
    3）从magic-pdf.json里读取到本地保存图片、Md等的临时目录位置,构造出LocalImageWriter，用来保存截图
    4）从magic-pdf.json里读取到本地保存图片、Md等的临时目录位置,构造出LocalIRdWriter，用来读写本地文件
    
    最后把以上步骤准备好的对象传入真正的解析API
    
第二个：
  接收1）pdf的本地路径。2）模型json文件（可选）。然后：
    1）根据~/magic-pdf.json读取到本地保存图片、md等临时目录的位置，构造出LocalImageWriter，用来保存截图
    2）从magic-pdf.json里读取到本地保存图片、Md等的临时目录位置,构造出LocalIRdWriter，用来读写本地文件
    3）根据约定，根据pdf本地路径，推导出pdf模型的json，并读入
    

效果：
python magicpdf.py --json  s3://llm-pdf-text/scihub/xxxx.json?bytes=0,81350 
python magicpdf.py --pdf  /home/llm/Downloads/xxxx.pdf --model /home/llm/Downloads/xxxx.json  或者 python magicpdf.py --pdf  /home/llm/Downloads/xxxx.pdf
"""




import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--json', type=str, help='输入一个S3路径')
def json_command(json):
    # 这里处理json相关的逻辑
    print(f'处理JSON: {json}')

@cli.command()
@click.option('--pdf', type=click.Path(exists=True), required=True, help='PDF文件的路径')
@click.option('--model', type=click.Path(exists=True), help='模型的路径')
def pdf_command(pdf, model):
    # 这里处理pdf和模型相关的逻辑
    print(f'处理PDF: {pdf}')
    print(f'加载模型: {model}')

if __name__ == '__main__':
    cli()
