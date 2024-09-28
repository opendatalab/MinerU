"""将MinerU部署为api服务"""
from magic_pdf.pipe.UNIPipe import UNIPipe
from fastapi import File, UploadFile
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
import json

from magic_pdf.tools.common import prepare_env
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

    

@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    # 添加一个curl命令测试该接口
    
    # 读取上传的PDF文件
    # 检查上传的文件是否为PDF
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "上传的文件不是PDF格式。请上传PDF文件。"}

    pdf_bytes = await file.read()
    output_dir = "./output"
    file_name = file.filename
    parse_method = "auto"
    # 创建内存写入器
    local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
    image_writer, md_writer = DiskReaderWriter(local_image_dir), DiskReaderWriter(local_md_dir)
    
    # 创建UNIPipe实例
    jso_useful_key = {"_pdf_type": "", "model_list": []}
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True)
    
    # 执行解析流程
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    markdown = pipe.pipe_mk_markdown(img_parent_path=local_image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD)
    
    # 将结果转换为JSON
    result = json.dumps(pipe.pdf_mid_data, ensure_ascii=False)
    
    return {"result": result, "markdown": markdown}


# 添加启动API的代码
if __name__ == "__main__":
    import uvicorn
    '''
    """
    使用以下curl命令测试该接口:
    
    curl -X POST "http://localhost:11000/parse" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@/eightT/open_source/lightMinerU/MinerU/demo/small_ocr.pdf"
    注意:
    1. 将 /path/to/your/file.pdf 替换为实际的PDF文件路径
    2. 确保服务器正在运行，且端口号为11000
    3. 如果在远程服务器上运行，请将 localhost 替换为服务器的IP地址或域名
    """
    '''
    # 设置服务器配置
    host = "0.0.0.0"  # 允许外部访问
    port = 11000  # 设置端口号
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port)
    
# 注意: 确保已安装uvicorn库
# 可以通过以下命令安装:
# pip install uvicorn

# 运行此脚本后,可以通过 http://localhost:8000 访问API
# 使用 POST 请求访问 http://localhost:8000/parse 来解析PDF文件


