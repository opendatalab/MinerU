import os
import json
from loguru import logger


if __name__ == '__main__':
    os.system('pip uninstall -y mineru')
    os.system('pip install git+https://github.com/myhloli/Magic-PDF.git@dev')
    os.system('mineru-models-download -s huggingface -m vlm')
    try:
        with open('/home/user/mineru.json', 'r+') as file:
            config = json.load(file)
            
            delimiters = {
                'display': {'left': '\\[', 'right': '\\]'},
                'inline': {'left': '\\(', 'right': '\\)'}
            }
            
            config['latex-delimiter-config'] = delimiters
            
            if os.getenv('apikey'):
                config['llm-aided-config']['title_aided']['api_key'] = os.getenv('apikey')
                config['llm-aided-config']['title_aided']['enable'] = True
            
            file.seek(0)  # 将文件指针移回文件开始位置
            file.truncate()  # 截断文件，清除原有内容
            json.dump(config, file, indent=4)  # 写入新内容
    except Exception as e:
        logger.exception(e)
    os.system('mineru-gradio --enable-sglang-engine true --enable-api false')