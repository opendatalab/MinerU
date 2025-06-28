import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from mineru.cli.common import do_parse, read_fn, pdf_suffixes
from mineru.utils.enum_class import MakeMode

def convert_pdf_to_md_json(input_folder, output_folder=None):
    """
    将指定文件夹中的所有pdf文件转换为markdown和json文件
    
    参数:
    input_folder(str): 包含pdf文件的输入文件夹路径
    output_folder(str): 输出文件的文件夹路径。如果为None，则在输入文件夹中创建'md_json'子文件夹
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'md_json')
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("**未找到pdf文件**")
        return None

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    progress_bar = tqdm(pdf_files, desc="Processing PDFs", unit="file")
    for pdf_file in progress_bar:
        input_path = os.path.join(input_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        
        try:
            # 使用mineru解析PDF文件

            do_parse(
                output_dir=output_folder,
                pdf_file_names=[base_name],
                pdf_bytes_list=[read_fn(Path(input_path))],
                p_lang_list=["ch"],
                backend="pipeline",
                parse_method="ocr",
                p_formula_enable=True,
                p_table_enable=True,
                f_dump_md=True,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.MM_MD,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False
            )
            
            progress_bar.set_postfix(file=pdf_file, status="Converted")
            
        except Exception as e:
            progress_bar.set_postfix(file=pdf_file, status="Failed", error=str(e))
            print(f"Error processing {pdf_file}: {e}")
            continue

if __name__ == "__main__":
    input_dir = "./pdf_files"
    output_dir = "./md_json_files"
    convert_pdf_to_md_json(input_dir, output_dir)