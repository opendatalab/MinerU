from pathlib import Path

import click
import json

from demo.pdf2md import main


@click.command()
@click.option("--pdf-file-path", help="s3上pdf文件的路径")
@click.option("--pdf-name", help="pdf name")
def main_shell(pdf_file_path: str, pdf_name: str):
    with open('/mnt/petrelfs/share_data/ouyanglinke/OCR/OCR_validation_dataset_final_rotated_formulafix_highdpi_scihub.json', 'r') as f:
        samples = json.load(f)
    for sample in samples:
        pdf_file_path = sample['s3_path']
        pdf_bin_file_profile = "outsider"
        pdf_name = sample['pdf_name']
        pdf_model_dir = f"s3://llm-pdf-text/eval_1k/layout_res/{pdf_name}"
        pdf_model_profile = "langchao"

        p = Path(pdf_file_path)
        pdf_file_name = p.name  # pdf文件名字，含后缀

        #pdf_model_dir = join_path(pdf_model_parent_dir, pdf_file_name)

        main(
            pdf_file_path,
            pdf_bin_file_profile,
            pdf_model_dir,
            pdf_model_profile,
            debug_mode=True,
        )
    
    
if __name__ == "__main__":
    main_shell()
