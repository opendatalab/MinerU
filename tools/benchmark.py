import zipfile
import os
import shutil
code_path = os.environ.get('GITHUB_WORKSPACE')
#code_path = "/home/quyuan/actions-runner/_work/Magic-PDF/Magic-PDF.bk"
#评测集存放路径
pdf_dev_path = "/home/quyuan/data"
#magicpdf跑测结果
pdf_res_path = "/home/quyuan/code/Magic-PDF/Magic-PDF/Magic-PDF/ci/magic-pdf"
file_types = ["academic_literature", "atlas", "courseware", "colorful_textbook", "historical_documents", "notes", "ordinary_books", "ordinary_exam_paper", "ordinary_textbook", "research_report", "special_exam_paper"]
#file_types = ["academic_literature"]

def test_cli():
    magicpdf_path = os.path.join(pdf_dev_path, "output")
    cmd = 'cd %s && export PYTHONPATH=. && find %s -type f -name "*.pdf" | xargs -I{} python magic_pdf/cli/magicpdf.py  pdf-command  --pdf {}' % (code_path, magicpdf_path)
    os.system(cmd)
    for root, dirs, files in os.walk(pdf_res_path):
         for magic_file in files:
            for file_type in file_types:
                target_dir = os.path.join(pdf_dev_path, "ci", file_type, "magicpdf")
                if magic_file.endswith(".md") and magic_file.startswith(file_type):
                    source_file = os.path.join(root, magic_file)
                    target_file = os.path.join(pdf_dev_path, "ci", file_type, "magicpdf", magic_file)
                    if not os.path.exists(target_dir):
                         os.makedirs(target_dir) 
                    shutil.copy(source_file, target_file)   

def calculate_score():
    data_path = os.path.join(pdf_dev_path, "ci")
    cmd = "cd %s && export PYTHONPATH=. && python tools/clean_photo.py --tool_name annotations --download_dir %s" % (code_path, data_path)
    os.system(cmd)
    cmd = "cd %s && export PYTHONPATH=. && python tools/clean_photo.py --tool_name magicpdf --download_dir %s" % (code_path, data_path)
    os.system(cmd)
    cmd = "cd %s && export PYTHONPATH=. && python tools/markdown_calculate.py --tool_name magicpdf --download_dir %s --results %s" % (code_path, data_path, os.path.join(data_path, "result.json"))
    os.system(cmd)


def extrat_zip(zip_file_path, extract_to_path):
    if zipfile.is_zipfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f'Files extracted to {extract_to_path}')
    else:
        print(f'{zip_file_path} is not a zip file')


if __name__ == "__main__":
    extrat_zip(os.path.join(pdf_dev_path, 'output.zip'), os.path.join(pdf_dev_path))
    test_cli()
    calculate_score()
