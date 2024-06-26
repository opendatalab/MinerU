import zipfile
import os
import shutil
code_path = os.environ.get('GITHUB_WORKSPACE')
pdf_dev_path = "/home/quyuan/data"
pdf_res_path = "/home/quyuan/code/Magic-PDF/Magic-PDF/Magic-PDF/ci/magic-pdf"
def test_cli():
    magicpdf_path = os.path.join(pdf_dev_path, "output")
    if not os.path.exists(magicpdf_path):
        os.makedirs(magicpdf_path)
    cmd = 'cd %s && export PYTHONPATH=. && find %s -type f -name "*.pdf" | xargs -I{} python magic_pdf/cli/magicpdf.py  pdf-command  --pdf {}' % (code_path, magicpdf_path)
    os.system(cmd)
   
    for annotaion_name in os.walk(os.path.join(pdf_dev_path, "ci")):
        if annotaion_name.endswith('.md'):
            for pdf_res_path  in os.listdir(pdf_res_path):
                if annotaion_name in os.path.join(pdf_res_path, annotaion_name, "auto"):
                    prefix = annotaion_name.split('_')[-2]
                    if not os.path.exists(os.join(pdf_dev_path, prefix)):
                        #os.makedirs(os.path.join(pdf_dev_path, prefix))
                        shutil.copy(os.path.join(pdf_res_path, annotaion_name.strip(".md"), "auto", annotaion_name), os.join(pdf_dev_path, "ci", prefix, annotaion_name))
                   

def calculate_score():
    cmd = "cd %s && export PYTHONPATH=. && python tools/clean_photo.py --tool_name annotations --download_dir %s" % (code_path, pdf_dev_path)
    os.system(cmd)
    cmd = "cd %s && export PYTHONPATH=. && python tools/clean_photo.py --tool_name magicpdf --download_dir %s" % (code_path, pdf_dev_path)
    os.system(cmd)
    cmd = "cd %s && export PYTHONPATH=. && python tools/markdown_calculate.py --tool_name pdf-command --download_dir %s --results %s" % (code_path, pdf_dev_path, os.path.join(pdf_dev_path, "result.json"))
    os.system(cmd)


def extrat_zip(zip_file_path, extract_to_path):
    if zipfile.is_zipfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f'Files extracted to {extract_to_path}')
    else:
        print(f'{zip_file_path} is not a zip file')


if __name__ == "__main__":
    extrat_zip(os.path.join(pdf_dev_path, 'output.zip'), os.path.join(pdf_dev_path,'datasets'))
    test_cli()
    calculate_score()
