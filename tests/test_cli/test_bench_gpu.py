
import json
import os
import shutil

from conf import conf
from lib import calculate_score

pdf_res_path = conf.conf['pdf_res_path']
code_path = conf.conf['code_path']
pdf_dev_path = conf.conf['pdf_dev_path']
class TestCliCuda:
    """test cli cuda."""
    def test_pdf_sdk_cuda(self):
        """pdf sdk cuda."""
        clean_magicpdf(pdf_res_path)
        pdf_to_markdown()
        fr = open(os.path.join(pdf_dev_path, 'result.json'), 'r', encoding='utf-8')
        lines = fr.readlines()
        last_line = lines[-1].strip()
        last_score = json.loads(last_line)
        last_simscore = last_score['average_sim_score']
        last_editdistance = last_score['average_edit_distance']
        last_bleu = last_score['average_bleu_score']
        os.system(f'python tests/test_cli/lib/pre_clean.py --tool_name mineru --download_dir {pdf_dev_path}')
        now_score = get_score()
        print ('now_score:', now_score)
        if not os.path.exists(os.path.join(pdf_dev_path, 'ci')):
            os.makedirs(os.path.join(pdf_dev_path, 'ci'), exist_ok=True)
        fw = open(os.path.join(pdf_dev_path, 'ci', 'result.json'), 'w+', encoding='utf-8')
        fw.write(json.dumps(now_score) + '\n')
        now_simscore = now_score['average_sim_score']
        now_editdistance = now_score['average_edit_distance']
        now_bleu = now_score['average_bleu_score']
        assert last_simscore <= now_simscore
        assert last_editdistance <= now_editdistance
        assert last_bleu <= now_bleu

def pdf_to_markdown():
    """pdf to md."""
    demo_names = list()
    pdf_path = os.path.join(pdf_dev_path, 'pdf')
    for pdf_file in os.listdir(pdf_path):
        if pdf_file.endswith('.pdf'):
            demo_names.append(pdf_file.split('.')[0])
    for demo_name in demo_names:
        pdf_path = os.path.join(pdf_dev_path, 'pdf', f'{demo_name}.pdf')
        cmd = 'magic-pdf pdf-command --pdf %s --inside_model true' % (pdf_path)
        os.system(cmd)
        dir_path = os.path.join(pdf_dev_path, 'mineru')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        res_path = os.path.join(dir_path, f'{demo_name}.md')
        src_path = os.path.join(pdf_res_path, demo_name, 'auto', f'{demo_name}.md')
        shutil.copy(src_path, res_path)



def get_score():
    """get score."""
    score = calculate_score.Scoring(os.path.join(pdf_dev_path, 'result.json'))
    score.calculate_similarity_total('mineru', pdf_dev_path)
    res = score.summary_scores()
    return res


def clean_magicpdf(pdf_res_path):
    """clean magicpdf."""
    cmd = 'rm -rf %s' % (pdf_res_path)
    os.system(cmd)
