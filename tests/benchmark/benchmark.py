"""
bench
"""
import os
import shutil
import json
import calculate_score
code_path = os.environ.get('GITHUB_WORKSPACE')
#评测集存放路径
pdf_dev_path = "datasets/"
#magicpdf跑测结果
pdf_res_path = "/tmp/magic-pdf"

def test_cli():
    """
    test pdf-command cli
    """
    rm_cmd = f"rm -rf {pdf_res_path}"
    os.system(rm_cmd)
    os.makedirs(pdf_res_path)
    cmd = f'magic-pdf pdf-command --pdf {os.path.join(pdf_dev_path, "mineru")} --inside_model true'
    os.system(cmd)
    for root, dirs, files in os.walk(pdf_res_path):
         for magic_file in files:
            target_dir = os.path.join(pdf_dev_path, "mineru")
            if magic_file.endswith(".md"):
                source_file = os.path.join(root, magic_file)
                target_file = os.path.join(pdf_dev_path, "mineru", magic_file)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir) 
                shutil.copy(source_file, target_file)

def get_score():
    """
    get score
    """
    data_path = os.path.join(pdf_dev_path, "ci")
    score = calculate_score.Scoring(os.path.join(data_path, "result.json"))
    score.calculate_similarity_total("mineru", data_path)
    res = score.summary_scores()
    return res


def ci_ben():
    """
    ci benchmark
    """
    try:
        fr = open(os.path.join(pdf_dev_path, "result.json"), "r", encoding="utf-8")
        lines = fr.readlines()
        last_line = lines[-1].strip()
        last_score = json.loads(last_line)
        print ("last_score:", last_score)
        last_simscore = last_score["average_sim_score"]
        last_editdistance = last_score["average_edit_distance"]
        last_bleu = last_score["average_bleu_score"]
    except IOError:
        print ("result.json not exist")
    test_cli()
    os.system(f"python pre_clean.py --tool_name mineru --download_dir {pdf_dev_path}")
    now_score = get_score()
    print ("now_score:", now_score)
    now_simscore = now_score["average_sim_score"]
    now_editdistance = now_score["average_edit_distance"]
    now_bleu = now_score["average_bleu_score"]
    assert last_simscore <= now_simscore
    assert last_editdistance <= now_editdistance
    assert last_bleu <= now_bleu


if __name__ == "__main__":
    os.system("sh env.sh")
    ci_ben()
