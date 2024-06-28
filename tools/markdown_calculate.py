import os  
from Levenshtein import distance  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize  
import json 
import re
import scoring
import argparse

# 初始化列表来存储编辑距离和BLEU分数  
class Scoring:
    def __init__(self, result_path):
        self.edit_distances = []
        self.bleu_scores = []
        self.sim_scores = []
        self.filenames = []
        self.score_dict = {}
        self.anntion_cnt = 0
        self.fw = open(result_path, "w+")
    def simple_bleu_score(self, candidate, reference):  
        candidate_tokens = word_tokenize(candidate)  
        reference_tokens = word_tokenize(reference) 
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method1) 


    def preprocess_string(self, s):  
        sub_enter = re.sub(r'\n+', '\n', s)
        return re.sub(r'  ', ' ', sub_enter)
    
    def calculate_similarity(self, annotion, actual, tool_type):
        class_dict = {}
        edit_distances = []
        bleu_scores = []
        sim_scores = list()
        total_file = 0
        for filename in os.listdir(annotion):  
            if filename.endswith('.md') and not filename.startswith('.'):  # 忽略隐藏文件  
                total_file = total_file + 1
                # 读取A目录中的文件  
                with open(os.path.join(annotion, filename), 'r', encoding='utf-8') as file_a:  
                    content_a = file_a.read()
                self.anntion_cnt = self.anntion_cnt + 1
                filepath_b = os.path.join(actual, filename)  
                if os.path.exists(filepath_b):  
                    with open(filepath_b, 'r', encoding='utf-8') as file_b:  
                        content_b = file_b.read()
                        self.filenames.append(filename)
                        # 计算编辑距离
                        edit_dist = distance(self.preprocess_string(content_b),self.preprocess_string(content_a)) / max(len(content_a), len(content_b))
                        self.edit_distances.append(edit_dist)  
                        edit_distances.append(edit_dist)
                        #计算BLUE分数
                        bleu_score = self.simple_bleu_score(content_b, content_a)  
                        bleu_scores.append(bleu_score)
                        self.bleu_scores.append(bleu_score)  
                        #计算marker分数
                        score = scoring.score_text(content_b, content_a)
                        sim_scores.append(score)
                        self.sim_scores.append(score)
                        class_dict[filename] = {"edit_dist": edit_dist, "bleu_score": bleu_score, "sim_score": score}
                        self.score_dict[filename] = {"edit_dist": edit_dist, "bleu_score": bleu_score, "sim_score": score}
                else:  
                    print(f"File {filename} not found in actual directory.")  
        # 计算每类平均值
        class_average_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0  
        class_average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0  
        class_average_sim_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        self.fw.write(json.dumps(class_dict, ensure_ascii=False) + "\n")
        ratio = len(class_dict)/total_file
        self.fw.write(f"{tool_type} extract ratio:  {ratio}" + "\n")
        self.fw.write(f"{tool_type} Average Levenshtein Distance: {class_average_edit_distance}" + "\n")
        self.fw.write(f"{tool_type} Average BLEU Score: {class_average_bleu_score}" + "\n")
        self.fw.write(f"{tool_type} Average Sim Score: {class_average_sim_score}" + "\n")

        print (f"{tool_type} extract ratio: {ratio}")
        print (f"{tool_type} Average Levenshtein Distance: {class_average_edit_distance}")
        print (f"{tool_type} Average BLEU Score: {class_average_bleu_score}")
        print (f"{tool_type} Average Sim Score: {class_average_sim_score}")
        return self.score_dict
    
    def summary_scores(self):
         # 计算整体平均值
        over_all_dict = dict()
        average_edit_distance = sum(self.edit_distances) / len(self.edit_distances) if self.edit_distances else 0  
        average_bleu_score = sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0  
        average_sim_score = sum(self.sim_scores) / len(self.sim_scores) if self.sim_scores else 0
        over_all_dict["average_edit_distance"] = average_edit_distance
        over_all_dict["average_bleu_score"] = average_bleu_score
        over_all_dict["average_sim_score"] = average_sim_score
        self.fw.write(json.dumps(over_all_dict, ensure_ascii=False) + "\n")
        return over_all_dict

    def calculate_similarity_total(self, tool_type, file_types, download_dir):
        for file_type in file_types:
            annotion = os.path.join(download_dir, file_type, "annotations", "cleaned")
            actual = os.path.join(download_dir, file_type, tool_type, "cleaned")
            self.calculate_similarity(annotion, actual, file_type)

