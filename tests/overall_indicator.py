import json
import pandas as pd
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu
import time
import argparse
import os
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
from sklearn import metrics
from pandas import isnull


def indicator_cal(json_standard,json_test):

    json_standard = pd.DataFrame(json_standard)
    json_test = pd.DataFrame(json_test)



    '''数据集总体指标'''
    
    a=json_test[['id','mid_json']]
    b=json_standard[['id','mid_json','pass_label']]
    outer_merge=pd.merge(a,b,on='id',how='outer')
    outer_merge.columns=['id','standard_mid_json','test_mid_json','pass_label']
    standard_exist=outer_merge.standard_mid_json.apply(lambda x: not isnull(x))
    test_exist=outer_merge.test_mid_json.apply(lambda x: not isnull(x))

    overall_report = {}
    overall_report['accuracy']=metrics.accuracy_score(standard_exist,test_exist)
    overall_report['precision']=metrics.precision_score(standard_exist,test_exist)
    overall_report['recall']=metrics.recall_score(standard_exist,test_exist)
    overall_report['f1_score']=metrics.f1_score(standard_exist,test_exist)


    inner_merge=pd.merge(a,b,on='id',how='inner')
    inner_merge.columns=['id','standard_mid_json','test_mid_json','pass_label']
    json_standard = inner_merge['standard_mid_json']#check一下是否对齐
    json_test = inner_merge['test_mid_json']


    

    '''批量读取中间生成的json文件'''
    test_inline_equations=[]
    test_interline_equations=[]
    test_inline_euqations_bboxs=[]
    test_interline_equations_bboxs=[]
    test_dropped_text_bboxes=[]
    test_dropped_text_tag=[]
    test_dropped_image_bboxes=[]
    test_dropped_table_bboxes=[] 
    test_preproc_num=[]#阅读顺序
    test_para_num=[]
    test_para_text=[]

    for i in json_test:
        mid_json=pd.DataFrame(i)
        mid_json=mid_json.iloc[:,:-1]
        for j1 in mid_json.loc['inline_equations',:]:
            page_in_text=[]
            page_in_bbox=[]
            for k1 in j1:
                page_in_text.append(k1['latex_text'])
                page_in_bbox.append(k1['bbox'])
            test_inline_equations.append(page_in_text)
            test_inline_euqations_bboxs.append(page_in_bbox)
        for j2 in mid_json.loc['interline_equations',:]:
            page_in_text=[]
            page_in_bbox=[]
            for k2 in j2:
                page_in_text.append(k2['latex_text'])
            test_interline_equations.append(page_in_text)
            test_interline_equations_bboxs.append(page_in_bbox)

        for j3 in mid_json.loc['droped_text_block',:]:
            page_in_bbox=[]
            page_in_tag=[]
            for k3 in j3:
                page_in_bbox.append(k3['bbox'])
                #如果k3中存在tag这个key
                if 'tag' in k3.keys():
                    page_in_tag.append(k3['tag'])
                else:
                    page_in_tag.append('None')
            test_dropped_text_tag.append(page_in_tag)
            test_dropped_text_bboxes.append(page_in_bbox)
        for j4 in mid_json.loc['droped_image_block',:]:
                test_dropped_image_bboxes.append(j4)
        for j5 in mid_json.loc['droped_table_block',:]:
                test_dropped_table_bboxes.append(j5)
        for j6 in mid_json.loc['preproc_blocks',:]:
            page_in=[]
            for k6 in j6:
                page_in.append(k6['number'])
            test_preproc_num.append(page_in)

        test_pdf_text=[]     
        for j7 in mid_json.loc['para_blocks',:]:
            test_para_num.append(len(j7))  
            for k7 in j7:
                test_pdf_text.append(k7['text'])  
        test_para_text.append(test_pdf_text)



    standard_inline_equations=[]
    standard_interline_equations=[]
    standard_inline_euqations_bboxs=[]
    standard_interline_equations_bboxs=[]
    standard_dropped_text_bboxes=[]
    standard_dropped_text_tag=[]
    standard_dropped_image_bboxes=[]
    standard_dropped_table_bboxes=[] 
    standard_preproc_num=[]#阅读顺序
    standard_para_num=[]
    standard_para_text=[]

    for i in json_standard:
        mid_json=pd.DataFrame(i)
        mid_json=mid_json.iloc[:,:-1]
        for j1 in mid_json.loc['inline_equations',:]:
            page_in_text=[]
            page_in_bbox=[]
            for k1 in j1:
                page_in_text.append(k1['latex_text'])
                page_in_bbox.append(k1['bbox'])
            standard_inline_equations.append(page_in_text)
            standard_inline_euqations_bboxs.append(page_in_bbox)
        for j2 in mid_json.loc['interline_equations',:]:
            page_in_text=[]
            page_in_bbox=[]
            for k2 in j2:
                page_in_text.append(k2['latex_text'])
                page_in_bbox.append(k2['bbox'])
            standard_interline_equations.append(page_in_text)
            standard_interline_equations_bboxs.append(page_in_bbox)
        for j3 in mid_json.loc['droped_text_block',:]:
            page_in_bbox=[]
            page_in_tag=[]
            for k3 in j3:
                page_in_bbox.append(k3['bbox'])
                if 'tag' in k3.keys():
                    page_in_tag.append(k3['tag'])
                else:
                    page_in_tag.append('None')
            standard_dropped_text_bboxes.append(page_in_bbox)
            standard_dropped_text_tag.append(page_in_tag)
        for j4 in mid_json.loc['droped_image_block',:]:
                standard_dropped_image_bboxes.append(j4)
        for j5 in mid_json.loc['droped_table_block',:]:
                standard_dropped_table_bboxes.append(j5)
        for j6 in mid_json.loc['preproc_blocks',:]:
            page_in=[]
            for k6 in j6:
                page_in.append(k6['number'])
            standard_preproc_num.append(page_in)     

        standard_pdf_text=[]
        for j7 in mid_json.loc['para_blocks',:]:
            standard_para_num.append(len(j7))  
            for k7 in j7:
                standard_pdf_text.append(k7['text'])
        standard_para_text.append(standard_pdf_text)


    """
    在计算指标之前最好先确认基本统计信息是否一致
    """


    '''
    计算pdf之间的总体编辑距离和bleu
    这里只计算正例的pdf
    '''
    
    test_para_text=np.asarray(test_para_text, dtype = object)[inner_merge['pass_label']=='yes']
    standard_para_text=np.asarray(standard_para_text, dtype = object)[inner_merge['pass_label']=='yes']

    pdf_dis=[]
    pdf_bleu=[]
    for a,b in zip(test_para_text,standard_para_text):
        a1=[ ''.join(i) for i in a]
        b1=[ ''.join(i) for i in b]
        pdf_dis.append(Levenshtein_Distance(a1,b1))
        pdf_bleu.append(sentence_bleu([a1],b1))
    overall_report['pdf间的平均编辑距离']=np.mean(pdf_dis)
    overall_report['pdf间的平均bleu']=np.mean(pdf_bleu)


    '''行内公式编辑距离和bleu'''
    dis1=[]
    bleu1=[]

    test_inline_equations=[ ''.join(i) for i in test_inline_equations]
    standard_inline_equations=[ ''.join(i) for i in standard_inline_equations]
           
    for a,b in zip(test_inline_equations,standard_inline_equations):
        if len(a)==0 and len(b)==0:
            continue
        else:
            if a==b:
                dis1.append(0)
                bleu1.append(1)
            else:
                dis1.append(Levenshtein_Distance(a,b))
                bleu1.append(sentence_bleu([a],b))
    inline_equations_edit=np.mean(dis1)
    inline_equations_bleu=np.mean(bleu1)

    '''行内公式bbox匹配相关指标'''
    inline_equations_bbox_report=bbox_match_indicator(test_inline_euqations_bboxs,standard_inline_euqations_bboxs)


    '''行间公式编辑距离和bleu'''
    dis2=[]
    bleu2=[]

    test_interline_equations=[ ''.join(i) for i in test_interline_equations]
    standard_interline_equations=[ ''.join(i) for i in standard_interline_equations]

    for a,b in zip(test_interline_equations,standard_interline_equations):
        if len(a)==0 and len(b)==0:
            continue
        else:
            if a==b:
                dis2.append(0)
                bleu2.append(1)
            else:
                dis2.append(Levenshtein_Distance(a,b))
                bleu2.append(sentence_bleu([a],b))
    interline_equations_edit=np.mean(dis2)
    interline_equations_bleu=np.mean(bleu2)


    '''行间公式bbox匹配相关指标'''
    interline_equations_bbox_report=bbox_match_indicator(test_interline_equations_bboxs,standard_interline_equations_bboxs)




    '''可以先检查page和bbox数量是否一致'''

    '''dropped_text_block的bbox匹配相关指标'''
    test_text_bbox=[]
    standard_text_bbox=[]
    test_tag=[]
    standard_tag=[]


    index=0
    for a,b in zip(test_dropped_text_bboxes,standard_dropped_text_bboxes):
        test_page_tag=[]
        standard_page_tag=[]
        test_page_bbox=[]
        standard_page_bbox=[]
        if len(a)==0 and len(b)==0:
            pass
        else:
            for i in range(len(b)):
                judge=0
                standard_page_tag.append(standard_dropped_text_tag[index][i])
                standard_page_bbox.append(1)
                for j in range(len(a)):
                    if bbox_offset(b[i],a[j]):
                        judge=1
                        test_page_tag.append(test_dropped_text_tag[index][j])
                        test_page_bbox.append(1)
                        break
                if judge==0:
                    test_page_tag.append('None')
                    test_page_bbox.append(0)


            if len(test_dropped_text_tag[index])+test_page_tag.count('None')>len(standard_dropped_text_tag[index]):#有多删的情况出现
                test_page_tag1=test_page_tag.copy()
                if 'None' in test_page_tag:
                    test_page_tag1=test_page_tag1.remove('None')
                else:
                    test_page_tag1=test_page_tag

                diff=list((Counter(test_dropped_text_tag[index]) - Counter(test_page_tag1)).elements())
              
                test_page_tag.extend(diff)
                standard_page_tag.extend(['None']*len(diff))
                test_page_bbox.extend([1]*len(diff))
                standard_page_bbox.extend([0]*len(diff))

            test_tag.extend(test_page_tag)
            standard_tag.extend(standard_page_tag)
            test_text_bbox.extend(test_page_bbox)
            standard_text_bbox.extend(standard_page_bbox)

        index+=1

    
    text_block_report = {}
    text_block_report['accuracy']=metrics.accuracy_score(standard_text_bbox,test_text_bbox)
    text_block_report['precision']=metrics.precision_score(standard_text_bbox,test_text_bbox)
    text_block_report['recall']=metrics.recall_score(standard_text_bbox,test_text_bbox)
    text_block_report['f1_score']=metrics.f1_score(standard_text_bbox,test_text_bbox)

    '''删除的text_block的tag的准确率,召回率和f1-score'''
    text_block_tag_report = classification_report(y_true=standard_tag , y_pred=test_tag,output_dict=True)
    del text_block_tag_report['None']
    del text_block_tag_report["macro avg"]
    del text_block_tag_report["weighted avg"]


    '''dropped_image_block的bbox匹配相关指标'''
    '''有数据格式不一致的问题'''
    image_block_report=bbox_match_indicator(test_dropped_image_bboxes,standard_dropped_image_bboxes)
    
    
    '''dropped_table_block的bbox匹配相关指标'''
    table_block_report=bbox_match_indicator(test_dropped_table_bboxes,standard_dropped_table_bboxes)
    
   
    '''阅读顺序编辑距离的均值'''
    preproc_num_dis=[]
    for a,b in zip(test_preproc_num,standard_preproc_num):
        preproc_num_dis.append(Levenshtein_Distance(a,b))
    preproc_num_edit=np.mean(preproc_num_dis)



    '''分段准确率'''
    test_para_num=np.array(test_para_num)
    standard_para_num=np.array(standard_para_num)
    acc_para=np.mean(test_para_num==standard_para_num)

    
    output=pd.DataFrame()
    output['总体指标']=[overall_report]
    output['行内公式平均编辑距离']=[inline_equations_edit]
    output['行间公式平均编辑距离']=[interline_equations_edit]
    output['行内公式平均bleu']=[inline_equations_bleu]
    output['行间公式平均bleu']=[interline_equations_bleu]
    output['行内公式识别相关指标']=[inline_equations_bbox_report]
    output['行间公式识别相关指标']=[interline_equations_bbox_report]
    output['阅读顺序平均编辑距离']=[preproc_num_edit]
    output['分段准确率']=[acc_para]
    output['删除的text block的相关指标']=[text_block_report]
    output['删除的image block的相关指标']=[image_block_report]
    output['删除的table block的相关指标']=[table_block_report]
    output['删除的text block的tag相关指标']=[text_block_tag_report]
    

    return output

"""
计算编辑距离
"""
def Levenshtein_Distance(str1, str2):
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]


'''
计算bbox偏移量是否符合标准的函数
'''
def bbox_offset(b_t,b_s):
    '''b_t是test_doc里的bbox,b_s是standard_doc里的bbox'''
    x1_t,y1_t,x2_t,y2_t=b_t
    x1_s,y1_s,x2_s,y2_s=b_s
    x1=max(x1_t,x1_s)
    x2=min(x2_t,x2_s)
    y1=max(y1_t,y1_s)
    y2=min(y2_t,y2_s)
    area_overlap=(x2-x1)*(y2-y1)
    area_t=(x2_t-x1_t)*(y2_t-y1_t)+(x2_s-x1_s)*(y2_s-y1_s)-area_overlap
    if area_t-area_overlap==0 or area_overlap/(area_t-area_overlap)>0.95:
        return True
    else:
        return False
    

'''bbox匹配和对齐函数，输出相关指标'''
'''输入的是以page为单位的bbox列表'''
def bbox_match_indicator(test_bbox_list,standard_bbox_list):
    
    test_bbox=[]
    standard_bbox=[]
    for a,b in zip(test_bbox_list,standard_bbox_list):

        test_page_bbox=[]
        standard_page_bbox=[]
        if len(a)==0 and len(b)==0:
            pass
        else:
            for i in b:
                if len(i)!=4:
                    continue
                else:
                    judge=0
                    standard_page_bbox.append(1)
                    for j in a:
                        if bbox_offset(i,j):
                            judge=1
                            test_page_bbox.append(1)
                            break
                    if judge==0:
                        test_page_bbox.append(0)
                        
            diff_num=len(a)+test_page_bbox.count(0)-len(b)
            if diff_num>0:#有多删的情况出现
                test_page_bbox.extend([1]*diff_num)
                standard_page_bbox.extend([0]*diff_num)

          
            test_bbox.extend(test_page_bbox)
            standard_bbox.extend(standard_page_bbox)

    
    block_report = {}
    block_report['accuracy']=metrics.accuracy_score(standard_bbox,test_bbox)
    block_report['precision']=metrics.precision_score(standard_bbox,test_bbox)
    block_report['recall']=metrics.recall_score(standard_bbox,test_bbox)
    block_report['f1_score']=metrics.f1_score(standard_bbox,test_bbox)

    return block_report




   
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str)
parser.add_argument('--standard', type=str)
args = parser.parse_args()
pdf_json_test = args.test
pdf_json_standard = args.standard



if __name__ == '__main__':
    
   pdf_json_test = [json.loads(line) 
                        for line in open(pdf_json_test, 'r', encoding='utf-8')]
   pdf_json_standard = [json.loads(line) 
                    for line in open(pdf_json_standard, 'r', encoding='utf-8')]
   
   overall_indicator=indicator_cal(pdf_json_standard,pdf_json_test)

   '''计算的指标输出到overall_indicator_output.json中'''
   overall_indicator.to_json('overall_indicator_output.json',orient='records',lines=True,force_ascii=False)
    