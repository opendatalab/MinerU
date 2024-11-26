"""
去掉正文的引文引用marker
https://aicarrier.feishu.cn/wiki/YLOPwo1PGiwFRdkwmyhcZmr0n3d
"""
import re
# from magic_pdf.libs.nlp_utils import NLPModels


# __NLP_MODEL = NLPModels()

def check_1(spans, cur_span_i):
    """寻找前一个char,如果是句号，逗号，那么就是角标"""
    if cur_span_i==0:
        return False # 不是角标
    pre_span = spans[cur_span_i-1]
    pre_char = pre_span['chars'][-1]['c']
    if pre_char in ['。', '，', '.', ',']:
        return True
    
    return False


# def check_2(spans, cur_span_i):
#     """检查前面一个span的最后一个单词，如果长度大于5，全都是字母，并且不含大写，就是角标"""
#     pattern = r'\b[A-Z]\.\s[A-Z][a-z]*\b' # 形如A. Bcde, L. Bcde, 人名的缩写
#
#     if cur_span_i==0 and len(spans)>1:
#         next_span = spans[cur_span_i+1]
#         next_txt = "".join([c['c'] for c in next_span['chars']])
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(next_txt)
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         if re.findall(pattern, next_txt):
#             return True
#
#         return False # 不是角标
#     elif cur_span_i==0 and len(spans)==1: # 角标占用了整行？谨慎删除
#         return False
#
#     # 如果这个span是最后一个span,
#     if cur_span_i==len(spans)-1:
#         pre_span = spans[cur_span_i-1]
#         pre_txt = "".join([c['c'] for c in pre_span['chars']])
#         pre_word = pre_txt.split(' ')[-1]
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(pre_txt)
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         if re.findall(pattern, pre_txt):
#             return True
#
#         return len(pre_word) > 5 and pre_word.isalpha() and pre_word.islower()
#     else: # 既不是第一个span，也不是最后一个span，那么此时检查一下这个角标距离前后哪个单词更近就属于谁的角标
#         pre_span = spans[cur_span_i-1]
#         next_span = spans[cur_span_i+1]
#         cur_span = spans[cur_span_i]
#         # 找到前一个和后一个span里的距离最近的单词
#         pre_distance = 10000 # 一个很大的数
#         next_distance = 10000 # 一个很大的数
#         for c in pre_span['chars'][::-1]:
#             if c['c'].isalpha():
#                 pre_distance = cur_span['bbox'][0] - c['bbox'][2]
#                 break
#         for c in next_span['chars']:
#             if c['c'].isalpha():
#                 next_distance = c['bbox'][0] - cur_span['bbox'][2]
#                 break
#
#         if pre_distance<next_distance:
#             belong_to_span = pre_span
#         else:
#             belong_to_span = next_span
#
#         txt = "".join([c['c'] for c in belong_to_span['chars']])
#         pre_word = txt.split(' ')[-1]
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(txt)
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         if re.findall(pattern, txt):
#             return True
#
#         return len(pre_word) > 5 and pre_word.isalpha() and pre_word.islower()


def check_3(spans, cur_span_i):
    """上标里有[], 有*， 有-， 有逗号"""
    # 如[2-3],[22]  
    # 如 2,3,4
    cur_span_txt = ''.join(c['c'] for c in spans[cur_span_i]['chars']).strip()
    bad_char = ['[', ']', '*', ',']

    if any([c in cur_span_txt for c in bad_char]) and any(character.isdigit() for character in cur_span_txt):
        return True

    # 如2-3, a-b
    patterns = [r'\d+-\d+', r'[a-zA-Z]-[a-zA-Z]', r'[a-zA-Z],[a-zA-Z]']
    for pattern in patterns:  
        match = re.match(pattern, cur_span_txt)
        if match is not None:
            return True

    return False


def remove_citation_marker(with_char_text_blcoks):
    for blk in with_char_text_blcoks:
        for line in blk['lines']:
            # 如果span里的个数少于2个，那只能忽略，角标不可能自己独占一行
            if len(line['spans'])<=1:
                continue

            # 找到高度最高的span作为位置比较的基准
            max_hi_span = line['spans'][0]['bbox']
            min_font_sz = 10000 # line里最小的字体
            max_font_sz = 0   # line里最大的字体
                
            for s in line['spans']:
                if max_hi_span[3]-max_hi_span[1]<s['bbox'][3]-s['bbox'][1]:
                    max_hi_span = s['bbox']
                if min_font_sz>s['size']:
                    min_font_sz = s['size']
                if max_font_sz<s['size']:
                    max_font_sz = s['size']
                        
            base_span_mid_y = (max_hi_span[3]+max_hi_span[1])/2
            
            
            span_to_del = []
            for i, span in enumerate(line['spans']):
                span_hi = span['bbox'][3]-span['bbox'][1]
                span_mid_y = (span['bbox'][3]+span['bbox'][1])/2
                span_font_sz = span['size']
                
                if max_font_sz-span_font_sz<1: # 先以字体过滤正文，如果是正文就不再继续判断了
                    continue

                # 对被除数为0的情况进行过滤
                if span_hi==0 or min_font_sz==0:
                    continue

                if (base_span_mid_y-span_mid_y)/span_hi>0.2 or (base_span_mid_y-span_mid_y>0 and abs(span_font_sz-min_font_sz)/min_font_sz<0.1):
                    """
                    1. 它的前一个char如果是句号或者逗号的话，那么肯定是角标而不是公式
                    2. 如果这个角标的前面是一个单词（长度大于5）而不是任何大写或小写的短字母的话 应该也是角标
                    3. 上标里有数字和逗号或者数字+星号的组合，方括号，一般肯定就是角标了
                    4. 这个角标属于前文还是后文要根据距离来判断，如果距离前面的文本太近，那么就是前面的角标，否则就是后面的角标
                    """
                    if (check_1(line['spans'], i) or
                        # check_2(line['spans'], i) or
                        check_3(line['spans'], i)
                    ):
                        """删除掉这个角标：删除这个span, 同时还要更新line的text"""
                        span_to_del.append(span)
            if len(span_to_del)>0:
                for span in span_to_del:
                    line['spans'].remove(span)
                line['text'] = ''.join([c['c'] for s in line['spans'] for c in s['chars']])
    
    return with_char_text_blcoks
