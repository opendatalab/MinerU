import math


def __inc_dict_val(mp, key, val_inc:int):
    if mp.get(key):
        mp[key] = mp[key] + val_inc
    else:
        mp[key] = val_inc
        
    

def get_text_block_base_info(block):
    """
    获取这个文本块里的字体的颜色、字号、字体
    按照正文字数最多的返回
    """
    
    counter = {}
    
    for line in block['lines']:
        for span in line['spans']:
            color = span['color']
            size = round(span['size'], 2)
            font = span['font']
            
            txt_len = len(span['text'])
            __inc_dict_val(counter, (color, size, font), txt_len)
            
    
    c, s, ft = max(counter, key=counter.get)
    
    return c, s, ft
    