

class MagicModel():
    """
    每个函数没有得到元素的时候返回空list
    
    """
    def __fix_axis():
        # TODO 计算
        self.__model_list = xx
        
    def __init__(model_list:list, page:Page):
        self.__model_list = model_list
        self.__fix_axis()
        self.__page = page
        
    def get_imgs(self, page_no:int): # @许瑞
        
        image_block = {
            
        }
        image_block['bbox'] = [x0, y0, x1, y1]# 计算出来
        image_block['img_body_bbox'] = [x0, y0, x1, y1]
        image_blcok['img_caption_bbox'] =  [x0, y0, x1, y1] # 如果没有就是None，但是保证key存在
        image_blcok['img_caption_text']=  [x0, y0, x1, y1] # 如果没有就是空字符串，但是保证key存在
        
        
        return [image_block,]
        
    def get_tables(self, page_no:int) ->list: # 3个坐标， caption, table主体，table-note
        pass # 许瑞, 结构和image一样
        
    def get_equations(self, page_no:int)->list: # 有坐标，也有字
        return inline_equations, interline_equations  # @凯文
        
    def get_discarded(self, page_no:int)->list: # 自研模型，只有坐标 
        pass # @凯文
        
    def get_text_blocks(self, page_no:int)->list: # 自研模型搞的，只有坐标，没有字
        pass # @凯文
        
    def get_title_blocks(self, page_no:int)->list: # 自研模型，只有坐标，没字
        pass # @凯文
        
    def get_ocr_text(self, page_no:int)->list: # paddle 搞的，有字也有坐标
        pass  # @小蒙
        
    def get_ocr_spans(self, page_no:int)->list:
        pass   # @小蒙
       