# 最终版：把那种text_block有重叠，且inline_formula位置在重叠部分的，认定整个页面都有问题，所有的inline_formula都改成no_check
from pdf_tools.libs import fitz


def check_inline_formula(page, inline_formula_boxes):
    """
    :param page :fitz读取的当前页的内容
    :param inline_formula_boxes: list类型，每一个元素是一个元祖 (L, U, R, D)

    :return: inline_formula_check: list类型，每一个元素是一个类别，其顺序对应输入的inline_formula_boxes，给每个行内公式打一个标签，包括：
        - nocheck_inline_formula：这个公式框没有与任何span相交，有可能存在问题
        - wrong_text_block：这个公式框同时存在多个block里，可能页面的text block存在问题
        - false_inline_formula：只涉及一个span并且只占据这个span的小部分面积，判断可能不是公式
        - true_inline_formula：两种情况判断为公式，一是横跨多个span，二是只涉及一个span但是几乎占据了这个span大部分的面积
    """

    # count = defaultdict(int)
    ## ------------------------ Text --------------------------------------------
    blocks = page.get_text(
            "dict",
            flags=fitz.TEXTFLAGS_TEXT,
            #clip=clip,
        )["blocks"]
    
    # iterate over the bboxes
    inline_formula_check = []
    for result in inline_formula_boxes:
        (x1, y1, x2, y2) = (result[0], result[1], result[2], result[3])
        ## 逐个block##
        in_block = 0
        for bbox in blocks:
            # image = cv2.rectangle(image, (int(bbox['bbox'][0]), int(bbox['bbox'][1])), (int(bbox['bbox'][2]), int(bbox['bbox'][3])), (0, 255, 0), 1)
            if (y1 >= bbox['bbox'][1] and y2 <= bbox['bbox'][3]) and (x1 >= bbox['bbox'][0] and x2 <= bbox['bbox'][2]):       # 判定公式在哪一个block
                in_block += 1
                intersect = []
                # ## 逐个span###
                for line in bbox['lines']:
                    if line['bbox'][1] <= ((y2 - y1) / 2) + y1 <= line['bbox'][3]:   # 判断公式在哪一行
                        for item in line['spans']:
                            (t_x1, t_y1, t_x2, t_y2) = item['bbox']
                            if not ((t_x1 < x1 and t_x2 < x1) or (t_x1 > x2 and t_x2 > x2) or (t_y1 < y1 and t_y2 < y1) or (t_y1 > y2 and t_y2 > y2)):   # 判断是否相交
                                intersect.append(item['bbox'])
                                # image = cv2.rectangle(image, (int(t_x1), int(t_y1)), (int(t_x2), int(t_y2)), (0, 255, 0), 1)    # 可视化涉及到的span

                # 可视化公式的分类
                if len(intersect) == 0:  # 没有与任何一个span有相交，这个span或者这个inline_formula_box可能有问题
                    # print(f'Wrong location, check {img_path}')
                    inline_formula_check_result = "nocheck_inline_formula"
                    # count['not_in_line'] += 1
                elif len(intersect) == 1:  
                    if abs((intersect[0][2] - intersect[0][0]) - (x2 - x1)) < (x2 - x1)*0.5: # 只涉及一个span但是几乎占据了这个span大部分的面积，判定为公式
                        # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)   
                        inline_formula_check_result = "true_inline_formula"
                        # count['one_span_large'] += 1
                    else:  # 只涉及一个span并且只占据这个span的小部分面积，判断可能不是公式
                        # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                        inline_formula_check_result = "false_inline_formula"
                        # count['fail'] += 1
                else:  # 横跨多个span,判定为公式
                    # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    inline_formula_check_result = "true_inline_formula"
                    # count['multi_span'] += 1
                            
        if in_block == 0:  # 这个公式没有在任何的block里，这个公式可能有问题
            # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
            inline_formula_check_result = "nocheck_inline_formula"
            # count['not_in_block'] += 1
        elif in_block > 1: # 这个公式存在于多个block里，这个页面可能有问题
            inline_formula_check_result = "wrong_text_block"

        inline_formula_check.append(inline_formula_check_result)

    return inline_formula_check
