from magic_pdf.libs.boxbase import _is_in, _is_in_or_part_overlap
import collections      # 统计库



def is_below(bbox1, bbox2):
    # 如果block1的上边y坐标大于block2的下边y坐标，那么block1在block2下面
    return bbox1[1] > bbox2[3]


def merge_bboxes(bboxes):
    # 找出所有blocks的最小x0，最大y1，最大x1，最小y0，这就是合并后的bbox
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)
    return [x0, y0, x1, y1]


def merge_footnote_blocks(page_info, main_text_font):
    page_info['merged_bboxes'] = []
    for layout in page_info['layout_bboxes']:
        # 找出layout中的所有footnote blocks和preproc_blocks
        footnote_bboxes = [block for block in page_info['footnote_bboxes_tmp'] if _is_in(block, layout['layout_bbox'])]
        # 如果没有footnote_blocks，就跳过这个layout
        if not footnote_bboxes:
            continue

        preproc_blocks = [block for block in page_info['preproc_blocks'] if _is_in(block['bbox'], layout['layout_bbox'])]
        # preproc_bboxes = [block['bbox'] for block in preproc_blocks]
        font_names = collections.Counter()
        if len(preproc_blocks) > 0:
            # 存储每一行的文本块大小的列表
            line_sizes = []
            # 存储每个文本块的平均行大小
            block_sizes = []
            for block in preproc_blocks:
                block_line_sizes = []
                block_fonts = collections.Counter()
                for line in block['lines']:
                    # 提取每个span的size属性，并计算行大小
                    span_sizes = [span['size'] for span in line['spans'] if 'size' in span]
                    if span_sizes:
                        line_size = sum(span_sizes) / len(span_sizes)
                        line_sizes.append(line_size)
                        block_line_sizes.append(line_size)
                    span_font = [(span['font'], len(span['text'])) for span in line['spans'] if
                                 'font' in span and len(span['text']) > 0]
                    if span_font:
                        # # todo main_text_font应该用基于字数最多的字体而不是span级别的统计
                        # font_names.append(font_name for font_name in span_font)
                        # block_fonts.append(font_name for font_name in span_font)
                        for font, count in span_font:
                            # font_names.extend([font] * count)
                            # block_fonts.extend([font] * count)
                            font_names[font] += count
                            block_fonts[font] += count
                if block_line_sizes:
                    # 计算文本块的平均行大小
                    block_size = sum(block_line_sizes) / len(block_line_sizes)
                    block_font = block_fonts.most_common(1)[0][0]
                    block_sizes.append((block, block_size, block_font))

            # 计算main_text_size
            # main_text_font = font_names.most_common(1)[0][0]
            main_text_size = collections.Counter(line_sizes).most_common(1)[0][0]
        else:
            continue

        need_merge_bboxes = []
        # 任何一个下面有正文block的footnote bbox都是假footnote
        for footnote_bbox in footnote_bboxes:
            # 检测footnote下面是否有正文block(正文block需满足，block平均size大于等于main_text_size，且block行数大于等于5)
            main_text_bboxes_below = [block['bbox'] for block, size, block_font in block_sizes if
                                      is_below(block['bbox'], footnote_bbox) and
                                      sum([size >= main_text_size,
                                           len(block['lines']) >= 5,
                                           block_font == main_text_font])
                                      >= 2]
            # 如果main_text_bboxes_below不为空，说明footnote下面有正文block，这个footnote不成立，跳过
            if len(main_text_bboxes_below) > 0:
                continue
            else:
                # 否则，说明footnote下面没有正文block，这个footnote成立，添加到待merge的footnote_bboxes中
                need_merge_bboxes.append(footnote_bbox)
        if len(need_merge_bboxes) == 0:
            continue
        # 找出最靠上的footnote block
        top_footnote_bbox = min(need_merge_bboxes, key=lambda bbox: bbox[1])
        # 找出所有在top_footnote_block下面的preproc_blocks，并确保这些preproc_blocks的平均行大小小于main_text_size
        bboxes_below = [block['bbox'] for block, size, block_font in block_sizes if is_below(block['bbox'], top_footnote_bbox)]
        # # 找出所有在top_footnote_block下面的preproc_blocks
        # bboxes_below = [bbox for bbox in preproc_bboxes if is_below(bbox, top_footnote_bbox)]
        # 合并top_footnote_block和blocks_below
        merged_bbox = merge_bboxes([top_footnote_bbox] + bboxes_below)
        # 添加到新的footnote_bboxes_tmp中
        page_info['merged_bboxes'].append(merged_bbox)
    return page_info


def remove_footnote_blocks(page_info):
    if page_info.get('merged_bboxes'):
        # 从文字中去掉footnote
        remain_text_blocks, removed_footnote_text_blocks = remove_footnote_text(page_info['preproc_blocks'], page_info['merged_bboxes'])
        # 从图片中去掉footnote
        image_blocks, removed_footnote_imgs_blocks = remove_footnote_image(page_info['images'], page_info['merged_bboxes'])
        # 更新page_info
        page_info['preproc_blocks'] = remain_text_blocks
        page_info['images'] = image_blocks
        page_info['droped_text_block'].extend(removed_footnote_text_blocks)
        page_info['droped_image_block'].extend(removed_footnote_imgs_blocks)
        # 删除footnote_bboxes_tmp和merged_bboxes
        del page_info['merged_bboxes']
    del page_info['footnote_bboxes_tmp']
    return page_info


def remove_footnote_text(raw_text_block, footnote_bboxes):
    """
    :param raw_text_block: str类型，是当前页的文本内容
    :param footnoteBboxes: list类型，是当前页的脚注bbox
    """
    footnote_text_blocks = []
    for block in raw_text_block:
        text_bbox = block['bbox']
        # TODO 更严谨点在line级别做
        if any([_is_in_or_part_overlap(text_bbox, footnote_bbox) for footnote_bbox in footnote_bboxes]):
            # if any([text_bbox[3]>=footnote_bbox[1] for footnote_bbox in footnote_bboxes]):
            block['tag'] = 'footnote'
            footnote_text_blocks.append(block)
            # raw_text_block.remove(block)

    # 移除，不能再内部移除，否则会出错
    for block in footnote_text_blocks:
        raw_text_block.remove(block)

    return raw_text_block, footnote_text_blocks


def remove_footnote_image(image_blocks, footnote_bboxes):
    """
    :param image_bboxes: list类型，是当前页的图片bbox(结构体)
    :param footnoteBboxes: list类型，是当前页的脚注bbox
    """
    footnote_imgs_blocks = []
    for image_block in image_blocks:
        if any([_is_in(image_block['bbox'], footnote_bbox) for footnote_bbox in footnote_bboxes]):
            footnote_imgs_blocks.append(image_block)

    for footnote_imgs_block in footnote_imgs_blocks:
        image_blocks.remove(footnote_imgs_block)

    return image_blocks, footnote_imgs_blocks