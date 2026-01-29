from io import BytesIO
from typing import Final, BinaryIO, Optional

from lxml import etree
from pptx import Presentation, presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.oxml.text import CT_TextLineBreak
from loguru import logger
from PIL import Image, UnidentifiedImageError, WmfImagePlugin

from mineru.utils.enum_class import BlockType
from mineru.utils.pdf_reader import image_to_b64str


class PptxConverter:

    def __init__(self):
        self.namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        self.file_stream = None
        self.pptx_obj = None
        self.pages = []
        self.cur_page = []
        self.list_block_stack: list = []  # 列表块堆栈

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        self.file_stream = file_stream
        self.pptx_obj = Presentation(self.file_stream)
        self.pages.append(self.cur_page)
        if self.pptx_obj:
            self._walk_linear(self.pptx_obj)
        if self.pages[-1] == []:
            self.pages.pop()

    def _walk_linear(self, pptx_obj: presentation.Presentation):
        slide_width = pptx_obj.slide_width
        slide_height = pptx_obj.slide_height

        # 遍历每一张幻灯片
        for _, slide in enumerate(pptx_obj.slides):

            def handle_shapes(shape):
                handle_groups(shape)
                if shape.has_table:
                    # 处理表格
                    self._handle_tables(shape)
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    # 处理图片
                    if hasattr(shape, "image"):
                        self._handle_pictures(shape)
                # 如果形状没有任何文本，则继续处理下一个形状
                if not hasattr(shape, "text"):
                    return
                if shape.text is None:
                    return
                if len(shape.text.strip()) == 0:
                    return
                if not shape.has_text_frame:
                    logger.warning("Warning: shape has text but not text_frame")
                    return
                # 处理其他文本元素，包括列表(项目符号列表、编号列表等)
                self._handle_text_elements(shape)
                return

            def handle_groups(shape):
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for groupedshape in shape.shapes:
                        handle_shapes(groupedshape)

            # 遍历幻灯片中的每一个形状
            for shape in slide.shapes:
                handle_shapes(shape)

            self.cur_page = []
            self.pages.append(self.cur_page)

    def _handle_tables(self, shape):
        """将PowerPoint表格转换为HTML格式。

        Args:
            shape: 包含表格的形状对象。
            parent_slide: 父幻灯片组。
            slide_ind: 当前幻灯片索引。
            doc: 文档对象(此实现中未使用)。
            slide_size: 幻灯片尺寸。

        Returns:
            str: 表格的HTML字符串，如果没有表格则返回None。
        """
        if not shape.has_table:
            return None

        table = shape.table
        table_xml = shape._element

        # 开始构建HTML表格
        html_parts = ['<table border="1">']

        # 跟踪已被合并单元格占用的位置
        # 格式: {(row, col): True}
        occupied_cells = {}

        for row_idx, row in enumerate(table.rows):
            html_parts.append("  <tr>")

            for col_idx, cell in enumerate(row.cells):
                # 跳过被合并占用的单元格
                if (row_idx, col_idx) in occupied_cells:
                    continue
                # 获取单元格XML以读取跨度信息
                cell_xml = table_xml.xpath(
                    f".//a:tbl/a:tr[{row_idx + 1}]/a:tc[{col_idx + 1}]"
                )

                if not cell_xml:
                    continue

                cell_xml = cell_xml[0]

                # 解析行跨度和列跨度
                row_span = cell_xml.get("rowSpan")
                col_span = cell_xml.get("gridSpan")

                row_span = int(row_span) if row_span else 1
                col_span = int(col_span) if col_span else 1

                # 标记被此单元格占用的位置
                for r in range(row_idx, row_idx + row_span):
                    for c in range(col_idx, col_idx + col_span):
                        if (r, c) != (row_idx, col_idx):
                            occupied_cells[(r, c)] = True

                # 确定标签类型：第一行使用<th>，其他使用<td>
                tag = "th" if row_idx == 0 else "td"

                # 构建属性字符串
                attrs = []
                if row_span > 1:
                    attrs.append(f'rowspan="{row_span}"')
                if col_span > 1:
                    attrs.append(f'colspan="{col_span}"')

                attr_str = " " + " ".join(attrs) if attrs else ""

                # 获取单元格文本内容
                cell_text = cell.text.strip() if cell.text else ""
                # 转义HTML特殊字符，防止XSS
                cell_text = (
                    cell_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

                html_parts.append(f"    <{tag}{attr_str}>{cell_text}</{tag}>")

            html_parts.append("  </tr>")

        html_parts.append("</table>")

        self.cur_page.append(
            {
                "type": BlockType.TABLE,
                "content": "\n".join(html_parts),
            }
        )

        return None

    def _handle_pictures(self, shape):
        # 使用PIL打开图像
        try:
            # 获取图像字节数据
            image = shape.image
            image_bytes = image.blob
            im_dpi, _ = image.dpi
            pil_image = Image.open(BytesIO(image_bytes))

            if isinstance(pil_image, WmfImagePlugin.WmfStubImageFile):
                logger.warning(f"Skipping WMF image, size: {pil_image.size}")
                placeholder = Image.new("RGB", pil_image.size, (240, 240, 240))
                img_base64 = image_to_b64str(placeholder)
            else:
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                img_base64 = image_to_b64str(pil_image)
            image_block = {
                "type": BlockType.IMAGE,
                "content": img_base64,
            }
            self.cur_page.append(image_block)

        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Warning: image cannot be loaded by Pillow: {e}")
        return

    def _handle_text_elements(self, shape):
        is_list_group_created = False
        enum_list_item_value = 0
        new_list = None

        # 遍历段落以构建文本
        for paragraph in shape.text_frame.paragraphs:
            is_a_list, bullet_type = self._is_list_item(paragraph)
            p = paragraph._element

            # 将换行符转换为空格并累积文本
            p_text = ""
            for e in p.content_children:
                if isinstance(e, CT_TextLineBreak):
                    p_text += " "
                else:
                    p_text += e.text

            if is_a_list:
                enum_marker = ""
                enumerated = bullet_type == "Numbered"

                if not is_list_group_created:
                    if enumerated:
                        list_attribute = "ordered"
                    else:
                        list_attribute = "unordered"
                    new_list_block = {
                        "type": BlockType.LIST,
                        "attribute": list_attribute,
                        "list_items": [],
                    }
                    self.cur_page.append(new_list_block)
                    self.list_block_stack.append(new_list_block)
                    is_list_group_created = True
                    enum_list_item_value = 0

                if enumerated:
                    enum_list_item_value += 1
                    enum_marker = str(enum_list_item_value) + ". "

                list_item = {
                    "type": BlockType.TEXT,
                    "content": enum_marker + p_text,
                }
                self.list_block_stack[-1]["list_items"].append(list_item)
            else:  # 段落不是列表项
                if is_list_group_created:
                    is_list_group_created = False
                    new_list = None
                    enum_list_item_value = 0
                    self.list_block_stack.pop()
                # 根据文本类型分配标签(标题/部分标题/段落等)
                label = BlockType.TEXT
                if shape.is_placeholder:
                    placeholder_type = shape.placeholder_format.type
                    if placeholder_type in [
                        PP_PLACEHOLDER.CENTER_TITLE,
                        PP_PLACEHOLDER.TITLE,
                        PP_PLACEHOLDER.SUBTITLE,
                    ]:
                        label = BlockType.TITLE

                # 输出累积的内联文本
                self.cur_page.append(
                    {
                        "type": label,
                        "content": p_text,
                    }
                )
        return

    def _is_list_item(self, paragraph) -> tuple[bool, str]:
        """
        判断段落是否应被视为列表项。
        该方法首先尝试通过拥有该段落的形状来解析列表样式信息。
        如果无法做到，则回退到基于段落属性和级别的更简单检查。
        Args:
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回一个2元组(`is_list`, `bullet_type`)，其中：
            `is_list` - 若段落被视为列表项，为True，否则为False；
            `bullet_type` - 为以下之一：'Bullet'(项目符号)、'Numbered'(编号)或'None'，
            描述列表标记类型。
        """
        p = paragraph._element

        # 尝试从段落获取形状（包含该段落的对象），如果可能的话
        shape = None
        try:
            # 这个路径适用于python-pptx段落对象
            # 首先获取文本框架(段落的父对象)
            text_frame = paragraph._parent
            # 然后获取形状(文本框架的父对象)
            shape = text_frame._parent
        except AttributeError:
            pass

        if shape is not None:
            marker_info = self._get_effective_list_marker(shape, paragraph)

            # 检查这是否肯定是一个列表项
            if marker_info["is_list"] is True or marker_info["kind"] in (
                "buChar",
                "buAutoNum",
                "buBlip",
            ):
                if marker_info["kind"] == "buChar":
                    return (True, "Bullet")
                elif marker_info["kind"] == "buAutoNum":
                    return (True, "Numbered")
                else:
                    return (True, "None")

            # 检查这是否肯定不是列表项
            if marker_info["is_list"] is False:
                return (False, "None")

            # 回退到段落级别检查(缩进级别大于0时视为列表项)
            if paragraph.level > 0:
                return (True, "None")

            return (False, "None")

        # 如果无法获取形状，使用更简单的检查方式
        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return (True, "Bullet")
        elif (
            p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]}) is not None
        ):
            return (True, "Numbered")
        elif paragraph.level > 0:
            # 很可能是子列表项(缩进表示嵌套)
            return (True, "None")
        else:
            return (False, "None")

    def _get_effective_list_marker(self, shape, paragraph) -> dict:
        """
        返回描述段落的有效列表标记的字典。
        列表标记信息可以来自多个来源：直接段落属性、形状级别的列表样式、
        布局占位符或主幻灯片文本样式。此辅助方法解析所有这些层，并返回
        有效标记的统一视图。

        Args:
            shape: 包含段落的形状对象。
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回列表标记信息的字典，其中：
            `is_list` - True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None，描述标记类型；
            `detail` - 项目符号字符或编号类型字符串，或如果不适用则为None；
            `level` - 段落级别，范围在(0, 8)内。
        """
        p = paragraph._element
        lvl = self._get_paragraph_level(p)

        # 1) 直接段落属性
        pPr = p.find("a:pPr", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(pPr)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 2) 形状级别的列表样式(txBody/a:lstStyle)
        txBody = shape._element.find(".//p:txBody", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
            txBody, lvl
        )
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 3) 布局占位符列表样式(如果这是一个占位符)
        layout_result = None
        if shape.is_placeholder:
            idx = shape.placeholder_format.idx
            layout = shape.part.slide.slide_layout
            layout_ph = None
            try:
                layout_ph = layout.placeholders.get(idx)
            except Exception:
                layout_ph = None

            if layout_ph is not None:
                layout_tx = layout_ph._element.find(
                    ".//p:txBody", namespaces=self.namespaces
                )
                is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
                    layout_tx, lvl
                )

                # 仅在is_list明确为True/False时使用布局结果
                if is_list is not None:
                    layout_result = {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

                # 4) 解析主文本样式
                ph_type = shape.placeholder_format.type
                master = shape.part.slide.slide_layout.slide_master
                is_list, kind, detail = self._parse_bullet_from_master_text_styles(
                    master, ph_type, lvl
                )

                # 检查主样式是否有标记信息
                if kind in ("buChar", "buAutoNum", "buBlip"):
                    return {
                        "is_list": True,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }
                elif is_list is not None:
                    return {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

            # If layout has explicit is_list value but master didn't override it, use layout
            # 如果布局有显式的is_list值但主样式没有覆盖它，则使用布局结果
            if layout_result is not None:
                return layout_result

        return {
            "is_list": None,
            "kind": None,
            "detail": None,
            "level": lvl,
        }

    def _get_paragraph_level(self, paragraph) -> int:
        """
        返回段落XML元素的缩进级别。
        段落可以有不同的缩进级别(0-8)。级别存储在段落属性XML元素的'lvl'属性中。

        Args:
            paragraph: 需要提取级别的段落XML元素。

        Returns:
            返回范围在(0, 8)内的段落级别。当找不到'a:pPr'元素、没有'lvl'属性
            或'lvl'属性值无效时，返回0。
        """
        pPr = paragraph.find("a:pPr", namespaces=self.namespaces)
        if pPr is not None and "lvl" in pPr.attrib:
            try:
                return int(pPr.get("lvl"))
            except ValueError:
                pass
        return 0

    def _parse_bullet_from_paragraph_properties(
        self, pPr
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从段落属性节点解析项目符号或编号信息。
        检查'a:pPr'或'a:lvlXpPr'元素，并提取关于项目符号字符、自动编号、
        图片项目符号或显式'buNone'标记的信息。

        Args:
            pPr: 段落属性XML元素('a:pPr'或'a:lvlXpPr')。

        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`(项目符号字符)、`buAutoNum`(自动编号)、
            `buBlip`(图片项目符号)、`buNone`(无标记)或None，描述标记类型；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        if pPr is None:
            return (None, None, None)

        # 显式指定无项目符号
        if pPr.find("a:buNone", namespaces=self.namespaces) is not None:
            return (False, "buNone", None)

        # 项目符号字符
        buChar = pPr.find("a:buChar", namespaces=self.namespaces)
        if buChar is not None:
            return (True, "buChar", buChar.get("char"))

        # 自动编号
        buAuto = pPr.find("a:buAutoNum", namespaces=self.namespaces)
        if buAuto is not None:
            return (True, "buAutoNum", buAuto.get("type"))

        # 图片项目符号
        buBlip = pPr.find("a:buBlip", namespaces=self.namespaces)
        if buBlip is not None:
            return (True, "buBlip", "image")

        return (None, None, None)

    def _parse_bullet_from_text_body_list_style(
        self, txBody, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从文本体的列表样式中解析项目符号或编号信息。
        在'txBody'下搜索'a:lstStyle/a:lvl{lvl+1}pPr'，并使用级别特定的段落属性
        推断项目符号或编号信息。

        Args:
            txBody: 文本体XML元素'p:txBody'。
            lvl: 段落级别，范围在(0, 8)内。
        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        if txBody is None:
            return (None, None, None)
        lstStyle = txBody.find("a:lstStyle", namespaces=self.namespaces)
        lvl_pPr = self._find_level_properties_in_list_style(lstStyle, lvl)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _parse_bullet_from_master_text_styles(
        self, slide_master, placeholder_type, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从主幻灯片的文本样式中解析项目符号或编号信息。
        在主幻灯片的'p:txStyles'中查找相应的样式bucket('titleStyle'、'bodyStyle'或
        'otherStyle')，并为给定的级别提取项目符号或编号信息。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        style = self._get_master_text_style_node(slide_master, placeholder_type)
        if style is None:
            return (None, None, None)

        lvl_pPr = style.find(f".//a:lvl{lvl + 1}pPr", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _find_level_properties_in_list_style(self, lstStyle, lvl: int):
        """Find the level-specific paragraph properties node from a list style.
        从列表样式中查找指定级别的段落属性节点。

        This looks for an `a:lvl{lvl+1}pPr` node inside an `a:lstStyle` element, where
        在'a:lstStyle'元素内查找'a:lvl{lvl+1}pPr'节点，其中'a:lvl1pPr'对应级别0，
        `a:lvl1pPr` corresponds to level 0, `a:lvl2pPr` to level 1, and so on.
        'a:lvl2pPr'对应级别1，依此类推。

        Args:
            lstStyle: List style XML element `a:lstStyle`.
            lstStyle: 列表样式XML元素'a:lstStyle'。
            lvl: Paragraph level in the range (0, 8).
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            Matching `a:lvl{lvl+1}pPr` XML element, or None if no matching element is
            匹配的'a:lvl{lvl+1}pPr'XML元素，如果未找到匹配元素则返回None。
                found.
        """
        if lstStyle is None:
            return None
        tag = f"a:lvl{lvl + 1}pPr"
        return lstStyle.find(tag, namespaces=self.namespaces)

    def _get_master_text_style_node(
        self, slide_master, placeholder_type
    ) -> Optional[etree._Element]:
        """
        获取占位符的相应主文本样式节点。
        大多数内容占位符(BODY/OBJECT)使用'p:bodyStyle'，而标题使用'p:titleStyle'。
        所有其他占位符默认使用'p:otherStyle'。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。

        Returns:
            从主幻灯片的'p:txStyles'中匹配的样式节点('p:bodyStyle'、'p:titleStyle'或'p:otherStyle')，或当未定义样式时返回None。
        """
        txStyles = slide_master._element.find(
            ".//p:txStyles", namespaces=self.namespaces
        )
        if txStyles is None:
            return None

        if placeholder_type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
            return txStyles.find("p:bodyStyle", namespaces=self.namespaces)

        if placeholder_type == PP_PLACEHOLDER.TITLE:
            return txStyles.find("p:titleStyle", namespaces=self.namespaces)

        return txStyles.find("p:otherStyle", namespaces=self.namespaces)
