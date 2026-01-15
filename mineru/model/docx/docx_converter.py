import os
import re
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union, Any, Final

from PIL import Image, WmfImagePlugin
from loguru import logger
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.text.paragraph import Paragraph
from docx.text.hyperlink import Hyperlink
from docx.text.run import Run
from lxml import etree
from pydantic import AnyUrl
from mammoth.conversion import convert_document_element_to_html
from mammoth.docx import body_xml

from mineru.model.docx.tools.office_xml import read_str
from mineru.model.docx.tools.math.omml import oMath2Latex
from mineru.utils.docx_fomatting import Formatting, Script
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_reader import image_to_b64str

ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]


class DocxConverter:
    _BLIP_NAMESPACES: Final = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
        "v": "urn:schemas-microsoft-com:vml",
        "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
        "w10": "urn:schemas-microsoft-com:office:word",
        "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    }
    """
    Word 文档中使用的 XML 命名空间映射。

    这些命名空间用于解析 DOCX 文件中的各种元素，包括：
    - a: DrawingML 主命名空间
    - r: Office 文档关系命名空间
    - w: WordprocessingML 主命名空间
    - wp: Wordprocessing Drawing 命名空间
    - mc: 标记兼容性命名空间
    - v: VML (Vector Markup Language) 命名空间
    - wps: Wordprocessing Shape 命名空间
    - w10: Office Word 命名空间
    - a14: Office 2010 Drawing 命名空间
    """

    def __init__(self):
        self.XML_KEY = (
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
        )
        self.xml_namespaces = {
            "w": "http://schemas.microsoft.com/office/word/2003/wordml"
        }
        self.blip_xpath_expr = etree.XPath(
            ".//a:blip", namespaces=DocxConverter._BLIP_NAMESPACES
        )

        self.docx_obj = None
        self.max_levels: int = 10  # 文档层次结构的最大层级数
        self.parents: dict = {}  # 父级节点映射
        self.pages = []
        self.cur_page = []
        self.pre_num_id: int = -1  # 上一个处理元素的 numId
        self.pre_ilevel: int = -1  # 上一个处理元素的缩进等级, 用于判断列表层级
        self.list_block_stack: list = []  # 列表块堆栈
        self.list_counters: dict[tuple[int, int], int] = (
            {}
        )  # 列表计数器 (numId, ilvl) -> count
        self.equation_bookends: str = "<eq>{EQ}</eq>"  # 公式标记格式

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        self.docx_obj = Document(file_stream)
        self.pages.append(self.cur_page)
        self._walk_linear(self.docx_obj.element.body)
        self._add_header_footer(self.docx_obj)

    def _walk_linear(
        self,
        body: BaseOxmlElement,
    ):
        for element in body:
            # 获取元素的标签名（去除命名空间前缀）
            tag_name = etree.QName(element).localname
            # 检查是否存在内联图像（blip元素）
            drawing_blip = self.blip_xpath_expr(element)

            # 查找所有绘图元素（用于处理DrawingML）
            drawingml_els = element.findall(
                ".//w:drawing", namespaces=DocxConverter._BLIP_NAMESPACES
            )

            if tag_name == "tbl":
                try:
                    # 处理表格元素
                    self._handle_tables(element)
                except Exception:
                    # 如果表格解析失败，记录调试信息
                    logger.debug("could not parse a table, broken docx table")
            # 检查图片元素
            elif drawing_blip:
                # 处理图片元素
                self._handle_pictures(drawing_blip)
            # 检查文本段落元素
            elif tag_name == "p":
                # 处理文本元素（包括段落属性如"tcPr", "sectPr"等）
                self._handle_text_elements(element)

            # 忽略其他未知元素并记录日志
            else:
                logger.debug(f"Ignoring element in DOCX with tag: {tag_name}")

    def _handle_tables(self, element: BaseOxmlElement):
        """
        处理表格。

        Args:
            element: 元素对象
        Returns:
            list[RefItem]: 元素引用列表
        """
        table = read_str(element.xml)
        body_reader = body_xml.reader()
        t = body_reader.read_all([table])
        res = convert_document_element_to_html(t.value[0])
        table_block = {
            "image_source": "",
            "html": res.value,
            "table_caption": "",
            "table_footnote": "",
            "table_type": "",
            "table_nest_level": "",
        }
        self.cur_page.append(table_block)

    def _handle_text_elements(
        self,
        element: BaseOxmlElement,
    ):
        """
        处理文本元素。

        Args:
            element: 元素对象
            doc: DoclingDocument 对象

        Returns:

        """
        is_section_end = False
        if element.find(".//w:sectPr", namespaces=DocxConverter._BLIP_NAMESPACES):
            # 如果没有text内容
            if element.text == "":
                self.cur_page = []
                self.pages.append(self.cur_page)
            else:
                # 标记本节结束，处理完文本之后再分节
                is_section_end = True
        paragraph = Paragraph(element, self.docx_obj)
        paragraph_elements = self._get_paragraph_elements(paragraph)
        text, equations = self._handle_equations_in_text(
            element=element, text=paragraph.text
        )

        if text is None:
            return None
        text = text.strip()

        # 常见的项目符号和编号列表样式。
        # "List Bullet", "List Number", "List Paragraph"
        # 识别列表是否为编号列表
        p_style_id, p_level = self._get_label_and_level(paragraph)
        numid, ilevel = self._get_numId_and_ilvl(paragraph)

        if numid == 0:
            numid = None

        # 处理列表
        if (
            numid is not None
            and ilevel is not None
            and p_style_id not in ["Title", "Heading"]
        ):
            # 通过检查 numFmt 来确认这是否实际上是编号列表
            is_numbered = self._is_numbered_list(numid, ilevel)

            self._add_list_item(
                numid=numid,
                ilevel=ilevel,
                elements=paragraph_elements,
                is_numbered=is_numbered,
                text=text,
                equations=equations,
            )
            # 列表项已处理，返回
            return None
        elif (  # 列表结束处理
            numid is None
            and self.pre_num_id != -1
            and p_style_id not in ["Title", "Heading"]
        ):  # 关闭列表
            # 重置列表状态
            self.pre_num_id = -1
            self.pre_ilevel = -1
            self.list_block_stack = []
            self.list_counters = {}

        if p_style_id in ["Title"]:
            if text != "":
                title_block = {
                    "type": BlockType.TITLE,
                    "level": 1,
                    "is_numbered_style": False,
                    "content": text,
                }
                self.cur_page.append(title_block)

        elif "Heading" in p_style_id:
            style_element = getattr(paragraph.style, "element", None)
            if style_element is not None:
                is_numbered_style = (
                    "<w:numPr>" in style_element.xml or "<w:numPr>" in element.xml
                )
            else:
                is_numbered_style = False
            if text != "":
                h_block = {
                    "type": BlockType.TITLE,
                    "level": p_level + 1 if p_level is not None else 2,
                    "is_numbered_style": is_numbered_style,
                    "content": text,
                }
                self.cur_page.append(h_block)

        elif len(equations) > 0:
            if (paragraph.text is None or len(paragraph.text.strip()) == 0) and len(
                text
            ) > 0:
                # 独立公式
                eq_block = {
                    "type": BlockType.INTERLINE_EQUATION,
                    "content": text.replace("<eq>", "").replace("</eq>", ""),
                }
                self.cur_page.append(eq_block)
            else:
                # 包含行内公式的文本块
                text_with_inline_eq_block = {
                    "type": BlockType.TEXT,
                    "contnet": "",
                }
                self.cur_page.append(text_with_inline_eq_block)
                text_tmp = text
                for eq in equations:
                    if len(text_tmp) == 0:
                        break

                    split_text_tmp = text_tmp.split(eq.strip(), maxsplit=1)

                    pre_eq_text = split_text_tmp[0]
                    text_tmp = "" if len(split_text_tmp) == 1 else split_text_tmp[1]

                    if len(pre_eq_text) > 0:
                        text_with_inline_eq_block["contnet"] += pre_eq_text
                    text_with_inline_eq_block["contnet"] += eq

                if len(text_tmp) > 0:
                    text_with_inline_eq_block["contnet"] += text_tmp.strip()
        elif p_style_id in [
            "Paragraph",
            "Normal",
            "Subtitle",
            "Author",
            "DefaultText",
            "ListParagraph",
            "ListBullet",
            "Quote",
        ]:
            for text, format, hyperlink in paragraph_elements:
                if text != "":
                    text_block = {
                        "type": BlockType.TEXT,
                        "content": text,
                    }
                    self.cur_page.append(text_block)
        # 判断是否是 Caption
        elif self._is_caption(element):
            for text, format, hyperlink in paragraph_elements:
                if text != "":
                    caption_block = {
                        "type": "caption",
                        "content": text,
                    }
                    self.cur_page.append(caption_block)
        else:
            # 文本样式名称不仅有默认值，还可能有用户自定义值
            # 因此我们将所有其他标签视为纯文本
            for text, format, hyperlink in paragraph_elements:
                if text != "":
                    text_block = {
                        "type": BlockType.TEXT,
                        "content": text,
                    }
                    self.cur_page.append(text_block)

        if is_section_end:
            self.cur_page = []
            self.pages.append(self.cur_page)

    def _handle_pictures(self, drawing_blip: Any):
        """
        处理图片。

        Args:
            drawing_blip: 绘图 blip 对象

        Returns:

        """

        def get_docx_image(drawing_blip: Any) -> Optional[bytes]:
            """
            获取 DOCX 图像数据。

            Args:
                drawing_blip: 绘图 blip 对象

            Returns:
                Optional[bytes]: 图像数据
            """
            image_data: Optional[bytes] = None
            rId = drawing_blip[0].get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if rId in self.docx_obj.part.rels:
                # 使用关系 ID 访问图像部分
                image_part = self.docx_obj.part.rels[rId].target_part
                image_data = image_part.blob  # 获取二进制图像数据
            return image_data

        # 使用 PIL 打开 BytesIO 对象创建图像
        image_data: Optional[bytes] = get_docx_image(drawing_blip)
        if image_data is None:
            logger.warning("Warning: image cannot be found")
        else:
            image_bytes = BytesIO(image_data)
            pil_image = Image.open(image_bytes)
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

    def _get_paragraph_elements(self, paragraph: Paragraph):
        """
        提取段落元素及其格式和超链接信息。

        Args:
            paragraph: 段落对象

        Returns:
            list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path]]]]:
            段落元素列表，每个元素包含文本、格式和超链接信息
        """

        # 目前保留空段落以保持向后兼容性:
        if paragraph.text.strip() == "":
            return [("", None, None)]

        paragraph_elements: list[
            tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path]]]
        ] = []
        group_text = ""
        previous_format = None

        # 遍历段落的 runs 并按格式分组
        for c in paragraph.iter_inner_content():
            if isinstance(c, Hyperlink):
                text = c.text
                hyperlink = Path(c.address)
                format = (
                    self._get_format_from_run(c.runs[0])
                    if c.runs and len(c.runs) > 0
                    else None
                )
            elif isinstance(c, Run):
                text = c.text
                hyperlink = None
                format = self._get_format_from_run(c)
            else:
                continue

            if (len(text.strip()) and format != previous_format) or (
                hyperlink is not None
            ):
                # 如果非空文本的样式发生变化，则添加前一个组
                if len(group_text.strip()) > 0:
                    paragraph_elements.append(
                        (group_text.strip(), previous_format, None)
                    )
                group_text = ""

                # 如果有超链接，则立即添加
                if hyperlink is not None:
                    paragraph_elements.append((text.strip(), format, hyperlink))
                    text = ""
                else:
                    previous_format = format

            group_text += text

        # 格式化最后一个组
        if len(group_text.strip()) > 0:
            paragraph_elements.append((group_text.strip(), format, None))

        return paragraph_elements

    @classmethod
    def _get_format_from_run(cls, run: Run) -> Optional[Formatting]:
        """
        从 Run 对象获取格式信息。

        Args:
            run: Run 对象

        Returns:
            Optional[Formatting]: 格式对象
        """
        # .bold 和 .italic 属性是布尔值，但 .underline 可能是枚举
        # 如 WD_UNDERLINE.THICK (值为 6)，所以需要转换为布尔值
        is_bold = run.bold or False
        is_italic = run.italic or False
        is_strikethrough = run.font.strike or False
        # 将任何非 None 的下划线值转换为 True
        is_underline = bool(run.underline is not None and run.underline)
        is_sub = run.font.subscript or False
        is_sup = run.font.superscript or False
        script = Script.SUB if is_sub else Script.SUPER if is_sup else Script.BASELINE

        return Formatting(
            bold=is_bold,
            italic=is_italic,
            underline=is_underline,
            strikethrough=is_strikethrough,
            script=script,
        )

    def _handle_equations_in_text(self, element, text):
        """
        处理文本中的公式。

        Args:
            element: 元素对象
            text: 文本内容

        Returns:
            tuple: (处理后的文本, 公式列表)
        """
        only_texts = []
        only_equations = []
        texts_and_equations = []
        for subt in element.iter():
            tag_name = etree.QName(subt).localname
            if tag_name == "t" and "math" not in subt.tag:
                if isinstance(subt.text, str):
                    only_texts.append(subt.text)
                    texts_and_equations.append(subt.text)
            elif "oMath" in subt.tag and "oMathPara" not in subt.tag:
                latex_equation = str(oMath2Latex(subt)).strip()
                if len(latex_equation) > 0:
                    only_equations.append(
                        self.equation_bookends.format(EQ=latex_equation)
                    )
                    texts_and_equations.append(
                        self.equation_bookends.format(EQ=latex_equation)
                    )

        if len(only_equations) < 1:
            return text, []

        if (
            re.sub(r"\s+", "", "".join(only_texts)).strip()
            != re.sub(r"\s+", "", text).strip()
        ):
            # 如果我们无法重构初始原始文本
            # 不要尝试解析公式并返回原始文本
            return text, []

        # 将公式插入原始文本中
        # 这样做是为了保持空白结构
        output_text = text[:]
        init_i = 0
        for i_substr, substr in enumerate(texts_and_equations):
            if len(substr) == 0:
                continue

            if substr in output_text[init_i:]:
                init_i += output_text[init_i:].find(substr) + len(substr)
            else:
                if i_substr > 0:
                    output_text = output_text[:init_i] + substr + output_text[init_i:]
                    init_i += len(substr)
                else:
                    output_text = substr + output_text

        return output_text, only_equations

    def _get_label_and_level(self, paragraph: Paragraph) -> tuple[str, Optional[int]]:
        """
        获取段落的标签和层级。

        Args:
            paragraph: 段落对象

        Returns:
            tuple[str, Optional[int]]: (标签, 层级) 元组
        """
        if paragraph.style is None:
            return "Normal", None

        label = paragraph.style.style_id
        name = paragraph.style.name
        base_style_label = None
        base_style_name = None
        if base_style := getattr(paragraph.style, "base_style", None):
            base_style_label = base_style.style_id
            base_style_name = base_style.name

        if label is None:
            return "Normal", None

        if ":" in label:
            parts = label.split(":")
            if len(parts) == 2:
                return parts[0], self._str_to_int(parts[1], None)

        if "heading" in label.lower():
            return self._get_heading_and_level(label)
        if "heading" in name.lower():
            return self._get_heading_and_level(name)
        if base_style_label and "heading" in base_style_label.lower():
            return self._get_heading_and_level(base_style_label)
        if base_style_name and "heading" in base_style_name.lower():
            return self._get_heading_and_level(base_style_name)

        return name, None

    def _get_numId_and_ilvl(
        self, paragraph: Paragraph
    ) -> tuple[Optional[int], Optional[int]]:
        """
        获取段落的列表编号ID和层级。

        Args:
            paragraph: 段落对象

        Returns:
            tuple[Optional[int], Optional[int]]: (numId, ilvl) 元组
        """
        # 访问段落的XML元素
        numPr = paragraph._element.find(
            ".//w:numPr", namespaces=paragraph._element.nsmap
        )

        if numPr is not None:
            # 获取 numId 元素并提取值
            numId_elem = numPr.find("w:numId", namespaces=paragraph._element.nsmap)
            ilvl_elem = numPr.find("w:ilvl", namespaces=paragraph._element.nsmap)
            numId = numId_elem.get(self.XML_KEY) if numId_elem is not None else None
            ilvl = ilvl_elem.get(self.XML_KEY) if ilvl_elem is not None else None

            return self._str_to_int(numId, None), self._str_to_int(ilvl, None)

        return None, None  # 如果段落不是列表的一部分

    def _is_numbered_list(self, numId: int, ilvl: int) -> bool:
        """
        根据 numFmt 值检查列表是否为编号列表。

        Args:
            numId: 列表编号ID
            ilvl: 列表层级

        Returns:
            bool: 如果是编号列表返回 True，否则返回 False
        """
        try:
            # 访问文档的编号部分
            if not hasattr(self.docx_obj, "part") or not hasattr(
                self.docx_obj.part, "package"
            ):
                return False

            numbering_part = None
            # 查找编号部分
            for part in self.docx_obj.part.package.parts:
                if "numbering" in part.partname:
                    numbering_part = part
                    break

            if numbering_part is None:
                return False

            # 解析编号 XML
            numbering_root = numbering_part.element
            namespaces = {
                "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            }

            # 查找具有给定 numId 的编号定义
            num_xpath = f".//w:num[@w:numId='{numId}']"
            num_element = numbering_root.find(num_xpath, namespaces=namespaces)

            if num_element is None:
                return False

            # 从 num 元素获取 abstractNumId
            abstract_num_id_elem = num_element.find(
                ".//w:abstractNumId", namespaces=namespaces
            )
            if abstract_num_id_elem is None:
                return False

            abstract_num_id = abstract_num_id_elem.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
            )
            if abstract_num_id is None:
                return False

            # 查找抽象编号定义
            abstract_num_xpath = (
                f".//w:abstractNum[@w:abstractNumId='{abstract_num_id}']"
            )
            abstract_num_element = numbering_root.find(
                abstract_num_xpath, namespaces=namespaces
            )

            if abstract_num_element is None:
                return False

            # 查找给定 ilvl 的层级定义
            lvl_xpath = f".//w:lvl[@w:ilvl='{ilvl}']"
            lvl_element = abstract_num_element.find(lvl_xpath, namespaces=namespaces)

            if lvl_element is None:
                return False

            # 获取 numFmt 元素
            num_fmt_element = lvl_element.find(".//w:numFmt", namespaces=namespaces)
            if num_fmt_element is None:
                return False

            num_fmt = num_fmt_element.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
            )

            # 编号格式包括: decimal, lowerRoman, upperRoman, lowerLetter, upperLetter
            # 项目符号格式包括: bullet
            numbered_formats = {
                "decimal",
                "lowerRoman",
                "upperRoman",
                "lowerLetter",
                "upperLetter",
                "decimalZero",
            }

            return num_fmt in numbered_formats

        except Exception as e:
            logger.debug(f"Error determining if list is numbered: {e}")
            return False

    def _build_text_content_list_with_equations(
        self, text: str, equations: list
    ) -> list:
        """
        构建包含行内公式的 text_content_list。

        Args:
            text: 处理后的文本（包含公式标记，如 <eq>...</eq>）
            equations: 公式列表

        Returns:
            list: text_content_list 内容列表
        """
        # 直接返回包含 <eq></eq> 标记的文本内容，与其他 text/title 保持一致
        return [
            {
                "type": ContentType.TEXT,
                "content": text,
            }
        ]

    def _add_list_item(
        self,
        *,
        numid: int,
        ilevel: int,
        elements: list,
        is_numbered: bool = False,
        text: str = "",
        equations: list = None,
    ) -> list:
        """
        添加列表项。

        Args:
            doc: DoclingDocument 对象
            numid: 列表ID
            ilevel: 缩进等级
            elements: 元素列表
            is_numbered: 是否编号
            text: 处理后的文本（包含公式标记）
            equations: 公式列表

        Returns:
            list[RefItem]: 元素引用列表
        """
        if equations is None:
            equations = []
        elem_ref: list = []
        if not elements:
            return elem_ref

        # 情况 1: 不存在上一个列表ID
        if self.pre_num_id == -1:
            # 为新编号序列重置计数器，确保编号从1开始
            self._reset_list_counters_for_new_sequence(numid)
            if is_numbered:
                list_attribute = "ordered"
            else:
                list_attribute = "unordered"
            list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "list_items": [],
                "ilevel": ilevel,
            }
            self.cur_page.append(list_block)
            # 入栈, 记录当前的列表块
            self.list_block_stack.append(list_block)
            # 记录当前引用
            elem_ref.append(id(list_block))

            # 构建 text_content_list，处理行内公式
            text_content_list = self._build_text_content_list_with_equations(
                text, equations
            )

            list_item = {
                "item_type": "text",
                "item_content": [
                    {
                        "type": BlockType.TEXT,
                        "text_content_list": text_content_list,
                    }
                ],
            }
            elem_ref.append(id(list_item))
            list_block["list_items"].append(list_item)
            self.pre_num_id = numid
            self.pre_ilevel = ilevel
        # 情况 2: 增加缩进，打开子列表, 嵌套列表的 num_id 和父列表的 num_id 相同, ilevel (缩进级别) 不同
        elif (
            self.pre_num_id == numid  # 同一个列表
            and self.pre_ilevel != -1  # 上一个缩进级别已知
            and self.pre_ilevel < ilevel  # 当前层级比之前更缩进
        ):
            # 为新增的缩进级别创建列表块
            if is_numbered:
                list_attribute = "ordered"
            else:
                list_attribute = "unordered"

            list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "list_items": [],
                "ilevel": ilevel,
            }
            # 获取栈顶的列表块
            parent_list_block = self.list_block_stack[-1]
            # 将新列表块添加为父列表块的最新列表项的子块
            newest_list_item = parent_list_block["list_items"][-1]
            newest_list_item["item_type"] = "list"  # 修改类型为列表
            newest_list_item["item_content"].append(list_block)
            # 入栈, 记录当前的列表块
            self.list_block_stack.append(list_block)
            elem_ref.append(id(list_block))

            # 构建 text_content_list，处理行内公式
            text_content_list = self._build_text_content_list_with_equations(
                text, equations
            )

            list_item = {
                "item_type": "text",
                "item_content": [
                    {
                        "type": BlockType.TEXT,
                        "text_content_list": text_content_list,
                    }
                ],
            }
            list_block["list_items"].append(list_item)
            # 更新目前缩进
            self.pre_ilevel = ilevel

        # 情况3: 减少缩进，关闭子列表
        elif (
            self.pre_num_id == numid  # 同一个列表
            and self.pre_ilevel != -1  # 上一个缩进级别已知
            and ilevel < self.pre_ilevel  # 当前层级比之前更少缩进
        ):
            # 出栈，直到找到匹配的 ilevel
            while self.list_block_stack:
                top_list_block = self.list_block_stack[-1]
                if top_list_block["ilevel"] == ilevel:
                    break
                self.list_block_stack.pop()
            list_block = self.list_block_stack[-1]

            # 构建 text_content_list，处理行内公式
            text_content_list = self._build_text_content_list_with_equations(
                text, equations
            )

            list_item = {
                "item_type": "text",
                "item_content": [
                    {
                        "type": BlockType.TEXT,
                        "text_content_list": text_content_list,
                    }
                ],
            }
            elem_ref.append(id(list_item))
            list_block["list_items"].append(list_item)
            self.pre_ilevel = ilevel

        # 情况 4: 同级列表项（相同缩进）
        elif self.pre_num_id == numid or self.pre_ilevel == ilevel:
            # 获取栈顶的列表块
            list_block = self.list_block_stack[-1]

            # 构建 text_content_list，处理行内公式
            text_content_list = self._build_text_content_list_with_equations(
                text, equations
            )

            list_item = {
                "item_type": "text",
                "item_content": [
                    {
                        "type": BlockType.TEXT,
                        "text_content_list": text_content_list,
                    }
                ],
            }
            list_block["list_items"].append(list_item)
            elem_ref.append(id(list_item))

        return elem_ref

    def _find_ilevel_list_block(self, outer_block, ilevel: int):
        """
        查找某一 ilevel 的 list_block, 由于需要的总是最新的, 并且可能存在相隔开的两个同一 ilevel 的
        子列表,所以在 list_block 中需要倒序查询最近的子 list_block
        """
        pass

    def _reset_list_counters_for_new_sequence(self, numid: int):
        """
        开始新的编号序列时重置计数器。

        Args:
            numid: 列表编号ID
        """
        keys_to_reset = [key for key in self.list_counters.keys() if key[0] == numid]
        for key in keys_to_reset:
            self.list_counters[key] = 0

    def _get_heading_and_level(self, style_label: str) -> tuple[str, Optional[int]]:
        """
        从样式标签获取标题和层级。

        Args:
            style_label: 样式标签

        Returns:
            tuple[str, Optional[int]]: (标签字符串, 层级) 元组
        """
        parts = self._split_text_and_number(style_label)

        if len(parts) == 2:
            parts.sort()
            label_str: str = ""
            label_level: Optional[int] = 0
            if parts[0].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[1], None)
            if parts[1].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[0], None)
            return label_str, label_level

        return style_label, None

    def _split_text_and_number(self, input_string: str) -> list[str]:
        """
        分割字符串中的文本和数字部分。

        Args:
            input_string: 输入字符串

        Returns:
            list[str]: 分割后的部分列表
        """
        match = re.match(r"(\D+)(\d+)$|^(\d+)(\D+)", input_string)
        if match:
            parts = list(filter(None, match.groups()))
            return parts
        else:
            return [input_string]

    def _str_to_int(
        self, s: Optional[str], default: Optional[int] = 0
    ) -> Optional[int]:
        """
        将字符串转换为整数。

        Args:
            s: 要转换的字符串
            default: 默认值，转换失败时返回

        Returns:
            Optional[int]: 转换后的整数，转换失败时返回默认值
        """
        if s is None:
            return None
        try:
            return int(s)
        except ValueError:
            return default

    def _get_list_counter(self, numid: int, ilvl: int) -> int:
        """
        获取并递增特定 numId 和 ilvl 组合的计数器。

        Args:
            numid: 列表编号ID
            ilvl: 列表层级

        Returns:
            int: 当前计数器值
        """
        key = (numid, ilvl)
        if key not in self.list_counters:
            self.list_counters[key] = 0
        self.list_counters[key] += 1
        return self.list_counters[key]

    def _add_header_footer(self, docx_obj: DocxDocument) -> None:
        """
        处理页眉和页脚，按照分节顺序添加到 pages 列表中，过滤掉空字符串和纯数字内容
        分为整个文档是否启用奇偶页不同和每一节是否启用首页不同两种情况，
        暂不考虑包含图片和表格
        """
        is_odd_even_different = docx_obj.settings.odd_and_even_pages_header_footer
        for sec_idx, section in enumerate(docx_obj.sections):
            hdrs = [section.header]
            if is_odd_even_different:
                hdrs.append(section.even_page_header)
            if section.different_first_page_header_footer:
                hdrs.append(section.first_page_header)
            for hdr in hdrs:
                par = [
                    txt for txt in (par.text.strip() for par in hdr.paragraphs) if txt
                ]
                text = "".join(par)
                if text != "" and not text.isdigit():
                    try:
                        self.pages[sec_idx].append(
                            {
                                "type": BlockType.HEADER,
                                "content": text,
                            }
                        )
                    except IndexError:
                        logger.error("Section index out of range when adding header.")

            ftrs = [section.footer]
            if is_odd_even_different:
                ftrs.append(section.even_page_footer)
            if section.different_first_page_header_footer:
                ftrs.append(section.first_page_footer)
            for ftr in ftrs:
                par = [
                    txt for txt in (par.text.strip() for par in ftr.paragraphs) if txt
                ]
                text = "".join(par)
                if text != "" and not text.isdigit():
                    try:
                        self.pages[sec_idx].append(
                            {
                                "type": BlockType.FOOTER,
                                "content": text,
                            }
                        )
                    except IndexError:
                        logger.error("Section index out of range when adding header.")

    def _is_caption(self, element: BaseOxmlElement) -> bool:
        """
        根据 insertText 中是否有 SEQ 字段来判断是否为 caption

        Args:
            element: 段落元素对象

        Returns:
            bool: 如果是标题返回 True，否则返回 False
        """
        instr_texts = element.findall(
            ".//w:instrText", namespaces=DocxConverter._BLIP_NAMESPACES
        )

        for instr in instr_texts:
            if instr.text and "SEQ" in instr.text:
                return True
        else:
            return False
