import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union, Any, Final

import pandas as pd
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
from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.docx_formatting import Formatting, Script
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_reader import image_to_b64str

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
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
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

        # 存放文档字节数据，用于需要重读 ZIP 的辅助方法
        self._file_bytes: bytes = b''
        self.docx_obj = None
        self.pages = []
        self.cur_page = []
        self._mammoth_tables_html: list = []   # 完整文档 mammoth 预解析的表格 HTML 列表
        self._mammoth_table_idx: int = 0       # 当前预解析表格游标
        self.pre_num_id: int = -1  # 上一个处理元素的 numId
        self.pre_ilevel: int = -1  # 上一个处理元素的缩进等级, 用于判断列表层级
        self.list_block_stack: list = []  # 列表块堆栈
        self.list_counters: dict[tuple[int, int], int] = (
            {}
        )  # 列表计数器 (numId, ilvl) -> count
        self.index_block_stack: list = []  # 目录索引块堆栈
        self.pre_index_ilevel: int = -1  # 上一个目录项的缩进等级
        self.heading_list_numids: set = set()  # 用作章节标题的列表numId集合
        self.equation_bookends: str = "<eq>{EQ}</eq>"  # 公式标记格式
        self.chart_list = []  # 图表列表
        self.processed_textbox_elements: list = []

    @staticmethod
    def _escape_hyperlink_text(text: str) -> str:
        """
        转义超链接文本中的方括号。

        Args:
            text: 要转义的文本

        Returns:
            str: 转义后的文本
        """
        if not text:
            return text
        # 转义方括号
        text = text.replace("[", "\\[").replace("]", "\\]")
        return text

    @staticmethod
    def _minify_html(html: str) -> str:
        """
        移除HTML中的格式化空白（换行、缩进等）。

        Args:
            html: 要处理的HTML字符串

        Returns:
            str: 去除格式化后的HTML
        """
        if not html:
            return html
        # 移除标签之间的换行符和制表符
        html = re.sub(r'>\s+<', '><', html)
        # 移除行首尾无关的空白
        html = re.sub(r'\n\s*', '', html)
        return html

    @staticmethod
    def _escape_hyperlink_url(url: str) -> str:
        """
        转义超链接 URL 中的括号。

        Args:
            url: 要转义的 URL

        Returns:
            str: 转义后的 URL
        """
        if not url:
            return url
        # 对括号进行 URL 编码
        url = url.replace("(", "%28").replace(")", "%29")
        return url

    @staticmethod
    def _get_style_str_from_format(format_obj) -> Optional[str]:
        """
        从 Formatting 对象提取样式字符串。

        Args:
            format_obj: Formatting 对象

        Returns:
            Optional[str]: 样式字符串（如 "bold,italic"），无样式时返回 None
        """
        if format_obj is None:
            return None
        styles = []
        if format_obj.bold:
            styles.append('bold')
        if format_obj.italic:
            styles.append('italic')
        if format_obj.underline:
            styles.append('underline')
        if format_obj.strikethrough:
            styles.append('strikethrough')
        return ','.join(styles) if styles else None

    @staticmethod
    def _has_visible_style(format_obj) -> bool:
        """
        检查格式是否包含可见样式（下划线或删除线）。

        空白文本在有这些样式时仍然是可见的，应当保留。

        Args:
            format_obj: Formatting 对象

        Returns:
            bool: 是否包含可见样式
        """
        if format_obj is None:
            return False
        return bool(format_obj.underline or format_obj.strikethrough)

    @classmethod
    def _format_text_with_hyperlink(
        cls,
        text: str,
        hyperlink: Optional[Union[AnyUrl, Path, str]],
        style_str: Optional[str] = None,
    ) -> str:
        """
        将文本和超链接格式化，支持字体样式标记。

        无超链接时：有样式包裹为 <text style="...">文本</text>，无样式直接返回文本。
        有超链接时：格式化为 <hyperlink><text [style="..."]>文本</text><url>链接</url></hyperlink>。

        Args:
            text: 文本内容
            hyperlink: 超链接地址
            style_str: 样式字符串（如 "bold,italic"），无样式时为 None

        Returns:
            str: 格式化后的文本
        """
        if not text:
            return text

        # 检查超链接是否有效（非空）
        if hyperlink is None:
            # 无超链接：只有有样式时才包裹 <text> 标签
            if style_str:
                return f'<text style="{style_str}">{text}</text>'
            return text

        hyperlink_str = str(hyperlink)
        if not hyperlink_str or hyperlink_str.strip() == "" or hyperlink_str == ".":
            if style_str:
                return f'<text style="{style_str}">{text}</text>'
            return text

        # 有超链接：构建 <text> 标签（含可选样式）
        if style_str:
            text_tag = f'<text style="{style_str}">{text}</text>'
        else:
            text_tag = f'<text>{text}</text>'

        return f"<hyperlink>{text_tag}<url>{hyperlink_str}</url></hyperlink>"

    def _build_text_from_elements(
        self,
        paragraph_elements: list[
            tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]
        ],
    ) -> str:
        """
        从 paragraph_elements 重组文本，应用超链接格式和字体样式。

        Args:
            paragraph_elements: 段落元素列表

        Returns:
            str: 重组后的文本
        """
        result_parts = []
        for text, format_obj, hyperlink in paragraph_elements:
            if text:
                style_str = self._get_style_str_from_format(format_obj)
                formatted_text = self._format_text_with_hyperlink(text, hyperlink, style_str)
                result_parts.append(formatted_text)
        return "".join(result_parts) if result_parts else ""

    def _build_text_with_equations_and_hyperlinks(
        self,
        paragraph_elements: list[
            tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]
        ],
        text_with_equations: str,
        equations: list,
    ) -> str:
        """
        构建同时包含公式、超链接和字体样式的文本。

        Args:
            paragraph_elements: 段落元素列表，包含格式和超链接信息
            text_with_equations: 包含公式标记的原始文本
            equations: 公式列表

        Returns:
            str: 包含公式标记、超链接格式和字体样式的文本
        """
        if not equations:
            # 没有公式，直接返回带超链接和样式的文本
            return self._build_text_from_elements(paragraph_elements)

        # 检查是否有超链接
        has_hyperlink = any(
            hyperlink is not None and str(hyperlink).strip() not in ("", ".")
            for _, _, hyperlink in paragraph_elements
        )

        # 检查是否有字体样式
        has_style = any(
            fmt is not None and (fmt.bold or fmt.italic or fmt.underline or fmt.strikethrough)
            for _, fmt, _ in paragraph_elements
        )

        if not has_hyperlink and not has_style:
            # 没有超链接也没有样式，直接返回带公式的文本
            return text_with_equations

        # 同时有公式和超链接/样式，需要合并处理
        # 策略：在带公式的文本基础上，将样式/超链接标记插入到正确的位置

        # 1. 记录每个元素的原始文本和对应的格式化结果
        element_mappings = []
        for text, format_obj, hyperlink in paragraph_elements:
            if text:
                style_str = self._get_style_str_from_format(format_obj)
                formatted_text = self._format_text_with_hyperlink(text, hyperlink, style_str)
                element_mappings.append((text, formatted_text))

        # 2. 在 text_with_equations 中定位每个元素的原始文本，然后替换为格式化后的文本
        result_text = text_with_equations
        for original_text, formatted_text in element_mappings:
            if original_text != formatted_text:
                # 只有当文本被格式化（添加样式或超链接）时才需要替换
                result_text = self._replace_text_outside_equations(
                    result_text, original_text, formatted_text
                )

        return result_text

    def _replace_text_outside_equations(
        self, text: str, old_text: str, new_text: str
    ) -> str:
        """
        在公式标记外替换文本。

        Args:
            text: 原始文本
            old_text: 要替换的文本
            new_text: 替换后的文本

        Returns:
            str: 替换后的文本
        """
        # 分割文本为公式和非公式部分
        eq_pattern = re.compile(r"(<eq>.*?</eq>)")
        parts = eq_pattern.split(text)

        result_parts = []
        for part in parts:
            if part.startswith("<eq>") and part.endswith("</eq>"):
                # 公式部分，保持不变
                result_parts.append(part)
            else:
                # 非公式部分，进行替换
                result_parts.append(part.replace(old_text, new_text, 1))

        return "".join(result_parts)

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        # 重置所有实例状态，确保同一实例多次调用 convert() 时不会残留上次的数据
        self.pages = []
        self.cur_page = []
        self.pre_num_id = -1
        self.pre_ilevel = -1
        self.list_block_stack = []
        self.list_counters = {}
        self.index_block_stack = []
        self.pre_index_ilevel = -1
        self.heading_list_numids = set()
        self.chart_list = []
        self.processed_textbox_elements = []

        # 读取文件字节，以便 mammoth 和 python-docx 各自使用独立读取流
        file_bytes = file_stream.read()
        # 保存一份字节副本用于后续需要重新打开 ZIP 的方法
        self._file_bytes = file_bytes
        # 使用完整文档 mammoth 转换预解析所有表格，获得完整上下文（编号/图片/样式等）
        self._mammoth_tables_html = self._preparse_tables_with_mammoth(file_bytes)
        self._mammoth_table_idx = 0
        self.docx_obj = Document(BytesIO(file_bytes))
        # 预扫描文档，识别用作章节标题的列表numId
        self.heading_list_numids = self._detect_heading_list_numids()
        self.pages.append(self.cur_page)
        self._walk_linear(self.docx_obj.element.body)
        self._add_header_footer(self.docx_obj)
        self._add_chart_table()

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
            if drawingml_els:
                self._handle_drawingml(drawingml_els)

            # 检查文本框内容（支持多种文本框格式）
            # 仅当该元素之前未被处理时才处理
            if element not in self.processed_textbox_elements:
                # 现代 Word 文本框
                txbx_xpath = etree.XPath(
                    ".//w:txbxContent|.//v:textbox//w:p",
                    namespaces=DocxConverter._BLIP_NAMESPACES,
                )
                textbox_elements = txbx_xpath(element)

                # 未找到现代文本框，检查替代/旧版文本框格式
                if not textbox_elements and tag_name in ["drawing", "pict"]:
                    # 额外检查 DrawingML 和 VML 格式中的文本框
                    alt_txbx_xpath = etree.XPath(
                        ".//wps:txbx//w:p|.//w10:wrap//w:p|.//a:p//a:t",
                        namespaces=DocxConverter._BLIP_NAMESPACES,
                    )
                    textbox_elements = alt_txbx_xpath(element)

                    # 检查不在标准文本框内的形状文本
                    if not textbox_elements:
                        shape_text_xpath = etree.XPath(
                            ".//a:bodyPr/ancestor::*//a:t|.//a:txBody//a:t",
                            namespaces=DocxConverter._BLIP_NAMESPACES,
                        )
                        shape_text_elements = shape_text_xpath(element)
                        if shape_text_elements:
                            # 从形状文本创建自定义文本元素
                            text_content = " ".join(
                                [t.text for t in shape_text_elements if t.text]
                            )
                            if text_content.strip():
                                logger.debug(
                                    f"Found shape text: {text_content[:50]}..."
                                )
                                self.cur_page.append(
                                    {
                                        "type": BlockType.TEXT,
                                        "content": text_content,
                                    }
                                )
                if textbox_elements:
                    self.processed_textbox_elements.append(element)
                    for tb_element in textbox_elements:
                        self.processed_textbox_elements.append(tb_element)

                    logger.debug(
                        f"Found textbox content with {len(textbox_elements)} elements"
                    )
                    self._handle_textbox_content(textbox_elements)

            if tag_name == "tbl":
                # 表格是顶层块级元素，会中断活跃列表的上下文。
                # 若不重置列表状态，后续列表项会被追加到表格之前创建的列表块中，
                # 导致表格在 cur_page 中出现在那些列表项之后，产生顺序错乱。
                if self.pre_num_id != -1:
                    self.pre_num_id = -1
                    self.pre_ilevel = -1
                    self.list_block_stack = []
                    self.list_counters = {}
                try:
                    # 处理表格元素
                    self._handle_tables(element)
                except Exception:
                    # 如果表格解析失败，记录调试信息
                    logger.debug("could not parse a table, broken docx table")
            # 检查图片元素
            elif drawing_blip:
                # 判断图片是否为锚定（浮动）图片
                is_anchored = bool(
                    element.findall(
                        ".//wp:anchor",
                        namespaces=DocxConverter._BLIP_NAMESPACES,
                    )
                )
                # 锚定图片在段落中浮动定位，段落文本应出现在图片之前
                if is_anchored and tag_name == "p":
                    self._handle_text_elements(element)
                    self._handle_pictures(drawing_blip)
                else:
                    # 处理图片元素
                    self._handle_pictures(drawing_blip)
                    # 如果是段落元素，同时处理其中的文本内容（如描述性文字）
                    if tag_name == "p":
                        self._handle_text_elements(element)
            # 检查 sdt 元素
            elif tag_name == "sdt":
                sdt_content = element.find(
                    ".//w:sdtContent", namespaces=DocxConverter._BLIP_NAMESPACES
                )
                if sdt_content is not None:
                    if self._is_toc_sdt(element):
                        # 处理目录SDT，转换为INDEX块
                        self._handle_sdt_as_index(sdt_content)
                    else:
                        # 其他SDT元素，按普通文本处理
                        paragraphs = sdt_content.findall(
                            ".//w:p", namespaces=DocxConverter._BLIP_NAMESPACES
                        )
                        for p in paragraphs:
                            self._handle_text_elements(p)
            # 检查文本段落元素
            elif tag_name == "p":
                # 处理文本元素（包括段落属性如"tcPr", "sectPr"等）
                self._handle_text_elements(element)

            # 忽略其他未知元素并记录日志
            else:
                logger.debug(f"Ignoring element in DOCX with tag: {tag_name}")

    def _preparse_tables_with_mammoth(self, file_bytes: bytes) -> list:
        """
        使用 mammoth 完整文档转换预解析所有顶层表格的 HTML。

        孤立模式下（仅传入 <w:tbl> XML 片段），mammoth 缺少编号定义
        （word/numbering.xml）、样式（word/styles.xml）和关系
        （word/_rels/document.xml.rels）等上下文，在遇到含列表项或图片
        的单元格时会抛出 AttributeError。通过完整文档转换，mammoth 可
        获得完整上下文，从而正确处理这些情况。

        图片会被 mammoth 转换为内联 data-URI base64 格式（<img src="data:...">）。

        注意：mammoth 不支持 OMML（Office Math Markup Language）公式，会静默丢弃
        表格单元格内的公式。本方法在获取 mammoth HTML 后，会同步遍历原始 DOCX XML，
        将丢失的公式重新注入对应的 HTML 单元格。

        Returns:
            list[str]: 文档中所有顶层表格的 HTML 字符串列表，按文档顺序排列
        """
        try:
            import mammoth as _mammoth
            from bs4 import BeautifulSoup as _BeautifulSoup

            result = _mammoth.convert_to_html(BytesIO(file_bytes))
            soup = _BeautifulSoup(result.value, 'html.parser')

            # 仅保留顶层表格，排除嵌套在其他表格单元格内的子表格
            all_tables = soup.find_all('table')
            top_level_tables = [t for t in all_tables if not t.find_parent('table')]

            # 同步加载 DOCX XML，获取所有顶层表格元素，用于公式注入
            docx_obj = Document(BytesIO(file_bytes))
            xml_top_tables = [
                elem for elem in docx_obj.element.body
                if etree.QName(elem).localname == 'tbl'
            ]

            logger.debug(
                f"Pre-parsed {len(top_level_tables)} top-level tables via full mammoth conversion"
            )

            # 将 XML 表格中的 OMML 公式注入到 mammoth HTML 表格中
            result_tables = []
            for idx, html_table in enumerate(top_level_tables):
                if idx < len(xml_top_tables):
                    html_table = self._inject_equations_into_table(
                        html_table, xml_top_tables[idx]
                    )
                result_tables.append(str(html_table))
            return result_tables
        except Exception as e:
            logger.debug(f"Could not pre-parse tables with full mammoth conversion: {e}")
            return []

    def _inject_equations_into_table(self, html_table, xml_table):
        """
        将 DOCX XML 表格中的 OMML 公式注入到 mammoth 生成的 HTML 表格中。

        mammoth 会静默丢弃 OMML（Office Math Markup Language）公式，导致含公式
        的表格单元格在 HTML 中为空。本方法并行遍历 HTML 表格（BeautifulSoup 对象）
        和 XML 表格（lxml 元素），对含有 OMML 公式的单元格用包含公式占位符的内容
        替换原来的空内容。

        Args:
            html_table: BeautifulSoup 的 Tag 对象，代表 mammoth 生成的 <table> 元素
            xml_table: lxml 的 Element 对象，代表原始 DOCX 中对应的 <w:tbl> 元素

        Returns:
            BeautifulSoup Tag: 注入公式后的 <table> 元素（原地修改并返回）
        """
        OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
        W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        # 快速检查：该表格是否含有任何公式
        if not xml_table.findall(f".//{{{OMML_NS}}}oMath"):
            return html_table

        from bs4 import BeautifulSoup

        html_rows = html_table.find_all('tr')
        xml_rows = xml_table.findall(f"{{{W_NS}}}tr")

        if len(html_rows) != len(xml_rows):
            logger.debug(
                f"Table row count mismatch when injecting equations: "
                f"HTML {len(html_rows)} vs XML {len(xml_rows)}"
            )
            return html_table

        for html_row, xml_row in zip(html_rows, xml_rows):
            html_cells = html_row.find_all(['td', 'th'])
            xml_cells = xml_row.findall(f"{{{W_NS}}}tc")

            if len(html_cells) != len(xml_cells):
                continue

            for html_cell, xml_cell in zip(html_cells, xml_cells):
                if not xml_cell.findall(f".//{{{OMML_NS}}}oMath"):
                    continue

                # 该单元格含公式，重建其 HTML 内容以保留公式
                new_content = self._build_cell_html_with_equations(xml_cell)
                if new_content:
                    html_cell.clear()
                    new_soup = BeautifulSoup(new_content, 'html.parser')
                    for child in list(new_soup.children):
                        html_cell.append(child)

        return html_table

    def _build_cell_html_with_equations(self, xml_cell) -> str:
        """
        为含 OMML 公式的表格单元格构建 HTML 内容字符串。

        遍历单元格内的段落，将普通文本和 OMML 公式（转换为 LaTeX 占位符）
        混合在一起，生成与 mammoth 输出风格一致的 HTML 片段。

        Args:
            xml_cell: lxml Element，代表 DOCX 中的 <w:tc> 元素

        Returns:
            str: 单元格内容的 HTML 字符串，如 "<p>text<eq>latex</eq></p>"；
                 若单元格为空则返回空字符串
        """
        W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        parts = []
        for child in xml_cell:
            child_tag = etree.QName(child).localname
            if child_tag == 'p':
                para_html = self._build_paragraph_html_with_equations(child)
                if para_html is not None:
                    parts.append(para_html)
            # 嵌套表格暂不处理，由外层逻辑负责
        return ''.join(parts)

    def _build_paragraph_html_with_equations(self, xml_para) -> Optional[str]:
        """
        为可能含 OMML 公式的段落构建 HTML 字符串。

        使用与 _handle_equations_in_text 相同的迭代逻辑：
        - 普通 <w:t> 元素的文本直接收集
        - <m:oMath> 元素转换为 LaTeX 并包装为公式占位符 <eq>...</eq>
        - <m:t> 等 math 命名空间下的 <t> 元素因标签中含 "math" 而被跳过，
          避免在 oMath2Latex 已处理整个 oMath 子树后重复提取

        Args:
            xml_para: lxml Element，代表 DOCX 中的 <w:p> 元素

        Returns:
            str | None: 格式为 "<p>...</p>" 的 HTML 字符串；段落为空时返回 None
        """
        items = []
        for subt in xml_para.iter():
            tag_name = etree.QName(subt).localname
            # 普通文本节点（排除 math 命名空间下的 <m:t>）
            if tag_name == 't' and 'math' not in subt.tag:
                if isinstance(subt.text, str) and subt.text:
                    items.append(subt.text)
            # OMML 公式元素（排除 oMathPara 容器避免重复处理）
            elif 'oMath' in subt.tag and 'oMathPara' not in subt.tag:
                try:
                    latex = str(oMath2Latex(subt)).strip()
                    if latex:
                        items.append(self.equation_bookends.format(EQ=latex))
                except Exception as e:
                    logger.debug(f"Failed to convert OMML equation to LaTeX: {e}")

        if not items:
            return None
        return f'<p>{"".join(items)}</p>'

    def _handle_tables(self, element: BaseOxmlElement):
        """
        处理表格。

        优先使用完整文档 mammoth 转换的预解析结果（支持列表、图片、样式等
        复杂单元格内容），若预解析结果耗尽则回退到孤立 XML 解析模式。

        Args:
            element: 元素对象
        Returns:
            list[RefItem]: 元素引用列表
        """
        # 优先使用预解析表格（完整文档上下文，能正确处理列表/图片等）
        if self._mammoth_table_idx < len(self._mammoth_tables_html):
            html = self._mammoth_tables_html[self._mammoth_table_idx]
            self._mammoth_table_idx += 1
            html = self._normalize_table_colspans(html)
            table_block = {
                "type": BlockType.TABLE,
                "content": html,
            }
            self.cur_page.append(table_block)
            return

        # 回退：孤立 XML 解析模式（原始方案，不含文档上下文）
        table = read_str(element.xml)
        body_reader = body_xml.reader()
        t = body_reader.read_all([table])
        res = convert_document_element_to_html(t.value[0])
        html = self._normalize_table_colspans(res.value)
        table_block = {
            "type": BlockType.TABLE,
            "content": html,
        }
        self.cur_page.append(table_block)

    def _normalize_table_colspans(self, html: str) -> str:
        """
        修正 HTML 表格中因无线表/少线表导致的 colspan 不一致问题。

        在无边框或少边框的 DOCX 表格中，部分行的单元格包含 w:gridSpan 值，
        该值来自 Word 内部虚拟栅格，并不反映实际视觉列数。mammoth 将这些
        w:gridSpan 值直接转换为 HTML colspan 属性，导致不同行的有效列数
        （所有 colspan 之和）不一致，产生行列对不齐的问题。

        本方法检测此类不一致，并将有效列数过多的行的 colspan 缩减至
        最常见的目标列数，从而恢复表格的正确结构。

        算法：
        1. 计算每行的有效列数（该行所有单元格 colspan 之和）
        2. 取最常见的列数作为目标列数
        3. 对有效列数超过目标值的行，从第一个 colspan > 1 的单元格开始缩减

        Args:
            html: 包含表格的 HTML 字符串

        Returns:
            str: 修正后的 HTML 字符串
        """
        try:
            from bs4 import BeautifulSoup
            from collections import Counter

            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table')
            modified = False

            for table in tables:
                rows = table.find_all('tr')
                if not rows:
                    continue

                # 若表格中存在 rowspan > 1 的单元格，各行的显式 colspan 之和
                # 无法反映真实网格宽度（被 rowspan 占据的列不出现在后续行的 td
                # 列表中），此时算法的假设不成立，跳过该表格以避免误修改合法的
                # colspan。
                all_cells = table.find_all(['td', 'th'])
                if any(int(c.get('rowspan', 1)) > 1 for c in all_cells):
                    continue

                # 计算每行的有效列数（所有单元格的 colspan 之和）
                row_col_counts = []
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    total = sum(int(c.get('colspan', 1)) for c in cells)
                    row_col_counts.append(total)

                if not row_col_counts:
                    continue

                # 找到目标列数（出现最多的列数）
                count_freq = Counter(row_col_counts)
                if len(count_freq) == 1:
                    continue  # 各行列数已一致，无需修正

                target = count_freq.most_common(1)[0][0]

                # 修正有效列数超过目标值的行：缩减 colspan > 1 的单元格
                for row, col_count in zip(rows, row_col_counts):
                    if col_count <= target:
                        continue

                    excess = col_count - target
                    cells = row.find_all(['td', 'th'])

                    for cell in cells:
                        if excess <= 0:
                            break
                        span = int(cell.get('colspan', 1))
                        if span > 1:
                            reduce_by = min(span - 1, excess)
                            new_span = span - reduce_by
                            if new_span == 1:
                                if 'colspan' in cell.attrs:
                                    del cell['colspan']
                            else:
                                cell['colspan'] = str(new_span)
                            excess -= reduce_by
                            modified = True

            if modified:
                return str(soup)
            return html
        except Exception as e:
            logger.debug(f"Failed to normalize table colspans: {e}")
            return html

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
        if element.find(".//w:sectPr", namespaces=DocxConverter._BLIP_NAMESPACES) is not None:
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

            if numid in self.heading_list_numids:
                # 该列表被用作章节标题（列表项间穿插了正文内容），直接转换为title block
                # 先关闭任何活跃的普通列表
                if self.pre_num_id != -1:
                    self.pre_num_id = -1
                    self.pre_ilevel = -1
                    self.list_block_stack = []
                    self.list_counters = {}
                content_text = self._build_text_with_equations_and_hyperlinks(
                    paragraph_elements, text, equations
                )
                if content_text:
                    title_block = {
                        "type": BlockType.TITLE,
                        "level": ilevel + 1,
                        "is_numbered_style": is_numbered,
                        "content": content_text,
                    }
                    self.cur_page.append(title_block)
            else:
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
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(
                paragraph_elements, text, equations
            )
            if content_text != "":
                title_block = {
                    "type": BlockType.TITLE,
                    "level": 1,
                    "is_numbered_style": False,
                    "content": content_text,
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
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(
                paragraph_elements, text, equations
            )
            if content_text != "":
                h_block = {
                    "type": BlockType.TITLE,
                    "level": p_level if p_level is not None else 2,
                    "is_numbered_style": is_numbered_style,
                    "content": content_text,
                }
                self.cur_page.append(h_block)

        elif len(equations) > 0:
            if (paragraph.text is None or len(paragraph.text.strip()) == 0) and len(
                text
            ) > 0:
                # 独立公式
                eq_block = {
                    "type": BlockType.EQUATION,
                    "content": text.replace("<eq>", "").replace("</eq>", ""),
                }
                self.cur_page.append(eq_block)
            else:
                # 包含行内公式的文本块，同时支持超链接
                content_text = self._build_text_with_equations_and_hyperlinks(
                    paragraph_elements, text, equations
                )
                text_with_inline_eq_block = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                self.cur_page.append(text_with_inline_eq_block)
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
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(
                paragraph_elements, text, equations
            )
            if content_text != "":
                text_block = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                self.cur_page.append(text_block)
        # 判断是否是 Caption
        elif self._is_caption(element):
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(
                paragraph_elements, text, equations
            )
            if content_text != "":
                caption_block = {
                    "type": BlockType.CAPTION,
                    "content": content_text,
                }
                self.cur_page.append(caption_block)
        else:
            # 文本样式名称不仅有默认值，还可能有用户自定义值
            # 因此我们将所有其他标签视为纯文本
            # 构建包含公式和超链接的文本
            content_text = self._build_text_with_equations_and_hyperlinks(
                paragraph_elements, text, equations
            )
            if content_text != "":
                text_block = {
                    "type": BlockType.TEXT,
                    "content": content_text,
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

        def get_docx_image(image: Any) -> Optional[bytes]:
            """
            获取 DOCX 图像数据。

            Args:
                image: 单个 blip 元素

            Returns:

                Optional[bytes]: 图像数据
            """
            image_data: Optional[bytes] = None
            rId = image.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if rId in self.docx_obj.part.rels:
                # 使用关系 ID 访问图像部分
                image_part = self.docx_obj.part.rels[rId].target_part
                image_data = image_part.blob  # 获取二进制图像数据
            return image_data

        # 遍历所有 blip 元素，支持 group images（多个 blip）
        for image in drawing_blip:
            image_data: Optional[bytes] = get_docx_image(image)
            if image_data is None:
                logger.warning("Warning: image cannot be found")
            else:
                image_bytes = BytesIO(image_data)
                pil_image = Image.open(image_bytes)
                if pil_image.format in ("WMF", "EMF"):
                    if is_windows_environment():
                        # 在 Windows 上，Pillow 依赖底层的 Image.core.drawwmf 渲染
                        # 有时需要显式调用 .load() 确保矢量图被光栅化到内存中
                        try:
                            pil_image.load()
                            img_base64 = image_to_b64str(pil_image, image_format="PNG")
                        except OSError as e:
                            logger.warning(f"Failed to render {pil_image.format} image: {e}, size: {pil_image.size}. Using placeholder instead.")
                            # 如果渲染失败，创建与原图同样大小的浅灰色占位图
                            placeholder = Image.new("RGB", pil_image.size, (240, 240, 240))
                            img_base64 = image_to_b64str(placeholder, image_format="JPEG")
                    else:
                        logger.warning(f"Skipping {pil_image.format} image on non-Windows environment, size: {pil_image.size}")
                        # 创建与原图同样大小的浅灰色占位图
                        placeholder = Image.new("RGB", pil_image.size, (240, 240, 240))
                        img_base64 = image_to_b64str(placeholder, image_format="JPEG")
                else:
                    # 处理常规图片
                    if pil_image.mode != "RGB":
                        # RGBA, P, L 等模式保留原貌并存为 PNG (PNG支持透明度)
                        img_base64 = image_to_b64str(pil_image, image_format="PNG")
                    else:
                        # 纯 RGB 图片存为 JPEG 以减小体积
                        img_base64 = image_to_b64str(pil_image, image_format="JPEG")
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
            list[tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]]:
            段落元素列表，每个元素包含文本、格式和超链接信息
        """

        # 目前保留空段落以保持向后兼容性:
        if paragraph.text.strip() == "":
            # 检查是否存在带可见样式（下划线或删除线）的空白文本 run。
            # 有可见样式的空白文本（如带下划线的空格）在视觉上是可见的，应予保留，
            # 因此跳过提前返回，交由后续完整 run 处理流程处理。
            has_visible_style_run = any(
                isinstance(c, Run) and c.text and self._has_visible_style(self._get_format_from_run(c))
                for c in paragraph.iter_inner_content()
            )
            if not has_visible_style_run:
                return [("", None, None)]

        paragraph_elements: list[
            tuple[str, Optional[Formatting], Optional[Union[AnyUrl, Path, str]]]
        ] = []
        group_text = ""
        previous_format = None

        # 字段代码超链接内联检测状态（处理 w:fldChar + w:instrText 形式的超链接）
        _W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        _field_in = False       # 当前是否在字段域内
        _field_url = None       # 当前字段域解析出的 URL
        _field_phase = None     # 'instr' 或 'result'
        _field_acc_text = ""    # 累积的显示文本
        _field_acc_format = None  # 首个显示 run 的格式

        # 遍历段落的 runs 并按格式分组
        for c in paragraph.iter_inner_content():
            if isinstance(c, Hyperlink):
                text = c.text
                # 若地址为 URL（含 ://），直接保留字符串，避免 Path 将 // 规范化为 /
                address = c.address
                if address and "://" in address:
                    hyperlink = address
                else:
                    hyperlink = Path(address) if address else Path(".")
                format = (
                    self._get_format_from_run(c.runs[0])
                    if c.runs and len(c.runs) > 0
                    else None
                )
            elif isinstance(c, Run):
                # ---- 字段代码超链接内联检测 ----
                fld_char = c._element.find(f"{{{_W_NS}}}fldChar")
                if fld_char is not None:
                    fld_type = fld_char.get(f"{{{_W_NS}}}fldCharType")
                    if fld_type == "begin":
                        _field_in = True
                        _field_url = None
                        _field_phase = "instr"
                        _field_acc_text = ""
                        _field_acc_format = None
                        continue
                    elif fld_type == "separate":
                        _field_phase = "result"
                        continue
                    elif fld_type == "end":
                        if _field_url and _field_acc_text.strip():
                            # 将累积的字段代码超链接作为一个整体处理
                            text = _field_acc_text
                            hyperlink = _field_url
                            format = _field_acc_format
                        else:
                            _field_in = False
                            _field_url = None
                            _field_phase = None
                            _field_acc_text = ""
                            _field_acc_format = None
                            continue
                        _field_in = False
                        _field_url = None
                        _field_phase = None
                        _field_acc_text = ""
                        _field_acc_format = None
                        # 继续执行下方的 hyperlink 统一处理逻辑
                    else:
                        continue
                else:
                    instr_elem = c._element.find(f"{{{_W_NS}}}instrText")
                    if instr_elem is not None and _field_phase == "instr":
                        # 捕获 HYPERLINK 指令中的 URL
                        if instr_elem.text:
                            m = re.search(r'HYPERLINK\s+"([^"]+)"', instr_elem.text)
                            if m:
                                _field_url = m.group(1)
                        continue

                    if _field_in and _field_phase == "result":
                        # 显示文本 run：累积到字段文本
                        t_elem = c._element.find(f"{{{_W_NS}}}t")
                        if t_elem is not None:
                            _field_acc_text += c.text
                            if _field_acc_format is None:
                                _field_acc_format = self._get_format_from_run(c)
                        continue

                    # 普通 run
                    text = c.text
                    hyperlink = None
                    format = self._get_format_from_run(c)
            else:
                continue

            # 当新 run 有可见内容（非空或带可见样式的空白）且格式变化时触发分组
            has_visible_content = len(text.strip()) > 0 or self._has_visible_style(format)
            if (has_visible_content and format != previous_format) or (
                hyperlink is not None
            ):
                # 前一组有实质内容（非空或带可见样式的空白）时才保存
                prev_has_visible = len(group_text.strip()) > 0 or (
                    group_text and self._has_visible_style(previous_format)
                )
                if prev_has_visible:
                    paragraph_elements.append(
                        (group_text, previous_format, None)
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
        # 注意：使用 previous_format（当前累积组的格式），而非 format（最后一次循环迭代的格式）。
        # 最后一次迭代可能是无样式的空 run，若使用 format 会导致样式丢失。
        last_has_visible = len(group_text.strip()) > 0 or (
            group_text and self._has_visible_style(previous_format)
        )
        if last_has_visible:
            paragraph_elements.append((group_text, previous_format, None))

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

        # 检测着重符号 (w:em)：若存在非 none 的 em 值，则视为下划线样式
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        rPr = run._element.find(f'{{{_W}}}rPr')
        if rPr is not None:
            em = rPr.find(f'{{{_W}}}em')
            if em is not None:
                em_val = em.get(f'{{{_W}}}val', '')
                if em_val and em_val != 'none':
                    is_underline = True

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

        生成的列表结构：
        {
            "type": "list",
            "attribute": "ordered" / "unordered",
            "ilevel": 0,
            "content": [
                {"type": "text", "content": "列表项文本"},
                {"type": "list", "attribute": "...", "ilevel": 1, "content": [...]},
                {"type": "text", "content": "另一个列表项"}
            ]
        }

        Args:
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
        if not elements:
            return None

        # 构建 content_text，处理行内公式和超链接
        content_text = self._build_text_with_equations_and_hyperlinks(
            elements, text, equations
        )

        # 确定列表属性
        list_attribute = "ordered" if is_numbered else "unordered"

        # 情况 1: 不存在上一个列表ID，创建新的顶层列表
        if self.pre_num_id == -1:
            # 为新编号序列重置计数器，确保编号从1开始
            self._reset_list_counters_for_new_sequence(numid)

            list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "content": [],
                "ilevel": ilevel,
            }
            self.cur_page.append(list_block)
            # 入栈, 记录当前的列表块
            self.list_block_stack.append(list_block)

            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }

            list_block["content"].append(list_item)
            self.pre_num_id = numid
            self.pre_ilevel = ilevel

        # 情况 2: 增加缩进，打开子列表
        elif (
            self.pre_num_id == numid  # 同一个列表
            and self.pre_ilevel != -1  # 上一个缩进级别已知
            and self.pre_ilevel < ilevel  # 当前层级比之前更缩进
        ):
            # 创建新的子列表块
            child_list_block = {
                "type": BlockType.LIST,
                "attribute": list_attribute,
                "content": [],
                "ilevel": ilevel,
            }

            # 获取栈顶的列表块，将子列表直接添加到其content中
            parent_list_block = self.list_block_stack[-1]
            parent_list_block["content"].append(child_list_block)

            # 入栈, 记录当前的列表块
            self.list_block_stack.append(child_list_block)

            # 添加当前列表项到子列表
            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            child_list_block["content"].append(list_item)

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

            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            list_block["content"].append(list_item)
            self.pre_ilevel = ilevel

        # 情况 4: 同级列表项（相同缩进）
        elif self.pre_num_id == numid or self.pre_ilevel == ilevel:
            # 获取栈顶的列表块
            list_block = self.list_block_stack[-1]


            list_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            list_block["content"].append(list_item)

    def _detect_heading_list_numids(self) -> set:
        """
        预扫描文档，检测用作章节标题的列表numId。

        判断依据（需同时满足两个条件）：
        1. 该numId的列表项之间穿插了非列表的正文内容（段落/表格等）；
        2. 该numId的列表项出现在**多个不同的缩进层级**（ilevel > 1种），
           即为真正的多级列表结构，而非普通的单级内容条目列表。

        这样可以避免将"多段内容条目之间穿插了小标签"的单级列表误判为标题列表。

        Returns:
            set: 应当转换为标题块的列表numId集合
        """
        heading_numids = set()
        # 收集文档元素序列：("list", numid, ilevel) 或 ("content",)
        items = []
        # 记录每个numId出现过的所有ilevel，用于判断是否为真正的多级列表
        numid_ilvels: dict[int, set] = {}

        for element in self.docx_obj.element.body:
            tag_name = etree.QName(element).localname
            if tag_name == "p":
                try:
                    paragraph = Paragraph(element, self.docx_obj)
                    p_style_id, _ = self._get_label_and_level(paragraph)
                    numid, ilevel = self._get_numId_and_ilvl(paragraph)
                    if numid == 0:
                        numid = None
                    text = paragraph.text.strip() if paragraph.text else ""
                except Exception:
                    continue

                if (
                    numid is not None
                    and ilevel is not None
                    and p_style_id not in ["Title", "Heading"]
                    and text
                ):
                    items.append(("list", numid, ilevel))
                    if numid not in numid_ilvels:
                        numid_ilvels[numid] = set()
                    numid_ilvels[numid].add(ilevel)
                elif p_style_id not in ["Title", "Heading"] and text:
                    items.append(("content", None, None))
            elif tag_name == "tbl":
                items.append(("content", None, None))

        # 对每个numId，检测其列表项之间是否有正文内容穿插
        # seen_numids[numid] = True 表示该numId的最后一个列表项之后出现了正文内容
        seen_numids: dict[int, bool] = {}

        for item_type, numid, ilevel in items:
            if item_type == "list":
                if numid in seen_numids and seen_numids[numid]:
                    # 上次列表项之后出现了正文内容，满足条件1
                    heading_numids.add(numid)
                seen_numids[numid] = False  # 重置：记录该numId出现了新列表项
            elif item_type == "content":
                # 将所有已见numId标记为"之后出现了正文内容"
                for nid in seen_numids:
                    seen_numids[nid] = True

        # 条件2：只保留真正的多级列表（出现过多于1种ilevel的numId）
        # 单级列表（如只有ilevel=0的内容条目列表）即使有正文段落穿插也不应转换为标题
        heading_numids = {
            nid for nid in heading_numids
            if len(numid_ilvels.get(nid, set())) > 1
        }

        if heading_numids:
            logger.debug(
                f"Detected heading-style list numIds (will convert to title blocks): {heading_numids}"
            )

        return heading_numids

    def _reset_list_counters_for_new_sequence(self, numid: int):
        """
        开始新的编号序列时重置计数器。

        Args:
            numid: 列表编号ID
        """
        keys_to_reset = [key for key in self.list_counters.keys() if key[0] == numid]
        for key in keys_to_reset:
            self.list_counters[key] = 0

    def _is_toc_sdt(self, element: BaseOxmlElement) -> bool:
        """
        检测SDT元素是否为目录(Table of Contents)。

        检测策略：
        1. 检查 w:sdtPr 中的 docPartGallery 或 tag 元素
        2. 回退到检查内容中的段落样式是否为 "TOC N" 格式

        Args:
            element: SDT XML元素

        Returns:
            bool: 如果是目录SDT返回 True，否则返回 False
        """
        # 方法1: 检查 w:sdtPr 中的 docPartGallery
        sdt_pr = element.find("w:sdtPr", namespaces=DocxConverter._BLIP_NAMESPACES)
        if sdt_pr is not None:
            doc_part_gallery = sdt_pr.find(
                ".//w:docPartGallery", namespaces=DocxConverter._BLIP_NAMESPACES
            )
            if doc_part_gallery is not None:
                val = doc_part_gallery.get(self.XML_KEY, "")
                if "Table of Contents" in val or "toc" in val.lower():
                    return True

            # 检查 tag 元素的值
            tag_elem = sdt_pr.find("w:tag", namespaces=DocxConverter._BLIP_NAMESPACES)
            if tag_elem is not None:
                val = tag_elem.get(self.XML_KEY, "").lower().replace(" ", "")
                if "toc" in val or "contents" in val or "tableofcontents" in val:
                    return True

        # 方法2: 检查内容段落的样式是否为 "TOC N" 格式
        sdt_content = element.find(
            "w:sdtContent", namespaces=DocxConverter._BLIP_NAMESPACES
        )
        if sdt_content is not None:
            paragraphs = sdt_content.findall(
                "w:p", namespaces=DocxConverter._BLIP_NAMESPACES
            )
            for p in paragraphs[:5]:  # 只检查前5个段落即可判断
                try:
                    p_obj = Paragraph(p, self.docx_obj)
                    if p_obj.style and p_obj.style.name:
                        style_name = p_obj.style.name
                        if re.match(r'^TOC\s*\d+$', style_name, re.IGNORECASE) or \
                           re.match(r'^目录\s*\d+$', style_name):
                            return True
                except Exception:
                    continue

        return False

    def _get_toc_item_level(self, paragraph: Paragraph) -> Optional[int]:
        """
        从段落样式中获取目录项的层级（0-based）。

        "TOC 1" -> 0
        "TOC 2" -> 1
        "目录 1" -> 0

        Args:
            paragraph: 段落对象

        Returns:
            Optional[int]: 层级（0-based），如果不是目录样式则返回 None
        """
        if paragraph.style is None:
            return None
        style_name = paragraph.style.name
        if style_name:
            match = re.match(r'^(?:TOC|目录)\s*(\d+)$', style_name, re.IGNORECASE)
            if match:
                level = int(match.group(1))
                return level - 1  # 转换为 0-based
        return None

    def _is_flat_list_toc(
        self, items: list[tuple[int, str, list, list]]
    ) -> bool:
        """
        检测目录是否为扁平列表（插图清单、列表清单等），
        这类目录的所有条目应在同一层级，不应嵌套。

        策略：检查是否超过 50% 的条目以"图"或"表"开头。
        """
        match_count = 0
        total_count = 0
        for _level, text, _elements, _equations in items:
            stripped = text.strip()
            if not stripped:
                continue
            total_count += 1
            if re.match(r'^[图表][\d\s.]', stripped) or re.match(
                r'^(Figure|Table)\s+\d', stripped, re.IGNORECASE
            ):
                match_count += 1
        if total_count == 0:
            return False
        return match_count / total_count > 0.5

    def _correct_toc_level_by_text(self, toc_level: int, text: str) -> int:
        """
        通过文本中的编号深度修正目录项的层级。

        仅对 toc_level > 0 的条目进行修正，避免影响顶层章节标题。
        例如：
        - "1.1 LYSO..." (toc 3 → ilevel=2) → text depth 2 → 返回 1
        - "1.1.1 LYSO..." (toc 3 → ilevel=2) → text depth 3 → 返回 2
        - "本章小结" (toc 1 → ilevel=0) → 返回 0（不修正）
        """
        if toc_level == 0:
            return 0
        stripped = text.strip()
        match = re.match(r'^(\d+(?:\.\d+)*)', stripped)
        if match:
            parts = match.group(1).split('.')
            # "1.1" -> 2 parts -> level 1; "1.1.1" -> 3 parts -> level 2
            return len(parts) - 1
        return toc_level

    def _add_index_item(
        self,
        *,
        ilevel: int,
        elements: list,
        text: str = "",
        equations: list = None,
    ) -> None:
        """
        添加目录项到索引块。

        生成的索引结构：
        {
            "type": "index",
            "ilevel": 0,
            "content": [
                {"type": "text", "content": "目录项文本"},
                {"type": "index", "ilevel": 1, "content": [...]},
            ]
        }

        Args:
            ilevel: 缩进等级（0-based）
            elements: 元素列表
            text: 处理后的文本（包含公式标记）
            equations: 公式列表
        """
        if equations is None:
            equations = []
        if not elements:
            return

        content_text = self._build_text_with_equations_and_hyperlinks(
            elements, text, equations
        )

        # 情况 1: 首个目录项，创建新的顶层索引块
        if self.pre_index_ilevel == -1:
            index_block = {
                "type": BlockType.INDEX,
                "content": [],
                "ilevel": ilevel,
            }
            self.cur_page.append(index_block)
            self.index_block_stack.append(index_block)

            index_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 2: 增加缩进，打开子索引块
        elif self.pre_index_ilevel < ilevel:
            child_index_block = {
                "type": BlockType.INDEX,
                "content": [],
                "ilevel": ilevel,
            }
            parent_index_block = self.index_block_stack[-1]
            parent_index_block["content"].append(child_index_block)
            self.index_block_stack.append(child_index_block)

            index_item = {
                "type": BlockType.TEXT,
                "content": content_text,
            }
            child_index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 3: 减少缩进，关闭子索引块
        elif ilevel < self.pre_index_ilevel:
            while self.index_block_stack:
                top_block = self.index_block_stack[-1]
                if top_block["ilevel"] == ilevel:
                    break
                self.index_block_stack.pop()
            if self.index_block_stack:
                index_block = self.index_block_stack[-1]
                index_item = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                index_block["content"].append(index_item)
            self.pre_index_ilevel = ilevel

        # 情况 4: 同级目录项
        else:
            if self.index_block_stack:
                index_block = self.index_block_stack[-1]
                index_item = {
                    "type": BlockType.TEXT,
                    "content": content_text,
                }
                index_block["content"].append(index_item)

    def _handle_sdt_as_index(self, sdt_content: BaseOxmlElement) -> None:
        """
        处理目录SDT内容，将其转换为层级化的INDEX块。

        两阶段处理：
        1. 收集所有段落及其层级；
        2. 检测目录类型（常规目录 vs 扁平列表），对层级进行修正后写入索引块。

        Args:
            sdt_content: w:sdtContent XML元素
        """
        paragraphs = sdt_content.findall(
            ".//w:p", namespaces=DocxConverter._BLIP_NAMESPACES
        )

        # --- 第一阶段：收集所有条目 ---
        toc_items: list[tuple[int, str, list, list]] = []
        for p in paragraphs:
            try:
                p_obj = Paragraph(p, self.docx_obj)
                paragraph_elements = self._get_paragraph_elements(p_obj)
                text, equations = self._handle_equations_in_text(
                    element=p, text=p_obj.text
                )
                if text is None:
                    continue
                text = text.strip()
                if not text:
                    continue

                toc_level = self._get_toc_item_level(p_obj)
                if toc_level is None:
                    toc_level = 0

                toc_items.append((toc_level, text, paragraph_elements, equations))
            except Exception as e:
                logger.debug(f"Error collecting TOC paragraph: {e}")
                continue

        # --- 第二阶段：修正层级并写入索引块 ---
        is_flat = self._is_flat_list_toc(toc_items)

        # 重置索引状态，开始新的目录块
        self.index_block_stack = []
        self.pre_index_ilevel = -1

        for toc_level, text, elements, equations in toc_items:
            if is_flat:
                # 插图/列表清单：强制全部扁平（层级 0）
                corrected_level = 0
            else:
                # 常规目录：依据文本编号深度修正层级，解决 docx 跳级问题
                corrected_level = self._correct_toc_level_by_text(toc_level, text)

            self._add_index_item(
                ilevel=corrected_level,
                elements=elements,
                text=text,
                equations=equations,
            )

        # 处理完成后重置索引状态
        self.index_block_stack = []
        self.pre_index_ilevel = -1

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

    def _process_header_footer_paragraph(self, paragraph: Paragraph) -> str:
        """
        处理页眉/页脚中的单个段落，支持行内公式和超链接。

        Args:
            paragraph: 段落对象

        Returns:
            str: 处理后的文本内容（包含公式标记和超链接格式）
        """
        paragraph_elements = self._get_paragraph_elements(paragraph)
        text, equations = self._handle_equations_in_text(
            element=paragraph._element, text=paragraph.text
        )

        if text is None:
            return ""

        text = text.strip()
        if not text:
            return ""

        # 构建包含公式和超链接的文本
        content_text = self._build_text_with_equations_and_hyperlinks(
            paragraph_elements, text, equations
        )

        return content_text

    def _add_header_footer(self, docx_obj: DocxDocument) -> None:
        """
        处理页眉和页脚，按照分节顺序添加到 pages 列表中，过滤掉空字符串和纯数字内容
        分为整个文档是否启用奇偶页不同和每一节是否启用首页不同两种情况，
        支持行内公式和超链接，并根据类型去重
        """
        is_odd_even_different = docx_obj.settings.odd_and_even_pages_header_footer
        for sec_idx, section in enumerate(docx_obj.sections):
            # 用于去重的集合
            added_headers = set()
            added_footers = set()

            hdrs = [section.header]
            if is_odd_even_different:
                hdrs.append(section.even_page_header)
            if section.different_first_page_header_footer:
                hdrs.append(section.first_page_header)
            for hdr in hdrs:
                # 处理每个段落，支持公式和超链接
                processed_parts = []
                for par in hdr.paragraphs:
                    content = self._process_header_footer_paragraph(par)
                    if content:
                        processed_parts.append(content)
                text = " ".join(processed_parts)
                if text != "" and not text.isdigit() and text not in added_headers:
                    added_headers.add(text)
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
                # 处理每个段落，支持公式和超链接
                processed_parts = []
                for par in ftr.paragraphs:
                    content = self._process_header_footer_paragraph(par)
                    if content:
                        processed_parts.append(content)
                text = " ".join(processed_parts)
                if text != "" and not text.isdigit() and text not in added_footers:
                    added_footers.add(text)
                    try:
                        self.pages[sec_idx].append(
                            {
                                "type": BlockType.FOOTER,
                                "content": text,
                            }
                        )
                    except IndexError:
                        logger.error("Section index out of range when adding footer.")

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

        return False
    
    def _handle_drawingml(self, elements: list[BaseOxmlElement]):
        """
        处理 DrawingML 元素，目前先处理 chart 元素。

        Args:
            elements: 包含 DrawingML 元素的列表

        Returns:

        """
        for element in elements:
            chart = element.find(
                ".//c:chart", namespaces=DocxConverter._BLIP_NAMESPACES
            )
            if chart is not None:
                # 如果找到 chart 元素，构造空的表格块，后续回填html
                table_block = {
                    "type": BlockType.TABLE,
                    "content": "",
                }
                self.cur_page.append(table_block)
                self.chart_list.append(table_block)

    def _add_chart_table(self):
        idx_xlsx_map = {}
        rel_pattern = re.compile(r"word/charts/_rels/chart(\d+)\.xml\.rels$")

        # 定义命名空间
        namespaces = {
            "r": "http://schemas.openxmlformats.org/package/2006/relationships"
        }

        # first pass: read relationships from rewindable byte buffer
        with zipfile.ZipFile(BytesIO(self._file_bytes), "r") as zf:
            for name in zf.namelist():
                match = rel_pattern.match(name)
                if match:
                    # 读取 .rels 文件内容
                    rels_content = zf.read(name)
                    # 解析 XML
                    rels_root = etree.fromstring(rels_content)

                    # 查找所有 Relationship 元素
                    for rel in rels_root.findall(
                        ".//r:Relationship", namespaces=namespaces
                    ):
                        target = rel.get("Target")
                        if target and target.endswith(".xlsx"):
                            path = Path(target)
                            idx_xlsx_map[path.name] = int(match.group(1))

        # second pass: again open buffer rather than original stream
        with zipfile.ZipFile(BytesIO(self._file_bytes), "r") as zf:
            for name in zf.namelist():
                if name.startswith("word/embeddings/"):
                    for path_name, chart_idx in idx_xlsx_map.items():
                        if name.endswith(path_name):
                            content = zf.read(name)
                            excel_data = pd.read_excel(BytesIO(content))
                            html = excel_data.to_html(index=False, header=True)
                            self.chart_list[chart_idx - 1]["content"] = self._minify_html(html)

    def _handle_textbox_content(
        self,
        textbox_elements: list,
    ):
        """
        处理文本框内容并将其添加到文档结构。
        """
        # 收集并组织段落
        container_paragraphs = self._collect_textbox_paragraphs(textbox_elements)

        # 处理所有段落
        all_paragraphs = []

        # 对每个容器内的段落进行排序，然后按容器顺序处理
        for paragraphs in container_paragraphs.values():
            # 按容器内的垂直位置进行排序
            sorted_container_paragraphs = sorted(
                paragraphs,
                key=lambda x: (
                    x[1] is None,
                    x[1] if x[1] is not None else float("inf"),
                ),
            )

            # 将排序后的段落添加到待处理列表
            all_paragraphs.extend(sorted_container_paragraphs)

        # 跟踪已处理段落以避免重复（相同内容和位置）
        processed_paragraphs = set()

        # 处理所有段落
        for p, position in all_paragraphs:
            # 创建 Paragraph 对象以获取文本内容
            paragraph = Paragraph(p, self.docx_obj)
            text_content = paragraph.text

            # 基于内容和位置创建唯一标识
            paragraph_id = (text_content, position)

            # 如果该段落（相同内容和位置）已处理，则跳过
            if paragraph_id in processed_paragraphs:
                logger.debug(
                    f"Skipping duplicate paragraph: content='{text_content[:50]}...', position={position}"
                )
                continue

            # 将该段落标记为已处理
            processed_paragraphs.add(paragraph_id)

            self._handle_text_elements(p)
        return

    def _collect_textbox_paragraphs(self, textbox_elements):
        """
        从文本框元素中收集并组织段落。
        """
        processed_paragraphs = []
        container_paragraphs = {}

        for element in textbox_elements:
            element_id = id(element)
            # 如果已处理相同元素，则跳过
            if element_id in processed_paragraphs:
                continue

            tag_name = etree.QName(element).localname
            processed_paragraphs.append(element_id)

            # 处理直接找到的段落（VML 文本框）
            if tag_name == "p":
                # 查找包含该段落的文本框或形状元素
                container_id = None
                for ancestor in element.iterancestors():
                    if any(ns in ancestor.tag for ns in ["textbox", "shape", "txbx"]):
                        container_id = id(ancestor)
                        break

                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []
                container_paragraphs[container_id].append(
                    (element, self._get_paragraph_position(element))
                )

            # 处理 txbxContent 元素（Word DrawingML 文本框）
            elif tag_name == "txbxContent":
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append(
                            (p, self._get_paragraph_position(p))
                        )
            else:
                # 尝试从未知元素中提取任何段落
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append(
                            (p, self._get_paragraph_position(p))
                        )

        return container_paragraphs

    def _get_paragraph_position(self, paragraph_element):
        """
        从段落元素提取垂直位置信息。
        """
        # 先尝试直接从包含顺序相关属性的 w:p 元素获取索引
        if (
            hasattr(paragraph_element, "getparent")
            and paragraph_element.getparent() is not None
        ):
            parent = paragraph_element.getparent()
            # 获取所有段落兄弟节点
            paragraphs = [
                p for p in parent.getchildren() if etree.QName(p).localname == "p"
            ]
            # 查找当前段落在其兄弟节点中的索引
            try:
                paragraph_index = paragraphs.index(paragraph_element)
                return paragraph_index  # 使用索引作为位置以保证一致的排序
            except ValueError:
                pass

        # 在元素及其祖先中查找位置提示属性
        for elem in (*[paragraph_element], *paragraph_element.iterancestors()):
            # 检查直接的位置信息属性
            for attr_name in ["y", "top", "positionY", "y-position", "position"]:
                value = elem.get(attr_name)
                if value:
                    try:
                        # 移除任何非数字字符（如 'pt', 'px' 等）
                        clean_value = re.sub(r"[^0-9.]", "", value)
                        if clean_value:
                            return float(clean_value)
                    except (ValueError, TypeError):
                        pass

            # 检查 transform 属性中的位移信息
            transform = elem.get("transform")
            if transform:
                # 从 transform 矩阵中提取 translate 的第二个参数
                match = re.search(r"translate\([^,]+,\s*([0-9.]+)", transform)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

            # 检查 Word 格式中的锚点或相对位置指示器
            # 'dist' 类属性可以表示相对位置
            for attr_name in ["distT", "distB", "anchor", "relativeFrom"]:
                if elem.get(attr_name) is not None:
                    return elem.sourceline  # 使用 XML 源行号作为回退

        # 针对 VML 形状，查找特定属性
        for ns_uri in paragraph_element.nsmap.values():
            if "vml" in ns_uri:
                # 尝试从 style 属性提取 top 值
                style = paragraph_element.get("style")
                if style:
                    match = re.search(r"top:([0-9.]+)pt", style)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            pass

        # 如果没有更好的位置指示，则使用 XML 源行号作为顺序的代理
        return (
            paragraph_element.sourceline
            if hasattr(paragraph_element, "sourceline")
            else None
        )
