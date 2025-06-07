import unittest
import os
from PIL import Image
from lxml import etree

from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel


class TestppTableModel(unittest.TestCase):
    def test_image2html(self):
        img = Image.open(os.path.join(os.path.dirname(__file__), "assets/table.jpg"))
        atom_model_manager = AtomModelSingleton()
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            lang='ch'
        )
        table_model = RapidTableModel(ocr_engine, 'slanet_plus')
        html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(img)
        # 验证生成的 HTML 是否符合预期
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_code, parser)

        # 检查 HTML 结构
        assert tree.find('.//table') is not None, "HTML should contain a <table> element"
        assert tree.find('.//tr') is not None, "HTML should contain a <tr> element"
        assert tree.find('.//td') is not None, "HTML should contain a <td> element"

        # 检查具体的表格内容
        headers = tree.xpath('//table/tr[1]/td')
        assert len(headers) == 5, "Thead should have 5 columns"
        assert headers[0].text and headers[0].text.strip() == "Methods", "First header should be 'Methods'"
        assert headers[1].text and headers[1].text.strip() == "R", "Second header should be 'R'"
        assert headers[2].text and headers[2].text.strip() == "P", "Third header should be 'P'"
        assert headers[3].text and headers[3].text.strip() == "F", "Fourth header should be 'F'"
        assert headers[4].text and headers[4].text.strip() == "FPS", "Fifth header should be 'FPS'"

        # 检查第一行数据
        first_row = tree.xpath('//table/tr[2]/td')
        assert len(first_row) == 5, "First row should have 5 cells"
        assert first_row[0].text and 'SegLink' in first_row[0].text.strip(), "First cell should be 'SegLink [26]'"
        assert first_row[1].text and first_row[1].text.strip() == "70.0", "Second cell should be '70.0'"
        assert first_row[2].text and first_row[2].text.strip() == "86.0", "Third cell should be '86.0'"
        assert first_row[3].text and first_row[3].text.strip() == "77.0", "Fourth cell should be '77.0'"
        assert first_row[4].text and first_row[4].text.strip() == "8.9", "Fifth cell should be '8.9'"

        # 检查倒数第二行数据
        second_last_row = tree.xpath('//table/tr[position()=last()-1]/td')
        assert len(second_last_row) == 5, "second_last_row should have 5 cells"
        assert second_last_row[0].text and second_last_row[0].text.strip() == "Ours (SynText)", "First cell should be 'Ours (SynText)'"
        assert second_last_row[1].text and second_last_row[1].text.strip() == "80.68", "Second cell should be '80.68'"
        assert second_last_row[2].text and second_last_row[2].text.strip() == "85.40", "Third cell should be '85.40'"
        # assert second_last_row[3].text and second_last_row[3].text.strip() == "82.97", "Fourth cell should be '82.97'"
        # assert second_last_row[3].text and second_last_row[4].text.strip() == "12.68", "Fifth cell should be '12.68'"


if __name__ == "__main__":
    unittest.main()
