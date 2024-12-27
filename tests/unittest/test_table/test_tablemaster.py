import unittest
from PIL import Image
from lxml import etree

from magic_pdf.model.sub_modules.table.tablemaster.tablemaster_paddle import TableMasterPaddleModel


class TestppTableModel(unittest.TestCase):
    def test_image2html(self):
        img = Image.open("tests/unittest/test_table/assets/table.jpg")
        # 修改table模型路径
        config = {"device": "cuda",
                  "model_dir": "/home/quyuan/.cache/modelscope/hub/opendatalab/PDF-Extract-Kit/models/TabRec/TableMaster"}
        table_model = TableMasterPaddleModel(config)
        res = table_model.img2html(img)
        # 验证生成的 HTML 是否符合预期
        parser = etree.HTMLParser()
        tree = etree.fromstring(res, parser)

        # 检查 HTML 结构
        assert tree.find('.//table') is not None, "HTML should contain a <table> element"
        assert tree.find('.//thead') is not None, "HTML should contain a <thead> element"
        assert tree.find('.//tbody') is not None, "HTML should contain a <tbody> element"
        assert tree.find('.//tr') is not None, "HTML should contain a <tr> element"
        assert tree.find('.//td') is not None, "HTML should contain a <td> element"

        # 检查具体的表格内容
        headers = tree.xpath('//thead/tr/td/b')
        print(headers)  # Print headers for debugging
        assert len(headers) == 5, "Thead should have 5 columns"
        assert headers[0].text and headers[0].text.strip() == "Methods", "First header should be 'Methods'"
        assert headers[1].text and headers[1].text.strip() == "R", "Second header should be 'R'"
        assert headers[2].text and headers[2].text.strip() == "P", "Third header should be 'P'"
        assert headers[3].text and headers[3].text.strip() == "F", "Fourth header should be 'F'"
        assert headers[4].text and headers[4].text.strip() == "FPS", "Fifth header should be 'FPS'"

        # 检查第一行数据
        first_row = tree.xpath('//tbody/tr[1]/td')
        assert len(first_row) == 5, "First row should have 5 cells"
        assert first_row[0].text and first_row[0].text.strip() == "SegLink[26]", "First cell should be 'SegLink[26]'"
        assert first_row[1].text and first_row[1].text.strip() == "70.0", "Second cell should be '70.0'"
        assert first_row[2].text and first_row[2].text.strip() == "86.0", "Third cell should be '86.0'"
        assert first_row[3].text and first_row[3].text.strip() == "77.0", "Fourth cell should be '77.0'"
        assert first_row[4].text and first_row[4].text.strip() == "8.9", "Fifth cell should be '8.9'"

        # 检查倒数第二行数据
        second_last_row = tree.xpath('//tbody/tr[position()=last()-1]/td')
        assert len(second_last_row) == 5, "second_last_row should have 5 cells"
        assert second_last_row[0].text and second_last_row[
            0].text.strip() == "Ours (SynText)", "First cell should be 'Ours (SynText)'"
        assert second_last_row[1].text and second_last_row[1].text.strip() == "80.68", "Second cell should be '80.68'"
        assert second_last_row[2].text and second_last_row[2].text.strip() == "85.40", "Third cell should be '85.40'"
        assert second_last_row[3].text and second_last_row[3].text.strip() == "82.97", "Fourth cell should be '82.97'"
        assert second_last_row[3].text and second_last_row[4].text.strip() == "12.68", "Fifth cell should be '12.68'"


if __name__ == "__main__":
    unittest.main()
