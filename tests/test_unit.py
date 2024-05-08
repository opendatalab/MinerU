import pytest

from magic_pdf.libs.commons import mymax


# 输入一个列表，如果列表空返回0，否则返回最大元素
@pytest.mark.parametrize("list_input",
                         [
                             # [0, 0, 0, 0],
                             # [(1, 2), (3, 4), (1, 4)],
                             # [0],
                             [1, 2, 5, 8, 4],
                             # [],
                             # ["None", "True"]
                         ])
def test_list_max(list_input: list):
    """
    list_input: 输入列表元素
    """
    assert "True" == mymax(list_input)




