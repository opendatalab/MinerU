"""
get cov
"""
from bs4 import BeautifulSoup
import shutil
def get_covrage():
    """get covrage"""
    # 发送请求获取网页内容
    html_content = open("htmlcov/index.html", "r", encoding="utf-8").read()
    soup = BeautifulSoup(html_content, 'html.parser')

    # 查找包含"pc_cov"的span标签
    pc_cov_span = soup.find('span', class_='pc_cov')

    # 提取百分比值
    percentage_value = pc_cov_span.text.strip()
    percentage_float = float(percentage_value.rstrip('%'))
    print ("percentage_float:", percentage_float)
    assert percentage_float >= 0.2

if __name__ == '__main__':
    get_covrage()