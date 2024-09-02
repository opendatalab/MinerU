
from magic_pdf.dict2md.ocr_mkcontent import __is_hyphen_at_line_end


def test_hyphen_at_line_end():
    """
    测试行尾是不是一个连字符
    """
    test_cases_ok = [
        "I am zhang-",
        "you are zhang- ",
        "math-",
        "This is a TEST-",
        "This is a TESTing-",
        "美国人 hello-",
    ]
    test_cases_bad = [
        "This is a TEST$-",
        "This is a TEST21-",
        "中国人-",
        "美国人 hello人-",
        "this is 123-",
    ]
    for test_case in test_cases_ok:
        assert __is_hyphen_at_line_end(test_case)

    for test_case in test_cases_bad:
        assert not __is_hyphen_at_line_end(test_case)