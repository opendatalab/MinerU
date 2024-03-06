import unittest

"""
Execute the following command to run the tests under directory code-clean:

    python -m tests.test_para.test_para_pipeline
    
    or
    
    pytest -v -s app/pdf_toolbox/tests/test_para/test_para_pipeline.py
    
"""

from tests.test_para.test_pdf2text_recogPara_Common import (
    TestIsBboxOverlap,
    TestIsInBbox,
    TestIsBboxOverlap,
    TestIsLineLeftAlignedFromNeighbors,
    TestIsLineRightAlignedFromNeighbors,
)
from tests.test_para.test_pdf2text_recogPara_EquationsProcessor import TestCalcOverlapPct
from tests.test_para.test_pdf2text_recogPara_BlockInnerParasProcessor import TestIsConsistentLines
from tests.test_para.test_pdf2text_recogPara_BlockContinuationProcessor import (
    TestIsAlphabetChar,
    TestIsChineseChar,
    TestIsOtherLetterChar,
)
from tests.test_para.test_pdf2text_recogPara_TitleProcessor import TestTitleProcessor


# Test suite
suite = unittest.TestSuite()

# Test cases from test_pdf2text_recogPara_Common
suite.addTest(unittest.makeSuite(TestIsBboxOverlap))
suite.addTest(unittest.makeSuite(TestIsInBbox))
suite.addTest(unittest.makeSuite(TestIsBboxOverlap))
suite.addTest(unittest.makeSuite(TestIsLineLeftAlignedFromNeighbors))
suite.addTest(unittest.makeSuite(TestIsLineRightAlignedFromNeighbors))

# Test cases from test_pdf2text_recogPara_EquationsProcessor
suite.addTest(unittest.makeSuite(TestCalcOverlapPct))

# Test cases from test_pdf2text_recogPara_BlockInnerParasProcessor
suite.addTest(unittest.makeSuite(TestIsConsistentLines))

# Test cases from test_pdf2text_recogPara_BlockContinuationProcessor
suite.addTest(unittest.makeSuite(TestIsAlphabetChar))
suite.addTest(unittest.makeSuite(TestIsChineseChar))
suite.addTest(unittest.makeSuite(TestIsOtherLetterChar))

# Test cases from test_pdf2text_recogPara_TitleProcessor
suite.addTest(unittest.makeSuite(TestTitleProcessor))

# Run test suite
unittest.TextTestRunner(verbosity=2).run(suite)
