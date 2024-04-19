from enum import Enum

class ModelBlockTypeEnum(Enum):
    TITLE = 0
    PLAIN_TEXT = 1
    ABANDON = 2
    ISOLATE_FORMULA = 8
    EMBEDDING = 13
    ISOLATED = 14