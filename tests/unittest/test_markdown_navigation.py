from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from mineru.utils.enum_class import BlockType, ContentType, MakeMode


def test_markdown_navigation_links_and_reference_formatting():
    pdf_info = [
        {
            "page_idx": 0,
            "page_size": [100, 100],
            "para_blocks": [
                {
                    "type": BlockType.TITLE,
                    "level": 2,
                    "lines": [{"spans": [{"type": ContentType.TEXT, "content": "References"}]}],
                },
                {
                    "type": BlockType.REF_TEXT,
                    "lines": [{"spans": [{"type": ContentType.TEXT, "content": "1. Ref one"}]}],
                },
                {
                    "type": BlockType.REF_TEXT,
                    "lines": [{"spans": [{"type": ContentType.TEXT, "content": "2. Ref two"}]}],
                },
                {
                    "type": BlockType.TEXT,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "type": ContentType.TEXT,
                                    "content": "See Eq. (46), Eqs. (46) and (47), Fig. 3, Table 2, and [1].",
                                }
                            ]
                        }
                    ],
                },
                {
                    "type": BlockType.INTERLINE_EQUATION,
                    "lines": [{"spans": [{"type": ContentType.INTERLINE_EQUATION, "content": "x=1 \\tag{46}"}]}],
                },
                {
                    "type": BlockType.INTERLINE_EQUATION,
                    "lines": [{"spans": [{"type": ContentType.INTERLINE_EQUATION, "content": "y=2 \\tag{47}"}]}],
                },
                {
                    "type": BlockType.IMAGE,
                    "blocks": [
                        {
                            "type": BlockType.IMAGE_CAPTION,
                            "lines": [{"spans": [{"type": ContentType.TEXT, "content": "Fig. 3 Caption"}]}],
                        }
                    ],
                },
                {
                    "type": BlockType.TABLE,
                    "blocks": [
                        {
                            "type": BlockType.TABLE_CAPTION,
                            "lines": [{"spans": [{"type": ContentType.TEXT, "content": "Table 2 Caption"}]}],
                        }
                    ],
                },
            ],
        }
    ]

    markdown = union_make(pdf_info, MakeMode.MM_MD, "images")

    assert "<a id=\"ref-1\"></a>1. Ref one\n<a id=\"ref-2\"></a>2. Ref two" in markdown
    assert "[Eq. (46)](#eq-46)" in markdown
    assert "Eqs. ([46](#eq-46)) and ([47](#eq-47))" in markdown
    assert "[Fig. 3](#fig-3)" in markdown
    assert "[Table 2](#table-2)" in markdown
    assert "<a id=\"eq-46\"></a>\n$$\nx=1 \\tag{46}\n$$" in markdown
    assert "<a id=\"fig-3\"></a>\nFig. 3 Caption" in markdown
    assert "<a id=\"table-2\"></a>\nTable 2 Caption" in markdown
