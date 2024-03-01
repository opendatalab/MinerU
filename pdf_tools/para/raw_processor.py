class RawBlockProcessor:
    def __init__(self) -> None:
        self.y_tolerance = 2
        self.pdf_dic = {}

    def __span_flags_decomposer(self, span_flags):
        """
        Make font flags human readable.

        Parameters
        ----------
        self : object
            The instance of the class.

        span_flags : int
            span flags

        Returns
        -------
        l : dict
            decomposed flags
        """

        l = {
            "is_superscript": False,
            "is_italic": False,
            "is_serifed": False,
            "is_sans_serifed": False,
            "is_monospaced": False,
            "is_proportional": False,
            "is_bold": False,
        }

        if span_flags & 2**0:
            l["is_superscript"] = True  # 表示上标

        if span_flags & 2**1:
            l["is_italic"] = True  # 表示斜体

        if span_flags & 2**2:
            l["is_serifed"] = True  # 表示衬线字体
        else:
            l["is_sans_serifed"] = True  # 表示非衬线字体

        if span_flags & 2**3:
            l["is_monospaced"] = True  # 表示等宽字体
        else:
            l["is_proportional"] = True  # 表示比例字体

        if span_flags & 2**4:
            l["is_bold"] = True  # 表示粗体

        return l

    def __make_new_lines(self, raw_lines):
        """
        This function makes new lines.

        Parameters
        ----------
        self : object
            The instance of the class.

        raw_lines : list
            raw lines

        Returns
        -------
        new_lines : list
            new lines
        """
        new_lines = []
        new_line = None

        for raw_line in raw_lines:
            raw_line_bbox = raw_line["bbox"]
            raw_line_spans = raw_line["spans"]
            raw_line_text = "".join([span["text"] for span in raw_line_spans])
            raw_line_dir = raw_line.get("dir", None)

            decomposed_line_spans = []
            for span in raw_line_spans:
                raw_flags = span["flags"]
                decomposed_flags = self.__span_flags_decomposer(raw_flags)
                span["decomposed_flags"] = decomposed_flags
                decomposed_line_spans.append(span)

            if new_line is None:
                new_line = {
                    "bbox": raw_line_bbox,
                    "text": raw_line_text,
                    "dir": raw_line_dir if raw_line_dir else (0, 0),
                    "spans": decomposed_line_spans,
                }
            else:
                if (
                    abs(raw_line_bbox[1] - new_line["bbox"][1]) <= self.y_tolerance
                    and abs(raw_line_bbox[3] - new_line["bbox"][3]) <= self.y_tolerance
                ):
                    new_line["bbox"] = (
                        min(new_line["bbox"][0], raw_line_bbox[0]),  # left
                        new_line["bbox"][1],  # top
                        max(new_line["bbox"][2], raw_line_bbox[2]),  # right
                        raw_line_bbox[3],  # bottom
                    )
                    new_line["text"] += " " + raw_line_text
                    new_line["spans"].extend(raw_line_spans)
                    new_line["dir"] = (
                        new_line["dir"][0] + raw_line_dir[0],
                        new_line["dir"][1] + raw_line_dir[1],
                    )
                else:
                    new_lines.append(new_line)
                    new_line = {
                        "bbox": raw_line_bbox,
                        "text": raw_line_text,
                        "dir": raw_line_dir if raw_line_dir else (0, 0),
                        "spans": raw_line_spans,
                    }
        if new_line:
            new_lines.append(new_line)

        return new_lines

    def __make_new_block(self, raw_block):
        """
        This function makes a new block.

        Parameters
        ----------
        self : object
            The instance of the class.
        ----------
        raw_block : dict
            a raw block

        Returns
        -------
        new_block : dict

        Schema of new_block:
        {
            "block_id": "block_1",
            "bbox": [0, 0, 100, 100],
            "text": "This is a block.",
            "lines": [
                {
                    "bbox": [0, 0, 100, 100],
                    "text": "This is a line.",
                    "spans": [
                        {
                            "text": "This is a span.",
                            "font": "Times New Roman",
                            "size": 12,
                            "color": "#000000",
                        }
                    ],
                }
            ],
        }
        """
        new_block = {}

        block_id = raw_block["number"]
        block_bbox = raw_block["bbox"]
        block_text = " ".join(span["text"] for line in raw_block["lines"] for span in line["spans"])
        raw_lines = raw_block["lines"]
        block_lines = self.__make_new_lines(raw_lines)

        new_block["block_id"] = block_id
        new_block["bbox"] = block_bbox
        new_block["text"] = block_text
        new_block["lines"] = block_lines

        return new_block

    def batch_process_blocks(self, pdf_dic):
        """
        This function processes the blocks in batch.

        Parameters
        ----------
        self : object
            The instance of the class.
        ----------
        blocks : list
            Input block is a list of raw blocks. Schema can refer to the value of key ""preproc_blocks", demo file is app/pdf_toolbox/tests/preproc_2_parasplit_example.json.

        Returns
        -------
        result_dict : dict
            result dictionary
        """

        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):
                para_blocks = []
                if "preproc_blocks" in blocks.keys():
                    input_blocks = blocks["preproc_blocks"]
                    for raw_block in input_blocks:
                        new_block = self.__make_new_block(raw_block)
                        para_blocks.append(new_block)

                blocks["para_blocks"] = para_blocks

        return pdf_dic

