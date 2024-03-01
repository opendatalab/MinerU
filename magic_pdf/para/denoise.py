import math

from collections import defaultdict
from magic_pdf.para.commons import *

if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class HeaderFooterProcessor:
    def __init__(self) -> None:
        pass

    def get_most_common_bboxes(self, bboxes, page_height, position="top", threshold=0.25, num_bboxes=3, min_frequency=2):
        """
        This function gets the most common bboxes from the bboxes

        Parameters
        ----------
        bboxes : list
            bboxes
        page_height : float
            height of the page
        position : str, optional
            "top" or "bottom", by default "top"
        threshold : float, optional
            threshold, by default 0.25
        num_bboxes : int, optional
            number of bboxes to return, by default 3
        min_frequency : int, optional
            minimum frequency of the bbox, by default 2

        Returns
        -------
        common_bboxes : list
            common bboxes
        """
        # Filter bbox by position
        if position == "top":
            filtered_bboxes = [bbox for bbox in bboxes if bbox[1] < page_height * threshold]
        else:
            filtered_bboxes = [bbox for bbox in bboxes if bbox[3] > page_height * (1 - threshold)]

        # Find the most common bbox
        bbox_count = defaultdict(int)
        for bbox in filtered_bboxes:
            bbox_count[tuple(bbox)] += 1

        # Get the most frequently occurring bbox, but only consider it when the frequency exceeds min_frequency
        common_bboxes = [
            bbox for bbox, count in sorted(bbox_count.items(), key=lambda item: item[1], reverse=True) if count >= min_frequency
        ][:num_bboxes]
        return common_bboxes

    def detect_footer_header(self, result_dict, similarity_threshold=0.5):
        """
        This function detects the header and footer of the document.

        Parameters
        ----------
        result_dict : dict
            result dictionary

        Returns
        -------
        result_dict : dict
            result dictionary
        """

        def compare_bbox_with_list(bbox, bbox_list, tolerance=1):
            return any(all(abs(a - b) < tolerance for a, b in zip(bbox, common_bbox)) for common_bbox in bbox_list)

        def is_single_line_block(block):
            # Determine based on the width and height of the block
            block_width = block["X1"] - block["X0"]
            block_height = block["bbox"][3] - block["bbox"][1]

            # If the height of the block is close to the average character height and the width is large, it is considered a single line
            return block_height <= block["avg_char_height"] * 3 and block_width > block["avg_char_width"] * 3

        # Traverse all blocks in the document
        single_preproc_blocks = 0
        total_blocks = 0
        single_preproc_blocks = 0

        for page_id, blocks in result_dict.items():
            if page_id.startswith("page_"):
                for block_key, block in blocks.items():
                    if block_key.startswith("block_"):
                        total_blocks += 1
                        if is_single_line_block(block):
                            single_preproc_blocks += 1

        # If there are no blocks, skip the header and footer detection
        if total_blocks == 0:
            print("No blocks found. Skipping header/footer detection.")
            return result_dict

        # If most of the blocks are single-line, skip the header and footer detection
        if single_preproc_blocks / total_blocks > 0.5:  # 50% of the blocks are single-line
            return result_dict

        # Collect the bounding boxes of all blocks
        all_bboxes = []
        all_texts = []

        for page_id, blocks in result_dict.items():
            if page_id.startswith("page_"):
                for block_key, block in blocks.items():
                    if block_key.startswith("block_"):
                        all_bboxes.append(block["bbox"])

        # Get the height of the page
        page_height = max(bbox[3] for bbox in all_bboxes)

        # Get the most common bbox lists for headers and footers
        common_header_bboxes = self.get_most_common_bboxes(all_bboxes, page_height, position="top") if all_bboxes else []
        common_footer_bboxes = self.get_most_common_bboxes(all_bboxes, page_height, position="bottom") if all_bboxes else []

        # Detect and mark headers and footers
        for page_id, blocks in result_dict.items():
            if page_id.startswith("page_"):
                for block_key, block in blocks.items():
                    if block_key.startswith("block_"):
                        bbox = block["bbox"]
                        text = block["text"]

                        is_header = compare_bbox_with_list(bbox, common_header_bboxes)
                        is_footer = compare_bbox_with_list(bbox, common_footer_bboxes)

                        block["is_header"] = int(is_header)
                        block["is_footer"] = int(is_footer)

        return result_dict


class NonHorizontalTextProcessor:
    def __init__(self) -> None:
        pass

    def detect_non_horizontal_texts(self, result_dict):
        """
        This function detects watermarks and vertical margin notes in the document.

        Watermarks are identified by finding blocks with the same coordinates and frequently occurring identical texts across multiple pages.
        If these conditions are met, the blocks are highly likely to be watermarks, as opposed to headers or footers, which can change from page to page.
        If the direction of these blocks is not horizontal, they are definitely considered to be watermarks.

        Vertical margin notes are identified by finding blocks with the same coordinates and frequently occurring identical texts across multiple pages.
        If these conditions are met, the blocks are highly likely to be vertical margin notes, which typically appear on the left and right sides of the page.
        If the direction of these blocks is vertical, they are definitely considered to be vertical margin notes.


        Parameters
        ----------
        result_dict : dict
            The result dictionary.

        Returns
        -------
        result_dict : dict
            The updated result dictionary.
        """
        # Dictionary to store information about potential watermarks
        potential_watermarks = {}
        potential_margin_notes = {}

        for page_id, page_content in result_dict.items():
            if page_id.startswith("page_"):
                for block_id, block_data in page_content.items():
                    if block_id.startswith("block_"):
                        if "dir" in block_data:
                            coordinates_text = (block_data["bbox"], block_data["text"])  # Tuple of coordinates and text

                            angle = math.atan2(block_data["dir"][1], block_data["dir"][0])
                            angle = abs(math.degrees(angle))

                            if angle > 5 and angle < 85:  # Check if direction is watermarks
                                if coordinates_text in potential_watermarks:
                                    potential_watermarks[coordinates_text] += 1
                                else:
                                    potential_watermarks[coordinates_text] = 1

                            if angle > 85 and angle < 105:  # Check if direction is vertical
                                if coordinates_text in potential_margin_notes:
                                    potential_margin_notes[coordinates_text] += 1  # Increment count
                                else:
                                    potential_margin_notes[coordinates_text] = 1  # Initialize count

        # Identify watermarks by finding entries with counts higher than a threshold (e.g., appearing on more than half of the pages)
        watermark_threshold = len(result_dict) // 2
        watermarks = {k: v for k, v in potential_watermarks.items() if v > watermark_threshold}

        # Identify margin notes by finding entries with counts higher than a threshold (e.g., appearing on more than half of the pages)
        margin_note_threshold = len(result_dict) // 2
        margin_notes = {k: v for k, v in potential_margin_notes.items() if v > margin_note_threshold}

        # Add watermark information to the result dictionary
        for page_id, blocks in result_dict.items():
            if page_id.startswith("page_"):
                for block_id, block_data in blocks.items():
                    coordinates_text = (block_data["bbox"], block_data["text"])
                    if coordinates_text in watermarks:
                        block_data["is_watermark"] = 1
                    else:
                        block_data["is_watermark"] = 0

                    if coordinates_text in margin_notes:
                        block_data["is_vertical_margin_note"] = 1
                    else:
                        block_data["is_vertical_margin_note"] = 0

        return result_dict


class NoiseRemover:
    def __init__(self) -> None:
        pass

    def skip_data_noises(self, result_dict):
        """
        This function skips the data noises, including overlap blocks, header, footer, watermark, vertical margin note, title
        """
        filtered_result_dict = {}
        for page_id, blocks in result_dict.items():
            if page_id.startswith("page_"):
                filtered_blocks = {}
                for block_id, block in blocks.items():
                    if block_id.startswith("block_"):
                        if any(
                            block.get(key, 0)
                            for key in [
                                "is_overlap",
                                "is_header",
                                "is_footer",
                                "is_watermark",
                                "is_vertical_margin_note",
                                "is_block_title",
                            ]
                        ):
                            continue
                        filtered_blocks[block_id] = block
                if filtered_blocks:
                    filtered_result_dict[page_id] = filtered_blocks

        return filtered_result_dict
