import os
import csv
import json
import pandas as pd
from pandas import DataFrame as df
from matplotlib import pyplot as plt
from termcolor import cprint

"""
Execute this script in the following way:

1. Make sure there are pdf_dic.json files under the directory code-clean/tmp/unittest/md/, such as the following:

    code-clean/tmp/unittest/md/scihub/scihub_00500000/libgen.scimag00527000-00527999.zip_10.1002/app.25178/pdf_dic.json
    
2. Under the directory code-clean, execute the following command:

    $ python -m libs.calc_span_stats
    
"""


def print_green_on_red(text):
    cprint(text, "green", "on_red", attrs=["bold"], end="\n\n")


def print_green(text):
    print()
    cprint(text, "green", attrs=["bold"], end="\n\n")


def print_red(text):
    print()
    cprint(text, "red", attrs=["bold"], end="\n\n")


def safe_get(dict_obj, key, default):
    val = dict_obj.get(key)
    if val is None:
        return default
    else:
        return val


class SpanStatsCalc:
    """Calculate statistics of span."""

    def draw_charts(self, span_stats: pd.DataFrame, fig_num: int, save_path: str):
        """Draw multiple figures in one figure."""
        # make a canvas
        fig = plt.figure(fig_num, figsize=(20, 20))

        pass

    def calc_stats_per_dict(self, pdf_dict) -> pd.DataFrame:
        """Calculate statistics per pdf_dict."""
        span_stats = pd.DataFrame()

        span_stats = []
        span_id = 0
        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                if "para_blocks" in blocks.keys():
                    for para_block in blocks["para_blocks"]:
                        for line in para_block["lines"]:
                            for span in line["spans"]:
                                span_text = safe_get(span, "text", "")
                                span_font_name = safe_get(span, "font", "")
                                span_font_size = safe_get(span, "size", 0)
                                span_font_color = safe_get(span, "color", "")
                                span_font_flags = safe_get(span, "flags", 0)

                                span_font_flags_decoded = safe_get(span, "decomposed_flags", {})
                                span_is_super_script = safe_get(span_font_flags_decoded, "is_superscript", False)
                                span_is_italic = safe_get(span_font_flags_decoded, "is_italic", False)
                                span_is_serifed = safe_get(span_font_flags_decoded, "is_serifed", False)
                                span_is_sans_serifed = safe_get(span_font_flags_decoded, "is_sans_serifed", False)
                                span_is_monospaced = safe_get(span_font_flags_decoded, "is_monospaced", False)
                                span_is_proportional = safe_get(span_font_flags_decoded, "is_proportional", False)
                                span_is_bold = safe_get(span_font_flags_decoded, "is_bold", False)

                                span_stats.append(
                                    {
                                        "span_id": span_id,  # id of span
                                        "page_id": page_id,  # page number of pdf
                                        "span_text": span_text,  # text of span
                                        "span_font_name": span_font_name,  # font name of span
                                        "span_font_size": span_font_size,  # font size of span
                                        "span_font_color": span_font_color,  # font color of span
                                        "span_font_flags": span_font_flags,  # font flags of span
                                        "span_is_superscript": int(
                                            span_is_super_script
                                        ),  # indicate whether the span is super script or not
                                        "span_is_italic": int(span_is_italic),  # indicate whether the span is italic or not
                                        "span_is_serifed": int(span_is_serifed),  # indicate whether the span is serifed or not
                                        "span_is_sans_serifed": int(
                                            span_is_sans_serifed
                                        ),  # indicate whether the span is sans serifed or not
                                        "span_is_monospaced": int(
                                            span_is_monospaced
                                        ),  # indicate whether the span is monospaced or not
                                        "span_is_proportional": int(
                                            span_is_proportional
                                        ),  # indicate whether the span is proportional or not
                                        "span_is_bold": int(span_is_bold),  # indicate whether the span is bold or not
                                    }
                                )

                                span_id += 1

        span_stats = pd.DataFrame(span_stats)
        # print(span_stats)

        return span_stats


def __find_pdf_dic_files(
    jf_name="pdf_dic.json",
    base_code_name="code-clean",
    tgt_base_dir_name="tmp",
    unittest_dir_name="unittest",
    md_dir_name="md",
    book_names=[
        "scihub",
    ],  # other possible values: "zlib", "arxiv" and so on
):
    pdf_dict_files = []

    curr_dir = os.path.dirname(__file__)

    for i in range(len(curr_dir)):
        if curr_dir[i : i + len(base_code_name)] == base_code_name:
            base_code_dir_name = curr_dir[: i + len(base_code_name)]
            for book_name in book_names:
                search_dir_relative_name = os.path.join(tgt_base_dir_name, unittest_dir_name, md_dir_name, book_name)
                if os.path.exists(base_code_dir_name):
                    search_dir_name = os.path.join(base_code_dir_name, search_dir_relative_name)
                    for root, dirs, files in os.walk(search_dir_name):
                        for file in files:
                            if file == jf_name:
                                pdf_dict_files.append(os.path.join(root, file))
                break

    return pdf_dict_files


def combine_span_texts(group_df, span_stats):
    combined_span_texts = []
    for _, row in group_df.iterrows():
        curr_span_id = row.name
        curr_span_text = row["span_text"]

        pre_span_id = curr_span_id - 1
        pre_span_text = span_stats.at[pre_span_id, "span_text"] if pre_span_id in span_stats.index else ""

        next_span_id = curr_span_id + 1
        next_span_text = span_stats.at[next_span_id, "span_text"] if next_span_id in span_stats.index else ""

        # pointer_sign is a right arrow if the span is superscript, otherwise it is a down arrow
        pointer_sign = "→ → → "
        combined_text = "\n".join([pointer_sign + pre_span_text, pointer_sign + curr_span_text, pointer_sign + next_span_text])
        combined_span_texts.append(combined_text)

    return "\n\n".join(combined_span_texts)


# pd.set_option("display.max_colwidth", None)  # 设置为 None 来显示完整的文本
pd.set_option("display.max_rows", None)  # 设置为 None 来显示更多的行


def main():
    pdf_dict_files = __find_pdf_dic_files()
    # print(pdf_dict_files)

    span_stats_calc = SpanStatsCalc()

    for pdf_dict_file in pdf_dict_files:
        print("-" * 100)
        print_green_on_red(f"Processing {pdf_dict_file}")

        with open(pdf_dict_file, "r", encoding="utf-8") as f:
            pdf_dict = json.load(f)

            raw_df = span_stats_calc.calc_stats_per_dict(pdf_dict)
            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_raw.csv")
            raw_df.to_csv(save_path, index=False)

            filtered_df = raw_df[raw_df["span_is_superscript"] == 1]
            if filtered_df.empty:
                print("No superscript span found!")
                continue

            filtered_grouped_df = filtered_df.groupby(["span_font_name", "span_font_size", "span_font_color"])

            combined_span_texts = filtered_grouped_df.apply(combine_span_texts, span_stats=raw_df)  # type: ignore

            final_df = filtered_grouped_df.size().reset_index(name="count")
            final_df["span_texts"] = combined_span_texts.reset_index(level=[0, 1, 2], drop=True)

            print(final_df)

            final_df["span_texts"] = final_df["span_texts"].apply(lambda x: x.replace("\n", "\r\n"))

            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_final.csv")
            # 使用 UTF-8 编码并添加 BOM，确保所有字段被双引号包围
            final_df.to_csv(save_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

            # 创建一个 2x2 的图表布局
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            # 按照 span_font_name 分类作图
            final_df.groupby("span_font_name")["count"].sum().plot(kind="bar", ax=axs[0, 0], title="By Font Name")

            # 按照 span_font_size 分类作图
            final_df.groupby("span_font_size")["count"].sum().plot(kind="bar", ax=axs[0, 1], title="By Font Size")

            # 按照 span_font_color 分类作图
            final_df.groupby("span_font_color")["count"].sum().plot(kind="bar", ax=axs[1, 0], title="By Font Color")

            # 按照 span_font_name、span_font_size 和 span_font_color 共同分类作图
            grouped = final_df.groupby(["span_font_name", "span_font_size", "span_font_color"])
            grouped["count"].sum().unstack().plot(kind="bar", ax=axs[1, 1], title="Combined Grouping")

            # 调整布局
            plt.tight_layout()

            # 显示图表
            # plt.show()

            # 保存图表到 PNG 文件
            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_combined.png")
            plt.savefig(save_path)

            # 清除画布
            plt.clf()


if __name__ == "__main__":
    main()
