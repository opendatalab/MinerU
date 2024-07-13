"""
clean data
"""
import argparse
import os
import re
import htmltabletomd # type: ignore
import pypandoc
import argparse

parser = argparse.ArgumentParser(description="get tool type")
parser.add_argument(
    "--tool_name",
    type=str,
    required=True,
    help="input tool name",
)
parser.add_argument(
    "--download_dir",
    type=str,
    required=True,
    help="input download dir",
)
args = parser.parse_args()

def clean_markdown_images(content):
    """
    clean markdown images
    """
    pattern = re.compile(r'!\[[^\]]*\]\([^)]*\)', re.IGNORECASE)  
    cleaned_content = pattern.sub('', content)   
    return cleaned_content
   
def clean_ocrmath_photo(content):
    """
    clean ocrmath photo
    """
    pattern = re.compile(r'\\includegraphics\[.*?\]\{.*?\}', re.IGNORECASE)  
    cleaned_content = pattern.sub('', content)   
    return cleaned_content

def convert_html_table_to_md(html_table):
    """
    convert html table to markdown table
    """
    lines = html_table.strip().split('\n')  
    md_table = ''  
    if lines and '<tr>' in lines[0]:  
        in_thead = True  
        for line in lines:  
            if '<th>' in line:  
                cells = re.findall(r'<th>(.*?)</th>', line)  
                md_table += '| ' + ' | '.join(cells) + ' |\n'  
                in_thead = False  
            elif '<td>' in line and not in_thead:  
                cells = re.findall(r'<td>(.*?)</td>', line)  
                md_table += '| ' + ' | '.join(cells) + ' |\n'  
        md_table = md_table.rstrip() + '\n'    
    return md_table  
 
def convert_latext_to_md(content):
    """
    convert latex table to markdown table
    """
    tables = re.findall(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', content, re.DOTALL)  
    placeholders = []  
    for table in tables:  
        placeholder = f"<!-- TABLE_PLACEHOLDER_{len(placeholders)} -->"  
        replace_str = f"\\begin{{tabular}}{table}cl\\end{{tabular}}"
        content = content.replace(replace_str, placeholder)  
        try:
            pypandoc.convert_text(replace_str,  format="latex", to="md", outputfile="output.md", encoding="utf-8")
        except:
            markdown_string = replace_str
        else: 
            markdown_string = open('output.md', 'r', encoding='utf-8').read()
        placeholders.append((placeholder, markdown_string)) 
    new_content = content  
    for placeholder, md_table in placeholders:  
        new_content = new_content.replace(placeholder, md_table)  
        # 写入文件  
    return new_content

 
def convert_htmltale_to_md(content):
    """
    convert html table to markdown table
    """
    tables = re.findall(r'<table>(.*?)</table>', content, re.DOTALL)  
    placeholders = []
    for table in tables:
        placeholder = f"<!-- TABLE_PLACEHOLDER_{len(placeholders)} -->"  
        content = content.replace(f"<table>{table}</table>", placeholder)  
        try:
            convert_table = htmltabletomd.convert_table(table)
        except:
            convert_table = table
        placeholders.append((placeholder,convert_table)) 
    new_content = content  
    for placeholder, md_table in placeholders:  
        new_content = new_content.replace(placeholder, md_table)  
        # 写入文件  
    return new_content

def clean_data(prod_type, download_dir):
    """
    clean data
    """
    tgt_dir = os.path.join(download_dir, prod_type, "cleaned")
    if not os.path.exists(tgt_dir):  
        os.makedirs(tgt_dir) 
    source_dir = os.path.join(download_dir, prod_type)
    filenames = os.listdir(source_dir)
    for filename in filenames:
        if filename.endswith('.md'):
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(tgt_dir, "cleaned_" + filename)
            with open(input_file, 'r', encoding='utf-8') as fr:
                content = fr.read()
                new_content = clean_markdown_images(content)
                with open(output_file, 'w', encoding='utf-8') as fw:
                    fw.write(new_content)


if __name__ == '__main__':
    tool_type = args.tool_name
    download_dir = args.download_dir
    clean_data(tool_type, download_dir)
