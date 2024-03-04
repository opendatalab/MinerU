# pdf_toolbox
pdf 解析基础函数


## pdf是否是文字类型/扫描类型的区分

```shell
cat s3_pdf_path.example.pdf | parallel --colsep ' ' -j 10 "python pdf_meta_scan.py --s3-pdf-path {2} --s3-profile {1} >> {/}.jsonl"

find dir/to/jsonl/ -type f -name "*.jsonl" | parallel -j 10 "python pdf_classfy_by_type.py --json_file {} >> {/}.jsonl"

```

```shell
# 如果单独运行脚本，合并到code-clean之后需要运行，参考如下：
python -m pdf_meta_scan --s3-pdf-path "D:\pdf_files\内容排序测试_pdf\p3_图文混排 5.pdf" --s3-profile s2
```

## pdf
