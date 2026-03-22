# MinerU 本地复跑说明

这份文档记录了我在本机上成功跑通 `MinerU` 解析 PDF 的方式，后续你可以直接按这里的命令复跑。

## 这次验证过的环境

- 项目目录：`/Users/wangyc/Desktop/projects/MinerU`
- Python：`3.11`
- 系统：macOS
- 设备：Apple Silicon，使用 `mps`
- 模型源：`huggingface`
- 解析后端：`pipeline`

## 为什么这样跑

这次没有用轻量模式，而是保留了：

- 图片识别
- 公式识别
- 表格识别

因此使用的是完整的 `pipeline` 方案。

另外，实际测试里 `huggingface` 下载模型比 `modelscope` 更顺，所以后续建议优先用：

```bash
MINERU_MODEL_SOURCE=huggingface
```

## 第一次安装

在项目根目录执行：

```bash
cd /Users/wangyc/Desktop/projects/MinerU

uv venv --python /opt/homebrew/bin/python3.11 .venv
uv pip install --python .venv/bin/python -e '.[pipeline]'
```

说明：

- `.venv`：本地虚拟环境
- `-e '.[pipeline]'`：按源码安装，并安装 `pipeline` 后端所需依赖
- 不需要装 `.[all]`，因为本次本地解析只用到了 `pipeline`

## 复跑命令

### 1. 解析中文 PDF

```bash
cd /Users/wangyc/Desktop/projects/MinerU

MINERU_MODEL_SOURCE=huggingface HF_HUB_DISABLE_XET=1 \
.venv/bin/mineru \
-p '/Users/wangyc/Desktop/projects/MinerU/my_test/SVOLT-196Ah-3.2V-LiFePO4-Blade-Battery-Cell-SpecificationDatasheet.pdf' \
-o '/Users/wangyc/Desktop/projects/MinerU/my_test/output_hf_full' \
-b pipeline -d mps -l ch --source huggingface
```

### 2. 解析英文 PDF

```bash
cd /Users/wangyc/Desktop/projects/MinerU

MINERU_MODEL_SOURCE=huggingface HF_HUB_DISABLE_XET=1 \
.venv/bin/mineru \
-p '/Users/wangyc/Desktop/projects/MinerU/my_test/AAAI-26 camera ready version 22735.pdf' \
-o '/Users/wangyc/Desktop/projects/MinerU/my_test/output_hf_full' \
-b pipeline -d mps -l en --source huggingface
```

### 3. 解析整个目录

如果目录里文件语言差不多，也可以直接跑目录：

```bash
cd /Users/wangyc/Desktop/projects/MinerU

MINERU_MODEL_SOURCE=huggingface HF_HUB_DISABLE_XET=1 \
.venv/bin/mineru \
-p '/Users/wangyc/Desktop/projects/MinerU/my_test' \
-o '/Users/wangyc/Desktop/projects/MinerU/my_test/output_hf_full' \
-b pipeline -d mps -l ch --source huggingface
```

注意：

- 跑目录时只能统一传一个 `-l` 语言参数
- 如果目录里中英混合，建议还是像这次一样分开跑

## 参数解释

- `-p`：输入 PDF 路径，或者一个目录
- `-o`：输出目录
- `-b pipeline`：使用本地完整解析后端，包含版面、OCR、公式、表格等能力
- `-d mps`：在 mac 上使用 Apple GPU
- `-l ch`：中文文档
- `-l en`：英文文档
- `--source huggingface`：指定模型源为 Hugging Face
- `MINERU_MODEL_SOURCE=huggingface`：环境变量方式指定模型源
- `HF_HUB_DISABLE_XET=1`：禁用 Hugging Face 的 xet 下载路径，当前这台机器上这样更稳

## 输出结果在哪

每个 PDF 会生成一个单独目录，路径结构类似：

```text
my_test/output_hf_full/<PDF文件名>/auto/
```

里面通常会有这些文件：

- `<文件名>.md`：最终 Markdown
- `<文件名>_content_list.json`：结构化内容列表
- `<文件名>_middle.json`：中间结果
- `<文件名>_model.json`：模型输出
- `<文件名>_layout.pdf`：版面框可视化
- `<文件名>_span.pdf`：文本框可视化
- `<文件名>_origin.pdf`：原 PDF 副本
- `images/`：导出的图片资源

本次实际结果：

- 中文论文输出目录：`/Users/wangyc/Desktop/projects/MinerU/my_test/output_hf_full/2201210418-王宇晨-论文/auto`
- AAAI 论文输出目录：`/Users/wangyc/Desktop/projects/MinerU/my_test/output_hf_full/AAAI-26 camera ready version 22735/auto`

## 首次运行为什么会慢

第一次运行会自动下载模型到本地缓存，所以会明显更慢。

本次实际观察：

- `huggingface` 比 `modelscope` 更快
- 第一次完整跑会先下载几 GB 模型
- 下载完成后，后续再跑会快很多

Hugging Face 模型缓存位置通常在：

```text
~/.cache/huggingface/hub/
```

## 常见情况

### 1. 默认 `python3` 版本不对

这个项目要求：

```text
Python >= 3.10
```

如果系统默认还是 `3.9`，就不要直接用系统 Python，按上面的 `python3.11` 建虚拟环境即可。

### 2. mac 上会看到一条 `cv2/av` 警告

类似：

```text
Class AVFFrameReceiver is implemented in both ...
```

这次测试里这条警告没有影响最终解析，先不用管。

### 3. 想走 CPU

如果 `mps` 不稳定，可以改成：

```bash
-d cpu
```

但速度会慢很多。

## 我这次跑出来的简要结论

- 中文论文：封面、摘要、目录、章节结构提取正常，图片也成功导出
- 英文 AAAI 论文：标题、摘要、章节、图片、公式都能出来
- 少量问题仍然存在：个别字符 OCR 噪声、作者行/公式空格有时不够漂亮，但整体可用

## 最推荐的复跑方式

以后你自己复跑，优先用下面这个模板改路径：

```bash
cd /Users/wangyc/Desktop/projects/MinerU

MINERU_MODEL_SOURCE=huggingface HF_HUB_DISABLE_XET=1 \
.venv/bin/mineru \
-p '<你的PDF路径>' \
-o '<输出目录>' \
-b pipeline -d mps -l <ch或en> --source huggingface
```
