# MinerU 提取原理说明

这份说明聚焦我最关心的三类内容：

- 公式
- 图片
- 表格

目标是回答两个问题：

1. 它们是怎么从 PDF 里被识别出来的
2. 它们最后是怎么转成 Markdown 的

---

## 总体思路

MinerU 在 `pipeline` 后端里，不是把整页 PDF 直接丢给一个模型，然后一步生成 Markdown。

它的核心是多阶段流水线：

1. 把 PDF 按页渲染成图像
2. 对页面做版面检测，找出正文、标题、图片、表格、公式等区域
3. 不同类型走不同专用模型
4. 把模型输出整理成统一的中间结构
5. 再把中间结构导出成 Markdown

所以它效果好的原因，不只是模型，而是：

- 有专门的图片/公式/表格处理链路
- 有统一的中间层 `middle_json`
- 有大量后处理，比如分组、排序、去重、跨页表格合并

主流程入口：

- `mineru/cli/client.py`
- `mineru/cli/common.py`
- `mineru/backend/pipeline/pipeline_analyze.py`
- `mineru/backend/pipeline/batch_analyze.py`
- `mineru/backend/pipeline/model_json_to_middle_json.py`
- `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py`

---

## 图片提取原理

### 1. 图片先靠版面检测找到区域

图片不是先 OCR。

第一步是用版面检测模型识别出页面里哪些区域属于：

- 图片正文 `ImageBody`
- 图片标题 `ImageCaption`
- 图片脚注 `ImageFootnote`

相关代码：

- `mineru/backend/pipeline/model_init.py`
- `mineru/backend/pipeline/batch_analyze.py`

版面检测模型是：

- `DocLayoutYOLOModel`

它会对整页图像输出很多检测框，不只是图片，也包括标题、正文、表格、公式等。

### 2. 把图片正文、标题、脚注重新组装成一个“图片块”

识别出图片区域后，MinerU 不会把它当孤立框直接导出，而是会把附近的 caption 和 footnote 绑定到同一个图片对象上。

这一步在：

- `mineru/backend/pipeline/pipeline_magic_model.py`

关键函数：

- `get_imgs()`

它会做的事：

- 把 `ImageBody` 和最近的 `ImageCaption` 关联起来
- 把 `ImageBody` 和最近的 `ImageFootnote` 关联起来

所以最终导出时，图片不是只有一张图，而是一个完整结构：

- 图片本体
- 图片标题
- 图片脚注

### 3. 把图片区域真正裁出来保存成文件

在中间结构整理阶段，会把图片区域从页面图里裁出来，写到输出目录的 `images/` 下。

相关代码：

- `mineru/backend/pipeline/model_json_to_middle_json.py`
- `mineru/utils/cut_image.py`

这一步会把图片 span 加上 `image_path`，后面 Markdown 就能直接引用这个路径。

### 4. 转成 Markdown

图片最终导出逻辑在：

- `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py`

图片会被写成：

```md
![](images/xxx.jpg)
```

然后再把标题和脚注按顺序拼接进去。

所以图片链路是：

```text
整页图像
-> 版面检测找到图片区域
-> 绑定图片标题/脚注
-> 裁图保存到 images/
-> Markdown 中用 ![](...) 引用
```

---

## 公式提取原理

公式处理不是普通 OCR。

它的核心是两段式：

1. 先检测公式区域
2. 再把公式图识别成 LaTeX

### 1. 公式检测

公式检测模型是：

- `YOLOv8MFDModel`

代码位置：

- `mineru/model/mfd/yolo_v8.py`

主流程调用位置：

- `mineru/backend/pipeline/batch_analyze.py`

这一步输入整页图像，输出一批公式框。

这些框会标出：

- 行内公式
- 行间公式

### 2. 裁出每个公式区域的小图

公式检测完后，代码会把每个公式框从页面图像中裁出来，得到很多公式小图。

这部分逻辑在：

- `mineru/model/mfr/unimernet/Unimernet.py`
- `mineru/model/mfr/pp_formulanet_plus_m/predict_formula.py`

### 3. 用公式识别模型把公式图转成 LaTeX

默认这次实际跑到的是：

- `unimernet_small`

对应代码：

- `mineru/model/mfr/unimernet/Unimernet.py`

模型初始化位置：

- `mineru/backend/pipeline/model_init.py`

公式识别的核心不是字符级 OCR，而是“看图生成 LaTeX”的视觉到文本模型。

底层模型在：

- `mineru/model/mfr/unimernet/unimernet_hf/modeling_unimernet.py`

它本质上是：

- 视觉编码器：把公式图片编码成视觉特征
- 文本解码器：像翻译一样逐 token 生成 LaTeX

所以它更像：

```text
公式图片 -> 视觉编码 -> 序列生成 -> LaTeX
```

不是：

```text
公式图片 -> 普通 OCR -> 文本
```

### 4. 预处理为什么重要

在送入 UniMERNet 前，公式图会先做预处理：

- 裁掉多余留白
- 保持长宽比缩放
- pad 到固定大小
- 灰度化和归一化

相关代码：

- `mineru/model/mfr/unimernet/unimernet_hf/unimer_swin/image_processing_unimer_swin.py`

这对公式识别质量很关键，因为公式区域边界、留白和尺度变化会直接影响结果。

### 5. 识别结果如何进入文档结构

识别后的 LaTeX 会回填到检测框里，写入：

- `latex`

然后在中间层中被转换成：

- 行内公式 `INLINE_EQUATION`
- 行间公式 `INTERLINE_EQUATION`

相关代码：

- `mineru/backend/pipeline/pipeline_magic_model.py`

### 6. 转成 Markdown

最终导出到 Markdown 时：

- 行内公式会写成 `$...$`
- 行间公式会写成块公式

相关代码：

- `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py`

如果某个行间公式没有成功识别出 LaTeX，会退回成图片引用，而不是完全丢失。

所以公式链路是：

```text
整页图像
-> 公式检测
-> 裁出公式小图
-> 公式识别模型生成 LaTeX
-> 区分行内/行间公式
-> 写成 Markdown 公式
```

---

## 表格提取原理

表格是三类里最复杂的。

它不是只靠 OCR，而是：

1. 先找表格区域
2. 再对表格内部做 OCR
3. 再恢复表格结构
4. 最后优先输出 HTML 表格

### 1. 先检测表格区域

表格也是由版面检测模型先识别出：

- 表格正文 `TableBody`
- 表格标题 `TableCaption`
- 表格脚注 `TableFootnote`

相关代码：

- `mineru/backend/pipeline/pipeline_magic_model.py`

关键函数：

- `get_tables()`

它会把表格正文和对应 caption/footnote 绑定成一个表格块。

### 2. 先做表格方向分类和表格类型分类

表格在真正恢复结构之前，还会先判断：

- 表格方向是否旋转
- 表格属于 wired 还是 wireless

相关代码：

- `mineru/backend/pipeline/batch_analyze.py`

相关模型：

- `PaddleOrientationClsModel`
- `PaddleTableClsModel`

### 3. 对表格内部文字做 OCR

表格 OCR 分两步：

1. 先检测表格内部文字框
2. 再识别每个文字框的内容

相关代码：

- `mineru/backend/pipeline/batch_analyze.py`

这里用到的 OCR 还是 `PytorchPaddleOCR`，但它不是把整页表格直接当普通文本，而是先按检测框切成很多小块，再识别。

### 4. 恢复表格结构

这一步才是表格效果好的关键。

MinerU 不满足于把表格识别成一堆文本，而是继续用专门的表格模型恢复表格结构。

相关模型：

- `RapidTableModel`
- `UnetTableModel`

对应代码：

- `mineru/backend/pipeline/model_init.py`
- `mineru/backend/pipeline/batch_analyze.py`

处理方式大致是：

- 无线表格先走 `RapidTableModel`
- 有线表格再走 `UnetTableModel`

最终模型会输出表格 HTML，回填到表格对象上。

### 5. 转成 Markdown

Markdown 导出时，表格优先使用结构化 HTML，而不是普通文本拼接。

相关代码：

- `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py`

逻辑是：

- 如果表格 span 里已经有 `html`，就直接输出 HTML 表格
- 如果没有结构化结果，才退回成表格图片

所以表格链路是：

```text
整页图像
-> 检测表格区域
-> 表格方向分类
-> 表格类型分类
-> 表格内部 OCR
-> 表格结构恢复为 HTML
-> Markdown 中嵌入 HTML 表格
```

---

## 为什么这三类效果好

核心原因不是单一模型，而是“分而治之”：

- 图片：主要做区域检测、归组和裁图导出
- 公式：主要做区域检测和 LaTeX 生成
- 表格：主要做区域检测、OCR 和结构恢复

也就是说，这三类内容并不是统一走“整页 OCR”。

它们分别走了更适合自己的处理方式，所以最终效果会比简单的 PDF OCR 更稳定。

---

## 一句话总结

MinerU 对这三类内容的处理可以概括为：

- 图片：检测出来，裁出来，写成 `![](...)`
- 公式：检测出来，识别成 LaTeX，写成 Markdown 公式
- 表格：检测出来，OCR + 结构恢复成 HTML，再嵌入 Markdown

这也是它在学术论文、图文混排、含公式表格文档上效果比较好的核心原因。
