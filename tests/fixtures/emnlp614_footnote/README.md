# EMNLP 2022 第 7 页脚注回归

`page7.pdf` 是用户提供的 `2022.emnlp-main.614.pdf` 的第 7 页，保留原始字符与字体；原文件 SHA-256 见 `detection.json`。

`detection.json` 固化 200 DPI 渲染尺寸、Torch/ONNX 的脚注区域原始 layout 检测、保留行内公式遮罩后的 OCR 检测框，以及真实 VLM 文本识别结果。坐标均来自原页，测试不依赖模型权重或用户 Downloads 目录。

错误链路是 `inline_formula` 被脚注边界规则改成 `footnote`，小框先领取第一行，大框只剩第二行。回归必须同时验证公式身份、识别输入、两行回填及最终单个脚注，不能仅以删掉小框为通过条件。第二行 `s0 > s1` 的下标漏检不属于本次修复范围。
