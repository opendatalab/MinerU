# Rendering Contract

状态: Markdown v1
读者: render 开发者、backend 开发者、SDK 开发者
范围: 严格 MiddleJson 到 Markdown 的消费契约

## 公共入口

```python
from mineru.render import MarkdownRenderMode, render_markdown

markdown = render_markdown(
    middle_json,
    mode=MarkdownRenderMode.DEFAULT,
    asset_base_url="",
)
```

`render_markdown()` 只接受 `MiddleJson`，不兼容旧 dict、`list[PageInfo]`、
backend 参数或旧 `Line/Span` renderer。渲染过程基于深拷贝，不修改输入对象，
也不负责把图片写到文件系统。

图片优先使用 `image_path`，缺失时使用 `image_base64` data URI。
`asset_base_url` 只给相对 `image_path` 和 HTML 内相对 `img src` 添加前缀。

## 展示模式

`MarkdownRenderMode.DEFAULT`:

- 隐藏 `header/footer/page_number/aside_text/page_footnote`。
- 合并页内和跨页 `text.continues_prev`。
- 只合并跨页 `table.continues_prev`。
- 页面之间不输出分割线。

`MarkdownRenderMode.FULL`:

- 展示全部顶层 block。
- 只合并页内 `text.continues_prev`。
- 不合并跨页 text/table。
- 每两个相邻 `PageInfo` 之间输出 `---`，空白页边界同样保留。

text 合并允许跨越任意其他 block，后一个 text 的内容被吸收到前一个 text，
后块不再在原位置输出。table 只接受跨页合并；失败时保留两张原表。

## Block 规则

- `text/ref_text`: 解析行内公式、样式和超链接；普通 text 转义可能误触发的 Markdown block 前缀。
- `doc_title/paragraph_title`: 使用全局 `level`，Markdown 标题最多六级；anchor 输出为 HTML id。
- `list`: 递归缩进，直接使用 content 已内化的有序或无序前缀。
- `index`: 递归输出 GFM 列表；标题叶子使用 anchor 生成内部链接，不当作 heading 输出。
- `equation`: content 非空时输出行间 LaTeX，空 content 回退图片。
- `image/chart`: 图片优先，识别内容放入折叠 details；无图片时输出已有 HTML/GFM 内容。
- `table`: 简单单层 HTML 转 GFM；合并单元格或不可无损结构保留 HTML；空间投影文本使用动态 fenced block；空 content 回退图片。
- `code`: 使用 `guess_lang` 和动态 fenced block。
- `algorithm`: 使用等宽、`white-space: pre-wrap` 的 HTML div，只把 `<eq>` 转为行内公式。

视觉父块严格按 `content` 原顺序渲染，body、重复 caption 和 footnote 不重新排序。

## 行内语义

行内 parser 只解释白名单标签，包括 `<eq>`、`<text style>`、
`<hyperlink>/<url>`、`<sup>/<sub>`。未知或损坏标签作为普通文本转义保留。
普通字体样式使用 Markdown；underline、emphasis、上下标和复杂组合使用安全 HTML。

LaTeX 定界符来自 `$MINERU_HOME/config.yaml`:

```yaml
render:
  latex_delimiters:
    display:
      left: "$$"
      right: "$$"
    inline:
      left: "$"
      right: "$"
```

## Index anchor

Office model-list 在对象化前按 document-wide anchor 查找真实目标标题。匹配成功的
目录 text 叶子改成相同的 `doc_title/paragraph_title` 类型与 level，同时保留目录
显示文本；未匹配 anchor 降级为普通 TextBlock。已有旧 MiddleJson 不补造已丢失的 anchor。
