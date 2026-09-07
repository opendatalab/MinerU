# DocGale 依赖范围验证

本次将 MinerU 的依赖调整为 `docgale>=0.1.0,<1.0.0`，允许后续 0.x 版本，
不包含 1.0.0。依赖边界检查覆盖 0.1.0、0.2.0、0.99.99 和 1.0.0。

DocGale 将 PDFium 限定为 `pypdfium2>=5.10.1,<6`。MinerU 自身的
`pypdfium2>=5.10.1` 声明保持不变，安装时两者约束共同生效。
其他依赖声明、Python 支持范围和 MinerU 许可证保持不变。

DocGale 项目代码改用 MIT；包含 Apache-2.0 派生代码及 Droid 字体的发行包
使用 `MIT AND Apache-2.0`，保留必要的源码归属、许可证和字体原始 NOTICE。
DocGale 的产品说明已独立化，现有适配器和序列化协议不变。

本地验证使用重新构建并安装的 DocGale 0.1.0 wheel，安装只替换 DocGale，
没有升级其他依赖。Python 3.13.5、PDFium 5.10.1 环境中，路由、ParseResult、
旧协议适配、HTML 渲染及 HTML 原生往返测试共 **247 项通过、1 项排除**。
排除项仍为已有的
`test_legacy_schema_adapter.py::test_doclib_compaction_rejects_unknown_legacy_schema`，
不计入本次修复或通过结果。解析金标未修改。

DocGale 另完成 47 项接口与格式矩阵检查，以及独立 Python 3.14 wheel 环境中的
29 项协议和完整流程检查。新 wheel/sdist 保持 0.1.0，更新现有发行草稿，
本次不发布到 PyPI。
