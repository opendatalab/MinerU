# 旧平台适配（MinerU <4）

以下 12 个平台保留已有适配方案，**仅适用于 MinerU <4**。主线 4.0 的依赖、模型和命令变更不自动扩展这些平台的兼容范围。

9 个专用 Dockerfile 的 MinerU 依赖为原 extras 与 `>=3.4.0,<4`。例如：

```bash
python -m pip install "mineru[core]>=3.4.0,<4"
```

此命令仅说明版本约束；实际部署应按各平台 Dockerfile 保留厂商 Torch、引擎和补丁，不能在厂商镜像内盲目执行通用全量升级。已有 `==2.7.0` 或固定提交等更严格约束继续保留。源码安装必须选择原指南中的固定提交或相应旧版标签，不跟随默认分支。

## 平台指南

以下链接为已归档的中文历史快照，不作为当前主线安装说明。

- [昇腾 / Ascend](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Ascend.md)
- [平头哥 / THead](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/THead.md)
- [沐曦 / METAX](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/METAX.md)
- [海光 / Hygon](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Hygon.md)
- [燧原 / Enflame](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Enflame.md)
- [摩尔线程 / MooreThreads](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/MooreThreads.md)
- [天数智芯 / IluvatarCorex](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/IluvatarCorex.md)
- [寒武纪 / Cambricon](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Cambricon.md)
- [昆仑芯 / Kunlunxin](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Kunlunxin.md)
- [太初元碁 / Tecorigin](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Tecorigin.md)
- [壁仞 / Biren](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Biren.md)
- [瀚博 / VastAI](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/VastAI.md)

## 旧版入口与资料

这些适配使用 `mineru -p`、`mineru-api`、`mineru-gradio` 和 `mineru-models-download`，不使用 4.0 的 `mineru-kit`。旧版参考：

- [3.4.5 installation](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/zh/quick_start/index.md)
- [3.4.5 model configuration](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/zh/usage/model_source.md)

厂商预制镜像的内容由厂商控制，仓库内 `<4` 约束不会修改已有镜像。部署前在容器中检查：

```bash
python -c "from importlib.metadata import version; from packaging.specifiers import SpecifierSet; v = version('mineru'); print(v); assert v in SpecifierSet('<4'), 'Use the vendor image for MinerU <4'"
```
