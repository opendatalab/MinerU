# 旧平台适配（MinerU <4）

以下 13 个平台保留已有适配方案，**仅适用于 MinerU <4**。主线 4.0 的依赖、模型和命令变更不自动扩展这些平台的兼容范围。

9 个专用 Dockerfile 的 MinerU 依赖为原 extras 与 `>=3.4.0,<4`。例如：

```bash
python -m pip install "mineru[core]>=3.4.0,<4"
```

此命令仅说明版本约束；实际部署应按各平台 Dockerfile 保留厂商 Torch、引擎和补丁，不能在厂商镜像内盲目执行通用全量升级。已有 `==2.7.0` 或固定提交等更严格约束继续保留。源码安装必须选择原指南中的固定提交或相应旧版标签，不跟随默认分支。

## 平台指南

- [昇腾 / Ascend](acceleration_cards/Ascend.md)
- [平头哥 / THead](acceleration_cards/THead.md)
- [沐曦 / METAX](acceleration_cards/METAX.md)
- [海光 / Hygon](acceleration_cards/Hygon.md)
- [燧原 / Enflame](acceleration_cards/Enflame.md)
- [摩尔线程 / MooreThreads](acceleration_cards/MooreThreads.md)
- [天数智芯 / IluvatarCorex](acceleration_cards/IluvatarCorex.md)
- [寒武纪 / Cambricon](acceleration_cards/Cambricon.md)
- [昆仑芯 / Kunlunxin](acceleration_cards/Kunlunxin.md)
- [太初元碁 / Tecorigin](acceleration_cards/Tecorigin.md)
- [壁仞 / Biren](acceleration_cards/Biren.md)
- [AMD](acceleration_cards/AMD.md)
- [瀚博 / VastAI](acceleration_cards/VastAI.md)

## 旧版入口与资料

这些适配使用 `mineru -p`、`mineru-api`、`mineru-gradio` 和 `mineru-models-download`，不使用 4.0 的 `mineru-kit`。旧版参考：

- [3.4.5 installation](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/zh/quick_start/index.md)
- [3.4.5 model configuration](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/zh/usage/model_source.md)

厂商预制镜像的内容由厂商控制，仓库内 `<4` 约束不会修改已有镜像。部署前在容器中检查：

```bash
python -c "from importlib.metadata import version; from packaging.specifiers import SpecifierSet; v = version('mineru'); print(v); assert v in SpecifierSet('<4'), 'Use the vendor image for MinerU <4'"
```
