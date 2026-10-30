# Legacy Platform Adaptations (MinerU <4)

The following 12 platforms retain their existing adaptations, **for MinerU <4 only**. Mainline 4.0 dependency, model, and command changes do not extend these platforms' compatibility coverage.

The nine dedicated Dockerfiles retain their original extras and use `>=3.4.0,<4`, for example:

```bash
python -m pip install "mineru[core]>=3.4.0,<4"
```

This illustrates the version bound, not a universal vendor installation recipe. Follow each Dockerfile's Torch, engine, and patch requirements rather than applying a broad upgrade inside a vendor image. Keep stricter constraints such as `==2.7.0` or fixed commits. Source installs must use the original pinned commit or an appropriate legacy tag instead of following the default branch.

## Platform guides

The Chinese guides are archived snapshots for legacy versions, not current mainline instructions.

- [Ascend](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Ascend.md)
- [THead](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/THead.md)
- [METAX](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/METAX.md)
- [Hygon](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Hygon.md)
- [Enflame](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Enflame.md)
- [MooreThreads](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/MooreThreads.md)
- [IluvatarCorex](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/IluvatarCorex.md)
- [Cambricon](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Cambricon.md)
- [Kunlunxin](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Kunlunxin.md)
- [Tecorigin](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Tecorigin.md)
- [Biren](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/Biren.md)
- [VastAI](https://github.com/opendatalab/MinerU/blob/master/docs/archive/platforms/VastAI.md)

## Legacy commands and references

These adaptations use `mineru -p`, `mineru-api`, `mineru-gradio`, and `mineru-models-download`, not 4.0 `mineru-kit`. Legacy references:

- [3.4.5 installation](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/quick_start/index.md)
- [3.4.5 model configuration](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/usage/model_source.md)

Vendors control prebuilt image contents; a `<4` bound in this repository does not modify existing images. Check inside the container before deployment:

```bash
python -c "from importlib.metadata import version; from packaging.specifiers import SpecifierSet; v = version('mineru'); print(v); assert v in SpecifierSet('<4'), 'Use the vendor image for MinerU <4'"
```
