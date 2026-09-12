# Legacy Platform Adaptations (MinerU <4)

The following 13 platforms retain their existing adaptations, **for MinerU <4 only**. Mainline 4.0 dependency, model, and command changes do not extend these platforms' compatibility coverage.

The nine dedicated Dockerfiles retain their original extras and use `>=3.4.0,<4`, for example:

```bash
python -m pip install "mineru[core]>=3.4.0,<4"
```

This illustrates the version bound, not a universal vendor installation recipe. Follow each Dockerfile's Torch, engine, and patch requirements rather than applying a broad upgrade inside a vendor image. Keep stricter constraints such as `==2.7.0` or fixed commits. Source installs must use the original pinned commit or an appropriate legacy tag instead of following the default branch.

## Platform guides

The retained platform guides are in Chinese.

- [Ascend](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/)
- [THead](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/THead/)
- [METAX](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/METAX/)
- [Hygon](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Hygon/)
- [Enflame](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Enflame/)
- [MooreThreads](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/MooreThreads/)
- [IluvatarCorex](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/IluvatarCorex/)
- [Cambricon](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Cambricon/)
- [Kunlunxin](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Kunlunxin/)
- [Tecorigin](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Tecorigin/)
- [Biren](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Biren/)
- [AMD](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/AMD/)
- [VastAI](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/VastAI/)

## Legacy commands and references

These adaptations use `mineru -p`, `mineru-api`, `mineru-gradio`, and `mineru-models-download`, not 4.0 `mineru-kit`. Legacy references:

- [3.4.5 installation](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/quick_start/index.md)
- [3.4.5 model configuration](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/usage/model_source.md)

Vendors control prebuilt image contents; a `<4` bound in this repository does not modify existing images. Check inside the container before deployment:

```bash
python -c "from importlib.metadata import version; from packaging.specifiers import SpecifierSet; v = version('mineru'); print(v); assert v in SpecifierSet('<4'), 'Use the vendor image for MinerU <4'"
```
