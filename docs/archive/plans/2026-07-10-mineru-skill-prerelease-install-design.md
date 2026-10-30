# MinerU Skill Pre-release Installation Design

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

## Goal

Ensure the MinerU document-parser skill installs the current Alpha release instead of silently selecting an older stable release.

## Design

- Add a pre-release status notice immediately after the skill title.
- Explain that normal package resolution prefers stable releases and that MinerU installation commands must opt in to pre-releases.
- Add `--prerelease allow` to every `uv tool install` command that installs MinerU.
- Add `--pip-args="--pre"` to every `pipx install` command that installs MinerU.
- Add `--pre` to every `pip install` command that installs MinerU.
- Do not add pre-release flags to interpreter discovery, version inspection, or Python installation commands.
- Do not pin a specific Alpha version so later Alpha releases remain installable without another skill update.

## Validation

- Confirm every MinerU package installation command contains the installer-specific pre-release option.
- Confirm the skill passes `quick_validate.py`.
