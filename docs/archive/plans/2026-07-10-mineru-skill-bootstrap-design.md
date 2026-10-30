# MinerU Skill Bootstrap Design

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

## Goal

Place a concise Skill installation entry at the beginning of the root README for both users and agents.

## Design

- Keep YAML frontmatter first so the README remains a valid Skill artifact.
- Place installation instructions immediately after the title and language navigation.
- Use `npx skills add "opendatalab/MinerU#next" --global --yes` as the primary one-command installation path.
- Explain that a plain terminal that cannot detect an Agent should omit `--yes` and choose the target interactively.
- Provide curl and wget fallbacks that download the `next` branch root README from `raw.githubusercontent.com` and save it as `<agent-skills-dir>/mineru/SKILL.md`.
- Do not execute any installation command while editing or validating the documentation.

## Validation

- Validate the README frontmatter through the Skill validator.
- Confirm all installation URLs and destination filenames are present.
- Confirm no install command ran during validation.
