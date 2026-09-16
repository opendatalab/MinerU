# README and Skill Layout Design

## Goal

Make the MinerU agent guide the repository's primary README while preserving it as a discoverable Skill and retaining the existing English and Chinese project documentation under explicit language filenames.

## File layout

```text
README.md                         # canonical agent guide and former Skill content
README_en.md                      # former README.md
README_zh.md                      # former README_zh-CN.md
skills/mineru/SKILL.md            # symbolic link to ../../README.md
```

`README.md` remains a regular file so GitHub, PyPI, source archives, and build tools read it normally. The Skill path is the symbolic link so Skill discovery still finds the required filename and frontmatter without duplicating the content.

## References

- Add a common navigation line linking the agent guide, English documentation, and Chinese documentation.
- Update root README language links to the new filenames.
- Update repository-owned documentation that points to the renamed root Chinese README.
- Leave references inside nested third-party or independent projects unchanged.
- Keep `pyproject.toml` configured with `readme = "README.md"`.

## Validation

- Confirm `skills/mineru/SKILL.md` resolves to the root README.
- Run the Skill validator through the symbolic link.
- Confirm old root README filenames are no longer referenced by repository-owned documentation.
- Build package metadata locally and verify the configured README is readable.
