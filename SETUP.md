# MinerU Environment Setup Guide (dev)

A short guide for working **from source code** in this repo, because apparently life wasn’t chaotic enough already. For full details (Docker, GPU, individual backends), see the [English Quick Start](docs/en/quick_start/index.md) and [README](README.md).

## Requirements

* Python **3.10–3.13** (per `pyproject.toml`)
* Recommended: **uv** for faster virtual environment creation and package installation

## Clean Installation (remove old venv, then recreate)

From the repo root directory (where `pyproject.toml` is located):

```bash
rm -rf .venv
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -e ".[all]"
```

On Windows (PowerShell):

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
uv venv .venv --python python3.12
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[all]"
```

### Without `uv`

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[all]"
```

## Optional Package Sets

| Command                    | Meaning                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| `pip install -e ".[all]"`  | Full feature set (matches upstream quick start)                         |
| `pip install -e ".[core]"` | Lighter install: VLM + pipeline + Gradio, without all optional backends |

## PyPI Mirror (if needed)

Example of a commonly used China mirror from official docs:

```bash
uv pip install -e ".[all]" -i https://mirrors.aliyun.com/pypi/simple
```

## Post-Install Tips

* Activate the virtual environment for each work session: `source .venv/bin/activate`
* Check the CLI: `mineru --help`
* The repo README may mention `./setup.sh` or `./run.sh`; if your fork **does not** include those files, just use the commands above and run `mineru` directly after activating the venv.

## Cursor Skill `/init`

This repo includes an **init** skill at `.cursor/skills/init/SKILL.md`, which describes the workflow for deleting `.venv`, rebuilding it, and reinstalling `mineru[all]` when you need the environment reset.
