# Extending MinerU Functionality Through Configuration Files

- MinerU is designed to work out-of-the-box, but also supports extending functionality through configuration files. You can create a `mineru.json` file in your home directory and add custom configurations.
- The `mineru.json` file will be automatically generated when you use the built-in model download command `mineru-models-download`. Alternatively, you can create it by copying the [configuration template file](../../mineru.template.json) to your home directory and renaming it to `mineru.json`.
- Below are some available configuration options:
  - `latex-delimiter-config`: Used to configure LaTeX formula delimiters, defaults to the `$` symbol, and can be modified to other symbols or strings as needed.
  - `llm-aided-config`: Used to configure related parameters for LLM-assisted heading level detection, compatible with all LLM models supporting the `OpenAI protocol`. It defaults to Alibaba Cloud Qwen's `qwen2.5-32b-instruct` model. You need to configure an API key yourself and set `enable` to `true` to activate this feature.
  - `models-dir`: Used to specify local model storage directories. Please specify separate model directories for the `pipeline` and `vlm` backends. After specifying these directories, you can use local models by setting the environment variable `export MINERU_MODEL_SOURCE=local`.

---