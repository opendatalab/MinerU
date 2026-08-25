# Hugging Face Xet Dependency Design

## Context

MinerU exposes Hugging Face model downloads from the base installation. `huggingface-hub` 0.36.x intends to install `hf-xet` on
supported 64-bit platforms, but its dependency marker omits the `AMD64` value reported by Windows. Environments constrained to the
0.x line can therefore fall back to regular HTTP and repeatedly warn that `hf_xet` is unavailable.

## Decision

Enable the existing `hf_xet` extra on MinerU's base `huggingface-hub` dependency:

```toml
huggingface-hub[hf_xet]>=0.32.4
```

This keeps model download behavior available from the base installation and delegates the compatible `hf-xet` version to
`huggingface-hub`. Document `HF_HUB_DISABLE_XET=1` as the explicit fallback for networks that cannot reach Xet CAS.

## Consequences

- Windows model downloads use Xet without a separate installation step.
- Base installations include the `hf-xet` native wheel, including installations that only use ModelScope or remote parsing.
- Supported mainstream 64-bit platforms have prebuilt wheels; unsupported platforms may need to build `hf-xet` or may fail to install.
- Users behind incompatible proxies can disable Xet and retain regular HTTP downloads.

## Verification

Add a project metadata test that requires the `hf-xet` extra and resolve the project dependencies for Windows x86-64 before release.
