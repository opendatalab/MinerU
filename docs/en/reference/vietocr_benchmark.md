# VietOCR recognizer — benchmark

`lang='vi'` routes the recognition step to VietOCR (`vgg_transformer`) instead
of PaddleOCR's shared recognizer (see `mineru/model/ocr/vietocr_fast_batch.py`).
This page documents the decode optimization and its measured effect.

## Why a custom decoder

`vietocr.tool.predictor.Predictor.predict_batch()` has two problems on
full-page crop lists:

1. It groups crops by resized width and concatenates each group into a single
   unbounded `torch.cat()` — a full page can request a multi-GB contiguous
   allocation and OOM.
2. `translate()` decodes the whole batch autoregressively **without a
   KV-cache**, recomputing self-attention over the entire growing prefix every
   step (per-step cost grows with prefix length), and only stops when the
   *slowest* sequence in the batch emits EOS.

`predict_batch_grouped()` replaces it with:

- **Sub-batching** by a fixed size (bounds peak memory).
- **Length-aware grouping** (sort each width bucket by estimated text length so
  a sub-batch isn't a mix of very short and very long lines).
- **KV-cache** decode: self-attention K/V are cached per step; cross-attention
  K/V over the fixed encoder memory are projected once. Per-step cost stops
  growing with prefix length.
- **Early-exit**: sequences drop out of the active batch as soon as they emit
  EOS instead of being stepped to the longest sequence's length.

It is **architecture-guarded** — the hand-rolled KV-cache decode runs only on a
post-norm `nn.TransformerDecoderLayer` stack; any other architecture falls back
to a decoder that drives the model's own `forward_decoder` (correct for any
`norm_first`), so a differently-configured model cannot produce silently-wrong
output.

## Correctness

Verified byte-identical to per-image `Predictor.predict()` — 0 text mismatches
across thousands of real crops, for both the KV-cache path and the fallback
path. The architecture guard was checked to select the fast path for the
reference model and reject pre-norm / subclassed-layer / missing-norm variants.

## Speed

Isolated microbenchmark, 2000 real Vietnamese line crops, single RTX 5080,
0/2000 text mismatches vs stock:

| Decoder | ms/crop | total |
|---|---|---|
| `predict_batch()` (stock vietocr) | 155.4 | 310.8 s |
| `predict_batch_grouped` (KV-cache + early-exit) | **9.0** | **18.0 s** |

**≈ 17× faster** than stock `predict_batch()` on this run. In the full pipeline
(where recognition shares the GPU with layout/table/detection stages), the
largest OCR batches went from ~150 ms/crop to ~50 ms/crop (≈ 3×), with per-step
GPU-kernel time reduced ~4× by the KV-cache.

Numbers depend on document content (line lengths), GPU thermal state and
hardware, and vary run-to-run (measured 17–26× across runs); treat them as
representative rather than exact.
