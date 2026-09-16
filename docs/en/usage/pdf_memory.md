# PDF memory regression checks

Window lifetime improvements are always active. `MINERU_MALLOC_TRIM` independently enables optional CPU heap trimming and is off by default. Measure real workloads on Linux/glibc before claiming an RSS improvement or an OOM fix; macOS checks cannot establish either.

## Compare three configurations

Use the same host, Python environment, DocVortex version, models, settings, and concurrency. Create a separate worktree at the pre-change commit without switching or overwriting your development checkout.

Activate the project Python environment and install the diagnostic-only dependency with `uv pip install psutil`. Run the updated script from your development checkout. `--repo` selects the child process's MinerU source; third-party dependencies still come from the current Python environment.

```bash
python scripts/benchmark_pdf_memory.py --repo /tmp/mineru-memory-baseline --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 0 --output /tmp/mineru-memory-before
python scripts/benchmark_pdf_memory.py --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 0 --output /tmp/mineru-memory-after-off
python scripts/benchmark_pdf_memory.py --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 1 --output /tmp/mineru-memory-after-on
```

Each group starts a fresh parser process, warms up for three rounds, and measures twenty rounds. One document is processed per round, cycling through multiple inputs in order and releasing the preceding result. Output directories must not already exist.

- For repeated-document checks, pass one PDF. For alternating documents, pass two or more.
- For multiple windows, use a document longer than `--window-size`. This affects only the experiment, not product defaults.
- Exercise `basic/standard/advanced` with `txt/ocr`, plus `flash --mode ocr`, using the required models and VLM configuration.
- Without neural models, `flash --mode txt` can validate the harness, document cleanup, and selective rendering. It cannot validate inference-window allocation lifetimes or inference performance.

## Outputs and interpretation

- `rounds.jsonl`: input, page count, parse duration, and timestamps per round, marking warmup rounds.
- `samples.jsonl`: externally sampled RSS/USS for the parser and all child PIDs every 0.2 seconds. Unavailable USS is `null`; change frequency with `--interval`.
- `parse.log`: production logs with window ranges and render-worker PIDs for correlation.
- `summary.json`: measured-round median parse time, per-PID peaks, and simultaneous process-total peaks. Summed RSS can double-count shared pages, and sampling can miss brief peaks.

Inspect the parent and workers separately. Compare post-warmup trends, plateaus, peaks, and timing. Investigate a median slowdown above 5% with trimming disabled after excluding concurrent workloads, cold model startup, and sampling noise. Report enabled-trim overhead separately and repeat the complete experiment before drawing conclusions.

Trimming cannot release live results, model caches, other allocators' pools, another process's memory, or objects still referenced by exception tracebacks. DocVortex worker reclamation is outside this change.

## Output parity

In separate output directories, add `--warmup 0 --rounds 2 --fingerprints` to each configuration. Compare `middle_sha256` and `model_sha256` for matching inputs; hashes include full asset fields. Fingerprinting performs extra serialization, so do not use these runs for memory or performance conclusions.

If Linux, models, or USS access are unavailable, record that limitation explicitly and keep trimming disabled by default.
