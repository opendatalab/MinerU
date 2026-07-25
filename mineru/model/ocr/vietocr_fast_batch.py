import csv
import math
import os
import tempfile
import time
from collections import defaultdict

import torch
from loguru import logger
from torch.nn.functional import softmax
from vietocr.tool.translate import process_input, resize

_STEP_DEBUG = os.environ.get("MINERU_VIETOCR_STEP_DEBUG") == "1"
_STEP_DEBUG_CSV = os.environ.get(
    "MINERU_VIETOCR_STEP_DEBUG_CSV",
    os.path.join(tempfile.gettempdir(), "vietocr_step_debug.csv"),
)


def _uncapped_width(w, h, expected_height, round_to=10):
    # Same math as vietocr.tool.translate.resize(), minus the image_max_width
    # clamp -- used only as a sort key so crops squashed into the same
    # capped-width tensor bucket are still grouped by their real relative
    # length (short lines don't get batched with much longer ones).
    new_w = int(expected_height * float(w) / float(h))
    return max(math.ceil(new_w / round_to) * round_to, round_to)


def _translate_early_exit(img_batch, model, device, sos_token=1, eos_token=2, max_seq_length=128, stats=None):
    # Batched autoregressive decode that drops each sequence out of the
    # active set as soon as it emits eos_token, instead of vietocr's
    # translate() which keeps stepping the whole batch until every sequence
    # (including the slowest one) finishes. Compute shrinks as sequences finish.
    use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
    model.eval()
    with torch.no_grad():
        src = model.cnn(img_batch)
        memory = model.transformer.forward_encoder(src)  # (T, N, E)

        n = img_batch.size(0)
        sequences = [[sos_token] for _ in range(n)]
        prob_sums = [0.0] * n
        prob_counts = [0] * n
        active = list(range(n))

        step = 0
        while active and step < max_seq_length:
            active_idx = torch.tensor(active, dtype=torch.long, device=memory.device)
            mem_active = memory.index_select(1, active_idx)
            tgt_inp = torch.LongTensor([sequences[i] for i in active]).transpose(0, 1).to(device)

            if stats is not None:
                if use_cuda:
                    torch.cuda.synchronize()
                t_kernel0 = time.perf_counter()

            output, _ = model.transformer.forward_decoder(tgt_inp, mem_active)
            output = softmax(output, dim=-1)

            if stats is not None:
                if use_cuda:
                    torch.cuda.synchronize()
                step_elapsed = time.perf_counter() - t_kernel0
                stats['gpu_kernel'] += step_elapsed
                stats['steps'] += 1
                if 'step_log' in stats:
                    # (prefix_len, active_batch_size, elapsed_s) -- to check
                    # whether per-step cost grows with prefix length (no
                    # KV-cache -> self-attention recomputed over the whole
                    # tgt_inp every step -> expected O(prefix_len^2)).
                    stats['step_log'].append((tgt_inp.shape[0], len(active), step_elapsed))

            top_probs, top_tokens = output[:, -1, :].max(dim=-1)
            next_tokens = top_tokens.tolist()
            next_probs = top_probs.tolist()

            still_active = []
            for pos, orig_idx in enumerate(active):
                tok = next_tokens[pos]
                sequences[orig_idx].append(tok)
                if tok != eos_token:
                    prob_sums[orig_idx] += next_probs[pos]
                    prob_counts[orig_idx] += 1
                    still_active.append(orig_idx)
            active = still_active
            step += 1

        confidences = [
            (prob_sums[i] / prob_counts[i]) if prob_counts[i] > 0 else 1.0 for i in range(n)
        ]
        return sequences, confidences


def _split_heads(x, num_heads):
    # x: (T, N, E) -> (N*num_heads, T, head_dim), matching the layout
    # torch.nn.functional.scaled_dot_product_attention-style batched matmul wants.
    T, N, E = x.shape
    head_dim = E // num_heads
    x = x.view(T, N, num_heads, head_dim)
    return x.permute(1, 2, 0, 3).reshape(N * num_heads, T, head_dim)


def _merge_heads(x, num_heads, N):
    # x: (N*num_heads, T, head_dim) -> (T, N, E)
    _, T, head_dim = x.shape
    x = x.view(N, num_heads, T, head_dim)
    return x.permute(2, 0, 1, 3).reshape(T, N, num_heads * head_dim)


def _project_qkv(x, in_proj_weight, in_proj_bias, d_model):
    # x: (T, N, E). Mirrors nn.MultiheadAttention's internal in-projection
    # (in_proj_weight/bias is the same combined (3E, E) parameter the model
    # was trained with -- we're only reorganizing *when* it's applied, not
    # changing the weights or the math).
    w_q, w_k, w_v = in_proj_weight.split(d_model, dim=0)
    b_q, b_k, b_v = (in_proj_bias.split(d_model, dim=0) if in_proj_bias is not None else (None, None, None))
    q = torch.nn.functional.linear(x, w_q, b_q)
    k = torch.nn.functional.linear(x, w_k, b_k)
    v = torch.nn.functional.linear(x, w_v, b_v)
    return q, k, v


def _attn_step(q_1, k_full, v_full, num_heads, out_proj, d_model):
    # q_1: (1, N, E) query for the single new position.
    # k_full, v_full: (T, N, E) -- either the growing self-attn cache
    # (causally valid: T = positions 0..current, all allowed for this query)
    # or the fixed, precomputed encoder-memory projection for cross-attn.
    N = q_1.shape[1]
    head_dim = d_model // num_heads
    q = _split_heads(q_1, num_heads)  # (N*H, 1, hd)
    k = _split_heads(k_full, num_heads)  # (N*H, T, hd)
    v = _split_heads(v_full, num_heads)  # (N*H, T, hd)
    scores = torch.bmm(q, k.transpose(1, 2)) / (head_dim ** 0.5)  # (N*H, 1, T)
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v)  # (N*H, 1, hd)
    out = _merge_heads(out, num_heads, N)  # (1, N, E)
    return out_proj(out)


def _kv_cache_applicable(model):
    # The KV-cache decode below re-implements
    # torch.nn.TransformerDecoderLayer.forward by hand, so it is only correct
    # for a *post-norm* (norm_first=False) stack of plain
    # nn.TransformerDecoderLayer inside vietocr's LanguageTransformer (the
    # default `transformer` seq modeling). Anything else -- norm_first=True, a
    # subclassed decoder layer with different forward math, a non-transformer
    # seq model, or a missing decoder -- must use the architecture-agnostic
    # fallback (_translate_early_exit), which drives the model's own
    # forward_decoder and is therefore correct for any of these.
    lt = getattr(model, "transformer", None)
    inner = getattr(lt, "transformer", None)
    decoder = getattr(inner, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if not layers or getattr(decoder, "norm", None) is None:
        return False
    for layer in layers:
        # Exact type (not isinstance): a subclass may override forward math.
        if type(layer) is not torch.nn.TransformerDecoderLayer:
            return False
        if getattr(layer, "norm_first", False):
            return False
    return True


def _transformer_api_available(model):
    # _translate_early_exit needs vietocr's LanguageTransformer encode/decode
    # entry points; absent them (an exotic seq model) neither in-house decoder
    # applies and the caller should degrade to its own recognizer.
    lt = getattr(model, "transformer", None)
    return (
        hasattr(model, "cnn")
        and hasattr(lt, "forward_encoder")
        and hasattr(lt, "forward_decoder")
    )


def _decoder_layer_step(layer, x_1, self_k_cache, self_v_cache, mem_k, mem_v, num_heads, d_model):
    # Re-implements TransformerDecoderLayer.forward (norm_first=False; the
    # applicability of this is checked by _kv_cache_applicable before use)
    # for a single new position, using the layer's own weights. Self-attention
    # reuses cached K/V from previous steps instead of recomputing them;
    # cross-attention reuses the precomputed, unchanging encoder-memory K/V.
    q, k_new, v_new = _project_qkv(x_1, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias, d_model)
    self_k_cache = torch.cat([self_k_cache, k_new], dim=0)
    self_v_cache = torch.cat([self_v_cache, v_new], dim=0)
    sa_out = _attn_step(q, self_k_cache, self_v_cache, num_heads, layer.self_attn.out_proj, d_model)
    x = layer.norm1(x_1 + sa_out)

    q_c, _, _ = _project_qkv(x, layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias, d_model)
    mha_out = _attn_step(q_c, mem_k, mem_v, num_heads, layer.multihead_attn.out_proj, d_model)
    x = layer.norm2(x + mha_out)

    ff_out = layer.linear2(layer.activation(layer.linear1(x)))
    x = layer.norm3(x + ff_out)

    return x, self_k_cache, self_v_cache


def _translate_kv_cache(img_batch, model, device, sos_token=1, eos_token=2, max_seq_length=128, stats=None):
    # Same early-exit batching as _translate_early_exit, but each decoder
    # layer's self-attention reuses a growing K/V cache instead of
    # recomputing projections for the whole prefix every step, and
    # cross-attention K/V (over the fixed encoder memory) is computed once
    # up front instead of every step. Removes the O(prefix_len) per-step
    # cost growth measured in MINERU_VIETOCR_STEP_DEBUG (no more O(L^2)
    # total decode cost from re-projecting old positions every step).
    use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
    lt = model.transformer  # vietocr LanguageTransformer
    layers = lt.transformer.decoder.layers
    final_norm = lt.transformer.decoder.norm
    num_heads = layers[0].self_attn.num_heads
    d_model = lt.d_model
    n_layers = len(layers)

    model.eval()
    with torch.no_grad():
        src = model.cnn(img_batch)
        memory = lt.forward_encoder(src)  # (S, N, E)
        n = img_batch.size(0)

        # Cross-attention K/V depend only on `memory`, which never changes
        # across decode steps -- project once per layer instead of every step.
        mem_kv = []
        for layer in layers:
            _, k_mem, v_mem = _project_qkv(
                memory, layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias, d_model
            )
            mem_kv.append((k_mem, v_mem))

        sequences = [[sos_token] for _ in range(n)]
        prob_sums = [0.0] * n
        prob_counts = [0] * n
        active = list(range(n))

        # Per-layer self-attention K/V cache, full n-width (matches the
        # existing pattern of always index_select-ing from full-size state
        # rather than physically shrinking storage as `active` shrinks).
        self_k_cache = [torch.zeros(0, n, d_model, device=memory.device, dtype=memory.dtype) for _ in range(n_layers)]
        self_v_cache = [torch.zeros(0, n, d_model, device=memory.device, dtype=memory.dtype) for _ in range(n_layers)]

        step = 0
        while active and step < max_seq_length:
            active_idx = torch.tensor(active, dtype=torch.long, device=memory.device)
            last_tok = torch.LongTensor([[sequences[i][-1] for i in active]]).to(device)  # (1, N_active)

            if stats is not None:
                if use_cuda:
                    torch.cuda.synchronize()
                t_kernel0 = time.perf_counter()

            x = lt.pos_enc.dropout(
                lt.embed_tgt(last_tok) * (d_model ** 0.5) + lt.pos_enc.pe[step : step + 1, :]
            )  # (1, N_active, E)

            for li, layer in enumerate(layers):
                k_cache_active = self_k_cache[li].index_select(1, active_idx)
                v_cache_active = self_v_cache[li].index_select(1, active_idx)
                mem_k_active = mem_kv[li][0].index_select(1, active_idx)
                mem_v_active = mem_kv[li][1].index_select(1, active_idx)

                x, k_new_active, v_new_active = _decoder_layer_step(
                    layer, x, k_cache_active, v_cache_active, mem_k_active, mem_v_active, num_heads, d_model
                )

                new_k_full = torch.zeros(1, n, d_model, device=memory.device, dtype=memory.dtype)
                new_v_full = torch.zeros(1, n, d_model, device=memory.device, dtype=memory.dtype)
                new_k_full[:, active_idx, :] = k_new_active[-1:]
                new_v_full[:, active_idx, :] = v_new_active[-1:]
                self_k_cache[li] = torch.cat([self_k_cache[li], new_k_full], dim=0)
                self_v_cache[li] = torch.cat([self_v_cache[li], new_v_full], dim=0)

            x = final_norm(x)
            logits = lt.fc(x.transpose(0, 1))  # (N_active, 1, vocab)
            output = softmax(logits, dim=-1)

            if stats is not None:
                if use_cuda:
                    torch.cuda.synchronize()
                step_elapsed = time.perf_counter() - t_kernel0
                stats['gpu_kernel'] += step_elapsed
                stats['steps'] += 1
                if 'step_log' in stats:
                    stats['step_log'].append((step + 1, len(active), step_elapsed))

            top_probs, top_tokens = output[:, -1, :].max(dim=-1)
            next_tokens = top_tokens.tolist()
            next_probs = top_probs.tolist()

            still_active = []
            for pos, orig_idx in enumerate(active):
                tok = next_tokens[pos]
                sequences[orig_idx].append(tok)
                if tok != eos_token:
                    prob_sums[orig_idx] += next_probs[pos]
                    prob_counts[orig_idx] += 1
                    still_active.append(orig_idx)
            active = still_active
            step += 1

        confidences = [
            (prob_sums[i] / prob_counts[i]) if prob_counts[i] > 0 else 1.0 for i in range(n)
        ]
        return sequences, confidences


def _dump_step_debug(step_log, batch_n):
    # step_log entries: (prefix_len, active_batch_size, elapsed_s).
    # Append raw rows to CSV for offline analysis.
    file_exists = os.path.isfile(_STEP_DEBUG_CSV)
    with open(_STEP_DEBUG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["batch_n", "prefix_len", "active_n", "elapsed_ms", "ms_per_active"])
        for prefix_len, active_n, elapsed_s in step_log:
            ms = elapsed_s * 1000
            writer.writerow([batch_n, prefix_len, active_n, f"{ms:.4f}", f"{ms / active_n:.4f}"])

    # Bucket by prefix_len (width of tgt_inp at that step) to see whether
    # per-active-sequence cost grows with prefix length (expected if
    # forward_decoder recomputes self-attention over the full prefix each
    # step, i.e. no KV-cache -> ~O(prefix_len^2) per step).
    bucket_size = 16
    buckets = defaultdict(list)  # bucket_start -> list of ms_per_active
    for prefix_len, active_n, elapsed_s in step_log:
        b = (prefix_len // bucket_size) * bucket_size
        buckets[b].append((elapsed_s * 1000) / active_n)

    lines = [f"[STEP-DEBUG] batch_n={batch_n} steps={len(step_log)} -- ms/active_seq by prefix_len bucket:"]
    for b in sorted(buckets):
        vals = buckets[b]
        avg = sum(vals) / len(vals)
        lines.append(f"    prefix_len [{b:>3}-{b + bucket_size:>3}): n_steps={len(vals):<5} avg_ms_per_active={avg:.4f}")
    logger.info("\n".join(lines))


def predict_batch_grouped(predictor, imgs, sub_batch_size=256):
    # Replacement for vietocr.tool.predictor.Predictor.predict_batch() that
    # (a) sub-batches to bound peak memory (avoids the multi-GB torch.cat
    # OOM on large crop lists) and (b) sorts each width-bucket by estimated
    # real text length before chunking, so a sub-batch isn't a mix of very
    # short and very long lines waiting on each other -- combined with
    # early-exit decode above, short sub-batches finish in few steps instead
    # of being dragged to the bucket's worst case every time.
    cfg_ds = predictor.config['dataset']
    image_height = cfg_ds['image_height']
    image_min_width = cfg_ds['image_min_width']
    image_max_width = cfg_ds['image_max_width']
    device = predictor.device
    model = predictor.model
    vocab = predictor.vocab
    use_cuda = torch.cuda.is_available() and 'cuda' in str(device)

    # Pick the fastest decoder the model's architecture actually supports:
    # KV-cache (fastest) only for a post-norm nn.TransformerDecoderLayer stack,
    # otherwise the architecture-agnostic early-exit path. If neither applies,
    # raise so the caller can fall back to its own recognizer rather than
    # silently returning wrong text.
    if _kv_cache_applicable(model):
        decode_fn = _translate_kv_cache
    elif _transformer_api_available(model):
        logger.warning(
            "VietOCR KV-cache path not applicable for this model architecture; "
            "using the slower architecture-agnostic early-exit decoder."
        )
        decode_fn = _translate_early_exit
    else:
        raise RuntimeError(
            "VietOCR model exposes neither a supported decoder layer stack nor "
            "the LanguageTransformer encode/decode API; cannot batch-decode."
        )

    t_wall_start = time.perf_counter()

    # CPU-side: PIL resize + numpy/tensor prep, one image at a time.
    t_prep0 = time.perf_counter()
    prepped = []
    for i, im in enumerate(imgs):
        w, h = im.size
        capped_w, _ = resize(w, h, image_height, image_min_width, image_max_width)
        uncapped_w = _uncapped_width(w, h, image_height)
        tensor = process_input(im, image_height, image_min_width, image_max_width)
        prepped.append((i, tensor, capped_w, uncapped_w))
    t_prep = time.perf_counter() - t_prep0

    buckets = defaultdict(list)
    for item in prepped:
        buckets[item[2]].append(item)

    results = [None] * len(imgs)
    t_decode_wall = 0.0
    stats = {'gpu_kernel': 0.0, 'steps': 0}
    if _STEP_DEBUG:
        stats['step_log'] = []
    for items in buckets.values():
        items.sort(key=lambda t: t[3])
        for start in range(0, len(items), sub_batch_size):
            sub = items[start : start + sub_batch_size]
            idxs = [t[0] for t in sub]
            batch_tensor = torch.cat([t[1] for t in sub], dim=0).to(device)

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            sequences, confidences = decode_fn(batch_tensor, model, device, stats=stats)
            if use_cuda:
                torch.cuda.synchronize()
            t_decode_wall += time.perf_counter() - t0

            for idx, seq, conf in zip(idxs, sequences, confidences):
                results[idx] = (vocab.decode(seq), conf)

    if _STEP_DEBUG and stats['step_log']:
        _dump_step_debug(stats['step_log'], batch_n=len(imgs))

    t_wall = time.perf_counter() - t_wall_start
    t_other = t_wall - t_prep - t_decode_wall
    t_py_glue = t_decode_wall - stats['gpu_kernel']
    logger.debug(
        f"[TIMING] VietOCR fast-batch breakdown: n={len(imgs)}  "
        f"prep(CPU resize)={t_prep:.2f}s  "
        f"decode_total={t_decode_wall:.2f}s "
        f"[gpu_kernel(forward_decoder+softmax)={stats['gpu_kernel']:.2f}s  "
        f"py_glue(tensor-build/index_select/argmax/bookkeeping, {stats['steps']} steps)={t_py_glue:.2f}s]  "
        f"other(concat/vocab)={t_other:.2f}s  total={t_wall:.2f}s"
    )

    return results
