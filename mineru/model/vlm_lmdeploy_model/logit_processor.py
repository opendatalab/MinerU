import torch

def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    ngram_n = 5
    if prev_input_ids.shape[1] < ngram_size + ngram_n + 1:
        return generated_ngrams
    gen_tokens = prev_input_ids[:, -ngram_size - ngram_n - 1:].tolist()
    for idx in range(num_hypos):
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        #for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
        for ngram_i in range(ngram_n):
            ngram = gen_tokens[idx][-ngram_size:] if ngram_i == 0 \
                    else gen_tokens[idx][-ngram_size-ngram_i: -ngram_i]
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: torch.Tensor,
                              num_hypos: int, cur_len: int):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx],
                              prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


def lmdeploy_custom_logits_processor(all_ids, scores):
    ngram_size = 100
    if all_ids is not None:
        all_ids = all_ids.unsqueeze(0)
        scores = scores.unsqueeze(0)
        generated_ngrams = [{} for _ in range(all_ids.shape[0])]
        num_batch_hypotheses = scores.shape[0]
        cur_len = all_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(
            ngram_size, all_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")
        scores = scores_processed
        scores = scores.squeeze(0)
    return scores