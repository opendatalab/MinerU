from typing import List

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class Mineru2LogitProcessor(CustomLogitProcessor):
    """
    Stateless logit processor for Mineru2.

    (base-class: sglang.srt.sampling.custom_logit_processor.CustomLogitProcessor)

    This processor applies token-level constraints to prevent repetition during generation.
    It supports two main constraints:

    - no_repeat_ngram_size (int):
        Prevents repeating the same n-gram of specified size in the output.
        Inspired by Hugging Face's NoRepeatNGramLogitsProcessor.
        This implementation is slower due to its lack of specialized optimization.

    - no_repeat_token_count (int):
        (Placeholder for future logic)
        Intended to prevent repeating the same token multiple times.
        Not yet implemented in this version.
    """

    def __init__(self) -> None:
        super().__init__()
        self._generated_ngrams = {}  # Cache of generated n-grams by request ID
        self._time = {}  # Timestamp of the last update for each request
        self._gen_step = 0  # Global generation step counter

    def __call__(self, logits, batch_info: List[dict]):
        """
        Applies repetition constraints to the logits before sampling tokens.

        Args:
            logits (FloatTensor): A tensor of shape (batch_size, vocab_size) containing raw token logits.
            batch_info (List[dict]): A list of metadata dicts for each sample in the batch. Each dict must include:
                - "__req__": Request object containing request ID and output_ids.
                - "no_repeat_ngram_size": Size of n-gram to avoid repeating.

        Returns:
            FloatTensor: The modified logits tensor with banned token logits set to -inf.
        """
        from sglang.srt.managers.schedule_batch import Req

        self._gen_step += 1  # Update global generation step

        for idx, info in enumerate(batch_info):
            if not isinstance(info, dict) or "__req__" not in info:
                continue

            req: Req = info["__req__"]
            rid = req.rid
            output_ids = req.output_ids
            ngram_size = info.get("no_repeat_ngram_size", 0)

            # Skip if there are not enough tokens to form an n-gram
            if ngram_size <= 0 or len(output_ids) < ngram_size:
                continue

            # Record the current step for cache cleanup tracking
            self._time[rid] = self._gen_step

            # Initialize n-gram cache for this request if it doesn't exist
            if rid not in self._generated_ngrams:
                self._generated_ngrams[rid] = {}

            # Get the n-gram prefix (all but the last token)
            prev_ngram = tuple(output_ids[-ngram_size:-1])
            last_token = output_ids[-1]

            # Store this n-gram occurrence
            self._generated_ngrams[rid][prev_ngram] = self._generated_ngrams[rid].get(prev_ngram, []) + [last_token]

            # Get the next-token candidates to ban based on current prefix
            current_prefix = tuple(output_ids[-ngram_size + 1 :])
            banned_tokens = self._generated_ngrams[rid].get(current_prefix, [])

            # Set the logits of banned tokens to negative infinity
            for token in banned_tokens:
                logits[idx][token] = -float("inf")

        # Clean up cache for expired requests
        expired_rids = [rid for rid, last_used in self._time.items() if last_used < self._gen_step]
        for rid in expired_rids:
            self._generated_ngrams.pop(rid, None)
            self._time.pop(rid, None)

        return logits
