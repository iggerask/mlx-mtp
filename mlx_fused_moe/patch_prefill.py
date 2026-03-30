"""
Monkey-patch Model to only compute LM head on the last token during prefill.

During prefill, the LM head (vocab projection) processes all N tokens, but we
only need logits for the last position. For Qwen3.5-35B-A3B with 248K vocab,
the LM head weight matrix is ~270MB — loading it takes ~1.4ms per position.

At ctx=2048, this wastes ~400ms (13% of total prefill time).

This patch detects prefill (seq_len > 1) and only runs the LM head on the
last hidden state, returning logits of shape (batch, 1, vocab) instead of
(batch, seq, vocab). This is safe because during prefill we only need the
last token's logits for sampling.
"""

_original_model_call = None


_LAST_TOK_THRESHOLD = 32  # Only apply for sequences longer than this


def _make_last_tok_call(original_call):
    """Create patched __call__ that only computes LM head for last token.

    Only applies for long sequences (prefill). Short sequences (decode,
    MTP verify batches of 2-4 tokens) pass through unchanged since the
    caller may need logits at non-last positions.
    """
    import mlx.core as mx

    def patched_call(self, inputs, cache=None, input_embeddings=None):
        seq_len = inputs.shape[-1] if inputs.ndim > 1 else inputs.shape[0]

        if seq_len <= _LAST_TOK_THRESHOLD:
            return original_call(self, inputs, cache=cache, input_embeddings=input_embeddings)

        # Prefill: run backbone, then LM head only on last token
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        last_hidden = out[:, -1:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(last_hidden)
        else:
            logits = self.lm_head(last_hidden)
        return logits

    return patched_call


def patch_last_tok_head(model, verbose=True):
    """Patch TextModel to only compute LM head on last token during prefill."""
    global _original_model_call

    from mlx_lm.models.qwen3_5 import TextModel

    _original_model_call = TextModel.__call__
    TextModel.__call__ = _make_last_tok_call(_original_model_call)

    if verbose:
        print("[patch_prefill] Patched TextModel with last-token LM head for prefill")


def unpatch_last_tok_head():
    """Restore original TextModel.__call__."""
    global _original_model_call

    from mlx_lm.models.qwen3_5 import TextModel

    if _original_model_call is not None:
        TextModel.__call__ = _original_model_call
        _original_model_call = None
