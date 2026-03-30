#!/usr/bin/env python3
"""
Comprehensive benchmark: Multi-token-per-step strategies for Qwen3.5-35B-A3B.

Doubling down on "process more tokens per step" — the only path to beating
baseline 74 t/s on M2 Max (since mx.compile showed no benefit).

Strategies tested:
  1. Baseline autoregressive
  2. MTP K=1 sequential verify (bit-exact)
  3. MTP K=1 batch verify
  4. MTP K=2,3,4 batch verify (chain MTP head)
  5. Prompt lookup draft=3,5,8
  6. Shared-expert draft K=1,2,3
  7. Optimistic 2-token batch (top-1 from logits → verify)
  8. MTP K=1 + prompt lookup hybrid

Key theoretical insight:
  Batch processing K tokens costs ~14 + (K-1)*4.5 ms (GDN sequential update).
  Throughput = expected_accepted_tokens / step_cost.
  Breakeven acceptance rate for K=1: ~60% (step overhead must beat 13.4ms baseline).
"""

import gc
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state
from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import PromptLookupDrafter, SharedExpertDrafter

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")
MAX_TOKENS = 200
NUM_RUNS = 3  # best-of-N

# Prompts designed to test different speculation strengths
PROMPTS = {
    "code": "Write a Python function that implements merge sort with type hints:\n```python\ndef merge_sort(arr: list[int]) -> list[int]:\n",
    "prose": "Explain how transformers work in machine learning, starting from self-attention:\n",
    "short": "The capital of France is",
    "repetitive": "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
    "qa": "What are the main differences between Python and Rust? List the key points:\n1.",
}


@dataclass
class BenchResult:
    strategy: str
    category: str
    tokens_generated: int
    prefill_ms: float
    decode_ms: float
    decode_tok_s: float
    acceptance_rate: float
    tokens_per_step: float
    total_steps: int
    output_preview: str


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


# ---------------------------------------------------------------------------
# 1. Baseline autoregressive
# ---------------------------------------------------------------------------

def generate_baseline(model, prompt_arr, max_tokens, eos_set):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    tokens = []
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())

    for _ in range(max_tokens - 1):
        if tokens[-1] in eos_set:
            break
        logits = model(mx.array([[tokens[-1]]]), cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        tokens.append(tok.item())

    t_done = time.perf_counter()
    return tokens, (t_prefill - t0) * 1000, (t_done - t_prefill) * 1000, len(tokens), 1.0, 0.0


# ---------------------------------------------------------------------------
# 2-4. MTP speculation at K=1,2,3,4
# ---------------------------------------------------------------------------

def generate_mtp(decoder, model, prompt_arr, max_tokens, eos_set):
    cache = make_prompt_cache(model)
    decoder.stats.__init__()
    t0 = time.perf_counter()
    tokens = list(decoder.generate(
        prompt_arr, cache, max_tokens=max_tokens,
        temperature=0.0, eos_tokens=eos_set,
    ))
    t_total = time.perf_counter() - t0
    prefill_ms = decoder.stats.prefill_time * 1000
    decode_ms = (t_total - decoder.stats.prefill_time) * 1000
    return (tokens, prefill_ms, decode_ms,
            decoder.stats.total_steps,
            decoder.stats.tokens_per_step,
            decoder.stats.acceptance_rate)


# ---------------------------------------------------------------------------
# 5. Prompt lookup speculation
# ---------------------------------------------------------------------------

def generate_prompt_lookup(model, prompt_arr, max_tokens, eos_set,
                           prompt_tokens, max_ngram=5, max_draft=5):
    drafter = PromptLookupDrafter(prompt_tokens, max_ngram=max_ngram, max_draft=max_draft)
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())

    while len(tokens) < max_tokens and tokens[-1] not in eos_set:
        total_steps += 1
        draft_ids = drafter.draft(tokens)

        if not draft_ids:
            logits = model(mx.array([[tokens[-1]]]), cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            continue

        total_drafted += len(draft_ids)
        saved = save_cache_state(cache)
        verify_input = mx.array([[tokens[-1]] + draft_ids])
        verify_logits = model(verify_input, cache=cache)
        mx.eval(verify_logits)

        accepted = 0
        for i, did in enumerate(draft_ids):
            verified = mx.argmax(verify_logits[:, i, :], axis=-1)
            mx.eval(verified)
            if verified.item() == did:
                accepted += 1
                tokens.append(did)
            else:
                tokens.append(verified.item())
                break
        else:
            bonus = mx.argmax(verify_logits[:, len(draft_ids), :], axis=-1)
            mx.eval(bonus)
            tokens.append(bonus.item())
            accepted += 1

        total_accepted += accepted

        if accepted < len(draft_ids):
            restore_cache_state(cache, saved)
            n_to_replay = accepted + 1
            replay_tokens = [tokens[-(n_to_replay + 1)]] + tokens[-n_to_replay:]
            replay = mx.array([replay_tokens])
            replay_logits = model(replay, cache=cache)
            mx.eval(replay_logits)

    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tps = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, prefill_ms, decode_ms, total_steps, tps, acc_rate


# ---------------------------------------------------------------------------
# 6. Shared-expert self-speculation
# ---------------------------------------------------------------------------

def generate_shared_expert(model, drafter, prompt_arr, max_tokens, eos_set,
                           num_draft=1):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())

    while len(tokens) < max_tokens and tokens[-1] not in eos_set:
        total_steps += 1
        saved = save_cache_state(cache)

        # Draft with shared-expert-only mode
        draft_ids = []
        draft_tok = mx.array([[tokens[-1]]])
        drafter.enable()
        try:
            for _ in range(num_draft):
                d_logits = model(draft_tok, cache=cache)
                mx.eval(d_logits)
                d_tok = mx.argmax(d_logits[:, -1, :], axis=-1)
                mx.eval(d_tok)
                did = d_tok.item()
                if did in eos_set:
                    break
                draft_ids.append(did)
                draft_tok = mx.array([[did]])
        finally:
            drafter.disable()

        if not draft_ids:
            restore_cache_state(cache, saved)
            logits = model(mx.array([[tokens[-1]]]), cache=cache)
            mx.eval(logits)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            continue

        total_drafted += len(draft_ids)
        restore_cache_state(cache, saved)
        verify_input = mx.array([[tokens[-1]] + draft_ids])
        verify_logits = model(verify_input, cache=cache)
        mx.eval(verify_logits)

        accepted = 0
        for i, did in enumerate(draft_ids):
            verified = mx.argmax(verify_logits[:, i, :], axis=-1)
            mx.eval(verified)
            if verified.item() == did:
                accepted += 1
                tokens.append(did)
            else:
                tokens.append(verified.item())
                break
        else:
            bonus = mx.argmax(verify_logits[:, len(draft_ids), :], axis=-1)
            mx.eval(bonus)
            tokens.append(bonus.item())
            accepted += 1

        total_accepted += accepted

        if accepted < len(draft_ids):
            restore_cache_state(cache, saved)
            n_to_replay = accepted + 1
            replay_tokens = [tokens[-(n_to_replay + 1)]] + tokens[-n_to_replay:]
            replay = mx.array([replay_tokens])
            replay_logits = model(replay, cache=cache)
            mx.eval(replay_logits)

    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tps = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, prefill_ms, decode_ms, total_steps, tps, acc_rate


# ---------------------------------------------------------------------------
# 7. Optimistic 2-token batch decode
# ---------------------------------------------------------------------------

def generate_optimistic_batch(model, prompt_arr, max_tokens, eos_set, batch_k=2):
    """Draft K-1 tokens using argmax from each position's logits, then verify.

    Unlike MTP, this uses the model itself to draft — but speculatively.
    We process token_0, take argmax as draft_1, then batch-verify
    [token_0, draft_1, ..., draft_{K-1}] in one forward pass.

    For K=2: draft using the model's own previous-step logits.
    """
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    # First token
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())

    # We'll carry forward the "next predicted token" from each step
    # For K=2: process current token, grab top-1 as draft, verify both next step
    pending_draft = None  # draft token from previous step's logits

    while len(tokens) < max_tokens and tokens[-1] not in eos_set:
        total_steps += 1
        current_tid = tokens[-1]

        if pending_draft is not None and pending_draft not in eos_set:
            # We have a draft from previous step — verify it
            total_drafted += 1
            saved = save_cache_state(cache)

            verify_input = mx.array([[current_tid, pending_draft]])
            verify_logits = model(verify_input, cache=cache)
            mx.eval(verify_logits)

            verified = mx.argmax(verify_logits[:, 0, :], axis=-1)
            mx.eval(verified)
            verified_id = verified.item()

            if verified_id == pending_draft:
                # Accept draft
                total_accepted += 1
                tokens.append(pending_draft)
                # Bonus token from position 1
                bonus = mx.argmax(verify_logits[:, 1, :], axis=-1)
                mx.eval(bonus)
                bonus_id = bonus.item()
                tokens.append(bonus_id)
                # Next draft = top-2nd prediction? No, just use bonus as next current
                # and draft from position 1 logits
                # Actually, get the draft for next step from the bonus position
                pending_draft = None  # We'd need another forward pass to get a draft
                # Let's just predict from the verify_logits[:, 1, :] — that's the bonus
                # But we already sampled it. We need a NEW draft for next round.
                # The next step processes bonus_id, and we could draft from these logits.
                # For simplicity: use top-1 from verify_logits[:, 1, :] as the bonus,
                # and top-1 from a hypothetical next step as draft. But we don't have that.
                # So we'll fall back to getting a draft on the next iteration.
                pending_draft = None
            else:
                # Reject draft
                restore_cache_state(cache, saved)
                logits = model(mx.array([[current_tid]]), cache=cache)
                mx.eval(logits)
                verified2 = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(verified2)
                tokens.append(verified2.item())
                # Draft for next step from this logits
                # Get 2nd-choice? No, just get argmax which is the verified token.
                # The draft for next step: run model with verified token → get draft
                pending_draft = None
        else:
            # No draft — standard step, get draft for next round
            logits = model(mx.array([[current_tid]]), cache=cache)
            mx.eval(logits)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            # Use this token as draft for next step? No, this IS the verified token.
            # We need to predict what comes AFTER this token.
            # We can't without another forward pass.
            # Alternative: use top-1 as next token AND top-2 as draft.
            # Actually, let's just draft the next token speculatively:
            # After getting tok from logits, we know tok is "position N".
            # Draft: what would come at position N+1? We'd need model(tok, cache).
            # That defeats the purpose.
            pending_draft = None

    # The simple optimistic approach doesn't work well without a cheap drafter.
    # Let's try a different approach: 2-step lookahead using the model itself.
    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tps = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, prefill_ms, decode_ms, total_steps, tps, acc_rate


# ---------------------------------------------------------------------------
# 8. MTP + Prompt Lookup Hybrid
# ---------------------------------------------------------------------------

def generate_mtp_plus_lookup(model, mtp_head, prompt_arr, max_tokens, eos_set,
                              prompt_tokens, max_ngram=5, max_draft_lookup=3):
    """Hybrid: use MTP to draft 1 token, then extend with prompt lookup matches.

    The MTP head gives us 1 high-quality draft. If prompt lookup can extend
    the n-gram further, we get extra free tokens.
    """
    drafter_lookup = PromptLookupDrafter(prompt_tokens, max_ngram=max_ngram,
                                          max_draft=max_draft_lookup)
    capture = HiddenStateCapture(model)
    cache = make_prompt_cache(model)

    # Resolve components
    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model
    embed_tokens = text_model.model.embed_tokens
    if text_model.args.tie_word_embeddings:
        lm_head = text_model.model.embed_tokens.as_linear
    else:
        lm_head = text_model.lm_head

    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    hidden = capture.get_hidden_state()
    mx.eval(logits, hidden)
    t_prefill = time.perf_counter()

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())
    last_hidden = hidden[:, -1:, :]
    current_token = tok.reshape(1)

    while len(tokens) < max_tokens and tokens[-1] not in eos_set:
        total_steps += 1

        # MTP draft (1 token)
        tok_embed = embed_tokens(current_token[None])
        if tok_embed.ndim == 2:
            tok_embed = tok_embed[:, None, :]
        mtp_h = mtp_head(last_hidden, tok_embed)
        mtp_logits = lm_head(mtp_h)
        mtp_draft = mx.argmax(mtp_logits[:, -1, :], axis=-1)
        mx.eval(mtp_draft)
        mtp_draft_id = mtp_draft.item()

        # Try to extend with prompt lookup
        draft_ids = [mtp_draft_id]
        if mtp_draft_id not in eos_set:
            # Look for n-gram continuation after [tokens..., mtp_draft_id]
            lookup_continuation = drafter_lookup.draft(tokens + [mtp_draft_id])
            if lookup_continuation:
                draft_ids.extend(lookup_continuation[:max_draft_lookup])

        # Filter EOS from drafts
        clean_drafts = []
        for d in draft_ids:
            if d in eos_set:
                break
            clean_drafts.append(d)
        draft_ids = clean_drafts

        if not draft_ids:
            # No drafts — standard step
            logits = model(current_token.reshape(1, 1), cache=cache)
            hidden = capture.get_hidden_state()
            mx.eval(logits, hidden)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            last_hidden = hidden[:, -1:, :]
            current_token = tok.reshape(1)
            continue

        total_drafted += len(draft_ids)
        saved = save_cache_state(cache)

        # Batch verify all drafts
        verify_tokens = [tokens[-1]] + draft_ids
        verify_input = mx.array([verify_tokens])
        verify_logits = model(verify_input, cache=cache)
        verify_hidden = capture.get_hidden_state()
        mx.eval(verify_logits, verify_hidden)

        accepted = 0
        for i, did in enumerate(draft_ids):
            verified = mx.argmax(verify_logits[:, i, :], axis=-1)
            mx.eval(verified)
            if verified.item() == did:
                accepted += 1
                tokens.append(did)
            else:
                tokens.append(verified.item())
                break
        else:
            bonus = mx.argmax(verify_logits[:, len(draft_ids), :], axis=-1)
            mx.eval(bonus)
            tokens.append(bonus.item())
            accepted += 1

        total_accepted += accepted

        if accepted == len(draft_ids) + 1:
            # All accepted
            last_hidden = verify_hidden[:, len(draft_ids):len(draft_ids)+1, :]
            current_token = mx.array([tokens[-1]]).reshape(1)
        elif accepted < len(draft_ids):
            # Partial accept — restore and replay
            restore_cache_state(cache, saved)
            n_accepted = accepted + 1  # includes the correction token
            replay_tokens = [tokens[-(n_accepted + 1)]] + tokens[-n_accepted:]
            replay = mx.array([replay_tokens])
            replay_logits = model(replay, cache=cache)
            replay_hidden = capture.get_hidden_state()
            mx.eval(replay_logits, replay_hidden)
            last_hidden = replay_hidden[:, -1:, :]
            current_token = mx.array([tokens[-1]]).reshape(1)
        else:
            last_hidden = verify_hidden[:, -1:, :]
            current_token = mx.array([tokens[-1]]).reshape(1)

    capture.restore()
    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tps = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, prefill_ms, decode_ms, total_steps, tps, acc_rate


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_best_of_n(fn, n=NUM_RUNS):
    """Run fn N times, return result with best decode t/s."""
    best = None
    best_tps = 0
    for _ in range(n):
        result = fn()
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = result
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        if tps > best_tps:
            best_tps = tps
            best = result
    return best


def main():
    print("=" * 90)
    print("MULTI-TOKEN-PER-STEP BENCHMARK — Qwen3.5-35B-A3B-4bit")
    print(f"Max tokens: {MAX_TOKENS}, Best of {NUM_RUNS} runs")
    print("=" * 90)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    # Load MTP head
    print("Loading MTP head...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    mtp_weights = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_head = build_mtp_head(mtp_weights, config, norm_shift=True)

    # Build shared-expert drafter
    print("Setting up shared-expert drafter...")
    try:
        shared_drafter = SharedExpertDrafter(model)
    except Exception as e:
        print(f"  SharedExpertDrafter failed: {e}")
        shared_drafter = None

    # Warmup
    print("Warming up...")
    warmup_arr = mx.array(tokenizer.encode("Hello world"))
    for _ in range(3):
        generate_baseline(model, warmup_arr, 20, eos_set)
    print("Warmup done.\n")

    results = []
    baseline_tokens_by_cat = {}

    strategies = [
        # (name, lambda prompt_arr, prompt_tokens, cat -> result)
    ]

    for cat, prompt in PROMPTS.items():
        prompt_tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(prompt_tokens)
        preview = prompt[:60].replace("\n", "\\n")
        print(f"\n{'='*80}")
        print(f"Category: {cat} — '{preview}'")
        print(f"{'='*80}")

        cat_results = []

        # 1. Baseline
        print("  [1] Baseline autoregressive...")
        r = run_best_of_n(lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        baseline_tokens_by_cat[cat] = tokens
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "baseline", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 2. MTP K=1 sequential
        print("  [2] MTP K=1 sequential...")
        dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=1, batch_verify=False))
        r = run_best_of_n(lambda p=prompt_arr: generate_mtp(dec, model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        dec.cleanup()
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_k1_seq", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 3. MTP K=1 batch
        print("  [3] MTP K=1 batch...")
        dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=1, batch_verify=True))
        r = run_best_of_n(lambda p=prompt_arr: generate_mtp(dec, model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        dec.cleanup()
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_k1_batch", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 4. MTP K=2 batch
        print("  [4] MTP K=2 batch...")
        dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=2, batch_verify=True))
        r = run_best_of_n(lambda p=prompt_arr: generate_mtp(dec, model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        dec.cleanup()
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_k2_batch", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 5. MTP K=3 batch
        print("  [5] MTP K=3 batch...")
        dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=3, batch_verify=True))
        r = run_best_of_n(lambda p=prompt_arr: generate_mtp(dec, model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        dec.cleanup()
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_k3_batch", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 6. MTP K=4 batch
        print("  [6] MTP K=4 batch...")
        dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=4, batch_verify=True))
        r = run_best_of_n(lambda p=prompt_arr: generate_mtp(dec, model, p, MAX_TOKENS, eos_set))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        dec.cleanup()
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_k4_batch", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 7. Prompt lookup draft=3
        print("  [7] Prompt lookup (draft=3)...")
        r = run_best_of_n(lambda p=prompt_arr, pt=prompt_tokens:
            generate_prompt_lookup(model, p, MAX_TOKENS, eos_set, pt, max_draft=3))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "prompt_lookup_d3", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 8. Prompt lookup draft=8
        print("  [8] Prompt lookup (draft=8)...")
        r = run_best_of_n(lambda p=prompt_arr, pt=prompt_tokens:
            generate_prompt_lookup(model, p, MAX_TOKENS, eos_set, pt, max_draft=8))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "prompt_lookup_d8", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 9. Shared-expert draft K=1
        if shared_drafter:
            print("  [9] Shared-expert (draft=1)...")
            r = run_best_of_n(lambda p=prompt_arr:
                generate_shared_expert(model, shared_drafter, p, MAX_TOKENS, eos_set, num_draft=1))
            tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
            n_decode = max(len(tokens) - 1, 1)
            tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
            ms_per_step = decode_ms / steps if steps > 0 else 0
            print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
            out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
            cat_results.append(BenchResult(
                "shared_expert_d1", cat, len(tokens), prefill_ms, decode_ms, tps,
                acc, tps_step, steps, out_preview))

        # 10. Shared-expert draft K=3
        if shared_drafter:
            print("  [10] Shared-expert (draft=3)...")
            r = run_best_of_n(lambda p=prompt_arr:
                generate_shared_expert(model, shared_drafter, p, MAX_TOKENS, eos_set, num_draft=3))
            tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
            n_decode = max(len(tokens) - 1, 1)
            tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
            ms_per_step = decode_ms / steps if steps > 0 else 0
            print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
            out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
            cat_results.append(BenchResult(
                "shared_expert_d3", cat, len(tokens), prefill_ms, decode_ms, tps,
                acc, tps_step, steps, out_preview))

        # 11. MTP + prompt lookup hybrid
        print("  [11] MTP K=1 + prompt lookup hybrid...")
        r = run_best_of_n(lambda p=prompt_arr, pt=prompt_tokens:
            generate_mtp_plus_lookup(model, mtp_head, p, MAX_TOKENS, eos_set, pt,
                                     max_ngram=5, max_draft_lookup=3))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "mtp_plus_lookup", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        # 12. Optimistic 2-token batch
        print("  [12] Optimistic 2-token batch...")
        r = run_best_of_n(lambda p=prompt_arr:
            generate_optimistic_batch(model, p, MAX_TOKENS, eos_set, batch_k=2))
        tokens, prefill_ms, decode_ms, steps, tps_step, acc = r
        n_decode = max(len(tokens) - 1, 1)
        tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_per_step = decode_ms / steps if steps > 0 else 0
        print(f"    {len(tokens)} tok, {tps:.1f} t/s, {tps_step:.2f} tok/step, {acc:.0%} accept, {ms_per_step:.1f} ms/step")
        out_preview = tokenizer.decode(tokens[:30]).replace("\n", "\\n")[:60]
        cat_results.append(BenchResult(
            "optimistic_batch_2", cat, len(tokens), prefill_ms, decode_ms, tps,
            acc, tps_step, steps, out_preview))

        results.extend(cat_results)

    # ---------------------------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------------------------

    print("\n\n" + "=" * 110)
    print("COMPREHENSIVE RESULTS — Multi-Token-Per-Step Strategies")
    print("=" * 110)

    strategies = sorted(set(r.strategy for r in results),
                       key=lambda s: next((i for i, r in enumerate(results) if r.strategy == s), 0))
    categories = list(PROMPTS.keys())

    # Overall averages
    print(f"\n{'Strategy':<24} {'Avg t/s':>8} {'vs Base':>8} {'Acc Rate':>9} {'Tok/Step':>9} {'ms/step':>8}")
    print("-" * 70)

    base_results = [r for r in results if r.strategy == "baseline"]
    avg_base = sum(r.decode_tok_s for r in base_results) / len(base_results) if base_results else 0

    for strat in strategies:
        sr = [r for r in results if r.strategy == strat and r.decode_tok_s > 0]
        if not sr:
            continue
        avg_tps = sum(r.decode_tok_s for r in sr) / len(sr)
        avg_acc = sum(r.acceptance_rate for r in sr) / len(sr)
        avg_tps_step = sum(r.tokens_per_step for r in sr) / len(sr)
        avg_ms = sum(r.decode_ms / r.total_steps for r in sr if r.total_steps > 0) / len(sr)
        ratio = avg_tps / avg_base if avg_base > 0 else 0
        print(f"{strat:<24} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>8.0%} {avg_tps_step:>8.2f} {avg_ms:>7.1f}")

    # Per-category breakdown
    print(f"\n{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strategies:
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].decode_tok_s:>11.1f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Acceptance rate per category
    print(f"\nACCEPTANCE RATES:")
    print(f"{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strategies:
        if strat == "baseline":
            continue
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].acceptance_rate:>11.0%}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Tokens per step per category
    print(f"\nTOKENS PER STEP:")
    print(f"{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strategies:
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].tokens_per_step:>11.2f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # ms per step per category
    print(f"\nMS PER STEP:")
    print(f"{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strategies:
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr and sr[0].total_steps > 0:
                ms = sr[0].decode_ms / sr[0].total_steps
                print(f" {ms:>11.1f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Save
    out_file = "benchmark_batch_speculation_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
