#!/usr/bin/env python3
"""
Push MTP K=1 batch to maximum throughput.

From the comprehensive benchmark:
  - MTP K=1 batch: 78.8 t/s (1.09x baseline) — the only consistent winner
  - MTP K=3 on repetitive: 100.4 t/s (1.40x) — great when acceptance is high
  - Bottleneck: MTP head ~5ms + batch verify ~18ms = 23ms for 1.82 tokens

Optimizations to test:
  1. Lazy draft (remove mx.eval sync before batch verify)
  2. Quantized MTP head (4-bit: reduce 5ms → ~2-3ms)
  3. Adaptive K (K=1 normally, K=2-3 when recent acceptance > 90%)
  4. Combined: lazy + quantized + adaptive
  5. Pipelined: overlap MTP head with model forward via async eval
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state
from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder, MTPStats
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head

logging.basicConfig(level=logging.WARNING)

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")
MAX_TOKENS = 200
NUM_RUNS = 3

PROMPTS = {
    "code": "Write a Python function that implements merge sort with type hints:\n```python\ndef merge_sort(arr: list[int]) -> list[int]:\n",
    "prose": "Explain how transformers work in machine learning, starting from self-attention:\n",
    "repetitive": "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
    "qa": "What are the main differences between Python and Rust? List the key points:\n1.",
}


@dataclass
class Result:
    strategy: str
    category: str
    tokens: int
    prefill_ms: float
    decode_ms: float
    tok_s: float
    acceptance: float
    tok_per_step: float
    steps: int
    ms_per_step: float


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


# ---------------------------------------------------------------------------
# Baseline
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
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    n_steps = max(len(tokens) - 1, 1)
    return tokens, prefill_ms, decode_ms, n_steps, 1.0, 0.0


# ---------------------------------------------------------------------------
# Standard MTP K=1 batch (current implementation)
# ---------------------------------------------------------------------------

def generate_mtp_standard(model, mtp_head, prompt_arr, max_tokens, eos_set):
    dec = MTPDecoder(model, mtp_head, MTPConfig(num_speculative_tokens=1, batch_verify=True))
    cache = make_prompt_cache(model)
    dec.stats = MTPStats()
    t0 = time.perf_counter()
    tokens = list(dec.generate(prompt_arr, cache, max_tokens=max_tokens,
                               temperature=0.0, eos_tokens=eos_set))
    t_total = time.perf_counter() - t0
    prefill_ms = dec.stats.prefill_time * 1000
    decode_ms = (t_total - dec.stats.prefill_time) * 1000
    result = (tokens, prefill_ms, decode_ms,
              dec.stats.total_steps, dec.stats.tokens_per_step, dec.stats.acceptance_rate)
    dec.cleanup()
    return result


# ---------------------------------------------------------------------------
# Optimization 1: Lazy draft (no early eval of draft token)
# ---------------------------------------------------------------------------

def generate_mtp_lazy(model, mtp_head, prompt_arr, max_tokens, eos_set):
    """MTP K=1 batch with lazy draft evaluation.

    Key change: Don't mx.eval(draft_token) before building verify batch.
    Let MLX build the entire graph (MTP head + model forward) before evaluation.
    """
    capture = HiddenStateCapture(model)

    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model
    embed_tokens = text_model.model.embed_tokens
    if text_model.args.tie_word_embeddings:
        lm_head = text_model.model.embed_tokens.as_linear
    else:
        lm_head = text_model.lm_head

    cache = make_prompt_cache(model)
    stats = MTPStats()
    t0 = time.perf_counter()

    # Prefill
    logits = model(prompt_arr[None], cache=cache)
    hidden = capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])
    stats.prefill_time = time.perf_counter() - t0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens = [tok.item()]
    stats.total_tokens = 1
    last_hidden = hidden[:, -1:, :]
    current_token = tok.reshape(1)

    while len(tokens) < max_tokens:
        if tokens[-1] in eos_set:
            break
        stats.total_steps += 1

        # Draft (lazy — don't eval yet)
        tok_embed = embed_tokens(current_token[None])
        if tok_embed.ndim == 2:
            tok_embed = tok_embed[:, None, :]
        mtp_h = mtp_head(last_hidden, tok_embed)
        mtp_logits = lm_head(mtp_h)
        draft_token = mx.argmax(mtp_logits[:, -1, :], axis=-1)
        # NO mx.eval(draft_token) here!

        stats.draft_attempts += 1

        # Save cache state
        saved = save_cache_state(cache)

        # Build batch verify with lazy draft token
        verify_input = mx.concatenate(
            [current_token.reshape(1, 1), draft_token.reshape(1, 1)], axis=1
        )
        verify_logits = model(verify_input, cache=cache)
        verify_hidden = capture.get_hidden_state()

        # Sample verified token (lazy)
        verified_token = mx.argmax(verify_logits[:, 0, :], axis=-1)

        # Eval everything at once — MTP head + model forward in single graph
        mx.eval(draft_token, verified_token, verify_logits, verify_hidden)

        draft_id = draft_token.item()
        verified_id = verified_token.item()

        if draft_id in eos_set:
            # Draft was EOS — rare, restore and rerun
            restore_cache_state(cache, saved)
            logits = model(current_token.reshape(1, 1), cache=cache)
            hidden = capture.get_hidden_state()
            mx.eval(logits, hidden)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            stats.total_tokens += 1
            last_hidden = hidden[:, -1:, :]
            current_token = tok.reshape(1)
            continue

        if verified_id == draft_id:
            # ACCEPT
            stats.draft_accepted += 1
            bonus_token = mx.argmax(verify_logits[:, 1, :], axis=-1)
            mx.eval(bonus_token)
            tokens.append(draft_id)
            tokens.append(bonus_token.item())
            stats.total_tokens += 2
            last_hidden = verify_hidden[:, 1:2, :]
            current_token = bonus_token.reshape(1)
        else:
            # REJECT — restore cache
            restore_cache_state(cache, saved)
            rerun_logits = model(current_token.reshape(1, 1), cache=cache)
            rerun_hidden = capture.get_hidden_state()
            mx.eval(rerun_logits, rerun_hidden)
            tokens.append(verified_id)
            stats.total_tokens += 1
            last_hidden = rerun_hidden[:, -1:, :]
            current_token = verified_token.reshape(1)

    capture.restore()
    t_total = time.perf_counter() - t0
    prefill_ms = stats.prefill_time * 1000
    decode_ms = (t_total - stats.prefill_time) * 1000
    return (tokens, prefill_ms, decode_ms,
            stats.total_steps, stats.tokens_per_step, stats.acceptance_rate)


# ---------------------------------------------------------------------------
# Optimization 2: Lazy draft + don't restore on reject (just rerun)
# ---------------------------------------------------------------------------

def generate_mtp_lazy_norestore(model, mtp_head, prompt_arr, max_tokens, eos_set):
    """MTP K=1 batch with lazy draft and no cache restore on reject.

    On reject, instead of restore + rerun, we just process the verified token
    in the NEXT step. The cache contains [token_0, wrong_draft], which is
    incorrect. So we must restore and rerun to get correct cache state.

    Wait — we MUST restore because the cache state is wrong after processing
    a rejected draft. This optimization is NOT possible for correctness.

    Alternative: On reject, just do a single forward pass of token_0.
    Skip the restore since we haven't modified the saved cache.
    Actually, save_cache_state + batch verify modifies cache.
    On reject we must restore.

    NEW IDEA: Don't save/restore at all. Instead:
    - Always run model on JUST token_0 (no batch with draft)
    - If draft matches: also run model on draft_token
    - This is sequential verify, but with lazy draft eval

    This avoids the save_cache_state cost entirely.
    """
    capture = HiddenStateCapture(model)

    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model
    embed_tokens = text_model.model.embed_tokens
    if text_model.args.tie_word_embeddings:
        lm_head = text_model.model.embed_tokens.as_linear
    else:
        lm_head = text_model.lm_head

    cache = make_prompt_cache(model)
    stats = MTPStats()
    t0 = time.perf_counter()

    # Prefill
    logits = model(prompt_arr[None], cache=cache)
    hidden = capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])
    stats.prefill_time = time.perf_counter() - t0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens = [tok.item()]
    stats.total_tokens = 1
    last_hidden = hidden[:, -1:, :]
    current_token = tok.reshape(1)

    while len(tokens) < max_tokens:
        if tokens[-1] in eos_set:
            break
        stats.total_steps += 1

        # Draft (lazy)
        tok_embed = embed_tokens(current_token[None])
        if tok_embed.ndim == 2:
            tok_embed = tok_embed[:, None, :]
        mtp_h = mtp_head(last_hidden, tok_embed)
        mtp_logits = lm_head(mtp_h)
        draft_token = mx.argmax(mtp_logits[:, -1, :], axis=-1)
        stats.draft_attempts += 1

        # Process token_0 through model (builds graph lazily)
        step1_logits = model(current_token.reshape(1, 1), cache=cache)
        step1_hidden = capture.get_hidden_state()
        verified_token = mx.argmax(step1_logits[:, -1, :], axis=-1)

        # Eval MTP draft + model forward together
        mx.eval(draft_token, verified_token, step1_hidden)

        draft_id = draft_token.item()
        verified_id = verified_token.item()

        if draft_id in eos_set or verified_id in eos_set:
            tokens.append(verified_id)
            stats.total_tokens += 1
            last_hidden = step1_hidden[:, -1:, :]
            current_token = verified_token.reshape(1)
            continue

        if verified_id == draft_id:
            # ACCEPT — also process draft
            stats.draft_accepted += 1
            step2_logits = model(draft_token.reshape(1, 1), cache=cache)
            step2_hidden = capture.get_hidden_state()
            bonus_token = mx.argmax(step2_logits[:, -1, :], axis=-1)
            mx.eval(step2_logits, step2_hidden, bonus_token)
            tokens.append(draft_id)
            tokens.append(bonus_token.item())
            stats.total_tokens += 2
            last_hidden = step2_hidden[:, -1:, :]
            current_token = bonus_token.reshape(1)
        else:
            # REJECT — cache already has token_0, no restore needed!
            tokens.append(verified_id)
            stats.total_tokens += 1
            last_hidden = step1_hidden[:, -1:, :]
            current_token = verified_token.reshape(1)

    capture.restore()
    t_total = time.perf_counter() - t0
    prefill_ms = stats.prefill_time * 1000
    decode_ms = (t_total - stats.prefill_time) * 1000
    return (tokens, prefill_ms, decode_ms,
            stats.total_steps, stats.tokens_per_step, stats.acceptance_rate)


# ---------------------------------------------------------------------------
# Optimization 3: Adaptive K
# ---------------------------------------------------------------------------

def generate_mtp_adaptive(model, mtp_head, prompt_arr, max_tokens, eos_set,
                          window=10, k_low=1, k_high=3, threshold=0.85):
    """Adaptive K: switch between K=1 and K=3 based on recent acceptance.

    When recent acceptance rate > threshold, use K=k_high for more aggressive
    speculation. Otherwise use K=k_low for safety.
    """
    capture = HiddenStateCapture(model)

    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model
    embed_tokens = text_model.model.embed_tokens
    if text_model.args.tie_word_embeddings:
        lm_head = text_model.model.embed_tokens.as_linear
    else:
        lm_head = text_model.lm_head

    cache = make_prompt_cache(model)
    stats = MTPStats()
    t0 = time.perf_counter()

    # Prefill
    logits = model(prompt_arr[None], cache=cache)
    hidden = capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])
    stats.prefill_time = time.perf_counter() - t0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens = [tok.item()]
    stats.total_tokens = 1
    last_hidden = hidden[:, -1:, :]
    current_token = tok.reshape(1)

    # Track recent acceptance
    recent_accepts = []  # list of bools

    while len(tokens) < max_tokens:
        if tokens[-1] in eos_set:
            break
        stats.total_steps += 1

        # Decide K based on recent acceptance
        if len(recent_accepts) >= window:
            recent_rate = sum(recent_accepts[-window:]) / window
        else:
            recent_rate = 0.5  # start conservative
        K = k_high if recent_rate >= threshold else k_low

        # Draft K tokens
        drafts = []
        h = last_hidden
        t = current_token
        for _ in range(K):
            t_embed = embed_tokens(t[None])
            if t_embed.ndim == 2:
                t_embed = t_embed[:, None, :]
            h = mtp_head(h, t_embed)
            d_logits = lm_head(h)
            t = mx.argmax(d_logits[:, -1, :], axis=-1)
            drafts.append(t)
            stats.draft_attempts += 1

        if K == 1:
            # Use sequential verify (no cache save/restore needed)
            draft_token = drafts[0]
            step1_logits = model(current_token.reshape(1, 1), cache=cache)
            step1_hidden = capture.get_hidden_state()
            verified_token = mx.argmax(step1_logits[:, -1, :], axis=-1)
            mx.eval(draft_token, verified_token, step1_hidden)

            draft_id = draft_token.item()
            verified_id = verified_token.item()

            if verified_id == draft_id and draft_id not in eos_set:
                stats.draft_accepted += 1
                recent_accepts.append(True)
                step2_logits = model(draft_token.reshape(1, 1), cache=cache)
                step2_hidden = capture.get_hidden_state()
                bonus = mx.argmax(step2_logits[:, -1, :], axis=-1)
                mx.eval(step2_logits, step2_hidden, bonus)
                tokens.append(draft_id)
                tokens.append(bonus.item())
                stats.total_tokens += 2
                last_hidden = step2_hidden[:, -1:, :]
                current_token = bonus.reshape(1)
            else:
                recent_accepts.append(False)
                tokens.append(verified_id)
                stats.total_tokens += 1
                last_hidden = step1_hidden[:, -1:, :]
                current_token = verified_token.reshape(1)
        else:
            # Multi-token batch verify
            mx.eval(*drafts)
            draft_ids = [d.item() for d in drafts]

            # Filter EOS
            for ei, did in enumerate(draft_ids):
                if did in eos_set:
                    draft_ids = draft_ids[:ei]
                    drafts = drafts[:ei]
                    break

            if not draft_ids:
                logits = model(current_token.reshape(1, 1), cache=cache)
                hidden = capture.get_hidden_state()
                mx.eval(logits, hidden)
                tok = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(tok)
                tokens.append(tok.item())
                stats.total_tokens += 1
                last_hidden = hidden[:, -1:, :]
                current_token = tok.reshape(1)
                recent_accepts.append(False)
                continue

            saved = save_cache_state(cache)
            all_tokens = [current_token.reshape(1, 1)] + [d.reshape(1, 1) for d in drafts]
            verify_input = mx.concatenate(all_tokens, axis=1)
            verify_logits = model(verify_input, cache=cache)
            verify_hidden = capture.get_hidden_state()
            mx.eval(verify_logits, verify_hidden)

            accepted = []
            last_accept_pos = -1
            for i, did in enumerate(draft_ids):
                v = mx.argmax(verify_logits[:, i, :], axis=-1)
                mx.eval(v)
                vid = v.item()
                if vid == did:
                    stats.draft_accepted += 1
                    accepted.append(did)
                    last_accept_pos = i
                    recent_accepts.append(True)
                else:
                    accepted.append(vid)
                    recent_accepts.append(False)
                    break
            else:
                bonus = mx.argmax(verify_logits[:, len(draft_ids), :], axis=-1)
                mx.eval(bonus)
                accepted.append(bonus.item())

            for tid in accepted:
                tokens.append(tid)
            stats.total_tokens += len(accepted)

            if last_accept_pos == len(draft_ids) - 1:
                last_hidden = verify_hidden[:, len(draft_ids):len(draft_ids)+1, :]
                current_token = mx.array([accepted[-1]]).reshape(1)
            else:
                restore_cache_state(cache, saved)
                n_accepted = len(accepted)
                replay_input = mx.concatenate(
                    [current_token.reshape(1, 1)] + [mx.array([[t]]) for t in accepted[:-1]],
                    axis=1,
                )
                replay_logits = model(replay_input, cache=cache)
                replay_hidden = capture.get_hidden_state()
                mx.eval(replay_logits, replay_hidden)
                last_hidden = replay_hidden[:, -1:, :]
                current_token = mx.array([accepted[-1]]).reshape(1)

    capture.restore()
    t_total = time.perf_counter() - t0
    prefill_ms = stats.prefill_time * 1000
    decode_ms = (t_total - stats.prefill_time) * 1000
    return (tokens, prefill_ms, decode_ms,
            stats.total_steps, stats.tokens_per_step, stats.acceptance_rate)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_best_of_n(fn, n=NUM_RUNS):
    best = None
    best_tps = 0
    for _ in range(n):
        result = fn()
        tok_list, pfill, dec_ms, steps, tps_step, acc = result
        n_dec = max(len(tok_list) - 1, 1)
        tps = n_dec / (dec_ms / 1000) if dec_ms > 0 else 0
        if tps > best_tps:
            best_tps = tps
            best = result
    return best


def main():
    print("=" * 90)
    print("OPTIMIZED MTP BENCHMARK — Pushing K=1 batch to maximum throughput")
    print(f"Max tokens: {MAX_TOKENS}, Best of {NUM_RUNS} runs")
    print("=" * 90)

    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    print("Loading MTP heads...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # BF16 head
    weights_bf16 = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_bf16 = build_mtp_head(weights_bf16, config, norm_shift=True)

    # Q4 head
    weights_q4 = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_q4 = build_mtp_head(weights_q4, config, norm_shift=True)
    quantize_mtp_head(mtp_q4, bits=4, group_size=64)

    bf16_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_bf16.parameters()))
    q4_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_q4.parameters()))
    print(f"  MTP BF16: {bf16_bytes/1e6:.1f} MB, Q4: {q4_bytes/1e6:.1f} MB ({q4_bytes/bf16_bytes:.0%})")

    # Warmup
    print("Warming up...")
    warmup = mx.array(tokenizer.encode("Hello world"))
    for _ in range(3):
        generate_baseline(model, warmup, 20, eos_set)
    print("Warmup done.\n")

    results = []

    strategies = [
        ("baseline", lambda p: generate_baseline(model, p, MAX_TOKENS, eos_set)),
        ("mtp_k1_batch_std", lambda p: generate_mtp_standard(model, mtp_bf16, p, MAX_TOKENS, eos_set)),
        ("mtp_k1_lazy_batch", lambda p: generate_mtp_lazy(model, mtp_bf16, p, MAX_TOKENS, eos_set)),
        ("mtp_k1_lazy_seq", lambda p: generate_mtp_lazy_norestore(model, mtp_bf16, p, MAX_TOKENS, eos_set)),
        ("mtp_q4_batch_std", lambda p: generate_mtp_standard(model, mtp_q4, p, MAX_TOKENS, eos_set)),
        ("mtp_q4_lazy_batch", lambda p: generate_mtp_lazy(model, mtp_q4, p, MAX_TOKENS, eos_set)),
        ("mtp_q4_lazy_seq", lambda p: generate_mtp_lazy_norestore(model, mtp_q4, p, MAX_TOKENS, eos_set)),
        ("adaptive_bf16", lambda p: generate_mtp_adaptive(model, mtp_bf16, p, MAX_TOKENS, eos_set)),
        ("adaptive_q4", lambda p: generate_mtp_adaptive(model, mtp_q4, p, MAX_TOKENS, eos_set)),
    ]

    for cat, prompt in PROMPTS.items():
        prompt_arr = mx.array(tokenizer.encode(prompt))
        print(f"\n--- {cat} ---")

        for name, fn in strategies:
            r = run_best_of_n(lambda p=prompt_arr, f=fn: f(p))
            tok_list, pfill, dec_ms, steps, tps_step, acc = r
            n_dec = max(len(tok_list) - 1, 1)
            tps = n_dec / (dec_ms / 1000) if dec_ms > 0 else 0
            ms_step = dec_ms / steps if steps > 0 else 0
            print(f"  {name:<22} {tps:>6.1f} t/s  {tps_step:.2f} tok/step  {acc:.0%} accept  {ms_step:.1f} ms/step")
            results.append(Result(name, cat, len(tok_list), pfill, dec_ms, tps,
                                  acc, tps_step, steps, ms_step))

    # Report
    print("\n\n" + "=" * 100)
    print("SUMMARY — Optimized MTP Results")
    print("=" * 100)

    strat_names = list(dict.fromkeys(r.strategy for r in results))
    categories = list(PROMPTS.keys())

    base_results = [r for r in results if r.strategy == "baseline"]
    avg_base = sum(r.tok_s for r in base_results) / len(base_results) if base_results else 0

    print(f"\n{'Strategy':<24} {'Avg t/s':>8} {'vs Base':>8} {'Accept':>8} {'Tok/Step':>9} {'ms/step':>8}")
    print("-" * 70)

    for strat in strat_names:
        sr = [r for r in results if r.strategy == strat]
        avg_tps = sum(r.tok_s for r in sr) / len(sr)
        avg_acc = sum(r.acceptance for r in sr) / len(sr)
        avg_tps_step = sum(r.tok_per_step for r in sr) / len(sr)
        avg_ms = sum(r.ms_per_step for r in sr) / len(sr)
        ratio = avg_tps / avg_base if avg_base > 0 else 0
        print(f"{strat:<24} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>7.0%} {avg_tps_step:>8.2f} {avg_ms:>7.1f}")

    # Per-category tok/s
    print(f"\nPER-CATEGORY t/s:")
    print(f"{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strat_names:
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].tok_s:>11.1f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Speedup vs baseline
    print(f"\nSPEEDUP vs BASELINE:")
    print(f"{'Strategy':<24}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (24 + 13 * len(categories)))

    for strat in strat_names:
        if strat == "baseline":
            continue
        print(f"{strat:<24}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            br = [r for r in results if r.strategy == "baseline" and r.category == cat]
            if sr and br:
                ratio = sr[0].tok_s / br[0].tok_s
                print(f" {ratio:>11.2f}x", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    out_file = "benchmark_mtp_optimized_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
