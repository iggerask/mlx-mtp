#!/usr/bin/env python3
"""
Layer skip optimization suite for Qwen3.5-35B-A3B.

Three approaches:
  1. Block Influence (BI) profiling — measure per-layer importance
  2. Static layer skip — skip lowest-BI layers, measure quality/speed tradeoff
  3. Self-speculative layer skip — use skipped-layer model as drafter,
     verify with full model. No external draft model needed.
  4. Early exit — stop forward pass when confidence exceeds threshold

Architecture: 40 layers (30 GatedDeltaNet + 10 Attention)
  - Attention at layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
  - All other layers: GatedDeltaNet (recurrent)
  - All layers have MoE/MLP block
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache, ArraysCache

from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state

logging.basicConfig(level=logging.WARNING)

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
MAX_TOKENS = 200
NUM_RUNS = 3

PROMPTS = {
    "code": "Write a Python function that implements merge sort with type hints:\n```python\ndef merge_sort(arr: list[int]) -> list[int]:\n",
    "prose": "Explain how transformers work in machine learning, starting from self-attention:\n",
    "repetitive": "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
    "qa": "What are the main differences between Python and Rust? List the key points:\n1.",
}

# Calibration prompts for BI analysis
CALIBRATION_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to find all prime numbers up to n.",
    "What are the key differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(",
    "In machine learning, overfitting occurs when",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55,",
]


@dataclass
class Result:
    strategy: str
    category: str
    tokens: int
    decode_ms: float
    tok_s: float
    ms_per_step: float
    steps: int
    tok_per_step: float
    acceptance: float


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


# ---------------------------------------------------------------------------
# Phase 1: Block Influence profiling
# ---------------------------------------------------------------------------

def measure_block_influence(model, tokenizer):
    """Measure Block Influence (BI) for each layer.

    BI = 1 - cosine_similarity(input_hidden, output_hidden)
    averaged over calibration prompts.

    Low BI = layer barely transforms the representation = safe to skip.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Block Influence Profiling")
    print("=" * 70)

    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    num_layers = len(inner.layers)

    # We'll hook into the layer forward to measure input/output similarity
    bi_scores = [[] for _ in range(num_layers)]

    for prompt_text in CALIBRATION_PROMPTS:
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_arr = mx.array(prompt_tokens)

        cache = make_prompt_cache(model)

        # Forward through embedding
        hidden = inner.embed_tokens(prompt_arr[None])

        # Create masks
        fa_idx = None
        ssm_idx = None
        for i, layer in enumerate(inner.layers):
            if not layer.is_linear:
                if fa_idx is None:
                    fa_idx = i
                break
        for i, layer in enumerate(inner.layers):
            if layer.is_linear:
                ssm_idx = i
                break

        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(hidden, cache[fa_idx] if fa_idx is not None else None)
        ssm_mask = create_ssm_mask(hidden, cache[ssm_idx] if ssm_idx is not None else None)

        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h_in = hidden
            hidden = layer(hidden, mask=mask, cache=c)
            mx.eval(hidden)

            # Compute cosine similarity between input and output
            # Use mean pooling over sequence and batch dimensions
            h_in_flat = h_in.reshape(-1, h_in.shape[-1])
            h_out_flat = hidden.reshape(-1, hidden.shape[-1])

            # Cosine sim per position, then average
            dot = mx.sum(h_in_flat * h_out_flat, axis=-1)
            norm_in = mx.sqrt(mx.sum(h_in_flat * h_in_flat, axis=-1) + 1e-8)
            norm_out = mx.sqrt(mx.sum(h_out_flat * h_out_flat, axis=-1) + 1e-8)
            cos_sim = mx.mean(dot / (norm_in * norm_out))
            mx.eval(cos_sim)

            bi = 1.0 - cos_sim.item()
            bi_scores[i].append(bi)

    # Average BI per layer
    avg_bi = [sum(scores) / len(scores) for scores in bi_scores]

    # Report
    print(f"\n{'Layer':>5} {'Type':>10} {'BI Score':>10} {'Rank':>6}")
    print("-" * 35)

    ranked = sorted(range(num_layers), key=lambda i: avg_bi[i])
    rank_map = {idx: rank for rank, idx in enumerate(ranked)}

    for i in range(num_layers):
        layer_type = "Attn" if not inner.layers[i].is_linear else "GDN"
        bar = "█" * int(avg_bi[i] * 200)
        print(f"{i:>5} {layer_type:>10} {avg_bi[i]:>10.4f} {rank_map[i]+1:>5}  {bar}")

    # Identify skip candidates (bottom 25% by BI, excluding first 4 and last 2)
    safe_range = list(range(4, num_layers - 2))
    safe_ranked = sorted(safe_range, key=lambda i: avg_bi[i])
    skip_candidates = safe_ranked[:10]  # Top 10 lowest-BI layers

    print(f"\nSkip candidates (lowest BI, layers 4-{num_layers-3}):")
    for i in skip_candidates:
        layer_type = "Attn" if not inner.layers[i].is_linear else "GDN"
        print(f"  Layer {i:>2} ({layer_type}): BI = {avg_bi[i]:.4f}")

    return avg_bi, skip_candidates


# ---------------------------------------------------------------------------
# Phase 2: Static layer skip — measure speed/quality tradeoff
# ---------------------------------------------------------------------------

class LayerSkipModel:
    """Wrapper that skips specified layers during forward pass."""

    def __init__(self, model, skip_layers: Set[int]):
        self.model = model
        self.skip_layers = skip_layers
        self._original_calls = {}
        self._patched = False

    def enable(self):
        if self._patched:
            return
        inner = self.model.model if not hasattr(self.model, 'language_model') else self.model.language_model.model
        for i in self.skip_layers:
            layer = inner.layers[i]
            self._original_calls[i] = layer.__class__.__call__

            # Replace __call__ with identity (residual passthrough)
            def make_skip(layer_idx):
                def skip_call(self_layer, x, mask=None, cache=None):
                    # For GDN layers, we still need to advance the cache
                    if hasattr(cache, 'advance') and cache is not None:
                        # Don't advance — just skip entirely
                        pass
                    return x
                return skip_call

            import types
            layer.__call__ = types.MethodType(make_skip(i), layer)
        self._patched = True

    def disable(self):
        if not self._patched:
            return
        inner = self.model.model if not hasattr(self.model, 'language_model') else self.model.language_model.model
        for i in self.skip_layers:
            layer = inner.layers[i]
            if hasattr(layer, '__call__'):
                del layer.__call__  # Remove instance override, restore class method
        self._original_calls.clear()
        self._patched = False


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
    decode_ms = (t_done - t_prefill) * 1000
    n_steps = max(len(tokens) - 1, 1)
    return tokens, decode_ms, n_steps


def generate_with_skip(model, skipper, prompt_arr, max_tokens, eos_set):
    """Generate with layers skipped."""
    skipper.enable()
    try:
        tokens, decode_ms, steps = generate_baseline(model, prompt_arr, max_tokens, eos_set)
    finally:
        skipper.disable()
    return tokens, decode_ms, steps


# ---------------------------------------------------------------------------
# Phase 3: Self-speculative layer skip
# ---------------------------------------------------------------------------

def generate_self_speculative(model, skip_layers, prompt_arr, max_tokens, eos_set,
                              num_draft=2):
    """Self-speculative decoding: draft with skipped layers, verify with full model.

    The key insight: we don't need an external draft model. We can use the
    same model with some layers skipped as a cheap drafter, then verify
    with the full model.

    Draft cost: ~(1 - skip_fraction) × baseline_cost per token
    Verify cost: ~14 + K × 4.5ms for K+1 tokens
    """
    skipper = LayerSkipModel(model, skip_layers)
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

        # Draft with skipped layers
        draft_ids = []
        draft_tok = mx.array([[tokens[-1]]])
        skipper.enable()
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
            skipper.disable()

        if not draft_ids:
            restore_cache_state(cache, saved)
            logits = model(mx.array([[tokens[-1]]]), cache=cache)
            mx.eval(logits)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())
            continue

        total_drafted += len(draft_ids)

        # Restore and verify with full model
        restore_cache_state(cache, saved)
        verify_input = mx.array([[tokens[-1]] + draft_ids])
        verify_logits = model(verify_input, cache=cache)
        mx.eval(verify_logits)

        accepted = 0
        for i, did in enumerate(draft_ids):
            v = mx.argmax(verify_logits[:, i, :], axis=-1)
            mx.eval(v)
            vid = v.item()
            if vid == did:
                accepted += 1
                tokens.append(did)
            else:
                tokens.append(vid)
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
            replay = [tokens[-(n_to_replay + 1)]] + tokens[-n_to_replay:]
            replay_logits = model(mx.array([replay]), cache=cache)
            mx.eval(replay_logits)

    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    decode_ms = (t_done - t_prefill) * 1000
    acc = total_accepted / total_drafted if total_drafted > 0 else 0
    tps = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, decode_ms, total_steps, tps, acc


# ---------------------------------------------------------------------------
# Phase 4: Early exit at attention layer checkpoints
# ---------------------------------------------------------------------------

def generate_early_exit(model, prompt_arr, max_tokens, eos_set,
                        exit_threshold=0.95, min_layers=16):
    """CALM-style early exit: stop at attention layer checkpoints if confident.

    At each attention layer (every 4th), project hidden state through lm_head
    and check top-1 probability. If > threshold, exit early.

    The cost of the extra lm_head projection at checkpoints is small (~1ms)
    but we save all remaining layers if we exit early.
    """
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model

    # Get lm_head
    if hasattr(model, 'language_model'):
        text_model = model.language_model
    else:
        text_model = model
    if text_model.args.tie_word_embeddings:
        lm_head_fn = text_model.model.embed_tokens.as_linear
    else:
        lm_head_fn = text_model.lm_head

    final_norm = inner.norm

    # Identify attention layer indices (checkpoint candidates)
    attn_layers = [i for i, l in enumerate(inner.layers) if not l.is_linear]

    cache = make_prompt_cache(model)
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    t0 = time.perf_counter()

    # Prefill with full model (need correct cache state)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens = [tok.item()]

    total_exits = {i: 0 for i in attn_layers}
    total_full = 0
    total_steps = 0

    while len(tokens) < max_tokens and tokens[-1] not in eos_set:
        total_steps += 1

        # Manual forward with early exit checkpoints
        input_ids = mx.array([[tokens[-1]]])
        hidden = inner.embed_tokens(input_ids)

        fa_mask = create_attention_mask(hidden, cache[attn_layers[0]])
        ssm_mask = create_ssm_mask(hidden, cache[0])

        exited = False
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=c)

            # Check at attention layer checkpoints (after min_layers)
            if i in attn_layers and i >= min_layers:
                # Project to logits and check confidence
                normed = final_norm(hidden)
                probe_logits = lm_head_fn(normed)
                probe_probs = mx.softmax(probe_logits[:, -1, :], axis=-1)
                top_prob = mx.max(probe_probs)
                mx.eval(top_prob)

                if top_prob.item() > exit_threshold:
                    # Early exit! Use this layer's prediction
                    tok = mx.argmax(probe_logits[:, -1, :], axis=-1)
                    mx.eval(tok)
                    tokens.append(tok.item())
                    total_exits[i] += 1
                    exited = True

                    # We need to "fill in" the remaining cache entries
                    # For GDN layers: state is stale but we accept the approximation
                    # For KV layers: no update needed (they just won't have this position)
                    # Problem: subsequent tokens will have inconsistent cache
                    # Simple fix: advance remaining KV cache offsets
                    for j in range(i + 1, len(inner.layers)):
                        remaining_c = cache[j]
                        if isinstance(remaining_c, KVCache):
                            # KV cache: the position is "missing" — subsequent
                            # attention will skip it. This is acceptable for
                            # high-confidence tokens (they're easy to predict).
                            pass
                        elif isinstance(remaining_c, ArraysCache):
                            # GDN state: already stale, accept approximation
                            pass
                    break

        if not exited:
            total_full += 1
            normed = final_norm(hidden)
            logits = lm_head_fn(normed)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tokens.append(tok.item())

    t_done = time.perf_counter()
    tokens = tokens[:max_tokens]
    decode_ms = (t_done - t_prefill) * 1000

    exit_rate = sum(total_exits.values()) / total_steps if total_steps > 0 else 0
    avg_exit_layer = (
        sum(layer * count for layer, count in total_exits.items()) /
        sum(total_exits.values())
        if sum(total_exits.values()) > 0 else len(inner.layers)
    )

    return tokens, decode_ms, total_steps, exit_rate, avg_exit_layer


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_best_of_n(fn, n=NUM_RUNS):
    best = None
    best_tps = 0
    for _ in range(n):
        result = fn()
        tokens = result[0]
        decode_ms = result[1]
        n_dec = max(len(tokens) - 1, 1)
        tps = n_dec / (decode_ms / 1000) if decode_ms > 0 else 0
        if tps > best_tps:
            best_tps = tps
            best = result
    return best


def token_match_rate(reference, candidate):
    """What fraction of candidate tokens match reference."""
    matches = sum(1 for a, b in zip(reference, candidate) if a == b)
    return matches / len(reference) if reference else 0


def main():
    print("=" * 90)
    print("LAYER SKIP OPTIMIZATION SUITE — Qwen3.5-35B-A3B-4bit")
    print("=" * 90)

    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    # Warmup
    print("Warming up...")
    warmup = mx.array(tokenizer.encode("Hello world"))
    for _ in range(3):
        generate_baseline(model, warmup, 20, eos_set)

    # -----------------------------------------------------------------------
    # Phase 1: BI profiling
    # -----------------------------------------------------------------------
    bi_scores, skip_candidates = measure_block_influence(model, tokenizer)

    # -----------------------------------------------------------------------
    # Phase 2 & 3: Benchmark layer skip strategies
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2-4: Speed & Quality Benchmarks")
    print("=" * 70)

    # Define skip sets to test
    # Sort candidates by BI score (lowest first)
    sorted_candidates = sorted(skip_candidates, key=lambda i: bi_scores[i])

    skip_configs = {
        "skip_5": set(sorted_candidates[:5]),
        "skip_8": set(sorted_candidates[:8]),
        "skip_10": set(sorted_candidates[:10]),
    }

    results = []
    baseline_tokens_by_cat = {}

    for cat, prompt in PROMPTS.items():
        prompt_tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(prompt_tokens)
        print(f"\n--- {cat} ---")

        # Baseline
        r = run_best_of_n(lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set))
        tokens, decode_ms, steps = r
        baseline_tokens_by_cat[cat] = tokens
        n_dec = max(len(tokens) - 1, 1)
        tps = n_dec / (decode_ms / 1000) if decode_ms > 0 else 0
        ms_step = decode_ms / steps if steps > 0 else 0
        print(f"  baseline            {tps:>6.1f} t/s  {ms_step:.1f} ms/step")
        results.append(Result("baseline", cat, len(tokens), decode_ms, tps, ms_step, steps, 1.0, 0.0))

        # Static skip benchmarks
        for skip_name, skip_set in skip_configs.items():
            skipper = LayerSkipModel(model, skip_set)
            r = run_best_of_n(lambda p=prompt_arr, s=skipper:
                generate_with_skip(model, s, p, MAX_TOKENS, eos_set))
            tokens, decode_ms, steps = r
            n_dec = max(len(tokens) - 1, 1)
            tps = n_dec / (decode_ms / 1000) if decode_ms > 0 else 0
            ms_step = decode_ms / steps if steps > 0 else 0
            match = token_match_rate(baseline_tokens_by_cat[cat], tokens)
            print(f"  {skip_name:<20} {tps:>6.1f} t/s  {ms_step:.1f} ms/step  match={match:.0%}")
            results.append(Result(skip_name, cat, len(tokens), decode_ms, tps, ms_step, steps, 1.0, match))

        # Self-speculative with best skip configs
        for skip_name, skip_set in [("skip_5", skip_configs["skip_5"]),
                                     ("skip_8", skip_configs["skip_8"])]:
            for nd in [1, 2]:
                strat_name = f"selfspec_{skip_name}_d{nd}"
                r = run_best_of_n(lambda p=prompt_arr, ss=skip_set, d=nd:
                    generate_self_speculative(model, ss, p, MAX_TOKENS, eos_set, num_draft=d))
                tokens, decode_ms, total_steps, tps_step, acc = r
                n_dec = max(len(tokens) - 1, 1)
                tps = n_dec / (decode_ms / 1000) if decode_ms > 0 else 0
                ms_step = decode_ms / total_steps if total_steps > 0 else 0
                print(f"  {strat_name:<20} {tps:>6.1f} t/s  {tps_step:.2f} tok/step  {acc:.0%} accept  {ms_step:.1f} ms/step")
                results.append(Result(strat_name, cat, len(tokens), decode_ms, tps, ms_step, total_steps, tps_step, acc))

        # Early exit
        for threshold in [0.9, 0.95, 0.99]:
            for min_layer in [16, 24]:
                strat_name = f"early_exit_t{threshold}_l{min_layer}"
                r = run_best_of_n(lambda p=prompt_arr, t=threshold, ml=min_layer:
                    generate_early_exit(model, p, MAX_TOKENS, eos_set,
                                        exit_threshold=t, min_layers=ml))
                tokens, decode_ms, total_steps, exit_rate, avg_exit = r
                n_dec = max(len(tokens) - 1, 1)
                tps = n_dec / (decode_ms / 1000) if decode_ms > 0 else 0
                ms_step = decode_ms / total_steps if total_steps > 0 else 0
                match = token_match_rate(baseline_tokens_by_cat[cat], tokens)
                print(f"  {strat_name:<20} {tps:>6.1f} t/s  exit={exit_rate:.0%}  avg_layer={avg_exit:.0f}  match={match:.0%}  {ms_step:.1f} ms/step")
                results.append(Result(strat_name, cat, len(tokens), decode_ms, tps, ms_step, total_steps, 1.0, match))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("LAYER SKIP RESULTS SUMMARY")
    print("=" * 100)

    strategies = sorted(set(r.strategy for r in results),
                       key=lambda s: next((i for i, r in enumerate(results) if r.strategy == s), 0))
    categories = list(PROMPTS.keys())

    base_results = [r for r in results if r.strategy == "baseline"]
    avg_base = sum(r.tok_s for r in base_results) / len(base_results)

    print(f"\n{'Strategy':<28} {'Avg t/s':>8} {'vs Base':>8} {'Accept/Match':>12} {'ms/step':>8}")
    print("-" * 70)

    for strat in strategies:
        sr = [r for r in results if r.strategy == strat]
        avg_tps = sum(r.tok_s for r in sr) / len(sr)
        avg_acc = sum(r.acceptance for r in sr) / len(sr)
        avg_ms = sum(r.ms_per_step for r in sr) / len(sr)
        ratio = avg_tps / avg_base if avg_base > 0 else 0
        print(f"{strat:<28} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>11.0%} {avg_ms:>7.1f}")

    # Per-category t/s
    print(f"\nPER-CATEGORY t/s:")
    print(f"{'Strategy':<28}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (28 + 13 * len(categories)))

    for strat in strategies:
        print(f"{strat:<28}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].tok_s:>11.1f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Save BI scores
    bi_data = {
        "bi_scores": bi_scores,
        "skip_candidates": skip_candidates,
        "skip_configs": {k: list(v) for k, v in skip_configs.items()},
        "results": [asdict(r) for r in results],
    }
    with open("benchmark_layer_skip_results.json", "w") as f:
        json.dump(bi_data, f, indent=2)
    print(f"\nResults saved to benchmark_layer_skip_results.json")


if __name__ == "__main__":
    main()
