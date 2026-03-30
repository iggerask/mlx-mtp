#!/usr/bin/env python3
"""
Benchmark: Fused kernels + MTP speculative decoding (combined).

Tests configurations:
1. Baseline: Standard MLX inference
2. Fused gate+up (SwiGLU): SIMD-optimized gather_qmm_swiglu kernel
3. Fused full MoE: gate+up+SwiGLU + down_proj+reduce kernels
4. MTP K=1: Lazy-batch MTP speculative decoding (1 draft token)
5. MTP K=2: Lazy-batch MTP speculative decoding (2 draft tokens)
6. All K=1: Full fused MoE + MTP K=1
7. All K=2: Full fused MoE + MTP K=2
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Set

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder, MTPStats
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head
from mlx_fused_moe.patch import patch_model, unpatch_model
from mlx_fused_moe.patch_moe_full import patch_moe_full, unpatch_moe_full

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


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


# ---------------------------------------------------------------------------
# Baseline generation (no MTP, no fused kernel)
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
    return tokens, (t_prefill - t0) * 1000, (t_done - t_prefill) * 1000, len(tokens) - 1, 1.0, 0.0


# ---------------------------------------------------------------------------
# MTP Q4 lazy-batch generation
# ---------------------------------------------------------------------------

def generate_mtp(model, mtp_head, prompt_arr, max_tokens, eos_set, k=1,
                  cascade=False, adaptive=False, threshold=0.90, zero_replay=False):
    cfg = MTPConfig(
        num_speculative_tokens=k,
        batch_verify=True,
        lazy_draft=True,
        cascade_verify=cascade,
        adaptive_k=adaptive,
        adaptive_k_threshold=threshold,
        zero_replay=zero_replay,
    )
    dec = MTPDecoder(model, mtp_head, cfg)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    tokens = list(dec.generate(
        prompt_arr, cache, max_tokens=max_tokens, temperature=0.0, eos_tokens=eos_set,
    ))
    t_total = time.perf_counter() - t0
    prefill_ms = dec.stats.prefill_time * 1000
    decode_ms = (t_total - dec.stats.prefill_time) * 1000
    result = (
        tokens, prefill_ms, decode_ms,
        dec.stats.total_steps, dec.stats.tokens_per_step, dec.stats.acceptance_rate,
    )
    dec.cleanup()
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    strategy: str
    category: str
    tokens: int
    decode_ms: float
    tok_s: float
    acceptance: float
    tok_per_step: float
    steps: int


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
            best = (tok_list, pfill, dec_ms, steps, tps_step, acc, tps)
    return best


def main():
    print("=" * 80)
    print("Combined Benchmark: Fused SIMD Kernel + MTP Speculative Decoding")
    print(f"Max tokens: {MAX_TOKENS}, Best of {NUM_RUNS} runs per config")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    # Load MTP head (Q4)
    print("Loading MTP head (Q4)...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    weights = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_head = build_mtp_head(weights, config, norm_shift=True)
    quantize_mtp_head(mtp_head, bits=4, group_size=64)
    q4_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_head.parameters()))
    print(f"  MTP Q4 head: {q4_bytes / 1e6:.1f} MB")

    # Warmup
    print("Warming up...")
    warmup = mx.array(tokenizer.encode("Hello world"))
    for _ in range(3):
        generate_baseline(model, warmup, 20, eos_set)
    print("Warmup done.\n")

    results = []

    # (name, moe_mode, mtp_kwargs)
    # moe_mode: None, "full"
    # mtp_kwargs: None or dict passed to generate_mtp
    configs = [
        ("baseline",       None,   None),
        ("all_k1",         "full", dict(k=1)),
        ("all_zr_k1",      "full", dict(k=1, zero_replay=True)),
        ("all_zr_k2",      "full", dict(k=2, zero_replay=True)),
        ("all_zr_k3",      "full", dict(k=3, zero_replay=True)),
    ]

    current_moe = None  # tracks active MoE patch

    for cat, prompt in PROMPTS.items():
        prompt_arr = mx.array(tokenizer.encode(prompt))
        print(f"--- {cat} ---")

        for name, moe_mode, mtp_kwargs in configs:
            # Patch/unpatch MoE kernels
            if moe_mode != current_moe:
                if current_moe == "full":
                    unpatch_moe_full(model)
                if moe_mode == "full":
                    patch_moe_full(model, verbose=False)
                current_moe = moe_mode

            if mtp_kwargs is not None:
                fn = lambda p=prompt_arr, kw=mtp_kwargs: generate_mtp(
                    model, mtp_head, p, MAX_TOKENS, eos_set, **kw
                )
            else:
                fn = lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set)

            r = run_best_of_n(fn)
            tok_list, pfill, dec_ms, steps, tps_step, acc, tps = r
            print(f"  {name:<16} {tps:>6.1f} t/s  {tps_step:.2f} tok/step  {acc:.0%} accept  ({len(tok_list)} tokens)")
            results.append(Result(name, cat, len(tok_list), dec_ms, tps, acc, tps_step, steps))

        # Unpatch after each category to start clean
        if current_moe == "full":
            unpatch_moe_full(model)
        current_moe = None
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    strat_names = [c[0] for c in configs]
    categories = list(PROMPTS.keys())

    # Average t/s across categories
    base_results = [r for r in results if r.strategy == "baseline"]
    avg_base = sum(r.tok_s for r in base_results) / len(base_results) if base_results else 0

    print(f"\n{'Strategy':<18} {'Avg t/s':>8} {'vs Base':>8} {'Avg Accept':>11} {'Avg Tok/Step':>13}")
    print("-" * 60)

    for strat in strat_names:
        sr = [r for r in results if r.strategy == strat]
        avg_tps = sum(r.tok_s for r in sr) / len(sr)
        avg_acc = sum(r.acceptance for r in sr) / len(sr)
        avg_tps_step = sum(r.tok_per_step for r in sr) / len(sr)
        ratio = avg_tps / avg_base if avg_base > 0 else 0
        print(f"{strat:<18} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>10.0%} {avg_tps_step:>12.2f}")

    # Per-category breakdown
    print(f"\nPER-CATEGORY t/s:")
    print(f"{'Strategy':<18}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print()
    print("-" * (18 + 13 * len(categories)))

    for strat in strat_names:
        print(f"{strat:<18}", end="")
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            if sr:
                print(f" {sr[0].tok_s:>11.1f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()

    # Speedup table
    print(f"\nSPEEDUP vs BASELINE:")
    print(f"{'Strategy':<18}", end="")
    for cat in categories:
        print(f" {cat:>12}", end="")
    print(f" {'average':>12}")
    print("-" * (18 + 13 * (len(categories) + 1)))

    for strat in strat_names:
        if strat == "baseline":
            continue
        print(f"{strat:<18}", end="")
        ratios = []
        for cat in categories:
            sr = [r for r in results if r.strategy == strat and r.category == cat]
            br = [r for r in results if r.strategy == "baseline" and r.category == cat]
            if sr and br and br[0].tok_s > 0:
                ratio = sr[0].tok_s / br[0].tok_s
                ratios.append(ratio)
                print(f" {ratio:>11.2f}x", end="")
            else:
                print(f" {'n/a':>12}", end="")
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        print(f" {avg_ratio:>11.2f}x")

    print()


if __name__ == "__main__":
    main()
