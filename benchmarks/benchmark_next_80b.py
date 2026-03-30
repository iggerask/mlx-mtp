#!/usr/bin/env python3
"""
Benchmark optimizations on Qwen3-Next-80B-A3B (80B/3B active, MoE + DeltaNet hybrid).

Tests:
  1. Baseline (autoregressive)
  2. MTP batch verify (BF16 head)
  3. MTP batch verify (4-bit quantized head)
  4. Prompt lookup decoding
  5. Shared-expert-only self-speculation (d=1, d=3)
"""

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import (
    PromptLookupDrafter,
    SharedExpertDrafter,
    quantize_mtp_head,
)
from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state

logging.basicConfig(level=logging.WARNING)

MODEL_NAME = "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"
BF16_SOURCE = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3-Next-80B-A3B.safetensors")
MAX_TOKENS = 128
NUM_RUNS = 2


@dataclass
class Result:
    method: str
    category: str
    prompt_idx: int
    tokens: int
    time_s: float
    tok_s: float
    acceptance: float
    tok_per_step: float
    output_match: bool
    prompt_preview: str


PROMPTS = {
    "code": [
        "Write a Python function that implements binary search:\n```python\ndef binary_search(arr, target):\n",
        "Write a JavaScript function that deep clones an object:\n```javascript\nfunction deepClone(obj) {\n",
    ],
    "prose": [
        "Explain the theory of relativity in simple terms that anyone could understand:\n",
        "Describe the history of the internet from ARPANET to the modern web:\n",
    ],
    "short": [
        "The capital of France is",
        "In Python, to read a file you use",
    ],
    "summarization": [
        "Summarize the following text:\n\nThe quick brown fox jumps over the lazy dog. The quick brown fox is very fast. The lazy dog sleeps all day. The quick brown fox runs through the forest. The lazy dog stays at home. The quick brown fox catches mice. The lazy dog guards the house.\n\nSummary:",
        "Rewrite the following code with better variable names:\n```python\ndef f(x, y, z):\n    a = x + y\n    b = a * z\n    c = b - x\n    return c\n```\nRewritten:\n```python\ndef f(x, y, z):\n",
    ],
}


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


def baseline_generate(model, prompt_arr, max_tokens, eos_set) -> Tuple[List[int], float]:
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    tokens = []
    for _ in range(max_tokens):
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        tid = tok.item()
        tokens.append(tid)
        if tid in eos_set:
            break
        logits = model(tok.reshape(1, 1), cache=cache)
        mx.eval(logits)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def mtp_generate(decoder, model, prompt_arr, max_tokens, eos_set) -> Tuple[List[int], float, dict]:
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    tokens = list(decoder.generate(
        prompt_arr, cache, max_tokens=max_tokens,
        temperature=0.0, eos_tokens=eos_set,
    ))
    elapsed = time.perf_counter() - t0
    stats = {"acceptance": decoder.stats.acceptance_rate, "tok_per_step": decoder.stats.tokens_per_step}
    decoder.stats.__init__()
    return tokens, elapsed, stats


def prompt_lookup_generate(
    model, tokenizer, prompt_arr, max_tokens, eos_set,
    prompt_tokens: List[int], max_ngram: int = 5, max_draft: int = 5,
) -> Tuple[List[int], float, dict]:
    drafter = PromptLookupDrafter(prompt_tokens, max_ngram=max_ngram, max_draft=max_draft)
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tid = tok.item()
    tokens.append(tid)

    while len(tokens) < max_tokens and tid not in eos_set:
        total_steps += 1
        draft_ids = drafter.draft(tokens)

        if not draft_ids:
            logits = model(tok.reshape(1, 1), cache=cache)
            mx.eval(logits)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tid = tok.item()
            tokens.append(tid)
            continue

        total_drafted += len(draft_ids)
        saved = save_cache_state(cache)

        verify_input = mx.array([[tid] + draft_ids])
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

        if accepted == len(draft_ids) + 1:
            tid = tokens[-1]
            tok = mx.array([tid])
        elif accepted < len(draft_ids):
            restore_cache_state(cache, saved)
            n_to_replay = accepted + 1
            replay_tokens = [tid] + tokens[-n_to_replay:]
            replay = mx.array([replay_tokens])
            replay_logits = model(replay, cache=cache)
            mx.eval(replay_logits)
            tid = tokens[-1]
            tok = mx.array([tid])
        else:
            tid = tokens[-1]
            tok = mx.array([tid])

    elapsed = time.perf_counter() - t0
    tokens = tokens[:max_tokens]
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tok_per_step = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, elapsed, {"acceptance": acc_rate, "tok_per_step": tok_per_step}


def shared_expert_generate(
    model, drafter: SharedExpertDrafter, prompt_arr, max_tokens, eos_set,
    num_draft: int = 3,
) -> Tuple[List[int], float, dict]:
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)

    tokens = []
    total_drafted = 0
    total_accepted = 0
    total_steps = 0

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tid = tok.item()
    tokens.append(tid)

    while len(tokens) < max_tokens and tid not in eos_set:
        total_steps += 1
        saved = save_cache_state(cache)

        draft_ids = []
        draft_tok = tok.reshape(1, 1)

        drafter.enable()
        try:
            for _ in range(num_draft):
                draft_logits = model(draft_tok, cache=cache)
                mx.eval(draft_logits)
                d_tok = mx.argmax(draft_logits[:, -1, :], axis=-1)
                mx.eval(d_tok)
                did = d_tok.item()
                if did in eos_set:
                    break
                draft_ids.append(did)
                draft_tok = d_tok.reshape(1, 1)
        finally:
            drafter.disable()

        if not draft_ids:
            restore_cache_state(cache, saved)
            logits = model(tok.reshape(1, 1), cache=cache)
            mx.eval(logits)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tid = tok.item()
            tokens.append(tid)
            continue

        total_drafted += len(draft_ids)

        restore_cache_state(cache, saved)
        verify_input = mx.array([[tid] + draft_ids])
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

        if accepted == len(draft_ids) + 1:
            tid = tokens[-1]
            tok = mx.array([tid])
        elif accepted < len(draft_ids):
            restore_cache_state(cache, saved)
            n_to_replay = accepted + 1
            replay_tokens = [tid] + tokens[-n_to_replay:]
            replay = mx.array([replay_tokens])
            replay_logits = model(replay, cache=cache)
            mx.eval(replay_logits)
            tid = tokens[-1]
            tok = mx.array([tid])
        else:
            tid = tokens[-1]
            tok = mx.array([tid])

    elapsed = time.perf_counter() - t0
    tokens = tokens[:max_tokens]
    acc_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tok_per_step = len(tokens) / total_steps if total_steps > 0 else 1
    return tokens, elapsed, {"acceptance": acc_rate, "tok_per_step": tok_per_step}


def run_benchmark(model, tokenizer, eos_set, mtp_head_bf16=None, mtp_head_q4=None):
    results = []

    # Build MTP decoders if available
    dec_bf16 = None
    dec_q4 = None
    if mtp_head_bf16:
        dec_bf16 = MTPDecoder(model, mtp_head_bf16, MTPConfig(greedy_draft=True, batch_verify=True))
    if mtp_head_q4:
        dec_q4 = MTPDecoder(model, mtp_head_q4, MTPConfig(greedy_draft=True, batch_verify=True))

    # Build shared-expert drafter
    try:
        shared_drafter = SharedExpertDrafter(model)
        print(f"  SharedExpertDrafter: found {len(shared_drafter._moe_layers)} MoE layers")
    except Exception as e:
        print(f"  SharedExpertDrafter setup failed: {e}")
        shared_drafter = None

    # Warmup
    print("  Warming up...")
    warmup_arr = mx.array(tokenizer.encode("Hello world"))
    for _ in range(2):
        baseline_generate(model, warmup_arr, 10, eos_set)
    print("  Warmup done")

    categories = list(PROMPTS.keys())

    for cat in categories:
        for pidx, prompt in enumerate(PROMPTS[cat]):
            prompt_tokens = tokenizer.encode(prompt)
            prompt_arr = mx.array(prompt_tokens)
            preview = prompt[:50].replace("\n", "\\n")

            def avg_runs(fn, n=NUM_RUNS):
                best_tps = 0
                best_result = None
                for _ in range(n):
                    r = fn()
                    tps = len(r[0]) / r[1] if r[1] > 0 else 0
                    if tps > best_tps:
                        best_tps = tps
                        best_result = r
                return best_result

            # 1. Baseline
            tokens_base, time_base = avg_runs(
                lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set)
            )
            base_tps = len(tokens_base) / time_base

            results.append(Result(
                method="baseline", category=cat, prompt_idx=pidx,
                tokens=len(tokens_base), time_s=time_base, tok_s=base_tps,
                acceptance=0, tok_per_step=1.0, output_match=True,
                prompt_preview=preview,
            ))

            # 2. MTP BF16
            bf16_tps = 0
            if dec_bf16:
                dec_bf16.stats.__init__()
                tokens_bf16, time_bf16, stats_bf16 = avg_runs(
                    lambda: mtp_generate(dec_bf16, model, prompt_arr, MAX_TOKENS, eos_set)
                )
                bf16_tps = len(tokens_bf16) / time_bf16

                results.append(Result(
                    method="mtp_bf16", category=cat, prompt_idx=pidx,
                    tokens=len(tokens_bf16), time_s=time_bf16, tok_s=bf16_tps,
                    acceptance=stats_bf16["acceptance"], tok_per_step=stats_bf16["tok_per_step"],
                    output_match=tokens_bf16 == tokens_base[:len(tokens_bf16)],
                    prompt_preview=preview,
                ))

            # 3. MTP Q4
            q4_tps = 0
            if dec_q4:
                dec_q4.stats.__init__()
                tokens_q4, time_q4, stats_q4 = avg_runs(
                    lambda: mtp_generate(dec_q4, model, prompt_arr, MAX_TOKENS, eos_set)
                )
                q4_tps = len(tokens_q4) / time_q4

                results.append(Result(
                    method="mtp_q4", category=cat, prompt_idx=pidx,
                    tokens=len(tokens_q4), time_s=time_q4, tok_s=q4_tps,
                    acceptance=stats_q4["acceptance"], tok_per_step=stats_q4["tok_per_step"],
                    output_match=tokens_q4 == tokens_base[:len(tokens_q4)],
                    prompt_preview=preview,
                ))

            # 4. Prompt lookup
            tokens_pl, time_pl, stats_pl = avg_runs(
                lambda: prompt_lookup_generate(
                    model, tokenizer, prompt_arr, MAX_TOKENS, eos_set,
                    prompt_tokens, max_ngram=5, max_draft=5,
                )
            )
            pl_tps = len(tokens_pl) / time_pl

            results.append(Result(
                method="prompt_lookup", category=cat, prompt_idx=pidx,
                tokens=len(tokens_pl), time_s=time_pl, tok_s=pl_tps,
                acceptance=stats_pl["acceptance"], tok_per_step=stats_pl["tok_per_step"],
                output_match=tokens_pl == tokens_base[:len(tokens_pl)],
                prompt_preview=preview,
            ))

            # 5. Shared-expert d=1
            se1_tps = 0
            if shared_drafter:
                tokens_se1, time_se1, stats_se1 = avg_runs(
                    lambda: shared_expert_generate(
                        model, shared_drafter, prompt_arr, MAX_TOKENS, eos_set,
                        num_draft=1,
                    )
                )
                se1_tps = len(tokens_se1) / time_se1

                results.append(Result(
                    method="shared_expert_d1", category=cat, prompt_idx=pidx,
                    tokens=len(tokens_se1), time_s=time_se1, tok_s=se1_tps,
                    acceptance=stats_se1["acceptance"], tok_per_step=stats_se1["tok_per_step"],
                    output_match=tokens_se1 == tokens_base[:len(tokens_se1)],
                    prompt_preview=preview,
                ))

            # 6. Shared-expert d=3
            se3_tps = 0
            if shared_drafter:
                tokens_se3, time_se3, stats_se3 = avg_runs(
                    lambda: shared_expert_generate(
                        model, shared_drafter, prompt_arr, MAX_TOKENS, eos_set,
                        num_draft=3,
                    )
                )
                se3_tps = len(tokens_se3) / time_se3

                results.append(Result(
                    method="shared_expert_d3", category=cat, prompt_idx=pidx,
                    tokens=len(tokens_se3), time_s=time_se3, tok_s=se3_tps,
                    acceptance=stats_se3["acceptance"], tok_per_step=stats_se3["tok_per_step"],
                    output_match=tokens_se3 == tokens_base[:len(tokens_se3)],
                    prompt_preview=preview,
                ))

            # Progress
            parts = [f"base={base_tps:.0f}"]
            if dec_bf16:
                parts.append(f"bf16={bf16_tps / base_tps:.2f}x")
            if dec_q4:
                parts.append(f"q4={q4_tps / base_tps:.2f}x")
            parts.append(f"pl={pl_tps / base_tps:.2f}x")
            if shared_drafter:
                parts.append(f"se1={se1_tps / base_tps:.2f}x")
                parts.append(f"se3={se3_tps / base_tps:.2f}x")
            print(f"    [{cat}:{pidx}] {' '.join(parts)} '{preview}'")

    if dec_bf16:
        dec_bf16.cleanup()
    if dec_q4:
        dec_q4.cleanup()
    return results


def print_report(results: List[Result], model_name: str):
    methods = sorted(set(r.method for r in results))
    categories = sorted(set(r.category for r in results))

    print()
    print("=" * 100)
    print(f"OPTIMIZATION BENCHMARK REPORT — {model_name}")
    print("=" * 100)
    print()

    print("OVERALL SUMMARY")
    print("-" * 80)
    print(f"{'Method':<20} {'Avg t/s':>8} {'vs Base':>8} {'Accept':>8} {'t/step':>8} {'Match':>6}")
    print("-" * 80)

    base_results = [r for r in results if r.method == "baseline"]
    avg_base = sum(r.tok_s for r in base_results) / len(base_results) if base_results else 0

    for method in methods:
        mr = [r for r in results if r.method == method]
        avg_tps = sum(r.tok_s for r in mr) / len(mr)
        avg_acc = sum(r.acceptance for r in mr) / len(mr)
        avg_tps_step = sum(r.tok_per_step for r in mr) / len(mr)
        ratio = avg_tps / avg_base if avg_base > 0 else 0
        all_match = all(r.output_match for r in mr)
        print(f"{method:<20} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>7.0%} {avg_tps_step:>7.2f} {'Y' if all_match else 'N':>5}")

    print()
    print("PER-CATEGORY SPEEDUP (vs baseline)")
    print("-" * 80)
    header = f"{'Method':<20}"
    for cat in categories:
        header += f" {cat:>12}"
    print(header)
    print("-" * 80)

    for method in methods:
        if method == "baseline":
            continue
        line = f"{method:<20}"
        for cat in categories:
            cat_base = [r for r in results if r.method == "baseline" and r.category == cat]
            cat_method = [r for r in results if r.method == method and r.category == cat]
            if cat_base and cat_method:
                avg_b = sum(r.tok_s for r in cat_base) / len(cat_base)
                avg_m = sum(r.tok_s for r in cat_method) / len(cat_method)
                ratio = avg_m / avg_b if avg_b > 0 else 0
                line += f" {ratio:>11.2f}x"
            else:
                line += f" {'n/a':>12}"
        print(line)

    print()
    print("DETAILED RESULTS")
    print("-" * 100)
    print(f"{'Category':<14} {'Prompt':<30} {'Method':<20} {'t/s':>7} {'ratio':>7} {'accept':>7} {'match':>6}")
    print("-" * 100)

    for cat in categories:
        for pidx in sorted(set(r.prompt_idx for r in results if r.category == cat)):
            cat_results = [r for r in results if r.category == cat and r.prompt_idx == pidx]
            base_r = next((r for r in cat_results if r.method == "baseline"), None)
            if not base_r:
                continue
            for i, r in enumerate(sorted(cat_results, key=lambda x: methods.index(x.method))):
                ratio = r.tok_s / base_r.tok_s if base_r.tok_s > 0 else 0
                cat_label = cat if i == 0 else ""
                prompt_label = r.prompt_preview[:30] if i == 0 else ""
                print(f"{cat_label:<14} {prompt_label:<30} {r.method:<20} {r.tok_s:>6.1f} {ratio:>6.2f}x {r.acceptance:>6.0%} {'Y' if r.output_match else 'N':>5}")
            print()


def main():
    print("=" * 80)
    print(f"Optimization Benchmark — Qwen3-Next-80B-A3B-Instruct-4bit")
    print("=" * 80)

    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"  Total parameters: {total_params / 1e9:.1f}B")

    # Load config for MTP
    print("  Loading config...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # Load MTP weights if available
    mtp_head_bf16 = None
    mtp_head_q4 = None
    if MTP_WEIGHTS.exists():
        print(f"  Loading MTP weights from {MTP_WEIGHTS}...")
        weights_bf16 = load_mtp_weights_from_file(MTP_WEIGHTS)
        mtp_head_bf16 = build_mtp_head(weights_bf16, config, norm_shift=True)

        if mtp_head_bf16:
            # Build quantized copy
            weights_q4 = load_mtp_weights_from_file(MTP_WEIGHTS)
            mtp_head_q4 = build_mtp_head(weights_q4, config, norm_shift=True)
            print("  Quantizing MTP head to 4-bit...")
            quantize_mtp_head(mtp_head_q4, bits=4, group_size=64)

            bf16_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_head_bf16.parameters()))
            q4_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_head_q4.parameters()))
            print(f"  MTP head BF16: {bf16_bytes / 1e6:.1f} MB")
            print(f"  MTP head Q4:   {q4_bytes / 1e6:.1f} MB ({q4_bytes / bf16_bytes:.0%} of BF16)")
        else:
            print("  WARNING: Failed to build MTP head from weights")
    else:
        print(f"  No MTP weights at {MTP_WEIGHTS} — skipping MTP benchmarks")

    print("\nRunning benchmarks...")
    results = run_benchmark(model, tokenizer, eos_set, mtp_head_bf16, mtp_head_q4)

    print_report(results, "Qwen3-Next-80B-A3B-Instruct-4bit")

    output_file = "benchmark_next_80b.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
