#!/usr/bin/env python3
"""
Benchmark optimizations on Nemotron-3-Nano models.

Tests:
  - Nemotron-3-Nano-30B-A3B (MoE, 30B/3B active): baseline, prompt lookup, shared-expert
  - Nemotron-3-Nano-4B (dense): baseline, prompt lookup

No MTP available for either model.
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
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.optimizations import PromptLookupDrafter, SharedExpertDrafter
from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state

logging.basicConfig(level=logging.WARNING)

MAX_TOKENS = 128
NUM_RUNS = 2


@dataclass
class Result:
    model: str
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


def benchmark_model(model_name: str, has_moe: bool):
    """Benchmark a single model."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    eos_set = resolve_eos(tokenizer)

    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"  Total parameters: {total_params / 1e9:.1f}B")

    # Build shared-expert drafter if MoE
    shared_drafter = None
    if has_moe:
        try:
            shared_drafter = SharedExpertDrafter(model)
            print(f"  SharedExpertDrafter: found {len(shared_drafter._moe_layers)} MoE layers")
        except Exception as e:
            print(f"  SharedExpertDrafter setup failed: {e}")

    # Warmup
    print("  Warming up...")
    warmup_arr = mx.array(tokenizer.encode("Hello world"))
    for _ in range(2):
        baseline_generate(model, warmup_arr, 10, eos_set)
    print("  Warmup done\n")

    results = []
    short_name = model_name.split("/")[-1]

    for cat in PROMPTS:
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

            # Baseline
            tokens_base, time_base = avg_runs(
                lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set)
            )
            base_tps = len(tokens_base) / time_base
            results.append(Result(
                model=short_name, method="baseline", category=cat, prompt_idx=pidx,
                tokens=len(tokens_base), time_s=time_base, tok_s=base_tps,
                acceptance=0, tok_per_step=1.0, output_match=True, prompt_preview=preview,
            ))

            # Prompt lookup
            tokens_pl, time_pl, stats_pl = avg_runs(
                lambda: prompt_lookup_generate(
                    model, tokenizer, prompt_arr, MAX_TOKENS, eos_set,
                    prompt_tokens, max_ngram=5, max_draft=5,
                )
            )
            pl_tps = len(tokens_pl) / time_pl
            results.append(Result(
                model=short_name, method="prompt_lookup", category=cat, prompt_idx=pidx,
                tokens=len(tokens_pl), time_s=time_pl, tok_s=pl_tps,
                acceptance=stats_pl["acceptance"], tok_per_step=stats_pl["tok_per_step"],
                output_match=tokens_pl == tokens_base[:len(tokens_pl)], prompt_preview=preview,
            ))

            # Shared expert (MoE only)
            se1_tps = 0
            se3_tps = 0
            if shared_drafter:
                for nd, label in [(1, "shared_expert_d1"), (3, "shared_expert_d3")]:
                    tokens_se, time_se, stats_se = avg_runs(
                        lambda nd=nd: shared_expert_generate(
                            model, shared_drafter, prompt_arr, MAX_TOKENS, eos_set,
                            num_draft=nd,
                        )
                    )
                    se_tps = len(tokens_se) / time_se
                    if nd == 1:
                        se1_tps = se_tps
                    else:
                        se3_tps = se_tps
                    results.append(Result(
                        model=short_name, method=label, category=cat, prompt_idx=pidx,
                        tokens=len(tokens_se), time_s=time_se, tok_s=se_tps,
                        acceptance=stats_se["acceptance"], tok_per_step=stats_se["tok_per_step"],
                        output_match=tokens_se == tokens_base[:len(tokens_se)], prompt_preview=preview,
                    ))

            # Progress
            parts = [f"base={base_tps:.0f}", f"pl={pl_tps / base_tps:.2f}x"]
            if shared_drafter:
                parts.append(f"se1={se1_tps / base_tps:.2f}x")
                parts.append(f"se3={se3_tps / base_tps:.2f}x")
            print(f"    [{cat}:{pidx}] {' '.join(parts)} '{preview}'")

    # Cleanup
    del model
    gc.collect()
    mx.clear_cache()

    return results


def print_report(all_results: List[Result]):
    models = sorted(set(r.model for r in all_results))

    for model_name in models:
        results = [r for r in all_results if r.model == model_name]
        methods = sorted(set(r.method for r in results))
        categories = sorted(set(r.category for r in results))

        print()
        print("=" * 100)
        print(f"BENCHMARK REPORT — {model_name}")
        print("=" * 100)
        print()

        print("OVERALL SUMMARY")
        print("-" * 80)
        print(f"{'Method':<20} {'Avg t/s':>8} {'vs Base':>8} {'Accept':>8} {'t/step':>8}")
        print("-" * 80)

        base_results = [r for r in results if r.method == "baseline"]
        avg_base = sum(r.tok_s for r in base_results) / len(base_results) if base_results else 0

        for method in methods:
            mr = [r for r in results if r.method == method]
            avg_tps = sum(r.tok_s for r in mr) / len(mr)
            avg_acc = sum(r.acceptance for r in mr) / len(mr)
            avg_tps_step = sum(r.tok_per_step for r in mr) / len(mr)
            ratio = avg_tps / avg_base if avg_base > 0 else 0
            print(f"{method:<20} {avg_tps:>7.1f} {ratio:>7.2f}x {avg_acc:>7.0%} {avg_tps_step:>7.2f}")

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


def main():
    print("=" * 80)
    print("Nemotron-3-Nano Benchmark Suite")
    print("=" * 80)

    all_results = []

    # Benchmark Nano-4B (dense — no MoE, no MTP)
    results_4b = benchmark_model("mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit", has_moe=False)
    all_results.extend(results_4b)

    # Benchmark Nano-30B-A3B (MoE — shared expert drafting available)
    results_30b = benchmark_model("mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit", has_moe=True)
    all_results.extend(results_30b)

    print_report(all_results)

    output_file = "benchmark_nemotron.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
