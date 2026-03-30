#!/usr/bin/env python3
"""
Master benchmark script — runs all models sequentially.

Models:
  1. Nemotron-3-Nano-4B (dense, no MTP) — baseline + prompt lookup
  2. Nemotron-3-Nano-30B-A3B (MoE, no MTP) — baseline + prompt lookup + shared expert
  3. Qwen3-Coder-Next (MoE + DeltaNet, no MTP) — baseline + prompt lookup + shared expert
  4. Qwen3-Next-80B-A3B (MoE + DeltaNet, MTP) — all optimizations
"""

import gc
import json
import logging
import os
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

# Monkey-patch transformers NemotronH config to support '-' (MLP) block type
try:
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    _orig = NemotronHConfig._pattern_to_list.__func__
    @staticmethod
    def _patched_pattern_to_list(pattern):
        mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [mapping[c] for c in pattern]
    NemotronHConfig._pattern_to_list = _patched_pattern_to_list
except Exception:
    pass

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


def baseline_generate(model, prompt_arr, max_tokens, eos_set):
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


def mtp_generate(decoder, model, prompt_arr, max_tokens, eos_set):
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


def prompt_lookup_generate(model, prompt_arr, max_tokens, eos_set, prompt_tokens):
    drafter = PromptLookupDrafter(prompt_tokens, max_ngram=5, max_draft=5)
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
            model(replay, cache=cache)
            mx.eval(cache)
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


def shared_expert_generate(model, drafter, prompt_arr, max_tokens, eos_set, num_draft=3):
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
            model(replay, cache=cache)
            mx.eval(cache)
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


def benchmark_model(model_name, short_name, has_moe=False, mtp_weights_path=None, bf16_source=None):
    """Benchmark a single model with all applicable optimizations."""
    print(f"\n{'='*80}")
    print(f"  {short_name}")
    print(f"{'='*80}")

    print(f"  Loading {model_name}...")
    model, tokenizer = load(model_name)
    eos_set = resolve_eos(tokenizer)
    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"  Parameters: {total_params / 1e9:.1f}B")

    # MTP heads
    dec_bf16 = dec_q4 = None
    if mtp_weights_path and Path(mtp_weights_path).exists():
        print(f"  Loading MTP weights from {mtp_weights_path}...")
        config_path = Path(snapshot_download(bf16_source, allow_patterns=["config.json"]))
        with open(config_path / "config.json") as f:
            config = json.load(f)
        # VL models nest text config under 'text_config'
        mtp_config = config.get("text_config", config)

        weights_bf16 = load_mtp_weights_from_file(Path(mtp_weights_path))
        mtp_head_bf16 = build_mtp_head(weights_bf16, mtp_config, norm_shift=True)
        if mtp_head_bf16:
            dec_bf16 = MTPDecoder(model, mtp_head_bf16, MTPConfig(greedy_draft=True, batch_verify=True))
            weights_q4 = load_mtp_weights_from_file(Path(mtp_weights_path))
            mtp_head_q4 = build_mtp_head(weights_q4, mtp_config, norm_shift=True)
            quantize_mtp_head(mtp_head_q4, bits=4, group_size=64)
            dec_q4 = MTPDecoder(model, mtp_head_q4, MTPConfig(greedy_draft=True, batch_verify=True))

            bf16_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_head_bf16.parameters()))
            q4_bytes = sum(v.nbytes for _, v in mlx.utils.tree_flatten(mtp_head_q4.parameters()))
            print(f"  MTP BF16: {bf16_bytes/1e6:.0f} MB, Q4: {q4_bytes/1e6:.0f} MB")

    # Shared expert drafter
    shared_drafter = None
    if has_moe:
        try:
            shared_drafter = SharedExpertDrafter(model)
            print(f"  SharedExpertDrafter: {len(shared_drafter._moe_layers)} MoE layers")
        except Exception as e:
            print(f"  SharedExpertDrafter failed: {e}")

    # Warmup
    print("  Warming up...")
    warmup = mx.array(tokenizer.encode("Hello world"))
    for _ in range(2):
        baseline_generate(model, warmup, 10, eos_set)
    print("  Running benchmarks...")

    results = []
    for cat in PROMPTS:
        for pidx, prompt in enumerate(PROMPTS[cat]):
            prompt_tokens = tokenizer.encode(prompt)
            prompt_arr = mx.array(prompt_tokens)
            preview = prompt[:50].replace("\n", "\\n")

            # Baseline
            tokens_base, time_base = avg_runs(
                lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set))
            base_tps = len(tokens_base) / time_base
            results.append(Result(model=short_name, method="baseline", category=cat,
                prompt_idx=pidx, tokens=len(tokens_base), time_s=time_base, tok_s=base_tps,
                acceptance=0, tok_per_step=1.0, output_match=True, prompt_preview=preview))

            parts = [f"base={base_tps:.0f}"]

            # MTP BF16
            if dec_bf16:
                dec_bf16.stats.__init__()
                t_bf16, tm_bf16, s_bf16 = avg_runs(
                    lambda: mtp_generate(dec_bf16, model, prompt_arr, MAX_TOKENS, eos_set))
                bf16_tps = len(t_bf16) / tm_bf16
                results.append(Result(model=short_name, method="mtp_bf16", category=cat,
                    prompt_idx=pidx, tokens=len(t_bf16), time_s=tm_bf16, tok_s=bf16_tps,
                    acceptance=s_bf16["acceptance"], tok_per_step=s_bf16["tok_per_step"],
                    output_match=t_bf16==tokens_base[:len(t_bf16)], prompt_preview=preview))
                parts.append(f"bf16={bf16_tps/base_tps:.2f}x")

            # MTP Q4
            if dec_q4:
                dec_q4.stats.__init__()
                t_q4, tm_q4, s_q4 = avg_runs(
                    lambda: mtp_generate(dec_q4, model, prompt_arr, MAX_TOKENS, eos_set))
                q4_tps = len(t_q4) / tm_q4
                results.append(Result(model=short_name, method="mtp_q4", category=cat,
                    prompt_idx=pidx, tokens=len(t_q4), time_s=tm_q4, tok_s=q4_tps,
                    acceptance=s_q4["acceptance"], tok_per_step=s_q4["tok_per_step"],
                    output_match=t_q4==tokens_base[:len(t_q4)], prompt_preview=preview))
                parts.append(f"q4={q4_tps/base_tps:.2f}x")

            # Prompt lookup
            t_pl, tm_pl, s_pl = avg_runs(
                lambda: prompt_lookup_generate(model, prompt_arr, MAX_TOKENS, eos_set, prompt_tokens))
            pl_tps = len(t_pl) / tm_pl
            results.append(Result(model=short_name, method="prompt_lookup", category=cat,
                prompt_idx=pidx, tokens=len(t_pl), time_s=tm_pl, tok_s=pl_tps,
                acceptance=s_pl["acceptance"], tok_per_step=s_pl["tok_per_step"],
                output_match=t_pl==tokens_base[:len(t_pl)], prompt_preview=preview))
            parts.append(f"pl={pl_tps/base_tps:.2f}x")

            # Shared expert d=1
            if shared_drafter:
                t_se, tm_se, s_se = avg_runs(
                    lambda: shared_expert_generate(model, shared_drafter, prompt_arr, MAX_TOKENS, eos_set, num_draft=1))
                se_tps = len(t_se) / tm_se
                results.append(Result(model=short_name, method="shared_expert_d1", category=cat,
                    prompt_idx=pidx, tokens=len(t_se), time_s=tm_se, tok_s=se_tps,
                    acceptance=s_se["acceptance"], tok_per_step=s_se["tok_per_step"],
                    output_match=t_se==tokens_base[:len(t_se)], prompt_preview=preview))
                parts.append(f"se1={se_tps/base_tps:.2f}x")

            print(f"    [{cat}:{pidx}] {' '.join(parts)} '{preview}'")

    # Cleanup
    if dec_bf16:
        dec_bf16.cleanup()
    if dec_q4:
        dec_q4.cleanup()
    del model
    gc.collect()
    mx.clear_cache()

    # Save incremental results
    output_file = f"benchmark_{short_name.replace('-', '_').replace('.', '_')}.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  Saved {output_file}")

    return results


def print_summary(all_results):
    models = sorted(set(r.model for r in all_results))

    print("\n" + "=" * 100)
    print("CROSS-MODEL SUMMARY")
    print("=" * 100)

    for model_name in models:
        results = [r for r in all_results if r.model == model_name]
        methods = sorted(set(r.method for r in results))
        base_results = [r for r in results if r.method == "baseline"]
        avg_base = sum(r.tok_s for r in base_results) / len(base_results) if base_results else 0

        print(f"\n  {model_name} (baseline: {avg_base:.1f} t/s)")
        for method in methods:
            if method == "baseline":
                continue
            mr = [r for r in results if r.method == method]
            avg_tps = sum(r.tok_s for r in mr) / len(mr)
            avg_acc = sum(r.acceptance for r in mr) / len(mr)
            ratio = avg_tps / avg_base if avg_base > 0 else 0
            print(f"    {method:<20} {avg_tps:>6.1f} t/s  {ratio:>5.2f}x  accept={avg_acc:.0%}")


MODELS = [
    {
        "name": "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit",
        "short": "Nemotron-3-Nano-4B",
        "moe": False,
    },
    {
        "name": "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
        "short": "Nemotron-3-Nano-30B-A3B",
        "moe": True,
    },
    {
        "name": "mlx-community/Qwen3-Coder-Next-4bit",
        "short": "Qwen3-Coder-Next",
        "moe": True,
    },
    {
        "name": "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        "short": "Qwen3-Next-80B-A3B",
        "moe": True,
        "mtp_weights": "mtp_weights/Qwen_Qwen3-Next-80B-A3B.safetensors",
        "bf16_source": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    },
]


def main():
    print("=" * 80)
    print("  MASTER BENCHMARK SUITE")
    print("  Models: Nemotron-3-Nano-4B, Nemotron-3-Nano-30B-A3B,")
    print("          Qwen3-Coder-Next, Qwen3-Next-80B-A3B")
    print("=" * 80)

    all_results = []

    for m in MODELS:
        try:
            results = benchmark_model(
                m["name"], m["short"],
                has_moe=m.get("moe", False),
                mtp_weights_path=m.get("mtp_weights"),
                bf16_source=m.get("bf16_source"),
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n  FAILED: {m['short']}: {e}")
            import traceback
            traceback.print_exc()

    print_summary(all_results)

    with open("benchmark_all_models.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nAll results saved to benchmark_all_models.json")


if __name__ == "__main__":
    main()
