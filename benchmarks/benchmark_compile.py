#!/usr/bin/env python3
"""
Benchmark mx.compile optimization for Qwen3.5-35B-A3B decode.

Tests multiple compilation strategies to reduce kernel dispatch overhead:
  1. Baseline (no extra compilation)
  2. Compile MoE/MLP blocks
  3. Compile per-layer norm+MLP path
  4. Compile full model step function (extract cache state as arrays)

The bottleneck: ~800 kernel dispatches × ~15μs each ≈ 12ms per decode step.
mx.compile fuses multiple ops into single dispatch, potentially cutting this
by 5-10x for compiled sub-graphs.
"""

import gc
import json
import logging
import time
import types
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
MAX_TOKENS = 128
WARMUP_TOKENS = 20
NUM_RUNS = 3  # best-of-N for each config

PROMPTS = [
    "Write a Python function that implements binary search:\n```python\ndef binary_search(arr, target):\n",
    "Explain the theory of relativity in simple terms:\n",
    "The capital of France is",
]


@dataclass
class BenchResult:
    strategy: str
    prompt_idx: int
    tokens_generated: int
    prefill_ms: float
    decode_ms: float
    decode_tok_s: float
    first_token_match: bool
    all_match: bool
    prompt_preview: str


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}
    return eos_set


# ---------------------------------------------------------------------------
# Baseline decode loop — no extra compilation
# ---------------------------------------------------------------------------

def generate_baseline(model, prompt_arr, max_tokens, eos_set):
    """Standard decode loop. Returns (tokens, prefill_ms, decode_ms)."""
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
        tid = tokens[-1]
        if tid in eos_set:
            break
        logits = model(mx.array([[tid]]), cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        tokens.append(tok.item())

    t_done = time.perf_counter()
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    return tokens, prefill_ms, decode_ms


# ---------------------------------------------------------------------------
# Strategy 1: Compile MoE/MLP blocks
# ---------------------------------------------------------------------------

def apply_compile_moe_blocks(model):
    """Wrap each MoE and MLP block's __call__ with mx.compile."""
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    compiled_count = 0
    original_calls = {}

    for i, layer in enumerate(inner.layers):
        mlp = layer.mlp
        # Save original
        original_calls[i] = mlp.__call__

        # Compile the forward — works because mlp(x) is pure array→array
        # (weights are captured via self closure, not passed as args)
        compiled_fn = mx.compile(mlp.__call__)
        layer.mlp = _WrapCompiled(mlp, compiled_fn)
        compiled_count += 1

    print(f"  Compiled {compiled_count} MLP/MoE blocks")
    return original_calls


class _WrapCompiled:
    """Wrapper that delegates __call__ to a compiled function but
    preserves the original module for attribute access."""

    def __init__(self, module, compiled_fn):
        self._module = module
        self._compiled_fn = compiled_fn

    def __call__(self, *args, **kwargs):
        return self._compiled_fn(*args, **kwargs)

    def __getattr__(self, name):
        if name in ('_module', '_compiled_fn'):
            return object.__getattribute__(self, name)
        return getattr(self._module, name)


def restore_moe_blocks(model, original_calls):
    """Restore original MLP/MoE __call__ methods."""
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    for i, layer in enumerate(inner.layers):
        if i in original_calls:
            # Unwrap if we wrapped
            if isinstance(layer.mlp, _WrapCompiled):
                layer.mlp = layer.mlp._module


# ---------------------------------------------------------------------------
# Strategy 2: Compile norm+MLP path per layer
# ---------------------------------------------------------------------------

def apply_compile_norm_mlp(model):
    """Compile the norm→MLP→residual path as a single fused function per layer."""
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    patched_layers = {}

    for i, layer in enumerate(inner.layers):
        norm = layer.post_attention_layernorm
        mlp = layer.mlp

        # Create a standalone function that captures norm+mlp
        def make_norm_mlp_fn(norm_mod, mlp_mod):
            @mx.compile
            def norm_mlp(h):
                return mlp_mod(norm_mod(h))
            return norm_mlp

        compiled_norm_mlp = make_norm_mlp_fn(norm, mlp)

        # Monkey-patch the layer's __call__ to use compiled norm+mlp
        original_call = layer.__class__.__call__
        patched_layers[i] = (layer, original_call)

        def make_patched_call(compiled_fn, is_linear):
            def patched_call(self, x, mask=None, cache=None):
                if is_linear:
                    r = self.linear_attn(self.input_layernorm(x), mask, cache)
                else:
                    r = self.self_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                out = h + compiled_fn(h)
                return out
            return patched_call

        layer.__call__ = types.MethodType(
            make_patched_call(compiled_norm_mlp, layer.is_linear), layer
        )

    print(f"  Compiled {len(patched_layers)} norm+MLP paths")
    return patched_layers


def restore_norm_mlp(model, patched_layers):
    """Restore original layer __call__ methods."""
    for i, (layer, original_call) in patched_layers.items():
        if hasattr(layer, '__call__'):
            del layer.__call__  # Remove instance override


# ---------------------------------------------------------------------------
# Strategy 3: Compile full model step (extract cache as arrays)
# ---------------------------------------------------------------------------

def make_compiled_step(model):
    """Create a compiled function for the full decode step.

    The trick: we don't pass cache as an argument. Instead we build
    a closure that captures the model and cache, and the compiled function
    takes only the token input array.

    This won't work directly because cache state changes each step.
    But we can compile the model forward by passing it as a traced module.
    """
    # Try compiling model.__call__ with shapeless
    # This works if all inputs are arrays and constants
    # Cache is the problem — let's try a different approach

    # Approach: Compile just the backbone loop (embed → layers → norm → lm_head)
    # without the cache objects. We'll extract per-layer cache state as array tuples.

    # Actually, the simplest approach that works: compile the full model forward
    # and let MLX trace through the cache operations as side effects.
    # MLX compile traces the computation graph, not the Python code.
    # As long as cache.update_and_fetch returns arrays, it should work.

    # Let's try it:
    try:
        compiled_model = mx.compile(model.__call__, shapeless=True)
        return compiled_model, "full_model_compiled"
    except Exception as e:
        print(f"  Full model compile failed: {e}")
        return None, str(e)


# ---------------------------------------------------------------------------
# Strategy 4: Compile model + sampling as one step
# ---------------------------------------------------------------------------

def make_compiled_generate_step(model):
    """Compile model forward + argmax sampling as single function."""
    def step(input_ids, cache):
        logits = model(input_ids, cache=cache)
        return mx.argmax(logits[:, -1, :], axis=-1), logits

    try:
        compiled_step = mx.compile(step, shapeless=True)
        return compiled_step, "step_compiled"
    except Exception as e:
        print(f"  Step compile failed: {e}")
        return None, str(e)


def generate_with_compiled_step(compiled_step, model, prompt_arr, max_tokens, eos_set):
    """Decode using a compiled step function."""
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
        tid = tokens[-1]
        if tid in eos_set:
            break
        next_tok, _ = compiled_step(mx.array([[tid]]), cache)
        mx.eval(next_tok)
        tokens.append(next_tok.item())

    t_done = time.perf_counter()
    prefill_ms = (t_prefill - t0) * 1000
    decode_ms = (t_done - t_prefill) * 1000
    return tokens, prefill_ms, decode_ms


# ---------------------------------------------------------------------------
# Strategy 5: Compile MoE routing logic only
# ---------------------------------------------------------------------------

def apply_compile_moe_routing(model):
    """Compile just the routing portion of MoE blocks (gate → softmax → argpartition → scores)."""
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    compiled_count = 0
    original_moe_calls = {}

    for i, layer in enumerate(inner.layers):
        mlp = layer.mlp
        # Only target MoE blocks (Qwen3NextSparseMoeBlock)
        if not hasattr(mlp, 'gate') or not hasattr(mlp, 'switch_mlp'):
            continue

        original_moe_calls[i] = mlp

        # Compile the routing: gate → softmax → argpartition → score extraction
        gate_mod = mlp.gate
        top_k = mlp.top_k
        norm_topk_prob = mlp.norm_topk_prob

        def make_routing_fn(gate, k, norm_topk):
            @mx.compile
            def routing(x):
                gates = gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if norm_topk:
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                return inds, scores
            return routing

        mlp._compiled_routing = make_routing_fn(gate_mod, top_k, norm_topk_prob)

        # Monkey-patch MoE __call__
        original_call = type(mlp).__call__

        def make_patched_moe(compiled_routing_fn):
            def patched_moe_call(self, x):
                inds, scores = compiled_routing_fn(x)
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                return y + shared_y
            return patched_moe_call

        mlp.__call__ = types.MethodType(
            make_patched_moe(mlp._compiled_routing), mlp
        )
        compiled_count += 1

    print(f"  Compiled routing in {compiled_count} MoE blocks")
    return original_moe_calls


def restore_moe_routing(model, original_moe_calls):
    """Restore original MoE __call__ methods."""
    inner = model.model if not hasattr(model, 'language_model') else model.language_model.model
    for i, mlp_mod in original_moe_calls.items():
        if hasattr(mlp_mod, '__call__'):
            del mlp_mod.__call__
        if hasattr(mlp_mod, '_compiled_routing'):
            del mlp_mod._compiled_routing


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_single(generate_fn, max_tokens, eos_set, num_runs=NUM_RUNS):
    """Run generate_fn multiple times, return best result."""
    best = None
    for _ in range(num_runs):
        tokens, prefill_ms, decode_ms = generate_fn()
        if best is None or decode_ms < best[2]:
            best = (tokens, prefill_ms, decode_ms)
    return best


def main():
    print("=" * 80)
    print("mx.compile Optimization Benchmark — Qwen3.5-35B-A3B-4bit")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = resolve_eos(tokenizer)

    # Warmup
    print("Warming up...")
    warmup_arr = mx.array(tokenizer.encode("Hello world"))
    for _ in range(3):
        generate_baseline(model, warmup_arr, WARMUP_TOKENS, eos_set)
    print("Warmup done.\n")

    results = []
    baseline_outputs = {}  # prompt_idx → token list

    # ---------------------------------------------------------------------------
    # Benchmark each strategy per prompt
    # ---------------------------------------------------------------------------

    for pidx, prompt in enumerate(PROMPTS):
        prompt_tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(prompt_tokens)
        preview = prompt[:60].replace("\n", "\\n")
        print(f"\n--- Prompt {pidx}: '{preview}' ---")

        # 1. Baseline
        print("  [1] Baseline (no extra compile)...")
        tokens, prefill_ms, decode_ms = run_single(
            lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set),
            MAX_TOKENS, eos_set,
        )
        baseline_outputs[pidx] = tokens
        n_decode = max(len(tokens) - 1, 1)
        decode_tps = n_decode / (decode_ms / 1000) if decode_ms > 0 else 0
        print(f"    {len(tokens)} tokens, prefill={prefill_ms:.0f}ms, decode={decode_ms:.0f}ms, {decode_tps:.1f} t/s")
        results.append(BenchResult(
            strategy="baseline", prompt_idx=pidx, tokens_generated=len(tokens),
            prefill_ms=prefill_ms, decode_ms=decode_ms, decode_tok_s=decode_tps,
            first_token_match=True, all_match=True, prompt_preview=preview,
        ))

        # 2. Compile MoE/MLP blocks
        print("  [2] Compile MoE/MLP blocks...")
        orig_moe = apply_compile_moe_blocks(model)
        # Warmup compiled path
        generate_baseline(model, warmup_arr, 5, eos_set)
        tokens2, prefill_ms2, decode_ms2 = run_single(
            lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set),
            MAX_TOKENS, eos_set,
        )
        restore_moe_blocks(model, orig_moe)
        n2 = max(len(tokens2) - 1, 1)
        tps2 = n2 / (decode_ms2 / 1000) if decode_ms2 > 0 else 0
        match2 = tokens2 == baseline_outputs[pidx]
        first_match2 = tokens2[0] == baseline_outputs[pidx][0] if tokens2 and baseline_outputs[pidx] else False
        print(f"    {len(tokens2)} tokens, prefill={prefill_ms2:.0f}ms, decode={decode_ms2:.0f}ms, {tps2:.1f} t/s, match={match2}")
        results.append(BenchResult(
            strategy="compile_moe_mlp", prompt_idx=pidx, tokens_generated=len(tokens2),
            prefill_ms=prefill_ms2, decode_ms=decode_ms2, decode_tok_s=tps2,
            first_token_match=first_match2, all_match=match2, prompt_preview=preview,
        ))

        # 3. Compile norm+MLP path
        print("  [3] Compile norm+MLP path...")
        patched = apply_compile_norm_mlp(model)
        # Warmup
        generate_baseline(model, warmup_arr, 5, eos_set)
        tokens3, prefill_ms3, decode_ms3 = run_single(
            lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set),
            MAX_TOKENS, eos_set,
        )
        restore_norm_mlp(model, patched)
        n3 = max(len(tokens3) - 1, 1)
        tps3 = n3 / (decode_ms3 / 1000) if decode_ms3 > 0 else 0
        match3 = tokens3 == baseline_outputs[pidx]
        first_match3 = tokens3[0] == baseline_outputs[pidx][0] if tokens3 and baseline_outputs[pidx] else False
        print(f"    {len(tokens3)} tokens, prefill={prefill_ms3:.0f}ms, decode={decode_ms3:.0f}ms, {tps3:.1f} t/s, match={match3}")
        results.append(BenchResult(
            strategy="compile_norm_mlp", prompt_idx=pidx, tokens_generated=len(tokens3),
            prefill_ms=prefill_ms3, decode_ms=decode_ms3, decode_tok_s=tps3,
            first_token_match=first_match3, all_match=match3, prompt_preview=preview,
        ))

        # 4. Compile MoE routing only
        print("  [4] Compile MoE routing only...")
        orig_routing = apply_compile_moe_routing(model)
        # Warmup
        generate_baseline(model, warmup_arr, 5, eos_set)
        tokens4, prefill_ms4, decode_ms4 = run_single(
            lambda p=prompt_arr: generate_baseline(model, p, MAX_TOKENS, eos_set),
            MAX_TOKENS, eos_set,
        )
        restore_moe_routing(model, orig_routing)
        n4 = max(len(tokens4) - 1, 1)
        tps4 = n4 / (decode_ms4 / 1000) if decode_ms4 > 0 else 0
        match4 = tokens4 == baseline_outputs[pidx]
        first_match4 = tokens4[0] == baseline_outputs[pidx][0] if tokens4 and baseline_outputs[pidx] else False
        print(f"    {len(tokens4)} tokens, prefill={prefill_ms4:.0f}ms, decode={decode_ms4:.0f}ms, {tps4:.1f} t/s, match={match4}")
        results.append(BenchResult(
            strategy="compile_moe_routing", prompt_idx=pidx, tokens_generated=len(tokens4),
            prefill_ms=prefill_ms4, decode_ms=decode_ms4, decode_tok_s=tps4,
            first_token_match=first_match4, all_match=match4, prompt_preview=preview,
        ))

        # 5. Try compiled step function (may fail)
        print("  [5] Compile full model step...")
        compiled_step, status = make_compiled_generate_step(model)
        if compiled_step is not None:
            try:
                # Warmup
                generate_with_compiled_step(compiled_step, model, warmup_arr, 5, eos_set)
                tokens5, prefill_ms5, decode_ms5 = run_single(
                    lambda p=prompt_arr: generate_with_compiled_step(
                        compiled_step, model, p, MAX_TOKENS, eos_set
                    ),
                    MAX_TOKENS, eos_set,
                )
                n5 = max(len(tokens5) - 1, 1)
                tps5 = n5 / (decode_ms5 / 1000) if decode_ms5 > 0 else 0
                match5 = tokens5 == baseline_outputs[pidx]
                first_match5 = tokens5[0] == baseline_outputs[pidx][0] if tokens5 and baseline_outputs[pidx] else False
                print(f"    {len(tokens5)} tokens, prefill={prefill_ms5:.0f}ms, decode={decode_ms5:.0f}ms, {tps5:.1f} t/s, match={match5}")
                results.append(BenchResult(
                    strategy="compile_full_step", prompt_idx=pidx, tokens_generated=len(tokens5),
                    prefill_ms=prefill_ms5, decode_ms=decode_ms5, decode_tok_s=tps5,
                    first_token_match=first_match5, all_match=match5, prompt_preview=preview,
                ))
            except Exception as e:
                print(f"    FAILED during generate: {e}")
                results.append(BenchResult(
                    strategy="compile_full_step", prompt_idx=pidx, tokens_generated=0,
                    prefill_ms=0, decode_ms=0, decode_tok_s=0,
                    first_token_match=False, all_match=False, prompt_preview=f"FAILED: {e}",
                ))
        else:
            print(f"    SKIPPED: {status}")
            results.append(BenchResult(
                strategy="compile_full_step", prompt_idx=pidx, tokens_generated=0,
                prefill_ms=0, decode_ms=0, decode_tok_s=0,
                first_token_match=False, all_match=False, prompt_preview=f"SKIPPED: {status}",
            ))

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("mx.compile BENCHMARK RESULTS")
    print("=" * 90)

    strategies = sorted(set(r.strategy for r in results))

    # Overall summary
    print(f"\n{'Strategy':<25} {'Avg t/s':>10} {'vs Base':>10} {'Avg decode ms':>15} {'Match':>8}")
    print("-" * 70)

    base_results = [r for r in results if r.strategy == "baseline" and r.decode_tok_s > 0]
    avg_base_tps = sum(r.decode_tok_s for r in base_results) / len(base_results) if base_results else 0

    for strat in strategies:
        sr = [r for r in results if r.strategy == strat and r.decode_tok_s > 0]
        if not sr:
            print(f"{strat:<25} {'N/A':>10} {'N/A':>10} {'N/A':>15} {'N/A':>8}")
            continue
        avg_tps = sum(r.decode_tok_s for r in sr) / len(sr)
        avg_decode = sum(r.decode_ms for r in sr) / len(sr)
        ratio = avg_tps / avg_base_tps if avg_base_tps > 0 else 0
        all_match = all(r.all_match for r in sr)
        print(f"{strat:<25} {avg_tps:>9.1f} {ratio:>9.2f}x {avg_decode:>14.0f} {'✓' if all_match else '✗':>7}")

    # Per-prompt detail
    print(f"\n{'Strategy':<25} {'Prompt':>5} {'Tokens':>7} {'Decode ms':>10} {'t/s':>8} {'ratio':>8} {'Match':>6}")
    print("-" * 75)

    for pidx in range(len(PROMPTS)):
        for strat in strategies:
            r = next((r for r in results if r.strategy == strat and r.prompt_idx == pidx), None)
            if not r:
                continue
            base_r = next((r for r in results if r.strategy == "baseline" and r.prompt_idx == pidx), None)
            ratio = r.decode_tok_s / base_r.decode_tok_s if base_r and base_r.decode_tok_s > 0 else 0
            print(f"{strat:<25} {pidx:>5} {r.tokens_generated:>7} {r.decode_ms:>9.0f} {r.decode_tok_s:>7.1f} {ratio:>7.2f}x {'Y' if r.all_match else 'N':>5}")
        print()

    # Save
    out_file = "benchmark_compile_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
