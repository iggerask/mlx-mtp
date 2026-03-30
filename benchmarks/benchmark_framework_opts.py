#!/usr/bin/env python3
"""Benchmark framework-level optimizations for Qwen3.5-35B-A3B.

Tests:
1. Baseline (stock MLX)
2. ZMLX patch (fused MoE gating + combine)
3. mx.compile with shapeless=True on model forward
4. Force sorted_indices for single-token decode
5. Combined: ZMLX + compile + sorted_indices
6. All above + MTP Q4 lazy batch
"""

import json
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")
PROMPT = "Write a Python function that checks if a number is prime:\n```python\ndef is_prime(n):\n"
MAX_TOKENS = 60
NUM_RUNS = 3  # Best of N


def baseline_generate(model, prompt_arr, max_tokens, eos_set):
    """Standard autoregressive decode — no optimizations."""
    cache = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
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
    return tokens


def async_generate(model, prompt_arr, max_tokens, eos_set):
    """Async eval pipeline — overlap CPU graph build with GPU execution."""
    cache = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])

    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens = [tok.item()]

    # Prime pipeline
    logits = model(mx.array([[tokens[-1]]]), cache=cache)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.async_eval(next_tok)

    for _ in range(max_tokens - 2):
        prev_tok = next_tok
        logits = model(mx.array([[prev_tok.item()]]), cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(next_tok)
        tokens.append(prev_tok.item())
        if tokens[-1] in eos_set:
            break

    mx.eval(next_tok)
    tokens.append(next_tok.item())
    return tokens


def compile_generate(model, prompt_arr, max_tokens, eos_set, compiled_model):
    """Use mx.compile on model forward."""
    cache = make_prompt_cache(model)
    logits = compiled_model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    tokens = []
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    tokens.append(tok.item())
    for _ in range(max_tokens - 1):
        if tokens[-1] in eos_set:
            break
        logits = compiled_model(mx.array([[tokens[-1]]]), cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        tokens.append(tok.item())
    return tokens


def mtp_generate(model, mtp_head, prompt_arr, max_tokens, eos_set, config):
    """MTP speculative decode."""
    from vllm_mlx_mtp.mtp_decoder import MTPDecoder
    dec = MTPDecoder(model, mtp_head, config)
    cache = make_prompt_cache(model)
    tokens = list(dec.generate(prompt_arr, cache, max_tokens=max_tokens,
                               temperature=0.0, eos_tokens=eos_set))
    stats = dec.stats
    dec.cleanup()
    return tokens, stats


def bench(fn, num_runs=NUM_RUNS):
    """Run function num_runs times, return best time and result."""
    best_time = float("inf")
    best_result = None
    for _ in range(num_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        result = fn()
        mx.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed < best_time:
            best_time = elapsed
            best_result = result
    return best_time, best_result


def force_sort_moe_indices(model):
    """Monkey-patch SwitchGLU to always sort indices (even for single token decode)."""
    tm = model.language_model if hasattr(model, "language_model") else model
    patched = 0
    for layer in tm.model.layers:
        moe = layer.get("mlp", None) if hasattr(layer, "get") else getattr(layer, "mlp", None)
        if moe is None:
            continue
        switch = getattr(moe, "switch_mlp", None)
        if switch is None:
            continue

        # Replace forward to always sort
        original_call = switch.__class__.__call__

        def make_sorted_call(orig):
            def sorted_call(self, x, indices):
                from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
                x = mx.expand_dims(x, (-2, -3))

                # Always sort (force sorted_indices=True even for small batches)
                x_sorted, idx_sorted, inv_order = _gather_sort(x, indices)
                if self.training:
                    idx_sorted = mx.stop_gradient(idx_sorted)
                x_up = self.up_proj(x_sorted, idx_sorted, sorted_indices=True)
                x_gate = self.gate_proj(x_sorted, idx_sorted, sorted_indices=True)
                x_out = self.down_proj(
                    self.activation(x_up, x_gate),
                    idx_sorted,
                    sorted_indices=True,
                )
                x_out = _scatter_unsort(x_out, inv_order, indices.shape)
                return x_out.squeeze(-2)
            return sorted_call

        switch.__class__.__call__ = make_sorted_call(original_call)
        patched += 1

    return patched


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}

    prompt_arr = mx.array(tokenizer.encode(PROMPT))

    # Warmup
    print("Warming up...")
    for _ in range(3):
        baseline_generate(model, mx.array(tokenizer.encode("Hello")), 10, eos_set)

    results = {}

    # ── 1. Baseline ──────────────────────────────────────────────
    print("\n1. Baseline (stock MLX)")
    t, tokens = bench(lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set))
    tps = len(tokens) / t
    print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
    print(f"   Output: {tokenizer.decode(tokens[:25])!r}...")
    results["baseline"] = {"tps": tps, "tokens": tokens}

    # ── 2. Async eval pipeline ───────────────────────────────────
    print("\n2. Async eval pipeline")
    t, tokens = bench(lambda: async_generate(model, prompt_arr, MAX_TOKENS, eos_set))
    tps = len(tokens) / t
    print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
    results["async_eval"] = {"tps": tps}

    # ── 3. ZMLX patch ───────────────────────────────────────────
    print("\n3. ZMLX patch")
    try:
        import zmlx
        n_patched = zmlx.patch(model)
        print(f"   Patched {n_patched} modules")
        t, tokens = bench(lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set))
        tps = len(tokens) / t
        match = sum(1 for a, b in zip(results["baseline"]["tokens"], tokens) if a == b)
        print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
        print(f"   Token match vs baseline: {match}/{len(results['baseline']['tokens'])}")
        results["zmlx"] = {"tps": tps, "patched": n_patched}

        # Undo ZMLX patch for subsequent tests
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["zmlx"] = {"tps": 0, "error": str(e)}
        model, tokenizer = load(MODEL_NAME)

    # ── 4. mx.compile with shapeless=True ────────────────────────
    print("\n4. mx.compile (shapeless=True)")
    try:
        # Try compiling just the model forward, not cache management
        compiled_model = mx.compile(model, shapeless=True)
        # Warmup compiled path
        for _ in range(3):
            compile_generate(model, mx.array(tokenizer.encode("Hi")), 5, eos_set, compiled_model)

        t, tokens = bench(lambda: compile_generate(model, prompt_arr, MAX_TOKENS, eos_set, compiled_model))
        tps = len(tokens) / t
        match = sum(1 for a, b in zip(results["baseline"]["tokens"], tokens) if a == b)
        print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
        print(f"   Token match vs baseline: {match}/{len(results['baseline']['tokens'])}")
        results["compile"] = {"tps": tps}
    except Exception as e:
        print(f"   FAILED: {e}")
        results["compile"] = {"tps": 0, "error": str(e)}

    # ── 5. Force sorted indices ──────────────────────────────────
    print("\n5. Force sorted_indices on MoE dispatch")
    try:
        n = force_sort_moe_indices(model)
        print(f"   Patched {n} MoE layers")

        # Warmup with new dispatch
        for _ in range(3):
            baseline_generate(model, mx.array(tokenizer.encode("Hi")), 5, eos_set)

        t, tokens = bench(lambda: baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set))
        tps = len(tokens) / t
        match = sum(1 for a, b in zip(results["baseline"]["tokens"], tokens) if a == b)
        print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
        print(f"   Token match vs baseline: {match}/{len(results['baseline']['tokens'])}")
        results["sorted_indices"] = {"tps": tps}

        # Reload to reset patches
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        print(f"   FAILED: {e}")
        results["sorted_indices"] = {"tps": 0, "error": str(e)}
        model, tokenizer = load(MODEL_NAME)

    # ── 6. ZMLX + MTP Q4 lazy batch ─────────────────────────────
    print("\n6. ZMLX + MTP Q4 lazy batch (combined)")
    try:
        import zmlx
        n_patched = zmlx.patch(model)
        print(f"   ZMLX patched: {n_patched} modules")

        # Load MTP head
        model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
        with open(model_path / "config.json") as f:
            config = json.load(f)
        from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
        from vllm_mlx_mtp.mtp_decoder import MTPConfig
        weights = load_mtp_weights_from_file(MTP_WEIGHTS)
        mtp_head = build_mtp_head(weights, config, norm_shift=True)

        cfg = MTPConfig(batch_verify=True, lazy_draft=True, quantize_head=True)

        # Warmup
        for _ in range(2):
            mtp_generate(model, mtp_head, mx.array(tokenizer.encode("Hello")), 10, eos_set, cfg)

        t, (tokens, stats) = bench(
            lambda: mtp_generate(model, mtp_head, prompt_arr, MAX_TOKENS, eos_set, cfg)
        )
        tps = len(tokens) / t
        match = sum(1 for a, b in zip(results["baseline"]["tokens"], tokens) if a == b)
        print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
        print(f"   {stats}")
        print(f"   Token match vs baseline: {match}/{len(results['baseline']['tokens'])}")
        results["zmlx_mtp"] = {"tps": tps}

        # Reload for clean MTP-only test
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback; traceback.print_exc()
        results["zmlx_mtp"] = {"tps": 0, "error": str(e)}
        model, tokenizer = load(MODEL_NAME)

    # ── 7. MTP Q4 lazy batch only (for comparison) ──────────────
    print("\n7. MTP Q4 lazy batch (no ZMLX)")
    try:
        model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
        with open(model_path / "config.json") as f:
            config = json.load(f)
        from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
        from vllm_mlx_mtp.mtp_decoder import MTPConfig
        weights = load_mtp_weights_from_file(MTP_WEIGHTS)
        mtp_head = build_mtp_head(weights, config, norm_shift=True)

        cfg = MTPConfig(batch_verify=True, lazy_draft=True, quantize_head=True)

        # Warmup
        for _ in range(2):
            mtp_generate(model, mtp_head, mx.array(tokenizer.encode("Hello")), 10, eos_set, cfg)

        t, (tokens, stats) = bench(
            lambda: mtp_generate(model, mtp_head, prompt_arr, MAX_TOKENS, eos_set, cfg)
        )
        tps = len(tokens) / t
        match = sum(1 for a, b in zip(results["baseline"]["tokens"], tokens) if a == b)
        print(f"   {len(tokens)} tokens in {t*1000:.0f}ms = {tps:.1f} tok/s")
        print(f"   {stats}")
        print(f"   Token match vs baseline: {match}/{len(results['baseline']['tokens'])}")
        results["mtp_only"] = {"tps": tps}
    except Exception as e:
        print(f"   FAILED: {e}")
        results["mtp_only"] = {"tps": 0, "error": str(e)}

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    base_tps = results["baseline"]["tps"]
    for name, data in results.items():
        tps = data.get("tps", 0)
        speedup = tps / base_tps if base_tps > 0 else 0
        err = data.get("error", "")
        status = f"FAILED: {err}" if err else f"{tps:.1f} tok/s ({speedup:.2f}x)"
        print(f"  {name:25s}: {status}")


if __name__ == "__main__":
    main()
