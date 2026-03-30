#!/usr/bin/env python3
"""
TurboQuant Benchmark for Qwen3.5-35B-A3B.

Compares three KV cache strategies at various context lengths:
  1. Baseline: Full-precision KV cache (float16)
  2. Naive KV Quant: mlx-lm's built-in QuantizedKVCache (quantize after prefill)
  3. TurboQuant: Hadamard rotation before quantization (our implementation)

Also tests combined with MTP speculative decoding.

Usage:
    .venv/bin/python benchmark_turboquant.py
"""

import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ["HF_HUB_DISABLE_XET"] = "1"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("turboquant")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

MODEL_PATH = "mlx-community/Qwen3.5-35B-A3B-4bit"
MTP_WEIGHTS_PATH = "mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors"
CONTEXT_LENGTHS = [512, 2048, 4096, 8192, 16384]
MAX_GEN_TOKENS = 128
RUNS_PER_CONFIG = 2  # Best of N

BASE_ARTICLE = """
Artificial intelligence has transformed numerous sectors of the economy in the past decade.
Machine learning models, particularly large language models, have demonstrated remarkable
capabilities in natural language understanding, code generation, and reasoning tasks. The
development of transformer architectures by Vaswani et al. in 2017 marked a pivotal moment,
enabling attention mechanisms to capture long-range dependencies efficiently. Subsequent models
like BERT, GPT-3, and ChatGPT brought these capabilities to mainstream applications, from
automated customer service to creative writing assistance. The rise of mixture-of-experts (MoE)
architectures has further improved the efficiency of these models by activating only a fraction
of parameters per token, achieving larger model capacity without proportional compute cost.
Research continues to advance in areas of model compression, speculative decoding, and
reinforcement learning from human feedback. Companies across industries are racing to integrate
AI capabilities into their products and workflows, while researchers and policymakers grapple
with questions of safety, alignment, and societal impact. The field moves at unprecedented speed,
with new breakthrough papers published weekly and open-source communities contributing significant
innovations. Hardware advances, particularly in GPU and custom AI accelerator design, continue
to push the boundaries of what's computationally feasible, while techniques like quantization
and distillation make large models accessible on consumer hardware. The democratization of AI
through open weights and fine-tuning APIs has created an ecosystem where individuals and small
teams can build specialized applications that would have required large research labs just a
few years ago. As we look ahead, the convergence of multimodal AI, improved reasoning
capabilities, and more efficient training methods promises to unlock even more transformative
applications across science, medicine, education, and creative industries.
"""


def create_long_prompt(tokenizer, target_tokens: int) -> str:
    base_tokens = tokenizer.encode(BASE_ARTICLE)
    n_base = len(base_tokens)

    if n_base >= target_tokens:
        truncated = tokenizer.decode(base_tokens[:target_tokens])
        return f"Summarize the key points of the following text:\n\n{truncated}\n\nSummary:"

    repeats = math.ceil(target_tokens / n_base)
    long_text = (BASE_ARTICLE.strip() + "\n\n") * repeats
    long_tokens = tokenizer.encode(long_text)[:target_tokens]
    long_text = tokenizer.decode(long_tokens)
    return f"Summarize the key points of the following text:\n\n{long_text}\n\nSummary:"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_memory_mb() -> float:
    try:
        return mx.metal.get_active_memory() / 1e6
    except Exception:
        return 0.0


def cache_nbytes(cache_list) -> int:
    total = 0
    for c in cache_list:
        try:
            total += c.nbytes
        except Exception:
            pass
    return total


def get_eos_tokens(tokenizer) -> set:
    eos = set()
    if hasattr(tokenizer, "eos_token_id"):
        if isinstance(tokenizer.eos_token_id, list):
            eos = set(tokenizer.eos_token_id)
        elif tokenizer.eos_token_id is not None:
            eos = {tokenizer.eos_token_id}
    return eos


@dataclass
class Result:
    method: str
    context_length: int
    actual_context: int
    tokens_generated: int
    prefill_s: float
    decode_s: float
    tok_s: float
    memory_mb: float
    kv_cache_mb: float
    output_tokens: List[int] = None
    acceptance_rate: float = 0.0
    tokens_per_step: float = 1.0


# ─────────────────────────────────────────────────────────────
# Generation functions
# ─────────────────────────────────────────────────────────────

def run_generation(
    model, tokenizer, prompt: str, max_tokens: int,
    cache_list: list, eos_tokens: set,
) -> Tuple[List[int], float, float]:
    """
    Core generation loop used by all methods.
    Cache must be pre-created (baseline, naive quant, or turboquant).
    Returns (output_tokens, prefill_s, decode_s).
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache_list)
    mx.eval(logits, *[c.state for c in cache_list if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    # Decode
    tokens_out = []
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    tokens_out.append(token.item())

    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        if tokens_out[-1] in eos_tokens:
            break
        logits = model(token.reshape(1, 1), cache=cache_list)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens_out.append(token.item())

    decode_s = time.perf_counter() - decode_start
    return tokens_out, prefill_s, decode_s


def run_baseline(model, tokenizer, prompt, max_tokens, eos_tokens) -> Result:
    """Baseline: standard FP16 KV cache."""
    cache = make_prompt_cache(model)
    tokens, prefill_s, decode_s = run_generation(
        model, tokenizer, prompt, max_tokens, cache, eos_tokens
    )
    tok_s = len(tokens) / decode_s if decode_s > 0 else 0
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    actual_ctx = len(tokenizer.encode(input_text))

    return Result(
        method="baseline", context_length=0, actual_context=actual_ctx,
        tokens_generated=len(tokens), prefill_s=prefill_s, decode_s=decode_s,
        tok_s=tok_s, memory_mb=get_memory_mb(),
        kv_cache_mb=cache_nbytes(cache) / 1e6,
        output_tokens=tokens,
    )


def run_naive_quant(model, tokenizer, prompt, max_tokens, eos_tokens, bits=4, group_size=64) -> Result:
    """Naive KV quantization: standard mlx-lm QuantizedKVCache (no rotation)."""
    cache = make_prompt_cache(model)

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))
    actual_ctx = input_ids.shape[0]

    # Prefill with full-precision cache
    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    # Convert to quantized cache after prefill
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=group_size, bits=bits)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    kv_mb = cache_nbytes(cache) / 1e6

    # Decode with quantized cache
    tokens_out = []
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    tokens_out.append(token.item())

    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        if tokens_out[-1] in eos_tokens:
            break
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens_out.append(token.item())

    decode_s = time.perf_counter() - decode_start
    tok_s = len(tokens_out) / decode_s if decode_s > 0 else 0

    return Result(
        method="", context_length=0, actual_context=actual_ctx,
        tokens_generated=len(tokens_out), prefill_s=prefill_s, decode_s=decode_s,
        tok_s=tok_s, memory_mb=get_memory_mb(), kv_cache_mb=kv_mb,
        output_tokens=tokens_out,
    )


def run_turboquant(model, tokenizer, prompt, max_tokens, eos_tokens,
                   make_cache_fn, bits=4) -> Result:
    """TurboQuant: Hadamard-rotated KV cache quantization."""
    cache = make_cache_fn()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))
    actual_ctx = input_ids.shape[0]

    # Prefill — TurboQuant cache rotates+quantizes on-the-fly
    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    kv_mb = cache_nbytes(cache) / 1e6

    # Decode
    tokens_out = []
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    tokens_out.append(token.item())

    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        if tokens_out[-1] in eos_tokens:
            break
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens_out.append(token.item())

    decode_s = time.perf_counter() - decode_start
    tok_s = len(tokens_out) / decode_s if decode_s > 0 else 0

    return Result(
        method="", context_length=0, actual_context=actual_ctx,
        tokens_generated=len(tokens_out), prefill_s=prefill_s, decode_s=decode_s,
        tok_s=tok_s, memory_mb=get_memory_mb(), kv_cache_mb=kv_mb,
        output_tokens=tokens_out,
    )


def run_mtp_with_cache(
    model, tokenizer, mtp_head, prompt, max_tokens, eos_tokens,
    cache_list: list,
) -> Result:
    """MTP speculative decoding with a pre-created cache (any type)."""
    from vllm_mlx_mtp.mtp_decoder import MTPDecoder, MTPConfig

    config = MTPConfig(method="mtp", num_speculative_tokens=1,
                       greedy_draft=True, batch_verify=False)
    decoder = MTPDecoder(model, mtp_head, config)

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))
    actual_ctx = input_ids.shape[0]

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache_list)
    hidden = decoder._capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache_list if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    kv_mb = cache_nbytes(cache_list) / 1e6

    # First token
    token_0 = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token_0)
    tokens_out = [token_0.item()]
    decoder.stats.total_tokens += 1
    decoder.stats.prefill_time = prefill_s

    last_hidden = hidden[:, -1:, :]
    current_token = token_0.reshape(1)

    decode_start = time.perf_counter()
    while len(tokens_out) < max_tokens:
        if current_token.item() in eos_tokens:
            break
        accepted, next_token, next_hidden = decoder.step(
            cache=cache_list, current_token=current_token,
            last_hidden=last_hidden, temperature=0.0, eos_tokens=eos_tokens,
        )
        for tid in accepted:
            if len(tokens_out) >= max_tokens:
                break
            tokens_out.append(tid)
            if tid in eos_tokens:
                break
        current_token = next_token
        last_hidden = next_hidden

    decode_s = time.perf_counter() - decode_start
    decoder.cleanup()

    tok_s = len(tokens_out) / decode_s if decode_s > 0 else 0
    return Result(
        method="", context_length=0, actual_context=actual_ctx,
        tokens_generated=len(tokens_out), prefill_s=prefill_s, decode_s=decode_s,
        tok_s=tok_s, memory_mb=get_memory_mb(), kv_cache_mb=kv_mb,
        output_tokens=tokens_out,
        acceptance_rate=decoder.stats.acceptance_rate,
        tokens_per_step=decoder.stats.tokens_per_step,
    )


# ─────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("TurboQuant Benchmark: Qwen3.5-35B-A3B")
    logger.info("=" * 70)

    # Load model
    logger.info(f"Loading {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    logger.info(f"Model loaded. Memory: {get_memory_mb():.0f} MB")

    eos_tokens = get_eos_tokens(tokenizer)

    # Load MTP head
    mtp_head = None
    if Path(MTP_WEIGHTS_PATH).exists():
        logger.info(f"Loading MTP head from {MTP_WEIGHTS_PATH}")
        from vllm_mlx_mtp.mtp_head import build_mtp_head
        import json as _json
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(MODEL_PATH, "config.json")
        with open(config_path) as f:
            config = _json.load(f)
        weights = dict(mx.load(MTP_WEIGHTS_PATH))
        mtp_head = build_mtp_head(weights, config)
        logger.info(f"MTP head loaded. Memory: {get_memory_mb():.0f} MB")
    else:
        logger.warning("MTP weights not found, skipping MTP methods")

    # Setup TurboQuant patches
    from vllm_mlx_mtp.turboquant import patch_model_for_turboquant

    # Warmup
    logger.info("Warmup...")
    _ = run_baseline(model, tokenizer, "Hello world", 16, eos_tokens)
    mx.clear_cache()
    gc.collect()

    all_results = []

    # Define methods
    methods = [
        ("baseline",       {"kv": None, "mtp": False}),
        ("naive_int8",     {"kv": "naive", "bits": 8, "mtp": False}),
        ("naive_int4",     {"kv": "naive", "bits": 4, "mtp": False}),
        ("turbo_int8",     {"kv": "turbo", "bits": 8, "mtp": False}),
        ("turbo_int4",     {"kv": "turbo", "bits": 4, "mtp": False}),
        ("mtp_bf16",       {"kv": None, "mtp": True}),
        ("mtp+naive_int4", {"kv": "naive", "bits": 4, "mtp": True}),
        ("mtp+turbo_int4", {"kv": "turbo", "bits": 4, "mtp": True}),
    ]

    for ctx_len in CONTEXT_LENGTHS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Context length: ~{ctx_len} tokens")
        logger.info(f"{'='*60}")

        prompt = create_long_prompt(tokenizer, ctx_len)

        for method_name, cfg in methods:
            use_mtp = cfg.get("mtp", False)
            kv_type = cfg.get("kv")
            bits = cfg.get("bits", 4)

            if use_mtp and mtp_head is None:
                continue

            logger.info(f"  {method_name}...")

            best = None
            for run_idx in range(RUNS_PER_CONFIG):
                try:
                    mx.clear_cache()
                    gc.collect()

                    # Determine cache and run method
                    if kv_type == "turbo":
                        # Patch model, run, unpatch
                        make_cache_fn, unpatch = patch_model_for_turboquant(
                            model, bits=bits, group_size=64
                        )
                        try:
                            if use_mtp:
                                cache = make_cache_fn()
                                result = run_mtp_with_cache(
                                    model, tokenizer, mtp_head, prompt,
                                    MAX_GEN_TOKENS, eos_tokens, cache,
                                )
                            else:
                                result = run_turboquant(
                                    model, tokenizer, prompt, MAX_GEN_TOKENS,
                                    eos_tokens, make_cache_fn, bits=bits,
                                )
                        finally:
                            unpatch()

                    elif kv_type == "naive":
                        if use_mtp:
                            # MTP with naive quantized cache
                            cache = make_prompt_cache(model)
                            # Prefill first, then quantize, then MTP decode
                            # We need a special flow: prefill -> quantize -> decode
                            result = _run_mtp_naive_quant(
                                model, tokenizer, mtp_head, prompt,
                                MAX_GEN_TOKENS, eos_tokens, bits=bits,
                            )
                        else:
                            result = run_naive_quant(
                                model, tokenizer, prompt, MAX_GEN_TOKENS,
                                eos_tokens, bits=bits,
                            )

                    else:
                        # Baseline or MTP without cache quant
                        if use_mtp:
                            cache = make_prompt_cache(model)
                            result = run_mtp_with_cache(
                                model, tokenizer, mtp_head, prompt,
                                MAX_GEN_TOKENS, eos_tokens, cache,
                            )
                        else:
                            result = run_baseline(
                                model, tokenizer, prompt, MAX_GEN_TOKENS, eos_tokens,
                            )

                    result.method = method_name
                    result.context_length = ctx_len

                    if best is None or result.tok_s > best.tok_s:
                        best = result

                    logger.info(
                        f"    run {run_idx+1}: {result.tok_s:.1f} t/s, "
                        f"prefill={result.prefill_s:.2f}s, "
                        f"KV={result.kv_cache_mb:.1f}MB, "
                        f"mem={result.memory_mb:.0f}MB"
                        + (f", accept={result.acceptance_rate:.0%}" if use_mtp else "")
                    )
                except Exception as e:
                    logger.error(f"    run {run_idx+1} FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            if best:
                all_results.append(best)

        # Quality comparison
        baseline_tokens = None
        for r in all_results:
            if r.context_length == ctx_len and r.method == "baseline":
                baseline_tokens = r.output_tokens
                break

        if baseline_tokens:
            logger.info(f"\n  Quality vs baseline (first {min(64, len(baseline_tokens))} tokens):")
            for r in all_results:
                if r.context_length == ctx_len and r.method != "baseline" and r.output_tokens:
                    n = min(len(baseline_tokens), len(r.output_tokens), 64)
                    matches = sum(1 for a, b in zip(baseline_tokens[:n], r.output_tokens[:n]) if a == b)
                    pct = matches / n * 100 if n > 0 else 0
                    logger.info(f"    {r.method:<20}: {pct:.0f}% match ({matches}/{n})")

    # Save results
    output_path = Path("benchmark_turboquant_results.json")
    serializable = []
    for r in all_results:
        serializable.append({
            "method": r.method,
            "context_length": r.context_length,
            "actual_context": r.actual_context,
            "tokens_generated": r.tokens_generated,
            "prefill_s": round(r.prefill_s, 3),
            "decode_s": round(r.decode_s, 3),
            "tok_s": round(r.tok_s, 1),
            "memory_mb": round(r.memory_mb, 0),
            "kv_cache_mb": round(r.kv_cache_mb, 1),
            "acceptance_rate": round(r.acceptance_rate, 3),
            "tokens_per_step": round(r.tokens_per_step, 2),
        })

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 110)
    print("SUMMARY: TurboQuant vs Naive KV Quantization for Qwen3.5-35B-A3B")
    print("=" * 110)
    print(f"{'Ctx':>6} | {'Method':<20} | {'t/s':>7} | {'vs base':>7} | "
          f"{'Prefill':>8} | {'KV MB':>7} | {'Mem MB':>7} | {'Accept':>6}")
    print("-" * 110)

    for ctx_len in CONTEXT_LENGTHS:
        ctx_results = [r for r in all_results if r.context_length == ctx_len]
        base_ts = next((r.tok_s for r in ctx_results if r.method == "baseline"), None)

        for r in ctx_results:
            ratio = f"{r.tok_s/base_ts:.2f}x" if base_ts else "-"
            accept = f"{r.acceptance_rate:.0%}" if r.acceptance_rate > 0 else "-"
            print(
                f"{r.context_length:>6} | {r.method:<20} | {r.tok_s:>7.1f} | "
                f"{ratio:>7} | {r.prefill_s:>7.2f}s | {r.kv_cache_mb:>7.1f} | "
                f"{r.memory_mb:>7.0f} | {accept:>6}"
            )
        if ctx_results:
            print("-" * 110)

    print("\nDone!")


def _run_mtp_naive_quant(model, tokenizer, mtp_head, prompt, max_tokens, eos_tokens, bits=4, group_size=64):
    """MTP with naive KV cache quantization: prefill -> quantize -> MTP decode."""
    from vllm_mlx_mtp.mtp_decoder import MTPDecoder, MTPConfig

    config = MTPConfig(method="mtp", num_speculative_tokens=1,
                       greedy_draft=True, batch_verify=False)
    decoder = MTPDecoder(model, mtp_head, config)

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))
    actual_ctx = input_ids.shape[0]

    cache = make_prompt_cache(model)

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache)
    hidden = decoder._capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    # Quantize cache after prefill
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=group_size, bits=bits)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    kv_mb = cache_nbytes(cache) / 1e6

    # MTP decode
    token_0 = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token_0)
    tokens_out = [token_0.item()]
    decoder.stats.total_tokens += 1
    decoder.stats.prefill_time = prefill_s

    last_hidden = hidden[:, -1:, :]
    current_token = token_0.reshape(1)

    decode_start = time.perf_counter()
    while len(tokens_out) < max_tokens:
        if current_token.item() in eos_tokens:
            break
        accepted, next_token, next_hidden = decoder.step(
            cache=cache, current_token=current_token,
            last_hidden=last_hidden, temperature=0.0, eos_tokens=eos_tokens,
        )
        for tid in accepted:
            if len(tokens_out) >= max_tokens:
                break
            tokens_out.append(tid)
            if tid in eos_tokens:
                break
        current_token = next_token
        last_hidden = next_hidden

    decode_s = time.perf_counter() - decode_start
    decoder.cleanup()

    tok_s = len(tokens_out) / decode_s if decode_s > 0 else 0
    return Result(
        method="", context_length=0, actual_context=actual_ctx,
        tokens_generated=len(tokens_out), prefill_s=prefill_s, decode_s=decode_s,
        tok_s=tok_s, memory_mb=get_memory_mb(), kv_cache_mb=kv_mb,
        output_tokens=tokens_out,
        acceptance_rate=decoder.stats.acceptance_rate,
        tokens_per_step=decoder.stats.tokens_per_step,
    )


if __name__ == "__main__":
    main()
