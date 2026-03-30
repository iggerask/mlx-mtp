#!/usr/bin/env python3
"""
Focused TurboQuant Lloyd-Max benchmark.
Tests TQ LM INT4 and INT8 at all context lengths.
"""

import gc
import json
import math
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

os.environ["HF_HUB_DISABLE_XET"] = "1"

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("tq_lm")

MODEL_PATH = "mlx-community/Qwen3.5-35B-A3B-4bit"
CONTEXT_LENGTHS = [512, 2048, 4096, 8192, 16384]
MAX_GEN_TOKENS = 128
RUNS_PER_CONFIG = 2

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
    actual_context: int = 0
    tokens_generated: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    tok_s: float = 0.0
    memory_mb: float = 0.0
    kv_cache_mb: float = 0.0
    output_tokens: List[int] = None


def _prepare_input(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(input_text))
    return input_ids, input_ids.shape[0]


def run_generation(model, tokenizer, prompt, max_tokens, eos_tokens, cache) -> Result:
    """Generic generation with any cache type."""
    input_ids, actual_ctx = _prepare_input(tokenizer, prompt)

    t0 = time.perf_counter()
    logits = model(input_ids[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    prefill_s = time.perf_counter() - t0

    kv_mb = cache_nbytes(cache) / 1e6

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


def main():
    logger.info("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    eos_tokens = get_eos_tokens(tokenizer)
    logger.info(f"Model loaded. Memory: {get_memory_mb():.0f} MB")

    from vllm_mlx_mtp.turboquant import patch_model_for_turboquant

    # Warmup
    logger.info("Warmup...")
    cache = make_prompt_cache(model)
    _ = run_generation(model, tokenizer, "Hello world", 16, eos_tokens, cache)
    mx.clear_cache()
    gc.collect()

    all_results = []

    for ctx_len in CONTEXT_LENGTHS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Context length: ~{ctx_len} tokens")
        logger.info(f"{'='*60}")

        prompt = create_long_prompt(tokenizer, ctx_len)

        # Baseline
        logger.info("  baseline...")
        best = None
        for i in range(RUNS_PER_CONFIG):
            mx.clear_cache(); gc.collect()
            cache = make_prompt_cache(model)
            r = run_generation(model, tokenizer, prompt, MAX_GEN_TOKENS, eos_tokens, cache)
            r.method = "baseline"; r.context_length = ctx_len
            if best is None or r.tok_s > best.tok_s: best = r
            logger.info(f"    run {i+1}: {r.tok_s:.1f} t/s, prefill={r.prefill_s:.2f}s, KV={r.kv_cache_mb:.1f}MB")
        all_results.append(best)

        # TQ LM INT4
        logger.info("  tq_lm_int4...")
        best = None
        for i in range(RUNS_PER_CONFIG):
            mx.clear_cache(); gc.collect()
            make_cache_fn, unpatch = patch_model_for_turboquant(model, bits=4, use_lloyd_max=True)
            try:
                cache = make_cache_fn()
                r = run_generation(model, tokenizer, prompt, MAX_GEN_TOKENS, eos_tokens, cache)
                r.method = "tq_lm_int4"; r.context_length = ctx_len
                if best is None or r.tok_s > best.tok_s: best = r
                logger.info(f"    run {i+1}: {r.tok_s:.1f} t/s, prefill={r.prefill_s:.2f}s, KV={r.kv_cache_mb:.1f}MB")
            except Exception as e:
                logger.error(f"    run {i+1} FAILED: {e}")
                import traceback; traceback.print_exc()
            finally:
                unpatch()
        if best: all_results.append(best)

        # TQ LM INT8
        logger.info("  tq_lm_int8...")
        best = None
        for i in range(RUNS_PER_CONFIG):
            mx.clear_cache(); gc.collect()
            make_cache_fn, unpatch = patch_model_for_turboquant(model, bits=8, use_lloyd_max=True)
            try:
                cache = make_cache_fn()
                r = run_generation(model, tokenizer, prompt, MAX_GEN_TOKENS, eos_tokens, cache)
                r.method = "tq_lm_int8"; r.context_length = ctx_len
                if best is None or r.tok_s > best.tok_s: best = r
                logger.info(f"    run {i+1}: {r.tok_s:.1f} t/s, prefill={r.prefill_s:.2f}s, KV={r.kv_cache_mb:.1f}MB")
            except Exception as e:
                logger.error(f"    run {i+1} FAILED: {e}")
                import traceback; traceback.print_exc()
            finally:
                unpatch()
        if best: all_results.append(best)

        # Quality check
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
                    logger.info(f"    {r.method:<16}: {pct:.0f}% match ({matches}/{n})")

    # Save
    output_path = Path("benchmark_tq_lm_results.json")
    serializable = [{
        "method": r.method, "context_length": r.context_length,
        "actual_context": r.actual_context, "tokens_generated": r.tokens_generated,
        "prefill_s": round(r.prefill_s, 3), "decode_s": round(r.decode_s, 3),
        "tok_s": round(r.tok_s, 1), "memory_mb": round(r.memory_mb, 0),
        "kv_cache_mb": round(r.kv_cache_mb, 1),
    } for r in all_results]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 90)
    print("TurboQuant Lloyd-Max Benchmark")
    print("=" * 90)
    print(f"{'Ctx':>6} | {'Method':<16} | {'t/s':>7} | {'vs base':>7} | {'Prefill':>8} | {'KV MB':>7} | {'Mem MB':>7}")
    print("-" * 90)
    for ctx_len in CONTEXT_LENGTHS:
        ctx_results = [r for r in all_results if r.context_length == ctx_len]
        base_ts = next((r.tok_s for r in ctx_results if r.method == "baseline"), None)
        for r in ctx_results:
            ratio = f"{r.tok_s/base_ts:.2f}x" if base_ts else "-"
            print(f"{r.context_length:>6} | {r.method:<16} | {r.tok_s:>7.1f} | {ratio:>7} | {r.prefill_s:>7.2f}s | {r.kv_cache_mb:>7.1f} | {r.memory_mb:>7.0f}")
        if ctx_results:
            print("-" * 90)
    print("Done!")


if __name__ == "__main__":
    main()
