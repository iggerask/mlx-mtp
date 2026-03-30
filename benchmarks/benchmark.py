#!/usr/bin/env python3
"""
MTP Benchmark Suite.

Comprehensive benchmarking of MTP speculative decoding across Qwen3.5 model
sizes, prompt categories, and verification modes.

Usage:
    # Single model
    python benchmark.py --model mlx-community/Qwen3.5-4B-4bit

    # Multiple models
    python benchmark.py --model mlx-community/Qwen3.5-4B-4bit mlx-community/Qwen3.5-9B-MLX-4bit

    # Specific categories
    python benchmark.py --model mlx-community/Qwen3.5-4B-4bit --categories code short

    # Include batch verify mode
    python benchmark.py --model mlx-community/Qwen3.5-4B-4bit --batch-verify
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.mtp_head import (
    build_mtp_head,
    detect_mtp_support,
    load_mtp_weights,
    load_mtp_weights_from_file,
)
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.hidden_capture import HiddenStateCapture


# ---------------------------------------------------------------------------
# Benchmark prompts by category
# ---------------------------------------------------------------------------

PROMPTS = {
    "code": [
        "Write a Python function that implements binary search:\n```python\ndef binary_search(arr, target):\n",
        "Write a Python class that implements a simple linked list with insert and delete:\n```python\nclass Node:\n",
        "Write a JavaScript function that deep clones an object:\n```javascript\nfunction deepClone(obj) {\n",
    ],
    "prose": [
        "Explain the theory of relativity in simple terms that anyone could understand:\n",
        "Write a detailed description of a sunset over the Pacific Ocean:\n",
        "Describe the history of the internet from ARPANET to the modern web:\n",
    ],
    "reasoning": [
        "What are the key differences between TCP and UDP? When would you use each?\n",
        "Compare and contrast microservices and monolithic architectures:\n",
        "Explain why quicksort has O(n log n) average time complexity:\n",
    ],
    "short": [
        "The capital of France is",
        "In Python, to read a file you use",
        "1 + 1 =",
    ],
    "repetitive": [
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,",
        "def f(x): return x\ndef f(x): return x\ndef f(x): return x\n",
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. ",
    ],
}

# Map quantized model names to their BF16 source (for MTP weights + config)
BF16_SOURCE_MAP = {
    "mlx-community/Qwen3.5-4B-4bit": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-4B-MLX-4bit": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-9B-4bit": "Qwen/Qwen3.5-9B",
    "mlx-community/Qwen3.5-9B-MLX-4bit": "Qwen/Qwen3.5-9B",
    "mlx-community/Qwen3.5-27B-4bit": "Qwen/Qwen3.5-27B",
    "mlx-community/Qwen3.5-35B-A3B-4bit": "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
}


@dataclass
class BenchResult:
    model: str
    category: str
    prompt_idx: int
    mode: str  # "baseline", "mtp_seq", "mtp_batch"
    tokens: int
    time_s: float
    tok_s: float
    acceptance: float
    tok_per_step: float
    output_match: bool
    prompt_preview: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_eos(tokenizer) -> Set[int]:
    eos_set = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, list):
            eos_set = set(eid)
        elif eid is not None:
            eos_set = {eid}
    return eos_set


def load_config(source_model: str) -> dict:
    model_path = Path(snapshot_download(source_model, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        return json.load(f)


def find_mtp_weights(source_model: str) -> Optional[dict]:
    safe_name = source_model.replace("/", "_")
    local_path = Path(f"mtp_weights/{safe_name}.safetensors")
    if local_path.exists():
        return load_mtp_weights_from_file(local_path)

    # Try loading from original model files
    try:
        model_path = Path(snapshot_download(
            source_model,
            allow_patterns=["*.json", "model*.safetensors", "mtp_weights.safetensors"],
        ))
        return load_mtp_weights(model_path)
    except Exception:
        return None


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


def mtp_generate(decoder, model, prompt_arr, max_tokens, eos_set) -> Tuple[List[int], float]:
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    tokens = list(decoder.generate(
        prompt_arr, cache, max_tokens=max_tokens,
        temperature=0.0, eos_tokens=eos_set,
    ))
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def benchmark_model(
    model_name: str,
    categories: List[str],
    max_tokens: int,
    include_batch: bool,
    num_warmup: int = 2,
) -> List[BenchResult]:
    """Benchmark a single model across all specified categories."""
    source_model = BF16_SOURCE_MAP.get(model_name)
    if not source_model:
        print(f"  Unknown model {model_name}, trying as BF16 source directly")
        source_model = model_name

    # Load config
    config = load_config(source_model)
    if not detect_mtp_support(config):
        print(f"  Model does not support MTP, skipping")
        return []

    # Load MTP weights
    mtp_weights = find_mtp_weights(source_model)
    if not mtp_weights:
        print(f"  No MTP weights found for {source_model}")
        print(f"  Run: python extract_mtp_weights.py --source {source_model}")
        return []

    # Load model
    print(f"  Loading model...")
    model, tokenizer = load(model_name)

    # Build MTP head
    mtp_head = build_mtp_head(mtp_weights, config, norm_shift=True)
    if mtp_head is None:
        print(f"  Failed to build MTP head")
        return []

    eos_set = resolve_eos(tokenizer)

    # Build decoders
    decoder_seq = MTPDecoder(model, mtp_head, MTPConfig(greedy_draft=True, batch_verify=False))
    decoder_batch = None
    if include_batch:
        decoder_batch = MTPDecoder(model, mtp_head, MTPConfig(greedy_draft=True, batch_verify=True))

    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    warmup_arr = mx.array(tokenizer.encode("Hello world"))
    for _ in range(num_warmup):
        baseline_generate(model, warmup_arr, 10, eos_set)
    print(f"  Warmup done")

    results = []

    for cat in categories:
        cat_prompts = PROMPTS.get(cat, [])
        if not cat_prompts:
            print(f"  Unknown category: {cat}")
            continue

        for pidx, prompt in enumerate(cat_prompts):
            prompt_arr = mx.array(tokenizer.encode(prompt))
            preview = prompt[:50].replace("\n", "\\n")

            # --- Baseline ---
            base_tokens, base_time = baseline_generate(model, prompt_arr, max_tokens, eos_set)
            base_tps = len(base_tokens) / base_time if base_time > 0 else 0

            results.append(BenchResult(
                model=model_name, category=cat, prompt_idx=pidx, mode="baseline",
                tokens=len(base_tokens), time_s=base_time, tok_s=base_tps,
                acceptance=0, tok_per_step=1.0, output_match=True,
                prompt_preview=preview,
            ))

            # --- MTP Sequential ---
            decoder_seq.stats.__init__()  # reset stats
            seq_tokens, seq_time = mtp_generate(decoder_seq, model, prompt_arr, max_tokens, eos_set)
            seq_tps = len(seq_tokens) / seq_time if seq_time > 0 else 0
            seq_stats = decoder_seq.stats
            seq_match = seq_tokens == base_tokens

            results.append(BenchResult(
                model=model_name, category=cat, prompt_idx=pidx, mode="mtp_seq",
                tokens=len(seq_tokens), time_s=seq_time, tok_s=seq_tps,
                acceptance=seq_stats.acceptance_rate, tok_per_step=seq_stats.tokens_per_step,
                output_match=seq_match,
                prompt_preview=preview,
            ))

            # --- MTP Batch (optional) ---
            if decoder_batch:
                decoder_batch.stats.__init__()
                batch_tokens, batch_time = mtp_generate(decoder_batch, model, prompt_arr, max_tokens, eos_set)
                batch_tps = len(batch_tokens) / batch_time if batch_time > 0 else 0
                batch_stats = decoder_batch.stats
                batch_match = batch_tokens == base_tokens

                results.append(BenchResult(
                    model=model_name, category=cat, prompt_idx=pidx, mode="mtp_batch",
                    tokens=len(batch_tokens), time_s=batch_time, tok_s=batch_tps,
                    acceptance=batch_stats.acceptance_rate, tok_per_step=batch_stats.tokens_per_step,
                    output_match=batch_match,
                    prompt_preview=preview,
                ))

            # Progress
            seq_ratio = seq_tps / base_tps if base_tps > 0 else 0
            batch_info = ""
            if decoder_batch:
                batch_ratio = batch_tps / base_tps if base_tps > 0 else 0
                batch_info = f" batch={batch_ratio:.2f}x"
            print(f"    [{cat}:{pidx}] base={base_tps:.0f}t/s seq={seq_ratio:.2f}x{batch_info} accept={seq_stats.acceptance_rate:.0%} '{preview}'")

    # Cleanup
    decoder_seq.cleanup()
    if decoder_batch:
        decoder_batch.cleanup()

    return results


def print_table(results: List[BenchResult], models: List[str]):
    """Print a formatted results table."""
    print()
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    for model_name in models:
        model_results = [r for r in results if r.model == model_name]
        if not model_results:
            continue

        short_model = model_name.split("/")[-1]
        print(f"\n{'─' * 100}")
        print(f"Model: {model_name}")
        print(f"{'─' * 100}")

        # Group by category
        categories = sorted(set(r.category for r in model_results))
        modes = sorted(set(r.mode for r in model_results if r.mode != "baseline"))

        # Header
        header = f"{'Category':<14} {'Prompt':<30}"
        header += f" {'Base t/s':>8}"
        for mode in modes:
            label = "Seq" if mode == "mtp_seq" else "Batch"
            header += f" {label + ' t/s':>8} {'ratio':>6} {'accept':>7}"
        header += f" {'match':>5}"
        print(header)
        print("─" * len(header))

        cat_summaries = {}

        for cat in categories:
            cat_base = [r for r in model_results if r.category == cat and r.mode == "baseline"]
            cat_seq = [r for r in model_results if r.category == cat and r.mode == "mtp_seq"]
            cat_batch = [r for r in model_results if r.category == cat and r.mode == "mtp_batch"]

            for i, base in enumerate(cat_base):
                line = f"{cat if i == 0 else '':<14} {base.prompt_preview[:30]:<30}"
                line += f" {base.tok_s:>7.0f}"

                for mode_results in [cat_seq, cat_batch]:
                    if mode_results and i < len(mode_results):
                        mr = mode_results[i]
                        ratio = mr.tok_s / base.tok_s if base.tok_s > 0 else 0
                        line += f" {mr.tok_s:>7.0f} {ratio:>5.2f}x {mr.acceptance:>6.0%}"
                    elif mode_results:
                        line += f" {'':>7} {'':>6} {'':>7}"

                # Match column (from seq)
                if cat_seq and i < len(cat_seq):
                    line += f"   {'Y' if cat_seq[i].output_match else 'N'}"
                print(line)

            # Category summary
            if cat_base:
                avg_base = sum(r.tok_s for r in cat_base) / len(cat_base)
                summary = {"base_tps": avg_base}
                if cat_seq:
                    avg_seq = sum(r.tok_s for r in cat_seq) / len(cat_seq)
                    avg_accept = sum(r.acceptance for r in cat_seq) / len(cat_seq)
                    summary["seq_tps"] = avg_seq
                    summary["seq_ratio"] = avg_seq / avg_base if avg_base > 0 else 0
                    summary["accept"] = avg_accept
                if cat_batch:
                    avg_batch = sum(r.tok_s for r in cat_batch) / len(cat_batch)
                    summary["batch_tps"] = avg_batch
                    summary["batch_ratio"] = avg_batch / avg_base if avg_base > 0 else 0
                cat_summaries[cat] = summary

        # Model summary
        all_base = [r for r in model_results if r.mode == "baseline"]
        all_seq = [r for r in model_results if r.mode == "mtp_seq"]
        all_batch = [r for r in model_results if r.mode == "mtp_batch"]

        if all_base and all_seq:
            print("─" * len(header))
            avg_base = sum(r.tok_s for r in all_base) / len(all_base)
            avg_seq = sum(r.tok_s for r in all_seq) / len(all_seq)
            avg_accept = sum(r.acceptance for r in all_seq) / len(all_seq)
            seq_match = all(r.output_match for r in all_seq)

            line = f"{'AVERAGE':<14} {'':30}"
            line += f" {avg_base:>7.0f}"
            line += f" {avg_seq:>7.0f} {avg_seq/avg_base:>5.2f}x {avg_accept:>6.0%}"
            if all_batch:
                avg_batch = sum(r.tok_s for r in all_batch) / len(all_batch)
                batch_match = all(r.output_match for r in all_batch)
                line += f" {avg_batch:>7.0f} {avg_batch/avg_base:>5.2f}x {'':>7}"
            line += f"   {'Y' if seq_match else 'N'}"
            print(line)

        # Per-category summary
        if len(cat_summaries) > 1:
            print(f"\n  Category breakdown for {short_model}:")
            for cat, s in cat_summaries.items():
                parts = f"    {cat:<14} base={s['base_tps']:.0f}t/s"
                if "seq_tps" in s:
                    parts += f"  seq={s['seq_ratio']:.2f}x"
                if "batch_tps" in s:
                    parts += f"  batch={s['batch_ratio']:.2f}x"
                if "accept" in s:
                    parts += f"  accept={s['accept']:.0%}"
                print(parts)


def save_results(results: List[BenchResult], output_file: str):
    data = [asdict(r) for r in results]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="MTP Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --model mlx-community/Qwen3.5-4B-4bit
  python benchmark.py --model mlx-community/Qwen3.5-4B-4bit mlx-community/Qwen3.5-9B-MLX-4bit
  python benchmark.py --model mlx-community/Qwen3.5-4B-4bit --categories code short --batch-verify
  python benchmark.py --model Qwen/Qwen3.5-4B --max-tokens 256
        """,
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Model name(s) to benchmark",
    )
    parser.add_argument(
        "--categories", nargs="+",
        choices=list(PROMPTS.keys()) + ["all"],
        default=["all"],
        help="Prompt categories (default: all)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--batch-verify", action="store_true",
        help="Include batch verification mode (faster, approximate on GDN models)",
    )
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    args = parser.parse_args()

    cats = list(PROMPTS.keys()) if "all" in args.categories else args.categories

    print(f"MTP Benchmark Suite")
    print(f"Models: {', '.join(args.model)}")
    print(f"Categories: {', '.join(cats)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch verify: {args.batch_verify}")
    print(f"Prompts per category: {len(PROMPTS[cats[0]])}")
    total = len(args.model) * len(cats) * len(PROMPTS[cats[0]])
    print(f"Total benchmark runs: {total}")
    print()

    all_results = []
    for model_name in args.model:
        print(f"\n{'#' * 80}")
        print(f"# Model: {model_name}")
        print(f"{'#' * 80}")

        results = benchmark_model(
            model_name=model_name,
            categories=cats,
            max_tokens=args.max_tokens,
            include_batch=args.batch_verify,
            num_warmup=args.warmup,
        )
        all_results.extend(results)

        # Save incremental results per model
        short = model_name.split("/")[-1]
        save_results(results, f"benchmark_results_{short}.json")

        # Force memory cleanup between models
        import gc
        gc.collect()
        mx.metal.clear_cache()

    print_table(all_results, args.model)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
