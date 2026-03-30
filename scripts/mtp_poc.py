#!/usr/bin/env python3
"""
MTP Speculative Decoding POC.

Standalone demo that loads a Qwen3.5 model and runs MTP speculative decoding,
comparing output and speed against baseline generation.

Usage:
    python mtp_poc.py [--model MODEL] [--max-tokens N] [--prompt TEXT]
"""

import argparse
import json
import time
from pathlib import Path

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


def find_mtp_weights(model_name: str) -> dict:
    """Find and load MTP weights for a model."""
    safe_name = model_name.replace("/", "_")
    local_path = Path(f"mtp_weights/{safe_name}.safetensors")
    if local_path.exists():
        print(f"Loading MTP weights from {local_path}")
        return load_mtp_weights_from_file(local_path)

    model_path = Path(snapshot_download(
        model_name,
        allow_patterns=["*.json", "model*.safetensors", "mtp_weights.safetensors"],
    ))
    weights = load_mtp_weights(model_path)
    if weights:
        return weights

    raise FileNotFoundError(
        f"No MTP weights found for {model_name}. "
        f"Run: python extract_mtp_weights.py --source <original-bf16-model>"
    )


def find_config(model_name: str) -> dict:
    """Load config, preferring the original BF16 model config."""
    bf16_map = {
        "mlx-community/Qwen3.5-4B-4bit": "Qwen/Qwen3.5-4B",
        "mlx-community/Qwen3.5-9B-4bit": "Qwen/Qwen3.5-9B",
        "mlx-community/Qwen3.5-9B-MLX-4bit": "Qwen/Qwen3.5-9B",
    }
    config_model = bf16_map.get(model_name, model_name)
    model_path = Path(snapshot_download(config_model, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        return json.load(f)


def baseline_generate(model, prompt_arr, max_tokens, eos_set):
    cache = make_prompt_cache(model)
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
    return tokens


def main():
    parser = argparse.ArgumentParser(description="MTP Speculative Decoding POC")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Write a Python function to compute fibonacci numbers:\n")
    parser.add_argument("--batch-verify", action="store_true",
                        help="Use batch verification (faster, approximate on GDN models)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch verify: {args.batch_verify}")
    print()

    print("Loading model...")
    model, tokenizer = load(args.model)

    config = find_config(args.model)
    if not detect_mtp_support(config):
        print(f"Model {args.model} does not support MTP")
        return

    # Find MTP weights - try the BF16 source model name
    source_model = config.get("_name_or_path", "Qwen/Qwen3.5-4B")
    mtp_weights = find_mtp_weights(source_model)

    mtp_head = build_mtp_head(mtp_weights, config, norm_shift=True)
    if mtp_head is None:
        print("Failed to build MTP head")
        return

    eos_set = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, list):
            eos_set = set(eid)
        elif eid is not None:
            eos_set = {eid}

    tokens = tokenizer.encode(args.prompt)
    prompt_arr = mx.array(tokens)

    print(f"Prompt: {args.prompt!r}")
    print(f"Prompt tokens: {len(tokens)}")
    print()

    # Warmup
    _ = baseline_generate(model, prompt_arr, max_tokens=5, eos_set=eos_set)

    # Baseline
    print("--- Baseline ---")
    t0 = time.perf_counter()
    base_tokens = baseline_generate(model, prompt_arr, args.max_tokens, eos_set)
    base_time = time.perf_counter() - t0
    base_text = tokenizer.decode(base_tokens)
    base_tps = len(base_tokens) / base_time
    print(f"Output: {base_text!r}")
    print(f"Tokens: {len(base_tokens)} in {base_time:.2f}s = {base_tps:.1f} tok/s")
    print()

    # MTP
    print("--- MTP Speculative Decoding ---")
    mtp_config = MTPConfig(greedy_draft=True, batch_verify=args.batch_verify)
    decoder = MTPDecoder(model, mtp_head, mtp_config)
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    mtp_tokens = list(decoder.generate(
        prompt_arr, cache, max_tokens=args.max_tokens,
        temperature=0.0, eos_tokens=eos_set,
    ))
    mtp_time = time.perf_counter() - t0
    mtp_text = tokenizer.decode(mtp_tokens)
    mtp_tps = len(mtp_tokens) / mtp_time

    print(f"Output: {mtp_text!r}")
    print(f"Tokens: {len(mtp_tokens)} in {mtp_time:.2f}s = {mtp_tps:.1f} tok/s")
    print(f"Stats: {decoder.stats}")
    print()

    print("--- Comparison ---")
    speedup = mtp_tps / base_tps if base_tps > 0 else 0
    match = mtp_tokens == base_tokens
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output match: {match}")
    if not match and not args.batch_verify:
        for i, (a, b) in enumerate(zip(mtp_tokens, base_tokens)):
            if a != b:
                print(f"First diff at pos {i}: MTP={tokenizer.decode([a])!r} vs Base={tokenizer.decode([b])!r}")
                break

    decoder.cleanup()


if __name__ == "__main__":
    main()
