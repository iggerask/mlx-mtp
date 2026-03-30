#!/usr/bin/env python3
"""Quick validation: compare all MTP paths produce correct output."""

import json
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")
PROMPT = "Write a Python function that checks if a number is prime:\n```python\ndef is_prime(n):\n"
MAX_TOKENS = 50


def baseline_generate(model, prompt_arr, max_tokens, eos_set):
    cache = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
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


def mtp_generate(model, mtp_head, prompt_arr, max_tokens, eos_set, config):
    dec = MTPDecoder(model, mtp_head, config)
    cache = make_prompt_cache(model)
    tokens = list(dec.generate(prompt_arr, cache, max_tokens=max_tokens,
                               temperature=0.0, eos_tokens=eos_set))
    stats = dec.stats
    dec.cleanup()
    return tokens, stats


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}

    print("Loading MTP head...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    weights = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_head = build_mtp_head(weights, config, norm_shift=True)

    prompt_arr = mx.array(tokenizer.encode(PROMPT))

    # Warmup
    print("Warmup...")
    for _ in range(2):
        baseline_generate(model, mx.array(tokenizer.encode("Hello")), 10, eos_set)

    # Baseline
    print("\n1. Baseline (no MTP):")
    t0 = time.perf_counter()
    base_tokens = baseline_generate(model, prompt_arr, MAX_TOKENS, eos_set)
    t1 = time.perf_counter()
    print(f"   {len(base_tokens)} tokens in {(t1-t0)*1000:.0f}ms")
    print(f"   Output: {tokenizer.decode(base_tokens[:30])!r}...")

    configs = [
        ("Sequential (bit-exact)", MTPConfig(batch_verify=False, lazy_draft=False)),
        ("Batch verify (legacy)", MTPConfig(batch_verify=True, lazy_draft=False)),
        ("Lazy batch (NEW DEFAULT)", MTPConfig(batch_verify=True, lazy_draft=True)),
        ("Lazy batch + Q4 head", MTPConfig(batch_verify=True, lazy_draft=True, quantize_head=True)),
    ]

    for name, cfg in configs:
        print(f"\n2. MTP: {name}")
        # Need fresh head for Q4 path
        if cfg.quantize_head:
            w = load_mtp_weights_from_file(MTP_WEIGHTS)
            head = build_mtp_head(w, config, norm_shift=True)
        else:
            head = mtp_head

        t0 = time.perf_counter()
        tokens, stats = mtp_generate(model, head, prompt_arr, MAX_TOKENS, eos_set, cfg)
        t1 = time.perf_counter()

        match = sum(1 for a, b in zip(base_tokens, tokens) if a == b)
        match_pct = match / len(base_tokens) * 100

        print(f"   {len(tokens)} tokens in {(t1-t0)*1000:.0f}ms")
        print(f"   {stats}")
        print(f"   Token match vs baseline: {match}/{len(base_tokens)} ({match_pct:.0f}%)")
        print(f"   Output: {tokenizer.decode(tokens[:30])!r}...")

    print("\nAll paths validated!")


if __name__ == "__main__":
    main()
