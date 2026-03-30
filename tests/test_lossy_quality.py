#!/usr/bin/env python3
"""Quality test: compare zero-replay vs standard MTP output (should be identical)."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")

PROMPTS = [
    "Write a Python function that computes fibonacci numbers:\n```python\ndef fib(n):\n",
    "Explain the theory of relativity in simple terms:\n",
    "Count from 1 to 30: 1, 2, 3, 4, 5,",
    "What are the key differences between TCP and UDP?\n1.",
]


def generate(model, mtp_head, tokenizer, prompt, max_tokens=100, use_zero_replay=False, k=1):
    eos_set = set()
    eid = tokenizer.eos_token_id
    if isinstance(eid, list):
        eos_set = set(eid)
    elif eid is not None:
        eos_set = {eid}

    cfg = MTPConfig(
        num_speculative_tokens=k,
        batch_verify=True,
        lazy_draft=True,
        zero_replay=use_zero_replay,
    )
    dec = MTPDecoder(model, mtp_head, cfg)
    cache = make_prompt_cache(model)
    prompt_arr = mx.array(tokenizer.encode(prompt))
    tokens = list(dec.generate(
        prompt_arr, cache, max_tokens=max_tokens, temperature=0.0, eos_tokens=eos_set,
    ))
    dec.cleanup()
    return tokens


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)

    print("Loading MTP head (Q4)...")
    model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        config = json.load(f)
    weights = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_head = build_mtp_head(weights, config, norm_shift=True)
    quantize_mtp_head(mtp_head, bits=4, group_size=64)

    for prompt in PROMPTS:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:60]}...")

        for k in [1, 2]:
            lossless = generate(model, mtp_head, tokenizer, prompt, use_zero_replay=False, k=k)
            zr_tokens = generate(model, mtp_head, tokenizer, prompt, use_zero_replay=True, k=k)

            lossless_text = tokenizer.decode(lossless)
            zr_text = tokenizer.decode(zr_tokens)

            match = lossless == zr_tokens
            n_diff = sum(1 for a, b in zip(lossless, zr_tokens) if a != b)
            first_diff = -1
            for i, (a, b) in enumerate(zip(lossless, zr_tokens)):
                if a != b:
                    first_diff = i
                    break

            print(f"\n  K={k}: {'EXACT MATCH' if match else f'DIVERGE at token {first_diff}, {n_diff} diffs'}")
            if not match:
                print(f"    Lossless    ({len(lossless)} tok): ...{lossless_text[max(0,first_diff*3-20):first_diff*3+40]}...")
                print(f"    Zero-replay ({len(zr_tokens)} tok): ...{zr_text[max(0,first_diff*3-20):first_diff*3+40]}...")
            else:
                print(f"    Output: {lossless_text[:80]}...")


if __name__ == "__main__":
    main()
